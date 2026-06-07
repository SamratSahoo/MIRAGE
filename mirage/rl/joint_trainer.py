from __future__ import annotations

import os
import random
import time
from sys import platform

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from mirage.encoder.load import load_encoder
from mirage.encoder.losses import (align_loss, forward_dyn_loss, info_nce,
                                   inverse_dyn_loss, recon_loss)
from mirage.encoder.models import (ForwardDynamics, InverseDynamics, MaskedStateEncoder,
                                   StateDecoder, build_mlp)
from mirage.envs.warp_antmaze import WarpAntMazeEnv
from mirage.paths import resolve_path
from mirage.utils.checkpoint import Checkpointer
from mirage.utils.running_norm import RunningMeanStd
from mirage.utils.wandb_session import WandbSession

from .agent import Agent, _layer_init
from .policy_input import (PolicyInputBuilder, _slice_proprio, _slice_proprio_contact,
                           _slice_desired_goal, _PROPRIO_27, _PROPRIO_DIM, _GOAL_DIM)


os.environ.setdefault("MUJOCO_GL", "glfw" if platform == "darwin" else "osmesa")

_RAW_DIM = _PROPRIO_DIM + _GOAL_DIM + _GOAL_DIM


def _value_head(obs_dim: int, hidden: int = 256, layernorm: bool = False) -> nn.Sequential:
    def block(d_in: int, d_out: int) -> list:
        layers = [_layer_init(nn.Linear(d_in, d_out))]
        if layernorm:
            layers.append(nn.LayerNorm(d_out))
        layers.append(nn.ReLU())
        return layers
    return nn.Sequential(
        *block(obs_dim, hidden), *block(hidden, hidden),
        _layer_init(nn.Linear(hidden, 1), std=1.0))


class JointPPOTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        env_cfg, ppo_cfg, wandb_cfg = cfg["env"], cfg["ppo"], cfg.get("wandb", {})
        jcfg = cfg.get("joint", {})

        self.exp_name = ppo_cfg["exp_name"]
        self.env_id = env_cfg["env_id"]
        self.seed = int(ppo_cfg["seed"])
        self.num_envs = int(ppo_cfg["num_envs"])
        self.num_steps = int(ppo_cfg["num_steps"])
        self.num_minibatches = int(ppo_cfg["num_minibatches"])
        self.update_epochs = int(ppo_cfg["update_epochs"])
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.learning_rate = float(ppo_cfg["learning_rate"])
        self.anneal_lr = bool(ppo_cfg["anneal_lr"])
        self.gamma = float(ppo_cfg["gamma"])
        self.gamma_int = float(jcfg.get("gamma_int", 0.99))
        self.gae_lambda = float(ppo_cfg["gae_lambda"])
        self.clip_coef = float(ppo_cfg["clip_coef"])
        self.ent_coef = float(ppo_cfg["ent_coef"])
        self.vf_coef = float(ppo_cfg["vf_coef"])
        self.max_grad_norm = float(ppo_cfg["max_grad_norm"])
        self.normalize_obs = bool(ppo_cfg.get("normalize_obs", True))
        self.load_default_checkpoint = bool(ppo_cfg.get("load_default_checkpoint", True))

        self.encoder_init_path = str(jcfg.get("encoder_init_path", "") or "")
        self.train_encoder = bool(jcfg.get("train_encoder", True))
        self.encoder_lr = float(jcfg.get("encoder_lr", 1e-4))
        self.rep_minibatches = int(jcfg.get("rep_minibatches", 4))
        self.intrinsic_coef = float(jcfg.get("intrinsic_coef", 1.0))
        self.rnd_lr = float(jcfg.get("rnd_lr", 1e-4))
        self.rnd_dim = int(jcfg.get("rnd_dim", 128))
        self.w_nce = float(jcfg.get("nce_weight", 1.0))
        self.w_fwd = float(jcfg.get("fwd_weight", 0.5))
        self.w_inv = float(jcfg.get("inv_weight", 1.0))
        self.w_recon = float(jcfg.get("recon_weight", 1.0))
        self.nce_temp = float(jcfg.get("nce_temperature", 0.1))
        self.grad_clip = float(jcfg.get("grad_clip", 10.0))
        self.use_graph_subgoals = bool(jcfg.get("use_graph_subgoals", False))
        self.subgoal_reward_value = float(jcfg.get("subgoal_reward_value", 0.1))
        self._graph_cfg = dict(jcfg.get("graph_planning", {}) or {})
        self.use_latent_critic = bool(jcfg.get("use_latent_critic", False))
        self.vlat_weight = float(jcfg.get("vlat_weight", 0.5))
        self.enc_mask_prob = float(jcfg.get("mask_prob", 0.3))
        self.align_weight = float(jcfg.get("align_weight", 0.1))

        self.run_name = self.exp_name
        self.run_dir = resolve_path(os.path.join("runs", self.run_name))
        os.makedirs(self.run_dir, exist_ok=True)
        self.checkpointer = Checkpointer(self.run_dir, self.exp_name)
        self.writer = None

        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.envs = WarpAntMazeEnv(env_id=self.env_id, num_envs=self.num_envs,
                                   device=self.device, seed=self.seed)
        self.act_dim = int(np.prod(self.envs.single_action_space.shape))

        self._build_modules()

        self.builder = PolicyInputBuilder("latent", ["latent_goal"], unified_encoder=self.encoder)
        self.graph_planner = None
        if self.use_graph_subgoals:
            from mirage.planning.graph_planning.latent_graph_planner import LatentGraphPlanner
            self.graph_planner = LatentGraphPlanner(
                self._graph_cfg, self.device, self.num_envs, "latent")
        self.obs_dim = (4 if self.use_graph_subgoals else 3) * self.latent_dim
        actor_hidden = int(jcfg.get("actor_hidden_dim", 256))
        actor_layers = int(jcfg.get("actor_n_layers", 2))
        actor_ln = bool(jcfg.get("actor_layernorm", False))
        self.actor_prenorm = bool(jcfg.get("actor_prenorm_latent", False))
        self.un_detach_actor = bool(jcfg.get("un_detach_actor", False))
        self.un_detach_encoder_lr = float(jcfg.get("un_detach_encoder_lr", 1e-5))
        self.agent = Agent(obs_dim=self.obs_dim, act_dim=self.act_dim,
                           hidden_dim=actor_hidden, n_layers=actor_layers,
                           layernorm=actor_ln).to(self.device)
        self.value_ext = _value_head(_RAW_DIM, layernorm=actor_ln).to(self.device)
        self.value_int = _value_head(_RAW_DIM, layernorm=actor_ln).to(self.device)
        self.policy_opt = optim.Adam(
            list(self.agent.parameters()) + list(self.value_ext.parameters())
            + list(self.value_int.parameters()), lr=self.learning_rate, eps=1e-5)
        if self.un_detach_actor:
            assert self.train_encoder, "un_detach_actor needs train_encoder=true (aux anchors)"
            self.policy_opt.add_param_group(
                {"params": list(self.encoder.parameters()), "lr": self.un_detach_encoder_lr})
        self.obs_norm = RunningMeanStd(self.obs_dim).to(self.device) if self.normalize_obs else None
        self.raw_norm = RunningMeanStd(_RAW_DIM).to(self.device)
        self.value_lat = _value_head(self.obs_dim).to(self.device) if self.use_latent_critic else None
        if self.value_lat is not None and self.rep_opt is not None:
            self.rep_opt.add_param_group({"params": list(self.value_lat.parameters())})

        self.session = WandbSession(
            run_dir=self.run_dir, project=str(wandb_cfg.get("project", "MIRAGE")),
            entity=wandb_cfg.get("entity"), run_name=self.run_name, config=cfg,
            enabled=bool(wandb_cfg.get("track", False)), sync_tensorboard=True)

        self._global_step = 0
        self._global_episodes = 0
        self._best_eval = None
        self._last_eval_step = 0
        self._first_goal_seen = False
        self._succ_ema = 0.0
        self._succ_best = 0.0
        self._stall = 0

        self.curriculum = bool(jcfg.get("curriculum", True))
        self.curr_thresh = float(jcfg.get("curriculum_success_thresh", 0.6))
        self.curr_patience = int(jcfg.get("curriculum_patience", 40))
        if self.curriculum:
            cd = self.envs._cell_dist
            pos = cd[cd > 1e-6]
            self._curr_min = float(pos.min())
            self._curr_max = float(cd.max())
            n_levels = int(jcfg.get("curriculum_levels", 8))
            self._curr_step = self._curr_min
            self._curr_dist = 0.0
            self.envs.curriculum_max_dist = self._curr_dist
        else:
            self._curr_dist = float("inf")

    def _build_modules(self):
        latent_dim, hidden, nh = 64, 512, 3
        enc_cfg = None
        if self.encoder_init_path:
            self.encoder, enc_cfg, ck = load_encoder(resolve_path(self.encoder_init_path),
                                                     self.device, eval_mode=False)
            latent_dim = int(self.encoder.latent_dim)
        else:
            self.encoder = MaskedStateEncoder(latent_dim=latent_dim, hidden_dim=hidden,
                                              n_hidden=nh, l2_normalize=True).to(self.device)
            ck = None
        self.latent_dim = latent_dim
        dyn_h = int(enc_cfg.get("dyn_hidden_dim", 256)) if enc_cfg else 256
        dyn_nh = int(enc_cfg.get("dyn_n_hidden", 2)) if enc_cfg else 2

        self.fwd = ForwardDynamics(latent_dim, self.act_dim, dyn_h, dyn_nh).to(self.device)
        self.inv = InverseDynamics(latent_dim, self.act_dim, dyn_h, dyn_nh).to(self.device)
        self.decoder = StateDecoder(latent_dim, _PROPRIO_27 + 2, hidden, 2).to(self.device)
        self.state_mean = torch.zeros(_PROPRIO_27 + 2, device=self.device)
        self.state_std = torch.ones(_PROPRIO_27 + 2, device=self.device)
        if ck is not None:
            if "fwd" in ck: self.fwd.load_state_dict(ck["fwd"])
            if "inv" in ck: self.inv.load_state_dict(ck["inv"])
            if "decoder" in ck:
                self.decoder.load_state_dict(ck["decoder"])
                self.state_mean = ck["state_mean"].to(self.device)
                self.state_std = ck["state_std"].to(self.device)

        self.rnd_target = build_mlp(latent_dim, 256, 2, self.rnd_dim).to(self.device)
        self.rnd_pred = build_mlp(latent_dim, 256, 2, self.rnd_dim).to(self.device)
        for p in self.rnd_target.parameters():
            p.requires_grad_(False)
        self.z_norm = RunningMeanStd(latent_dim).to(self.device)
        self.rint_rms = RunningMeanStd(1).to(self.device)

        rep_params = (list(self.encoder.parameters()) + list(self.fwd.parameters())
                      + list(self.inv.parameters()) + list(self.decoder.parameters()))
        self.rep_opt = optim.Adam(rep_params, lr=self.encoder_lr) if self.train_encoder else None
        self.rnd_opt = optim.Adam(self.rnd_pred.parameters(), lr=self.rnd_lr)
        self._rep_params = rep_params

    def _proprio29(self, obs_107, achieved_xy):
        return torch.cat([_slice_proprio(obs_107), achieved_xy], dim=-1)

    def _spatial_latent(self, achieved_xy):
        pad = torch.zeros(achieved_xy.shape[0], _PROPRIO_27,
                          device=achieved_xy.device, dtype=achieved_xy.dtype)
        return self.encoder.encode_goal(torch.cat([pad, achieved_xy], dim=-1))

    def _raw_critic_input(self, obs_107, achieved_xy, *, update_norm):
        x = torch.cat([_slice_proprio_contact(obs_107), achieved_xy,
                       _slice_desired_goal(obs_107)], dim=-1)
        if update_norm:
            self.raw_norm.update(x)
        return self.raw_norm.normalize(x)

    def _spatial_latent_prenorm(self, achieved_xy):
        pad = torch.zeros(achieved_xy.shape[0], _PROPRIO_27,
                          device=achieved_xy.device, dtype=achieved_xy.dtype)
        return self.encoder.encode_goal(torch.cat([pad, achieved_xy], dim=-1), prenorm=True)

    def _policy_input(self, obs_107, achieved_xy, *, update_norm, z_subgoal=None):
        pn = self.actor_prenorm
        with torch.no_grad():
            if pn:
                z_state = self.encoder.encode_full(
                    torch.cat([_slice_proprio(obs_107), achieved_xy], dim=-1), prenorm=True)
                z_ach = self._spatial_latent_prenorm(achieved_xy)
                z_goal = self._spatial_latent_prenorm(_slice_desired_goal(obs_107))
            else:
                z_state = self.encoder.encode_full(
                    torch.cat([_slice_proprio(obs_107), achieved_xy], dim=-1))
                z_ach = self._spatial_latent(achieved_xy)
                z_goal = self._spatial_latent(_slice_desired_goal(obs_107))
            if self.use_graph_subgoals:
                sg = z_subgoal if z_subgoal is not None else z_goal
                pi = torch.cat([z_state, z_ach, z_goal, sg], dim=-1)
            else:
                pi = torch.cat([z_state, z_ach, z_goal], dim=-1)
        if self.obs_norm is not None:
            if update_norm:
                self.obs_norm.update(pi)
            pi = self.obs_norm.normalize(pi)
        return pi

    def _policy_input_grad(self, proprio29, goal_xy):
        pn = self.actor_prenorm
        achieved_xy = proprio29[:, _PROPRIO_27:_PROPRIO_27 + _GOAL_DIM]
        z_state = self.encoder.encode_full(proprio29, prenorm=pn)
        if pn:
            z_ach = self._spatial_latent_prenorm(achieved_xy)
            z_goal = self._spatial_latent_prenorm(goal_xy)
        else:
            z_ach = self._spatial_latent(achieved_xy)
            z_goal = self._spatial_latent(goal_xy)
        pi = torch.cat([z_state, z_ach, z_goal], dim=-1)
        if self.obs_norm is not None:
            pi = self.obs_norm.normalize(pi)
        return pi

    def _z_normalize(self, z):
        return torch.clamp((z - self.z_norm.mean) / torch.sqrt(self.z_norm.var + 1e-8), -5.0, 5.0)

    @torch.no_grad()
    def _rnd_reward(self, z, *, update_stats):
        if update_stats:
            self.z_norm.update(z)
        zc = self._z_normalize(z)
        err = ((self.rnd_pred(zc) - self.rnd_target(zc)) ** 2).mean(-1)
        if update_stats:
            self.rint_rms.update(err.unsqueeze(-1))
        return err / (torch.sqrt(self.rint_rms.var + 1e-8)).squeeze(-1)

    def _aux_update(self, s, s_next, a, z_flat, goal_xy, ret_ext):
        logs = {}
        K = s.shape[0]
        for _ in range(self.rep_minibatches):
            sel = torch.randint(0, K, (min(self.minibatch_size, K),), device=self.device)
            s_b, sn_b, a_b = s[sel], s_next[sel], a[sel]
            if self.rep_opt is not None:
                mp = self.enc_mask_prob if self.use_latent_critic else 0.0
                z1 = self.encoder(s_b, mask_prob=mp); z2 = self.encoder(sn_b, mask_prob=mp)
                nce_l, _ = info_nce(z1, z2, temperature=self.nce_temp)
                z = self.encoder(s_b); z_n = self.encoder(sn_b)
                fwd_l, _ = forward_dyn_loss(self.fwd(z, a_b), z_n.detach())
                inv_l, _ = inverse_dyn_loss(self.inv(z, z_n), a_b)
                s_tgt = (s_b - self.state_mean) / self.state_std
                rec_l, _ = recon_loss(self.decoder(z), s_tgt)
                rep_loss = (self.w_nce * nce_l + self.w_fwd * fwd_l
                            + self.w_inv * inv_l + self.w_recon * rec_l)
                logs = {"rep/nce": nce_l.item(), "rep/fwd": fwd_l.item(),
                        "rep/inv": inv_l.item(), "rep/recon": rec_l.item()}
                if self.use_latent_critic:
                    al_l, _ = align_loss(z, self.encoder.encode_goal(s_b))
                    ach = s_b[:, _PROPRIO_27:_PROPRIO_27 + _GOAL_DIM]
                    pi_lat = torch.cat([z, self._spatial_latent(ach),
                                        self._spatial_latent(goal_xy[sel])], dim=-1)
                    if self.obs_norm is not None:
                        pi_lat = self.obs_norm.normalize(pi_lat)
                    vlat_l = 0.5 * ((self.value_lat(pi_lat).view(-1) - ret_ext[sel]) ** 2).mean()
                    rep_loss = rep_loss + self.align_weight * al_l + self.vlat_weight * vlat_l
                    logs["rep/align"] = al_l.item(); logs["rep/vlat"] = vlat_l.item()
                self.rep_opt.zero_grad(set_to_none=True)
                rep_loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self._rep_params, self.grad_clip)
                self.rep_opt.step()

            zb = z_flat[torch.randint(0, z_flat.shape[0], (self.minibatch_size,), device=self.device)]
            zc = self._z_normalize(zb)
            rnd_loss = ((self.rnd_pred(zc) - self.rnd_target(zc).detach()) ** 2).mean()
            self.rnd_opt.zero_grad(set_to_none=True)
            rnd_loss.backward()
            self.rnd_opt.step()
            logs["rnd/loss"] = rnd_loss.item()
        return logs

    def _ckpt_state(self):
        return {
            "agent": self.agent.state_dict(),
            "value_ext": self.value_ext.state_dict(), "value_int": self.value_int.state_dict(),
            "value_lat": self.value_lat.state_dict() if self.value_lat is not None else None,
            "policy_opt": self.policy_opt.state_dict(),
            "encoder": self.encoder.state_dict(), "fwd": self.fwd.state_dict(),
            "inv": self.inv.state_dict(), "decoder": self.decoder.state_dict(),
            "rnd_pred": self.rnd_pred.state_dict(), "rnd_target": self.rnd_target.state_dict(),
            "rep_opt": self.rep_opt.state_dict() if self.rep_opt else None,
            "rnd_opt": self.rnd_opt.state_dict(),
            "obs_norm": self.obs_norm.state_dict() if self.obs_norm is not None else None,
            "raw_norm": self.raw_norm.state_dict(),
            "z_norm": self.z_norm.state_dict(), "rint_rms": self.rint_rms.state_dict(),
            "global_step": self._global_step, "global_episodes": self._global_episodes,
            "best_eval": self._best_eval, "last_eval_step": self._last_eval_step,
            "first_goal_seen": self._first_goal_seen, "curr_dist": self._curr_dist,
            "succ_ema": self._succ_ema, "config": self.cfg,
        }

    def _load_ckpt(self, st):
        self.agent.load_state_dict(st["agent"])
        self.value_ext.load_state_dict(st["value_ext"]); self.value_int.load_state_dict(st["value_int"])
        if self.value_lat is not None and st.get("value_lat") is not None:
            self.value_lat.load_state_dict(st["value_lat"])
        self.policy_opt.load_state_dict(st["policy_opt"])
        self.encoder.load_state_dict(st["encoder"]); self.fwd.load_state_dict(st["fwd"])
        self.inv.load_state_dict(st["inv"]); self.decoder.load_state_dict(st["decoder"])
        self.rnd_pred.load_state_dict(st["rnd_pred"]); self.rnd_target.load_state_dict(st["rnd_target"])
        if self.rep_opt and st.get("rep_opt"): self.rep_opt.load_state_dict(st["rep_opt"])
        self.rnd_opt.load_state_dict(st["rnd_opt"])
        if self.obs_norm is not None and st.get("obs_norm"): self.obs_norm.load_state_dict(st["obs_norm"])
        self.raw_norm.load_state_dict(st["raw_norm"])
        self.z_norm.load_state_dict(st["z_norm"]); self.rint_rms.load_state_dict(st["rint_rms"])
        self._global_step = int(st["global_step"]); self._global_episodes = int(st["global_episodes"])
        self._best_eval = st.get("best_eval"); self._last_eval_step = int(st.get("last_eval_step", 0))
        self._first_goal_seen = bool(st.get("first_goal_seen", False))
        if self.curriculum and "curr_dist" in st:
            self._curr_dist = float(st["curr_dist"])
            self.envs.curriculum_max_dist = self._curr_dist
        self._succ_ema = float(st.get("succ_ema", 0.0))

    @torch.no_grad()
    def _evaluate(self, global_step, eval_episodes=256):
        self.agent.eval(); self.encoder.eval()
        if self.graph_planner is not None:
            self.graph_planner.reset_state()
        prev_curr = self.envs.curriculum_max_dist
        self.envs.curriculum_max_dist = float("inf")
        rets, goals = [], []
        obs = self.envs.reset(seed=self.seed + global_step)
        achieved = self.envs.achieved_xy.clone()
        eval_done = None
        max_steps = self.envs.max_episode_steps * (eval_episodes // max(1, self.num_envs) + 2)
        c = 0
        while len(rets) < eval_episodes and c < max_steps:
            c += 1
            z_sub = None
            if self.graph_planner is not None:
                z_sub = self.graph_planner.update(obs, achieved, done_mask=eval_done)["z_subgoal"]
            pi = self._policy_input(obs, achieved, update_norm=False, z_subgoal=z_sub)
            obs, _, _, _, infos = self.envs.step(self.agent.actor_mean(pi))
            achieved = infos["achieved_goal"]
            eval_done = infos["done"].bool()
            if bool(infos["done"].any()):
                di = torch.nonzero(infos["done"], as_tuple=False).squeeze(1)
                for r, g in zip(infos["episodic_return"][di].tolist(),
                                infos["episodic_goals_reached"][di].tolist()):
                    rets.append(float(r)); goals.append(int(g))
        self.agent.train(); self.encoder.train()
        self.envs.curriculum_max_dist = prev_curr
        if not rets:
            return
        goals = np.array(goals[:eval_episodes], dtype=np.float32)
        mg = float(goals.mean()); a1 = float((goals >= 1).mean())
        self.writer.add_scalar("eval/goals_reached_mean", mg, global_step)
        self.writer.add_scalar("eval/at_least_one_goal_rate", a1, global_step)
        tag = "  -> new best" if (self._best_eval is None or mg > self._best_eval) else ""
        if self._best_eval is None or mg > self._best_eval:
            self._best_eval = mg
            self.checkpointer.save_best(self._ckpt_state())
        print(f"[eval] step={global_step} goals={mg:.3f} >=1={a1:.1%}{tag}")

    def train(self, total_timesteps, save_model=True, save_freq=0,
              eval_freq=2_000_000, eval_episodes=256):
        num_iterations = total_timesteps // self.batch_size
        if self.load_default_checkpoint:
            st = self.checkpointer.load(map_location=self.device)
            if st is not None:
                self._load_ckpt(st); print(f"[checkpoint] resumed at step={self._global_step}")
            else:
                print("[checkpoint] training from scratch")
        self.session.init()
        self.writer = SummaryWriter(self.run_dir)
        dev = self.device
        S, N, D = self.num_steps, self.num_envs, _PROPRIO_27 + 2

        proprio = torch.zeros((S, N, D), device=dev)
        proprio_next = torch.zeros((S, N, D), device=dev)
        actions = torch.zeros((S, N, self.act_dim), device=dev)
        logprobs = torch.zeros((S, N), device=dev)
        rew_ext = torch.zeros((S, N), device=dev)
        rew_int = torch.zeros((S, N), device=dev)
        dones = torch.zeros((S, N), device=dev)
        val_ext = torch.zeros((S, N), device=dev)
        val_int = torch.zeros((S, N), device=dev)
        rnv_ext = torch.zeros((S, N), device=dev)
        rnv_int = torch.zeros((S, N), device=dev)
        pol_inputs = torch.zeros((S, N, self.obs_dim), device=dev)
        raw_inputs = torch.zeros((S, N, _RAW_DIM), device=dev)
        z_next_states = torch.zeros((S, N, self.latent_dim), device=dev)
        goal_xys = torch.zeros((S, N, _GOAL_DIM), device=dev)

        start_iter = (self._global_step // self.batch_size) + 1
        t0 = time.time()
        next_obs = self.envs.reset(seed=self.seed, stagger=self.curriculum)
        next_achieved = self.envs.achieved_xy.clone()
        next_done = torch.zeros(N, device=dev)

        for iteration in range(start_iter, num_iterations + 1):
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / max(num_iterations, 1)
                self.policy_opt.param_groups[0]["lr"] = frac * self.learning_rate

            iter_ep_returns, iter_ep_goals = [], []
            for step in range(S):
                self._global_step += N
                dones[step] = next_done
                proprio[step] = self._proprio29(next_obs, next_achieved)
                if self.use_latent_critic or self.un_detach_actor:
                    goal_xys[step] = _slice_desired_goal(next_obs)
                z_sub = None; sub_reached = None
                if self.use_graph_subgoals:
                    prev_ptr = self.graph_planner.ptr.clone()
                    gp = self.graph_planner.update(next_obs, next_achieved,
                                                   done_mask=next_done.bool())
                    z_sub = gp["z_subgoal"]
                    sub_reached = (self.graph_planner.ptr > prev_ptr).float()
                pi = self._policy_input(next_obs, next_achieved, update_norm=True, z_subgoal=z_sub)
                pol_inputs[step] = pi
                raw_in = self._raw_critic_input(next_obs, next_achieved, update_norm=True)
                raw_inputs[step] = raw_in
                with torch.no_grad():
                    action, logprob, _, _ = self.agent.get_action_and_value(pi)
                    val_ext[step] = self.value_ext(raw_in).flatten()
                    val_int[step] = self.value_int(raw_in).flatten()
                actions[step] = action; logprobs[step] = logprob

                clipped = torch.clamp(action, self.envs.action_low, self.envs.action_high)
                next_obs, reward, term, trunc, infos = self.envs.step(clipped)
                next_achieved = infos["achieved_goal"]
                proprio_next[step] = self._proprio29(infos["final_obs"], infos["achieved_goal"])
                rew_ext[step] = reward
                if sub_reached is not None:
                    rew_ext[step] = rew_ext[step] + self.subgoal_reward_value * sub_reached
                done = term | trunc
                with torch.no_grad():
                    z_next_states[step] = self._spatial_latent(infos["achieved_goal"])
                    raw_f = self._raw_critic_input(infos["final_obs"], infos["achieved_goal"],
                                                   update_norm=False)
                    rnv_ext[step] = self.value_ext(raw_f).flatten() * (1.0 - term.float())
                    rnv_int[step] = self.value_int(raw_f).flatten()
                next_done = done.float()
                if bool(done.any()):
                    di = done.bool()
                    iter_ep_returns += infos["episodic_return"][di].tolist()
                    iter_ep_goals += infos["episodic_goals_reached"][di].tolist()

            for step in range(S):
                rew_int[step] = self._rnd_reward(z_next_states[step], update_stats=True)

            if rew_ext.abs().sum() > 0:
                self._first_goal_seen = True

            with torch.no_grad():
                adv_ext = torch.zeros_like(rew_ext); adv_int = torch.zeros_like(rew_int)
                lae, lai = 0, 0
                for t in reversed(range(S)):
                    nnt = 1.0 - (next_done if t == S - 1 else dones[t + 1])
                    d_e = rew_ext[t] + self.gamma * rnv_ext[t] - val_ext[t]
                    adv_ext[t] = lae = d_e + self.gamma * self.gae_lambda * nnt * lae
                    d_i = rew_int[t] + self.gamma_int * rnv_int[t] - val_int[t]
                    adv_int[t] = lai = d_i + self.gamma_int * self.gae_lambda * nnt * lai
                ret_ext = adv_ext + val_ext
                ret_int = adv_int + val_int
                if self.curriculum or self._first_goal_seen:
                    a_e = (adv_ext - adv_ext.mean()) / (adv_ext.std() + 1e-8)
                else:
                    a_e = torch.zeros_like(adv_ext)
                a_i = (adv_int - adv_int.mean()) / (adv_int.std() + 1e-8)
                advantages = a_e + self.intrinsic_coef * a_i

            b_obs = pol_inputs.reshape(-1, self.obs_dim)
            b_raw = raw_inputs.reshape(-1, _RAW_DIM)
            b_proprio = proprio.reshape(-1, D)
            b_goal = goal_xys.reshape(-1, _GOAL_DIM)
            b_logp = logprobs.reshape(-1); b_act = actions.reshape(-1, self.act_dim)
            b_adv = advantages.reshape(-1)
            b_rext = ret_ext.reshape(-1); b_rint = ret_int.reshape(-1)
            inds = np.arange(self.batch_size)
            pg_loss = v_loss = ent_loss = approx_kl = torch.tensor(0.0)
            for _ in range(self.update_epochs):
                np.random.shuffle(inds)
                for s0 in range(0, self.batch_size, self.minibatch_size):
                    mb = inds[s0:s0 + self.minibatch_size]
                    if self.un_detach_actor:
                        pi_mb = self._policy_input_grad(b_proprio[mb], b_goal[mb])
                    else:
                        pi_mb = b_obs[mb]
                    _, newlogp, entropy, _ = self.agent.get_action_and_value(pi_mb, b_act[mb])
                    nv_e = self.value_ext(b_raw[mb]).view(-1)
                    nv_i = self.value_int(b_raw[mb]).view(-1)
                    logratio = newlogp - b_logp[mb]; ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                    mb_adv = b_adv[mb]
                    pg1 = -mb_adv * ratio
                    pg2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg1, pg2).mean()
                    v_e = 0.5 * ((nv_e - b_rext[mb]) ** 2).mean()
                    v_i = 0.5 * ((nv_i - b_rint[mb]) ** 2).mean()
                    v_loss = v_e + v_i
                    ent_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * ent_loss + self.vf_coef * v_loss
                    self.policy_opt.zero_grad(); loss.backward()
                    clip_params = (list(self.agent.parameters()) + list(self.value_ext.parameters())
                                   + list(self.value_int.parameters()))
                    if self.un_detach_actor:
                        clip_params += list(self.encoder.parameters())
                    nn.utils.clip_grad_norm_(clip_params, self.max_grad_norm)
                    self.policy_opt.step()

            rep_logs = self._aux_update(
                proprio.reshape(-1, D), proprio_next.reshape(-1, D),
                actions.reshape(-1, self.act_dim), z_next_states.reshape(-1, self.latent_dim),
                goal_xys.reshape(-1, _GOAL_DIM), ret_ext.reshape(-1))

            self._global_episodes += len(iter_ep_returns)
            if iter_ep_goals:
                succ = float((np.array(iter_ep_goals, dtype=np.float32) >= 1).mean())
                self._succ_ema = 0.95 * self._succ_ema + 0.05 * succ
                if self.curriculum and self._curr_dist < self._curr_max:
                    if self._succ_ema > self._succ_best + 0.005:
                        self._succ_best = self._succ_ema; self._stall = 0
                    else:
                        self._stall += 1
                    if self._succ_ema > self.curr_thresh or self._stall > self.curr_patience:
                        self._curr_dist = min(self._curr_max, self._curr_dist + self._curr_step)
                        self.envs.curriculum_max_dist = self._curr_dist
                        self._succ_ema = 0.0; self._succ_best = 0.0; self._stall = 0
            sps = int(self._global_step / max(time.time() - t0, 1e-8))
            yt = ret_ext.reshape(-1); yp = val_ext.reshape(-1)
            var_y = yt.var()
            ev = float("nan") if var_y == 0 else float(1 - (yt - yp).var() / var_y)
            ri = rew_int.reshape(-1)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self._global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), self._global_step)
            self.writer.add_scalar("losses/entropy", ent_loss.item(), self._global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self._global_step)
            self.writer.add_scalar("losses/explained_variance_ext", ev, self._global_step)
            self.writer.add_scalar("intrinsic/rnd_reward_mean", ri.mean().item(), self._global_step)
            self.writer.add_scalar("intrinsic/rnd_reward_std", ri.std().item(), self._global_step)
            self.writer.add_scalar("intrinsic/first_goal_seen", float(self._first_goal_seen), self._global_step)
            self.writer.add_scalar("rollout/reward_ext_mean", rew_ext.mean().item(), self._global_step)
            self.writer.add_scalar("rollout/action_abs_mean", actions.abs().mean().item(), self._global_step)
            for k, v in rep_logs.items():
                self.writer.add_scalar(k, v, self._global_step)
            if iter_ep_goals:
                g = np.array(iter_ep_goals, dtype=np.float32)
                self.writer.add_scalar("charts/goals_reached_mean", float(g.mean()), self._global_step)
                self.writer.add_scalar("charts/at_least_one_goal_rate", float((g >= 1).mean()), self._global_step)
            self.writer.add_scalar("charts/SPS", sps, self._global_step)
            if self.curriculum:
                self.writer.add_scalar("curriculum/max_dist", self._curr_dist, self._global_step)
                self.writer.add_scalar("curriculum/success_ema", self._succ_ema, self._global_step)
            if self.graph_planner is not None:
                self.writer.add_scalar("graph/active_subgoal_frac",
                                       float((self.graph_planner.seq_len > 1).float().mean()),
                                       self._global_step)

            if save_model and self._global_step - self._last_eval_step >= eval_freq:
                self._last_eval_step = self._global_step
                self._evaluate(self._global_step, eval_episodes)
                self.checkpointer.save_latest(self._ckpt_state())
                next_obs = self.envs.reset(seed=self.seed + self._global_step,
                                           stagger=self.curriculum)
                next_achieved = self.envs.achieved_xy.clone()
                next_done = torch.zeros(N, device=dev)
                if self.graph_planner is not None:
                    self.graph_planner.reset_state()

        if save_model:
            self._evaluate(self._global_step, eval_episodes)
            self.checkpointer.save_latest(self._ckpt_state())
        self.envs.close(); self.writer.close(); self.session.finish()
