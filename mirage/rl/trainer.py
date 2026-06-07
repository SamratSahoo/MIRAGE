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

from mirage.encoder.load import is_masked, load_encoder
from mirage.envs.warp_antmaze import WarpAntMazeEnv
from mirage.paths import resolve_path
from mirage.planning.motion_planning.cell_graph_planner import CellGraphPlanner
from mirage.planning.graph_planning.latent_graph_planner import LatentGraphPlanner
from mirage.utils.checkpoint import Checkpointer
from mirage.utils.running_norm import RunningMeanStd
from mirage.utils.wandb_session import WandbSession

from .agent import Agent
from .policy_input import PolicyInputBuilder


os.environ.setdefault("MUJOCO_GL", "glfw" if platform == "darwin" else "osmesa")


class PPOTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        env_cfg = cfg["env"]
        ppo_cfg = cfg["ppo"]
        wandb_cfg = cfg.get("wandb", {})

        self.env_type = env_cfg["env_type"]
        self.env_id = env_cfg["env_id"]
        self.exp_name = ppo_cfg["exp_name"]
        self.seed = int(ppo_cfg["seed"])
        self.num_envs = int(ppo_cfg["num_envs"])
        self.num_steps = int(ppo_cfg["num_steps"])
        self.num_minibatches = int(ppo_cfg["num_minibatches"])
        self.update_epochs = int(ppo_cfg["update_epochs"])
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.learning_rate = float(ppo_cfg["learning_rate"])
        self.anneal_lr = bool(ppo_cfg["anneal_lr"])
        self.target_kl = ppo_cfg["target_kl"]
        self.gamma = float(ppo_cfg["gamma"])
        self.gae_lambda = float(ppo_cfg["gae_lambda"])
        self.norm_adv = bool(ppo_cfg["norm_adv"])
        self.clip_coef = float(ppo_cfg["clip_coef"])
        self.clip_vloss = bool(ppo_cfg["clip_vloss"])
        self.ent_coef = float(ppo_cfg["ent_coef"])
        self.vf_coef = float(ppo_cfg["vf_coef"])
        self.max_grad_norm = float(ppo_cfg["max_grad_norm"])
        self.load_default_checkpoint = bool(ppo_cfg.get("load_default_checkpoint", True))
        self.normalize_obs = bool(ppo_cfg.get("normalize_obs", False))

        self.state_mode = str(ppo_cfg.get("state_mode", "raw"))
        self.unified_encoder_path = str(ppo_cfg.get("unified_encoder_path", "") or "")
        self.goal_conditioning = list(ppo_cfg.get("goal_conditioning", ["goal"]))
        self.subgoal_mode = str(ppo_cfg.get("subgoal_mode", "") or "")
        self.subgoal_radius = float(ppo_cfg.get("subgoal_radius", 1.5))

        self.run_name = self.exp_name
        self.run_dir = resolve_path(os.path.join("runs", self.run_name))
        os.makedirs(self.run_dir, exist_ok=True)
        self.checkpointer = Checkpointer(self.run_dir, self.exp_name)
        self.writer = None

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.graph_planning = self.subgoal_mode == "graph_planning"
        self.gp_cfg = dict(ppo_cfg.get("graph_planning", {}) or {})
        self.use_unified = bool(self.unified_encoder_path) and not self.graph_planning
        self.unified_encoder = None
        self.planner = None
        self.graph_planner = None
        if not self.graph_planning:
            needs_planner = self.subgoal_mode == "motion_planning" or any(
                c in ("subgoal", "latent_subgoal") for c in self.goal_conditioning
            )
            if needs_planner:
                if self.subgoal_mode != "motion_planning":
                    raise ValueError(f"unsupported subgoal_mode {self.subgoal_mode!r}")
                self.planner = CellGraphPlanner(self.env_id, device=self.device,
                                                subgoal_radius=self.subgoal_radius)

        self.envs = WarpAntMazeEnv(
            env_id=self.env_id,
            num_envs=self.num_envs,
            device=self.device,
            seed=self.seed,
            subgoal_planner=self.planner,
            subgoal_radius=self.subgoal_radius,
        )

        if self.use_unified:
            self.unified_encoder, _uecfg, _ = load_encoder(
                self.unified_encoder_path, self.device, eval_mode=True)
            if not is_masked(_uecfg):
                raise ValueError(
                    f"unified_encoder_path must be a masked (dual-input) encoder; "
                    f"'{self.unified_encoder_path}' is not masked")

        if self.graph_planning:
            self.graph_planner = LatentGraphPlanner(
                self.gp_cfg, self.device, self.num_envs, self.state_mode)

        self.builder = PolicyInputBuilder(
            state_mode=self.state_mode,
            goal_conditioning=self.goal_conditioning,
            unified_encoder=self.unified_encoder,
            graph_planning=self.graph_planning,
            planning_latent_dim=(self.graph_planner.latent_dim
                                 if self.graph_planner is not None else None),
        )
        self.obs_dim = self.builder.obs_dim
        self.act_dim = int(np.prod(self.envs.single_action_space.shape))

        self.agent = Agent(obs_dim=self.obs_dim, act_dim=self.act_dim).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
        self.obs_norm = (RunningMeanStd(self.obs_dim).to(self.device)
                         if self.normalize_obs else None)

        self._imagine_act_fn = (
            (lambda z, zk, gxy: self.agent.actor_mean(
                self.builder.assemble_from_latents(z, zk, zk, goal_xy=gxy)))
            if self.graph_planning else None
        )

        self.hyperparams = {
            **{f"env.{k}": v for k, v in env_cfg.items()},
            **{f"ppo.{k}": v for k, v in ppo_cfg.items() if not isinstance(v, dict)},
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "batch_size": self.batch_size,
            "minibatch_size": self.minibatch_size,
        }
        self.session = WandbSession(
            run_dir=self.run_dir,
            project=str(wandb_cfg.get("project", "MIRAGE")),
            entity=wandb_cfg.get("entity"),
            run_name=self.run_name,
            config=cfg,
            enabled=bool(wandb_cfg.get("track", False)),
            sync_tensorboard=True,
        )

        self._best_eval_return = None
        self._global_step = 0
        self._global_episodes = 0
        self._last_save_step = 0
        self._last_eval_step = 0

    def _assemble(self, obs, achieved, subgoal=None, *, update_norm: bool = False, **kw):
        pi = self.builder.assemble(obs, achieved, subgoal, **kw)
        if self.obs_norm is not None:
            if update_norm:
                self.obs_norm.update(pi)
            pi = self.obs_norm.normalize(pi)
        return pi

    def _capture_rng(self) -> dict:
        return {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

    def _restore_rng(self, rng: dict) -> None:
        torch.set_rng_state(rng["torch"].to("cpu", dtype=torch.uint8))
        cuda_state = rng.get("torch_cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.to("cpu", dtype=torch.uint8) for s in cuda_state])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])

    def _checkpoint_state(self) -> dict:
        state = {
            "agent": self.agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_norm": self.obs_norm.state_dict() if self.obs_norm is not None else None,
            "global_step": self._global_step,
            "global_episodes": self._global_episodes,
            "best_eval_return": self._best_eval_return,
            "last_save_step": self._last_save_step,
            "last_eval_step": self._last_eval_step,
            "rng": self._capture_rng(),
            "config": self.cfg,
        }
        return state

    def _load_checkpoint(self, state: dict) -> None:
        self.agent.load_state_dict(state["agent"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.obs_norm is not None and state.get("obs_norm") is not None:
            self.obs_norm.load_state_dict(state["obs_norm"])
        self._global_step = int(state["global_step"])
        self._global_episodes = int(state["global_episodes"])
        self._best_eval_return = state.get("best_eval_return")
        self._last_save_step = int(state.get("last_save_step", self._global_step))
        self._last_eval_step = int(state.get("last_eval_step", self._global_step))
        if "rng" in state:
            self._restore_rng(state["rng"])

    def _evaluate_agent(self, global_step: int, eval_episodes: int = 256) -> None:
        self.checkpointer.save_latest(self._checkpoint_state())
        self.agent.eval()
        eval_returns, eval_goals, eval_lengths = [], [], []
        obs = self.envs.reset(seed=self.seed + global_step)
        achieved = self.envs.achieved_xy.clone()
        subgoal = self.envs.subgoal_xy.clone() if self.planner is not None else None
        eval_done = None
        if self.graph_planner is not None:
            self.graph_planner.reset_state()
            eval_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        max_total_steps = self.envs.max_episode_steps * (eval_episodes // max(1, self.num_envs) + 2)
        step_count = 0
        with torch.no_grad():
            while len(eval_returns) < eval_episodes and step_count < max_total_steps:
                step_count += 1
                if self.graph_planner is not None:
                    gp = self.graph_planner.update(obs, achieved, done_mask=eval_done,
                                                   act_fn=self._imagine_act_fn)
                    policy_input = self._assemble(
                        obs, achieved, z_state=gp["z_state"], z_goal=gp["z_goal"],
                        z_subgoal=gp["z_subgoal"], update_norm=False)
                else:
                    policy_input = self._assemble(obs, achieved, subgoal, update_norm=False)
                action = self.agent.actor_mean(policy_input)
                clipped = torch.clamp(action, self.envs.action_low, self.envs.action_high)
                obs, _, _, _, infos = self.envs.step(clipped)
                achieved = infos["achieved_goal"]
                if self.planner is not None:
                    subgoal = infos["subgoal"]
                done = infos["done"]
                if self.graph_planner is not None:
                    eval_done = done.bool()
                if bool(done.any()):
                    done_idx = torch.nonzero(done, as_tuple=False).squeeze(1)
                    ep_ret = infos["episodic_return"][done_idx].cpu().numpy()
                    ep_g = infos["episodic_goals_reached"][done_idx].cpu().numpy()
                    ep_l = infos["episodic_length"][done_idx].cpu().numpy()
                    for r, g, l in zip(ep_ret.tolist(), ep_g.tolist(), ep_l.tolist()):
                        eval_returns.append(float(r))
                        eval_goals.append(int(g))
                        eval_lengths.append(int(l))
        self.agent.train()
        if not eval_returns:
            return
        rets = np.array(eval_returns[:eval_episodes], dtype=np.float32)
        goals = np.array(eval_goals[:eval_episodes], dtype=np.float32)
        lengths = np.array(eval_lengths[:eval_episodes], dtype=np.float32)
        mean_return = float(rets.mean())
        mean_goals = float(goals.mean())
        at_least_one = float((goals >= 1).mean())
        self.writer.add_scalar("eval/episodic_return_mean", mean_return, global_step)
        self.writer.add_scalar("eval/episodic_return_std", float(rets.std()), global_step)
        self.writer.add_scalar("eval/episodic_length_mean", float(lengths.mean()), global_step)
        self.writer.add_scalar("eval/goals_reached_mean", mean_goals, global_step)
        self.writer.add_scalar("eval/goals_reached_std", float(goals.std()), global_step)
        self.writer.add_scalar("eval/goals_reached_max", float(goals.max()), global_step)
        self.writer.add_scalar("eval/at_least_one_goal_rate", at_least_one, global_step)
        if self._best_eval_return is None or mean_goals > self._best_eval_return:
            self._best_eval_return = mean_goals
            self.checkpointer.save_best(self._checkpoint_state())
            print(f"[eval] step={global_step} goals={mean_goals:.2f} >=1={at_least_one:.1%} "
                  f"return={mean_return:.3f}  -> new best")
        else:
            print(f"[eval] step={global_step} goals={mean_goals:.2f} >=1={at_least_one:.1%} "
                  f"return={mean_return:.3f}  (best_goals={self._best_eval_return:.2f})")
        self.writer.add_scalar("eval/best_goals_reached_mean", self._best_eval_return, global_step)

    def train(self, total_timesteps: int, save_model: bool = True, save_freq: int = 0,
              eval_freq: int = 2_000_000, eval_episodes: int = 256):
        num_iterations = total_timesteps // self.batch_size

        if self.load_default_checkpoint:
            state = self.checkpointer.load(map_location=self.device)
            if state is not None:
                self._load_checkpoint(state)
                print(f"[checkpoint] resumed at global_step={self._global_step} "
                      f"best_eval_return={self._best_eval_return}")
            else:
                print("[checkpoint] no checkpoint found; training from scratch")
        else:
            print("[checkpoint] load_default_checkpoint=false; training from scratch")

        self.session.init()
        self.writer = SummaryWriter(self.run_dir)
        num_params = sum(p.numel() for p in self.agent.parameters())
        self.writer.add_scalar("charts/num_parameters", num_params, 0)
        self.writer.add_scalar("charts/num_iterations", num_iterations, 0)

        device = self.device
        policy_inputs = torch.zeros((self.num_steps, self.num_envs, self.obs_dim), device=device)
        actions = torch.zeros((self.num_steps, self.num_envs, self.act_dim), device=device)
        logprobs = torch.zeros((self.num_steps, self.num_envs), device=device)
        rewards = torch.zeros((self.num_steps, self.num_envs), device=device)
        dones = torch.zeros((self.num_steps, self.num_envs), device=device)
        values = torch.zeros((self.num_steps, self.num_envs), device=device)
        real_next_values = torch.zeros((self.num_steps, self.num_envs), device=device)

        action_low = self.envs.action_low
        action_high = self.envs.action_high

        start_iter = (self._global_step // self.batch_size) + 1
        start_time = time.time()
        next_obs = self.envs.reset(seed=self.seed)
        next_achieved = self.envs.achieved_xy.clone()
        next_subgoal = self.envs.subgoal_xy.clone()
        next_done = torch.zeros(self.num_envs, device=device)

        for iteration in range(start_iter, num_iterations + 1):
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / max(num_iterations, 1)
                self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

            iter_ep_returns, iter_ep_lengths, iter_ep_goals = [], [], []
            iter_terminations, iter_truncations = 0, 0
            action_clip_fracs = []
            rollout_start = time.time()

            for step in range(self.num_steps):
                self._global_step += self.num_envs
                dones[step] = next_done

                if self.graph_planner is not None:
                    gp = self.graph_planner.update(
                        next_obs, next_achieved, done_mask=next_done.bool(),
                        act_fn=self._imagine_act_fn)
                    pi = self._assemble(
                        next_obs, next_achieved,
                        z_state=gp["z_state"], z_goal=gp["z_goal"], z_subgoal=gp["z_subgoal"],
                        update_norm=True)
                else:
                    pi = self._assemble(next_obs, next_achieved, next_subgoal, update_norm=True)
                policy_inputs[step] = pi
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(pi)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                clipped_action = torch.clamp(action, action_low, action_high)
                action_clip_fracs.append((clipped_action != action).float().mean().item())

                next_obs, reward, terminations, truncations, infos = self.envs.step(clipped_action)
                next_achieved = infos["achieved_goal"]
                next_subgoal = infos["subgoal"]
                done = terminations | truncations
                iter_terminations += int(terminations.sum().item())
                iter_truncations += int((truncations & ~terminations).sum().item())
                rewards[step] = reward

                with torch.no_grad():
                    if self.graph_planner is not None:
                        zs_f, zg_f = self.graph_planner.encode(
                            infos["final_obs"], infos["achieved_goal"])
                        final_pi = self._assemble(
                            infos["final_obs"], infos["achieved_goal"],
                            z_state=zs_f, z_goal=zg_f, z_subgoal=gp["z_subgoal"],
                            update_norm=False)
                    else:
                        final_pi = self._assemble(
                            infos["final_obs"], infos["achieved_goal"], infos["subgoal"],
                            update_norm=False,
                        )
                    real_next_values[step] = (
                        self.agent.get_value(final_pi).flatten()
                        * (1.0 - terminations.float())
                    )
                next_done = done.float()

                if bool(done.any()):
                    ep_returns = infos["episodic_return"][done]
                    ep_lengths = infos["episodic_length"][done]
                    ep_goals = infos["episodic_goals_reached"][done]
                    for ep_r, ep_l, ep_g in zip(ep_returns.tolist(), ep_lengths.tolist(), ep_goals.tolist()):
                        self.writer.add_scalar("charts/episodic_return", ep_r, self._global_step)
                        self.writer.add_scalar("charts/episodic_length", ep_l, self._global_step)
                        self.writer.add_scalar("charts/episodic_goals_reached", ep_g, self._global_step)
                        iter_ep_returns.append(float(ep_r))
                        iter_ep_lengths.append(float(ep_l))
                        iter_ep_goals.append(int(ep_g))

            rollout_time = time.time() - rollout_start
            self._global_episodes += len(iter_ep_returns)
            update_start = time.time()

            with torch.no_grad():
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    nextnonterminal = 1.0 - (next_done if t == self.num_steps - 1 else dones[t + 1])
                    delta = rewards[t] + self.gamma * real_next_values[t] - values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            b_obs = policy_inputs.reshape(-1, self.obs_dim)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1, self.act_dim)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(self.batch_size)
            clipfracs = []
            grad_norms = []
            ratio_mins, ratio_maxs = [], []
            pg_losses, v_losses, entropy_losses, total_losses = [], [], [], []
            epochs_run = 0
            old_approx_kl = torch.tensor(0.0)
            approx_kl = torch.tensor(0.0)
            entropy_loss = torch.tensor(0.0)
            pg_loss = torch.tensor(0.0)
            v_loss = torch.tensor(0.0)
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                epoch_kls = []
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
                        epoch_kls.append(approx_kl.item())
                        ratio_mins.append(ratio.min().item())
                        ratio_maxs.append(ratio.max().item())

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    grad_norms.append(float(grad_norm))
                    pg_losses.append(pg_loss.item())
                    v_losses.append(v_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    total_losses.append(loss.item())

                epochs_run += 1
                if self.target_kl is not None and np.mean(epoch_kls) > self.target_kl:
                    break

            update_time = time.time() - update_start

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = float(np.var(y_true))
            explained_var = float("nan") if var_y == 0 else 1 - float(np.var(y_true - y_pred)) / var_y

            now = time.time()
            sps = int(self._global_step / max(now - start_time, 1e-8))
            iteration_time = now - rollout_start
            iteration_fps = self.batch_size / max(iteration_time, 1e-8)
            rollout_fps = self.batch_size / max(rollout_time, 1e-8)
            update_fps = self.batch_size / max(update_time, 1e-8)

            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self._global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), self._global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self._global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), self._global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self._global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self._global_step)
            self.writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), self._global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, self._global_step)
            self.writer.add_scalar("losses/grad_norm", float(np.mean(grad_norms)), self._global_step)
            self.writer.add_scalar("charts/epochs_run", epochs_run, self._global_step)
            self.writer.add_scalar("charts/SPS", sps, self._global_step)
            self.writer.add_scalar("charts/FPS", iteration_fps, self._global_step)
            self.writer.add_scalar("charts/iteration", iteration, self._global_step)
            self.writer.add_scalar("charts/global_step", self._global_step, self._global_step)
            self.writer.add_scalar("charts/total_episodes", self._global_episodes, self._global_step)
            self.writer.add_scalar("time/rollout_seconds", rollout_time, self._global_step)
            self.writer.add_scalar("time/update_seconds", update_time, self._global_step)
            self.writer.add_scalar("time/iteration_seconds", iteration_time, self._global_step)
            self.writer.add_scalar("time/rollout_fps", rollout_fps, self._global_step)
            self.writer.add_scalar("time/update_fps", update_fps, self._global_step)
            self.writer.add_scalar("time/iteration_fps", iteration_fps, self._global_step)

            done_total = iter_terminations + iter_truncations
            self.writer.add_scalar("charts/episodes_this_iter", len(iter_ep_returns), self._global_step)
            self.writer.add_scalar("charts/terminations", iter_terminations, self._global_step)
            self.writer.add_scalar("charts/truncations", iter_truncations, self._global_step)
            self.writer.add_scalar(
                "charts/termination_fraction",
                iter_terminations / max(done_total, 1),
                self._global_step,
            )
            if iter_ep_returns:
                ep_r = np.array(iter_ep_returns)
                ep_l = np.array(iter_ep_lengths)
                ep_g = np.array(iter_ep_goals, dtype=np.float32)
                self.writer.add_scalar("charts/episodic_return_mean", float(ep_r.mean()), self._global_step)
                self.writer.add_scalar("charts/episodic_return_std", float(ep_r.std()), self._global_step)
                self.writer.add_scalar("charts/episodic_length_mean", float(ep_l.mean()), self._global_step)
                self.writer.add_scalar("charts/goals_reached_mean", float(ep_g.mean()), self._global_step)
                self.writer.add_scalar("charts/goals_reached_std", float(ep_g.std()), self._global_step)
                self.writer.add_scalar("charts/goals_reached_max", float(ep_g.max()), self._global_step)
                self.writer.add_scalar("charts/at_least_one_goal_rate",
                                       float((ep_g >= 1).mean()), self._global_step)
            self.writer.add_scalar("rollout/reward_mean", rewards.mean().item(), self._global_step)
            self.writer.add_scalar("rollout/action_clip_fraction", float(np.mean(action_clip_fracs)), self._global_step)

            if save_model and save_freq > 0 and self._global_step - self._last_save_step >= save_freq:
                self._last_save_step = self._global_step
                path = self.checkpointer.save_step(self._checkpoint_state(), self._global_step)
                print(f"[checkpoint] step={self._global_step} -> {path}")

            if save_model and self._global_step - self._last_eval_step >= eval_freq:
                self._last_eval_step = self._global_step
                self._evaluate_agent(self._global_step, eval_episodes=eval_episodes)

        if save_model:
            self._evaluate_agent(self._global_step, eval_episodes=eval_episodes)
            self.checkpointer.save_latest(self._checkpoint_state())

        self.envs.close()
        self.writer.close()
        self.session.finish()
