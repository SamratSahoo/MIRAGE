from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import torch

from mirage.rl.agent import Agent
from mirage.rl.policy_input import PolicyInputBuilder
from mirage.rl.trainer import _load_encoder_bundle
from mirage.envs.env_config import configure_env
from mirage.envs.warp_antmaze import WarpAntMazeEnv
from mirage.planning.motion_planning.cell_graph_planner import CellGraphPlanner
from mirage.planning.graph_planning.latent_graph_planner import LatentGraphPlanner

from .warp_renderer import WarpRenderer


def _save_video(path: str, frames: list, fps: int = 30) -> None:
    import imageio.v2 as imageio
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


class PPOEvaluator:
    def __init__(self, cfg: dict, model_path: str, num_episodes: int = 256,
                 num_envs: int = 64, seed: int = 1234, deterministic: bool = True,
                 record_trajectory_episodes: int = 1, video: bool = True,
                 video_envs: int = 4, video_width: int = 640, video_height: int = 480,
                 video_fps: int = 30, output_dir: str | None = None):
        self.cfg = cfg
        self.model_path = model_path
        self.num_episodes = int(num_episodes)
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.record_trajectory_episodes = int(record_trajectory_episodes)
        self.video = bool(video)
        self.video_envs = int(video_envs)
        self.video_width = int(video_width)
        self.video_height = int(video_height)
        self.video_fps = int(video_fps)

        env_cfg = cfg["env"]
        ppo_cfg = cfg["ppo"]
        configure_env(
            reward_type=env_cfg["reward_type"],
            continuing_task=env_cfg["continuing_task"],
            reset_target=env_cfg["reset_target"],
            max_episode_steps=env_cfg["max_episode_steps"],
            capture_video=env_cfg["capture_video"],
            video_every=env_cfg["video_every"],
            use_cuda_graph=env_cfg.get("use_cuda_graph", True),
        )
        self.env_id = env_cfg["env_id"]
        self.env_cfg = env_cfg

        if output_dir is None:
            run_dir = os.path.dirname(os.path.abspath(model_path))
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(run_dir, f"eval_{ts}")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.video_dir = os.path.join(output_dir, "videos")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_mode = str(ppo_cfg.get("state_mode", "raw"))
        self.goal_conditioning = list(ppo_cfg.get("goal_conditioning", ["goal"]))
        self.subgoal_mode = str(ppo_cfg.get("subgoal_mode", "") or "")
        self.subgoal_radius = float(ppo_cfg.get("subgoal_radius", 1.5))
        self.latent_encoder_path = str(ppo_cfg.get("latent_encoder_path", "") or "")
        self.goal_encoder_path = str(ppo_cfg.get("goal_encoder_path", "") or "")
        self.graph_planning = self.subgoal_mode == "graph_planning"
        self.gp_cfg = dict(ppo_cfg.get("graph_planning", {}) or {})
        self.graph_planner = None

        self.state_encoder = None
        self.state_encoder_input_mode = None
        if self.state_mode == "latent" and not self.graph_planning:
            enc, _, _, ecfg = _load_encoder_bundle(self.latent_encoder_path, self.device, with_dynamics=False)
            self.state_encoder = enc
            self.state_encoder_input_mode = str(ecfg["input_mode"])

        self.goal_encoder = None
        self.goal_encoder_input_mode = None
        needs_goal_encoder = any(c in ("latent_goal", "latent_subgoal") for c in self.goal_conditioning)
        if needs_goal_encoder and not self.graph_planning:
            path = self.goal_encoder_path or self.latent_encoder_path
            enc, _, _, ecfg = _load_encoder_bundle(path, self.device, with_dynamics=False)
            self.goal_encoder = enc
            self.goal_encoder_input_mode = str(ecfg["input_mode"])

        self.planner = None
        if not self.graph_planning and (self.subgoal_mode == "motion_planning" or any(
            c in ("subgoal", "latent_subgoal") for c in self.goal_conditioning
        )):
            self.planner = CellGraphPlanner(self.env_id, device=self.device,
                                            subgoal_radius=self.subgoal_radius)

        self.envs = WarpAntMazeEnv(
            env_id=self.env_id, num_envs=self.num_envs, device=self.device, seed=self.seed,
            subgoal_planner=self.planner, subgoal_radius=self.subgoal_radius,
        )

        if self.graph_planning:
            self.graph_planner = LatentGraphPlanner(
                self.gp_cfg, self.device, self.num_envs, self.state_mode)

        self.builder = PolicyInputBuilder(
            state_mode=self.state_mode,
            goal_conditioning=self.goal_conditioning,
            state_encoder=self.state_encoder,
            state_encoder_input_mode=self.state_encoder_input_mode,
            goal_encoder=self.goal_encoder,
            goal_encoder_input_mode=self.goal_encoder_input_mode,
            graph_planning=self.graph_planning,
            planning_latent_dim=(self.graph_planner.latent_dim
                                 if self.graph_planner is not None else None),
        )
        self.obs_dim = self.builder.obs_dim
        self.act_dim = int(np.prod(self.envs.single_action_space.shape))

        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(state_dict, dict) and "agent" in state_dict:
            agent_state = state_dict["agent"]
        else:
            agent_state = state_dict
        self.agent = Agent(obs_dim=self.obs_dim, act_dim=self.act_dim).to(self.device)
        self.agent.load_state_dict(agent_state)
        self.agent.eval()

        self._imagine_act_fn = (
            (lambda z, zk, gxy: self.agent.actor_mean(
                self.builder.assemble_from_latents(z, zk, zk, goal_xy=gxy)))
            if self.graph_planning else None
        )

    def run(self) -> dict:
        envs = self.envs
        obs = envs.reset(seed=self.seed)
        per_episode = []
        trajectories: list[dict] = []
        traj_buffer = self._empty_traj()
        step_count = 0

        video_paths = {"success": None, "failure": None}
        renderer = None
        video_env_ids: list[int] = []
        frame_bufs: dict[int, list] = {}
        if self.video:
            os.makedirs(self.video_dir, exist_ok=True)
            renderer = WarpRenderer(self.env_id, self.env_cfg,
                                    width=self.video_width, height=self.video_height)
            video_env_ids = list(range(min(self.video_envs, self.num_envs)))
            frame_bufs = {i: [] for i in video_env_ids}

        def video_done() -> bool:
            return all(p is not None for p in video_paths.values())

        eval_done = None
        if self.graph_planner is not None:
            self.graph_planner.reset_state()
            eval_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        while len(per_episode) < self.num_episodes:
            achieved_xy = envs.achieved_xy.clone()
            subgoal_xy = envs.subgoal_xy.clone() if self.planner is not None else None
            with torch.no_grad():
                if self.graph_planner is not None:
                    gp = self.graph_planner.update(obs, achieved_xy, done_mask=eval_done,
                                                   act_fn=self._imagine_act_fn)
                    pi = self.builder.assemble(obs, achieved_xy, z_state=gp["z_state"],
                                               z_goal=gp["z_goal"], z_subgoal=gp["z_subgoal"])
                else:
                    pi = self.builder.assemble(obs, achieved_xy, subgoal_xy)
                if self.deterministic:
                    action = self.agent.actor_mean(pi)
                else:
                    action, _, _, _ = self.agent.get_action_and_value(pi)

            ag0 = envs._qpos[0, 0:2].detach().cpu().numpy().copy()
            dg0 = envs.desired_goal[0].detach().cpu().numpy().copy()
            sg0 = envs.subgoal_xy[0].detach().cpu().numpy().copy() if self.planner is not None else None

            if renderer is not None and not video_done():
                qpos_cpu = envs._qpos.detach().cpu().numpy()
                dg_cpu = envs.desired_goal.detach().cpu().numpy()
                sg_cpu = envs.subgoal_xy.detach().cpu().numpy() if self.planner is not None else None
                for i in video_env_ids:
                    if i in frame_bufs:
                        sg = sg_cpu[i] if sg_cpu is not None else None
                        frame_bufs[i].append(renderer.render(qpos_cpu[i], dg_cpu[i], sg))

            next_obs, reward, terminated, truncated, infos = envs.step(action)
            done = infos["done"]
            if self.graph_planner is not None:
                eval_done = done.bool()

            if len(trajectories) < self.record_trajectory_episodes:
                traj_buffer["obs"].append(obs[0].detach().cpu().numpy().copy())
                traj_buffer["action"].append(action[0].detach().cpu().numpy().copy())
                traj_buffer["reward"].append(float(reward[0].item()))
                traj_buffer["achieved_goal"].append(ag0)
                traj_buffer["desired_goal"].append(dg0)
                if sg0 is not None:
                    traj_buffer["subgoal"].append(sg0)
                traj_buffer["terminated"].append(bool(terminated[0].item()))
                traj_buffer["truncated"].append(bool(truncated[0].item()))

            obs = next_obs
            step_count += 1

            if bool(done.any()):
                done_mask = done.detach().cpu().numpy().astype(bool)
                ep_ret = infos["episodic_return"].detach().cpu().numpy()
                ep_len = infos["episodic_length"].detach().cpu().numpy()
                term_np = terminated.detach().cpu().numpy().astype(bool)
                trunc_np = truncated.detach().cpu().numpy().astype(bool)
                dg_per_env = infos["desired_goal"].detach().cpu().numpy()

                for i in np.where(done_mask)[0]:
                    if len(per_episode) >= self.num_episodes:
                        break
                    is_success = bool(term_np[i])
                    per_episode.append({
                        "episode": len(per_episode) + 1,
                        "env": int(i),
                        "return": float(ep_ret[i]),
                        "length": int(ep_len[i]),
                        "success": is_success,
                        "terminated": is_success,
                        "truncated": bool(trunc_np[i]),
                        "desired_goal": dg_per_env[i].tolist(),
                    })

                    if i == 0 and len(trajectories) < self.record_trajectory_episodes:
                        trajectories.append({k: np.array(v) for k, v in traj_buffer.items() if v})
                        traj_buffer = self._empty_traj()

                    if renderer is not None and int(i) in frame_bufs:
                        kind = "success" if is_success else "failure"
                        if video_paths[kind] is None and frame_bufs[int(i)]:
                            out_path = os.path.join(self.video_dir, f"{kind}.mp4")
                            _save_video(out_path, frame_bufs[int(i)], fps=self.video_fps)
                            video_paths[kind] = out_path
                        frame_bufs[int(i)] = []
                        if video_done():
                            renderer.close()
                            renderer = None
                            frame_bufs = {}

        envs.close()
        if renderer is not None:
            renderer.close()

        returns = np.array([e["return"] for e in per_episode], dtype=np.float32)
        lengths = np.array([e["length"] for e in per_episode], dtype=np.int64)
        successes = np.array([e["success"] for e in per_episode], dtype=bool)
        success_lengths = lengths[successes]

        summary = {
            "model_path": os.path.abspath(self.model_path),
            "env_id": self.env_id,
            "reward_type": self.env_cfg["reward_type"],
            "num_episodes": int(len(per_episode)),
            "num_envs": int(self.num_envs),
            "rollout_steps": int(step_count),
            "deterministic": bool(self.deterministic),
            "seed": int(self.seed),
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()),
            "return_min": float(returns.min()),
            "return_max": float(returns.max()),
            "length_mean": float(lengths.mean()),
            "length_min": int(lengths.min()),
            "length_max": int(lengths.max()),
            "success_rate": float(successes.mean()),
            "num_successes": int(successes.sum()),
            "success_length_mean": float(success_lengths.mean()) if success_lengths.size else None,
            "success_length_min": int(success_lengths.min()) if success_lengths.size else None,
            "video_success": video_paths["success"],
            "video_failure": video_paths["failure"],
        }
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump({"summary": summary, "episodes": per_episode}, f, indent=2)
        for k, traj in enumerate(trajectories):
            traj_path = os.path.join(self.output_dir, f"trajectory_{k:02d}.npz")
            np.savez_compressed(traj_path, **traj)
        return summary

    def _empty_traj(self) -> dict:
        return {"obs": [], "action": [], "reward": [], "achieved_goal": [],
                "desired_goal": [], "subgoal": [], "terminated": [], "truncated": []}
