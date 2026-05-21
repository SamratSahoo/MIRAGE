import gymnasium as gym
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw

from algorithms.utils import ENV_CONFIG

_OBS_DIM = 105
_GOAL_DIM = 2
_ACT_DIM = 8
_GOAL_RADIUS = 0.45
_CONTACT_CLIP = 1.0
_DEFAULT_MAX_EPISODE_STEPS = 700
_XY_NOISE = 1.0


class WarpAntMazeEnv:
    def __init__(self, env_id, num_envs, device, seed=0, njmax=512, use_cuda_graph=True):
        self.env_id = env_id
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.policy_obs_dim = _OBS_DIM + _GOAL_DIM
        self.use_cuda_graph = bool(use_cuda_graph)
        self._graph = None

        reward_type = ENV_CONFIG["reward_type"]
        if reward_type not in ("dense", "sparse"):
            raise ValueError(f"unsupported reward_type {reward_type!r}")
        self.reward_type = reward_type
        self.continuing_task = bool(ENV_CONFIG["continuing_task"])
        self.reset_target = bool(ENV_CONFIG["reset_target"])
        cfg_steps = ENV_CONFIG.get("max_episode_steps")
        self.max_episode_steps = int(cfg_steps) if cfg_steps else _DEFAULT_MAX_EPISODE_STEPS

        ref = gym.make(
            env_id,
            reward_type=reward_type,
            continuing_task=self.continuing_task,
            reset_target=self.reset_target,
        )
        ant = ref.unwrapped.ant_env
        mjm = ref.unwrapped.model
        self.frame_skip = int(ant.frame_skip)
        qpos0 = np.array(mjm.qpos0, dtype=np.float32)
        cells = np.asarray(ref.unwrapped.maze.unique_goal_locations, dtype=np.float32)
        ref.close()

        if mjm.nq != 15 or mjm.nv != 14 or mjm.nu != _ACT_DIM:
            raise RuntimeError(
                f"unexpected ant model dims nq={mjm.nq} nv={mjm.nv} nu={mjm.nu}"
            )

        self.wm = mjw.put_model(mjm)
        self.wd = mjw.make_data(mjm, nworld=self.num_envs, njmax=njmax)

        self._qpos = wp.to_torch(self.wd.qpos)
        self._qvel = wp.to_torch(self.wd.qvel)
        self._ctrl = wp.to_torch(self.wd.ctrl)
        self._cfrc = wp.to_torch(self.wd.cfrc_ext)

        self.qpos0 = torch.from_numpy(qpos0).to(self.device)
        self.cells = torch.from_numpy(cells).to(self.device)
        self.n_cells = self.cells.shape[0]
        self.desired_goal = torch.zeros((self.num_envs, _GOAL_DIM), device=self.device)
        self.episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gen = torch.Generator(device=self.device).manual_seed(int(seed))

        self.single_observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(-np.inf, np.inf, (_OBS_DIM,), np.float32),
                "desired_goal": gym.spaces.Box(-np.inf, np.inf, (_GOAL_DIM,), np.float32),
                "achieved_goal": gym.spaces.Box(-np.inf, np.inf, (_GOAL_DIM,), np.float32),
            }
        )
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (_ACT_DIM,), np.float32)
        self.action_low = torch.full((_ACT_DIM,), -1.0, device=self.device)
        self.action_high = torch.full((_ACT_DIM,), 1.0, device=self.device)

        if self.use_cuda_graph:
            self._run_physics()
            try:
                with wp.ScopedCapture() as capture:
                    self._run_physics()
                self._graph = capture.graph
            except Exception as exc:
                print(f"[WarpAntMazeEnv] CUDA graph capture failed ({exc}); "
                      f"using eager stepping")
                self.use_cuda_graph = False

    def _sample_goal_reset(self, n):
        goal_cell = torch.randint(0, self.n_cells, (n,), generator=self.gen, device=self.device)
        offset = torch.randint(1, self.n_cells, (n,), generator=self.gen, device=self.device)
        reset_cell = (goal_cell + offset) % self.n_cells
        goal_noise = (torch.rand((n, 2), generator=self.gen, device=self.device) * 2 - 1) * _XY_NOISE
        reset_noise = (torch.rand((n, 2), generator=self.gen, device=self.device) * 2 - 1) * _XY_NOISE
        goal_xy = self.cells[goal_cell] + goal_noise
        reset_xy = self.cells[reset_cell] + reset_noise
        return goal_xy, reset_xy

    def _write_reset(self, idx, goal_xy, reset_xy):
        self._qpos[idx, 0:2] = reset_xy
        self._qpos[idx, 2:15] = self.qpos0[2:15]
        self._qvel[idx] = 0.0
        self.desired_goal[idx] = goal_xy

    def _observation(self, contact_zero_mask=None):
        contact = torch.clamp(self._cfrc[:, 1:14, :], -_CONTACT_CLIP, _CONTACT_CLIP)
        contact = contact.reshape(self.num_envs, -1)
        if contact_zero_mask is not None:
            contact = contact.clone()
            contact[contact_zero_mask] = 0.0
        obs = torch.cat([self._qpos[:, 2:15], self._qvel[:, 0:14], contact], dim=1)
        return torch.cat([obs, self.desired_goal], dim=1)

    def _run_physics(self):
        for _ in range(self.frame_skip):
            mjw.step(self.wm, self.wd)
        mjw.rne_postconstraint(self.wm, self.wd)

    def _step_physics(self):
        if self.use_cuda_graph and self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._run_physics()

    @torch.no_grad()
    def reset(self, seed=None):
        if seed is not None:
            self.gen.manual_seed(int(seed))
        idx = torch.arange(self.num_envs, device=self.device)
        goal_xy, reset_xy = self._sample_goal_reset(self.num_envs)
        self._write_reset(idx, goal_xy, reset_xy)
        mjw.forward(self.wm, self.wd)
        self.episode_step.zero_()
        self.episode_return.zero_()
        self.episode_length.zero_()
        full = torch.arange(self.num_envs, device=self.device)
        return self._observation(contact_zero_mask=full)

    @torch.no_grad()
    def step(self, actions):
        self._ctrl[:] = torch.clamp(actions, self.action_low, self.action_high)
        self._step_physics()

        achieved_goal = self._qpos[:, 0:2]
        dist = torch.linalg.norm(achieved_goal - self.desired_goal, dim=1)
        if self.reward_type == "dense":
            reward = torch.exp(-dist)
        else:
            reward = (dist <= _GOAL_RADIUS).float()

        if self.continuing_task:
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            terminated = dist <= _GOAL_RADIUS
        self.episode_step += 1
        truncated = self.episode_step >= self.max_episode_steps
        done = terminated | truncated

        final_obs = self._observation()
        self.episode_return += reward
        self.episode_length += 1
        info = {
            "final_obs": final_obs,
            "episodic_return": self.episode_return.clone(),
            "episodic_length": self.episode_length.clone(),
            "done": done,
        }

        if bool(done.any()):
            done_idx = torch.nonzero(done, as_tuple=False).squeeze(1)
            goal_xy, reset_xy = self._sample_goal_reset(done_idx.numel())
            self._write_reset(done_idx, goal_xy, reset_xy)
            mjw.forward(self.wm, self.wd)
            self.episode_step[done_idx] = 0
            self.episode_return[done_idx] = 0.0
            self.episode_length[done_idx] = 0

        next_obs = self._observation(contact_zero_mask=done)
        return next_obs, reward, terminated, truncated, info

    def close(self):
        pass
