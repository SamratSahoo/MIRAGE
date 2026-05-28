import gymnasium as gym
import numpy as np

from .env_config import ENV_CONFIG, _MAZE_ENV_TYPES


def _build_single_env(env_type, env_id, render_mode):
    kwargs = {"render_mode": render_mode}
    if ENV_CONFIG["max_episode_steps"] is not None:
        kwargs["max_episode_steps"] = ENV_CONFIG["max_episode_steps"]
    if env_type in _MAZE_ENV_TYPES:
        kwargs["reward_type"] = ENV_CONFIG["reward_type"]
        kwargs["continuing_task"] = ENV_CONFIG["continuing_task"]
        kwargs["reset_target"] = ENV_CONFIG["reset_target"]
    return gym.make(env_id, **kwargs)


def make_env(env_type, env_id, idx, capture_video, run_name):
    record_video = bool(capture_video) and idx == 0 and ENV_CONFIG["capture_video"]

    def thunk():
        render_mode = "rgb_array" if record_video else None
        env = _build_single_env(env_type, env_id, render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if record_video:
            every = max(1, int(ENV_CONFIG["video_every"]))
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=f"videos/{run_name}",
                episode_trigger=lambda ep, n=every: ep % n == 0,
                name_prefix=str(env_id).replace("/", "_"),
                disable_logger=True,
            )
        return env

    return thunk


def transform_obs(obs, goal_size):
    if not isinstance(obs, dict):
        return np.asarray(obs, dtype=np.float32)
    observation = np.asarray(obs["observation"], dtype=np.float32)
    if goal_size and goal_size > 0:
        goal = np.asarray(obs["desired_goal"], dtype=np.float32)
        return np.concatenate([observation, goal], axis=-1)
    return observation
