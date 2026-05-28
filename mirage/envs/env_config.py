import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


ENV_CONFIG = {
    "reward_type": "dense",
    "continuing_task": False,
    "reset_target": False,
    "max_episode_steps": None,
    "capture_video": False,
    "video_every": 100,
    "use_cuda_graph": True,
}

_MAZE_ENV_TYPES = {"antmaze", "maze", "gymnasium_robotics"}


def configure_env(**overrides):
    unknown = set(overrides) - set(ENV_CONFIG)
    if unknown:
        raise KeyError(
            f"Unknown env config key(s): {sorted(unknown)}. "
            f"Valid keys: {sorted(ENV_CONFIG)}"
        )
    ENV_CONFIG.update(overrides)
    return dict(ENV_CONFIG)
