import argparse
import os
import sys

import yaml

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mirage.rl.trainer import PPOTrainer
from mirage.envs.env_config import ENV_CONFIG, configure_env


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config '{path}' did not parse to a mapping")
    for section in ("env", "ppo", "train"):
        if section not in cfg:
            raise KeyError(f"config '{path}' is missing the '{section}:' section")
    return cfg


def main():
    parser = argparse.ArgumentParser(description="PPO training on AntMaze")
    parser.add_argument(
        "--config",
        default=os.path.join(_PROJECT_ROOT, "config", "ppo", "raw_state_raw_goal.yaml"),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["ppo"]["exp_name"] = os.path.splitext(os.path.basename(args.config))[0]
    env_cfg = cfg["env"]
    train_cfg = cfg["train"]
    configure_env(
        reward_type=env_cfg["reward_type"],
        continuing_task=env_cfg["continuing_task"],
        reset_target=env_cfg["reset_target"],
        max_episode_steps=env_cfg["max_episode_steps"],
        capture_video=env_cfg["capture_video"],
        video_every=env_cfg["video_every"],
    )

    print("=" * 66)
    print(f"AntMaze PPO  |  config: {args.config}")
    print(f"  env_id           : {env_cfg['env_id']}  ({ENV_CONFIG['reward_type']} reward)")
    print(f"  state_mode       : {cfg['ppo'].get('state_mode', 'raw')}")
    print(f"  goal_conditioning: {cfg['ppo'].get('goal_conditioning')}")
    print(f"  subgoal_mode     : {cfg['ppo'].get('subgoal_mode', '')!r}")
    print(f"  total_timesteps  : {int(train_cfg['total_timesteps']):,}")
    print("=" * 66)

    trainer = PPOTrainer(cfg)
    trainer.train(
        total_timesteps=int(train_cfg["total_timesteps"]),
        save_model=bool(train_cfg["save_model"]),
        save_freq=int(train_cfg.get("save_freq", 0)),
        eval_freq=int(train_cfg.get("eval_freq", 2_000_000)),
        eval_episodes=int(train_cfg.get("eval_episodes", 256)),
    )
    print(f"[done] run '{trainer.run_name}'  ->  runs/{trainer.run_name}")


if __name__ == "__main__":
    main()
