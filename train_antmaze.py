import argparse
import os
import sys

import yaml

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from algorithms.utils import ENV_CONFIG, configure_env  
from ppo import PPOTrainer 


def load_config(path):
    """Load and minimally validate the YAML config."""
    with open(path) as handle:
        cfg = yaml.safe_load(handle)
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
        default=os.path.join(_PROJECT_ROOT, "config.yaml"),
        help="path to the YAML config (default: config.yaml beside this script)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg, ppo_cfg, train_cfg = cfg["env"], cfg["ppo"], cfg["train"]
    wandb_cfg = cfg.get("wandb", {})

    # Push AntMaze-specific options into algorithms.utils. make_env() reads
    # ENV_CONFIG when the PPOTrainer builds its vector envs, so this has to run
    # *before* PPOTrainer(...) is constructed.
    configure_env(
        reward_type=env_cfg["reward_type"],
        continuing_task=env_cfg["continuing_task"],
        reset_target=env_cfg["reset_target"],
        max_episode_steps=env_cfg["max_episode_steps"],
        capture_video=env_cfg["capture_video"],
        video_every=env_cfg["video_every"],
    )

    total_timesteps = int(train_cfg["total_timesteps"])
    print("=" * 66)
    print(f"AntMaze PPO  |  config: {args.config}")
    print(f"  env_id           : {env_cfg['env_id']}  ({ENV_CONFIG['reward_type']} reward)")
    print(f"  continuing_task  : {ENV_CONFIG['continuing_task']}")
    print(f"  goal_size        : {env_cfg['goal_size']}")
    print(f"  num_envs         : {ppo_cfg['num_envs']}   num_steps: {ppo_cfg['num_steps']}")
    print(f"  total_timesteps  : {total_timesteps:,}")
    print("=" * 66)

    # Every argument is passed by keyword; PPOTrainer defines sensible defaults
    # but config.yaml is the single source of truth here.
    trainer = PPOTrainer(
        env_type=env_cfg["env_type"],
        env_id=env_cfg["env_id"],
        goal_size=int(env_cfg["goal_size"]),
        seed=int(ppo_cfg["seed"]),
        num_envs=int(ppo_cfg["num_envs"]),
        num_steps=int(ppo_cfg["num_steps"]),
        num_minibatches=int(ppo_cfg["num_minibatches"]),
        update_epochs=int(ppo_cfg["update_epochs"]),
        exp_name=str(ppo_cfg["exp_name"]),
        learning_rate=float(ppo_cfg["learning_rate"]),
        anneal_lr=bool(ppo_cfg["anneal_lr"]),
        target_kl=ppo_cfg["target_kl"],
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        norm_adv=bool(ppo_cfg["norm_adv"]),
        clip_coef=float(ppo_cfg["clip_coef"]),
        clip_vloss=bool(ppo_cfg["clip_vloss"]),
        ent_coef=float(ppo_cfg["ent_coef"]),
        vf_coef=float(ppo_cfg["vf_coef"]),
        max_grad_norm=float(ppo_cfg["max_grad_norm"]),
        track=bool(wandb_cfg.get("track", False)),
        wandb_project_name=str(wandb_cfg.get("project", "mirage-ppo")),
        wandb_entity=wandb_cfg.get("entity"),
    )

    trainer.train(total_timesteps=total_timesteps, save_model=bool(train_cfg["save_model"]))
    print(f"[done] run '{trainer.run_name}'  ->  runs/{trainer.run_name}")


if __name__ == "__main__":
    main()
