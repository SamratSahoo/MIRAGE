import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim


def _warm_torch_cuda() -> None:
    if not torch.cuda.is_available():
        return
    dev = torch.device("cuda")
    _w = nn.Linear(2, 2).to(dev)
    _o = optim.Adam(_w.parameters(), lr=1.0)
    _y = _w(torch.zeros(1, 2, device=dev)).sum()
    _y.backward()
    _o.step()
    del _w, _o, _y
    torch.cuda.empty_cache()


_warm_torch_cuda()

import yaml

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mirage.rl.joint_trainer import JointPPOTrainer
from mirage.envs.env_config import ENV_CONFIG, configure_env


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for section in ("env", "ppo", "train"):
        if section not in cfg:
            raise KeyError(f"config '{path}' is missing the '{section}:' section")
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Joint end-to-end latent PPO on AntMaze")
    parser.add_argument("--config", default=os.path.join(_PROJECT_ROOT, "config", "ppo", "joint_latent.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["ppo"]["exp_name"] = os.path.splitext(os.path.basename(args.config))[0]
    env_cfg, train_cfg = cfg["env"], cfg["train"]
    configure_env(
        reward_type=env_cfg["reward_type"],
        continuing_task=env_cfg["continuing_task"],
        reset_target=env_cfg["reset_target"],
        max_episode_steps=env_cfg["max_episode_steps"],
        capture_video=env_cfg["capture_video"],
        video_every=env_cfg["video_every"],
        use_cuda_graph=env_cfg.get("use_cuda_graph", True),
    )

    print("=" * 66)
    print(f"AntMaze JOINT latent PPO  |  config: {args.config}")
    print(f"  env_id          : {env_cfg['env_id']}  ({ENV_CONFIG['reward_type']} reward)")
    print(f"  intrinsic_coef  : {cfg.get('joint', {}).get('intrinsic_coef')}")
    print(f"  train_encoder   : {cfg.get('joint', {}).get('train_encoder')}")
    print(f"  total_timesteps : {int(train_cfg['total_timesteps']):,}")
    print("=" * 66)

    trainer = JointPPOTrainer(cfg)
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
