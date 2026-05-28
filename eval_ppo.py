import argparse
import os
import sys
from sys import platform

os.environ.setdefault("MUJOCO_GL", "glfw" if platform == "darwin" else "egl")

import yaml

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mirage.eval.ppo_evaluator import PPOEvaluator


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config '{path}' did not parse to a mapping")
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Evaluate a PPO agent on WarpAntMazeEnv")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-episodes", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--record-trajectory-episodes", type=int, default=1)
    parser.add_argument("--video", action="store_true", default=True)
    parser.add_argument("--no-video", dest="video", action="store_false")
    parser.add_argument("--video-envs", type=int, default=4)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluator = PPOEvaluator(
        cfg=cfg,
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        num_envs=args.num_envs,
        seed=args.seed,
        deterministic=args.deterministic,
        record_trajectory_episodes=args.record_trajectory_episodes,
        video=args.video,
        video_envs=args.video_envs,
        video_width=args.video_width,
        video_height=args.video_height,
        video_fps=args.video_fps,
        output_dir=args.output_dir,
    )
    summary = evaluator.run()
    print("=" * 66)
    print(f"[eval] episodes        : {summary['num_episodes']}  "
          f"(rollout steps: {summary['rollout_steps']})")
    print(f"[eval] return  (mean)  : {summary['return_mean']:+.3f}  "
          f"std={summary['return_std']:.3f}")
    print(f"[eval] length  (mean)  : {summary['length_mean']:.1f}  "
          f"[{summary['length_min']}, {summary['length_max']}]")
    print(f"[eval] success rate    : {summary['success_rate']:.1%}  "
          f"({summary['num_successes']}/{summary['num_episodes']})")
    print(f"[eval] output_dir      : {evaluator.output_dir}")
    print("=" * 66)


if __name__ == "__main__":
    main()
