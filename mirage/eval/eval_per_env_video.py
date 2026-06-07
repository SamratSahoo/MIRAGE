import argparse
import os
import sys
from sys import platform

os.environ.setdefault("MUJOCO_GL", "glfw" if platform == "darwin" else "egl")

import numpy as np
import torch
import yaml
import mujoco_warp as mjw

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datetime import datetime

from mirage.eval.ppo_evaluator import PPOEvaluator, _save_video
from mirage.eval.warp_renderer import WarpRenderer

_XY_NOISE = 1.0


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def sample_constrained_goals(envs, n, dmin, dmax, gen):
    cells = envs.cells
    dev = cells.device
    reset_xy = torch.empty((n, 2), device=dev)
    goal_xy = torch.empty((n, 2), device=dev)
    filled = torch.zeros(n, dtype=torch.bool, device=dev)
    while not bool(filled.all()):
        need = torch.nonzero(~filled, as_tuple=False).squeeze(1)
        m = need.numel()
        rc = torch.randint(0, envs.n_cells, (m,), generator=gen, device=dev)
        gc = torch.randint(0, envs.n_cells, (m,), generator=gen, device=dev)
        rn = (torch.rand((m, 2), generator=gen, device=dev) * 2 - 1) * _XY_NOISE
        gn = (torch.rand((m, 2), generator=gen, device=dev) * 2 - 1) * _XY_NOISE
        rxy = cells[rc] + rn
        gxy = cells[gc] + gn
        dist = torch.linalg.norm(rxy - gxy, dim=1)
        ok = (dist >= dmin) & (dist <= dmax)
        sel = need[ok]
        reset_xy[sel] = rxy[ok]
        goal_xy[sel] = gxy[ok]
        filled[sel] = True
    return goal_xy, reset_xy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--horizon", type=int, default=1000,
                   help="steps to record (defaults to one episode)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--goal-dist-min", type=float, default=4.0)
    p.add_argument("--goal-dist-max", type=float, default=8.0)
    p.add_argument("--video-width", type=int, default=640)
    p.add_argument("--video-height", type=int, default=480)
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--stochastic", dest="deterministic", action="store_false", default=True)
    args = p.parse_args()

    cfg = load_config(args.config)

    ev = PPOEvaluator(
        cfg=cfg, model_path=args.model_path, num_episodes=args.num_envs,
        num_envs=args.num_envs, seed=args.seed, deterministic=args.deterministic,
        record_trajectory_episodes=0, video=False, video_envs=args.num_envs,
        output_dir=args.output_dir,
    )
    envs = ev.envs
    dev = ev.device

    if args.output_dir is None:
        run_dir = os.path.dirname(os.path.abspath(args.model_path))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(run_dir, f"eval_perenv_{ts}")
    else:
        out_dir = args.output_dir
    video_dir = os.path.join(out_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    obs = envs.reset(seed=args.seed)
    all_idx = torch.arange(envs.num_envs, device=dev)
    goal_xy, reset_xy = sample_constrained_goals(
        envs, envs.num_envs, args.goal_dist_min, args.goal_dist_max, envs.gen)
    envs._write_reset(all_idx, goal_xy, reset_xy)
    mjw.forward(envs.wm, envs.wd)
    envs._refresh_subgoals(all_idx)
    obs = envs._observation(contact_zero_mask=all_idx)

    init_pos = envs._qpos[:, 0:2].clone()
    init_goal = envs.desired_goal.clone()
    init_dist = torch.linalg.norm(init_pos - init_goal, dim=1).cpu().numpy()

    eval_done = None
    if ev.graph_planner is not None:
        ev.graph_planner.reset_state()
        eval_done = torch.zeros(envs.num_envs, dtype=torch.bool, device=dev)

    renderer = WarpRenderer(ev.env_id, ev.env_cfg,
                            width=args.video_width, height=args.video_height)
    frame_bufs = {i: [] for i in range(envs.num_envs)}
    reached_step = [None] * envs.num_envs

    for t in range(args.horizon):
        achieved_xy = envs.achieved_xy.clone()
        subgoal_xy = envs.subgoal_xy.clone() if ev.planner is not None else None
        with torch.no_grad():
            if ev.graph_planner is not None:
                gp = ev.graph_planner.update(obs, achieved_xy, done_mask=eval_done,
                                             act_fn=ev._imagine_act_fn)
                pi = ev.builder.assemble(obs, achieved_xy, z_state=gp["z_state"],
                                         z_goal=gp["z_goal"], z_subgoal=gp["z_subgoal"])
            else:
                pi = ev.builder.assemble(obs, achieved_xy, subgoal_xy)
            if ev.deterministic:
                action = ev.agent.actor_mean(pi)
            else:
                action, _, _, _ = ev.agent.get_action_and_value(pi)

        qpos_cpu = envs._qpos.detach().cpu().numpy()
        dg_cpu = envs.desired_goal.detach().cpu().numpy()
        sg_cpu = envs.subgoal_xy.detach().cpu().numpy() if ev.planner is not None else None
        for i in range(envs.num_envs):
            sg = sg_cpu[i] if sg_cpu is not None else None
            frame_bufs[i].append(renderer.render(qpos_cpu[i], dg_cpu[i], sg))

        next_obs, reward, terminated, truncated, infos = envs.step(action)
        rwd = reward.detach().cpu().numpy()
        for i in range(envs.num_envs):
            if reached_step[i] is None and rwd[i] > 0.5:
                reached_step[i] = t
        if ev.graph_planner is not None:
            eval_done = infos["done"].bool()
        obs = next_obs

    renderer.close()
    envs.close()

    paths = []
    for i in range(envs.num_envs):
        vp = os.path.join(video_dir, f"env_{i:02d}.mp4")
        _save_video(vp, frame_bufs[i], fps=args.video_fps)
        paths.append(vp)

    print("=" * 70)
    print(f"[eval] model      : {os.path.basename(args.model_path)}")
    print(f"[eval] horizon    : {args.horizon} steps  | envs: {envs.num_envs}")
    for i in range(envs.num_envs):
        rs = reached_step[i]
        rs_txt = f"reached goal @ step {rs}" if rs is not None else "goal not reached"
        print(f"[eval] env {i:02d}    : init goal dist = {init_dist[i]:.2f} m  | {rs_txt}")
    print(f"[eval] videos dir : {video_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
