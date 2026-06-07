from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")

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

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mirage.encoder.data import load_antmaze
from mirage.encoder.load import load_encoder
from mirage.paths import resolve_path
from mirage.rl.agent import Agent
from mirage.rl.policy_input import PolicyInputBuilder
from mirage.utils.checkpoint import Checkpointer
from mirage.utils.running_norm import RunningMeanStd

ENC = "runs_encoder/dual_input_masked_recon/encoder_best.pt"
_PROPRIO_CONTACT = 105
_PROPRIO_27 = 27


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", default="ab_latent_bc")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no_norm", action="store_true",
                    help="disable obs normalization (must match the PPO config)")
    args = ap.parse_args()
    norm = not args.no_norm
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    enc, _, _ = load_encoder(resolve_path(ENC), dev, eval_mode=True)
    builder = PolicyInputBuilder("latent", ["latent_goal"], unified_encoder=enc)
    obs_dim = builder.obs_dim
    print(f"[bc] obs_dim={obs_dim}")

    d = load_antmaze("D4RL/antmaze/umaze-v1", datasets_path="data/minari")
    ss = d.state_starts
    sel = np.concatenate([np.arange(ss[i], ss[i + 1] - 1) for i in range(d.n_ep)])
    assert sel.shape[0] == d.act.shape[0], (sel.shape, d.act.shape)
    proprio = d.obs[sel].astype(np.float32)
    des = d.des[sel].astype(np.float32)
    ach = d.ach[sel].astype(np.float32)
    act = torch.from_numpy(d.act.astype(np.float32))
    act_dim = act.shape[1]
    N = sel.shape[0]
    print(f"[bc] transitions={N}  act_dim={act_dim}")

    pad = np.zeros((N, _PROPRIO_CONTACT - _PROPRIO_27), dtype=np.float32)
    obs107 = np.concatenate([proprio, pad, des], axis=-1)

    pis = torch.empty((N, obs_dim), dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, N, 16384):
            o = torch.from_numpy(obs107[i:i + 16384]).to(dev)
            a = torch.from_numpy(ach[i:i + 16384]).to(dev)
            pis[i:i + 16384] = builder.assemble(o, a).cpu()
    print(f"[bc] precomputed policy inputs {tuple(pis.shape)}")

    obs_norm = RunningMeanStd(obs_dim).to(dev) if norm else None
    if obs_norm is not None:
        with torch.no_grad():
            for i in range(0, N, 16384):
                obs_norm.update(pis[i:i + 16384].to(dev))
        print(f"[bc] obs_norm fit: |mean|={obs_norm.mean.abs().mean():.3f} "
              f"std={obs_norm.var.sqrt().mean():.3f}")

    agent = Agent(obs_dim=obs_dim, act_dim=act_dim).to(dev)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    for epoch in range(args.epochs):
        perm = torch.randperm(N)
        tot, nb = 0.0, 0
        for s in range(0, N, args.batch):
            mb = perm[s:s + args.batch]
            x = pis[mb].to(dev)
            if obs_norm is not None:
                x = obs_norm.normalize(x)
            pred = agent.actor_mean(x)
            loss = ((pred - act[mb].to(dev)) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += loss.item(); nb += 1
        print(f"[bc] epoch {epoch + 1:2d}/{args.epochs}  mse={tot / nb:.4f}")

    state = {
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "obs_norm": obs_norm.state_dict() if obs_norm is not None else None,
        "global_step": 0,
        "global_episodes": 0,
        "best_eval_return": None,
        "last_save_step": 0,
        "last_eval_step": 0,
    }
    run_dir = resolve_path(os.path.join("runs", args.exp_name))
    Checkpointer(run_dir, args.exp_name).save_latest(state)
    print(f"[bc] saved warm-start checkpoint -> {run_dir}/{args.exp_name}.cleanrl_model")


if __name__ == "__main__":
    main()
