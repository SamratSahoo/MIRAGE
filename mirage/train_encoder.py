from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from mirage.data import load_antmaze
from mirage.data.antmaze import Sampler
from mirage.eval import full_eval
from mirage.losses import forward_dyn_loss, info_nce, inverse_dyn_loss
from mirage.models import ForwardDynamics, InverseDynamics, StateEncoder


def parse_args() -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/encoder_umaze.yaml")
    args, extra = p.parse_known_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    i = 0
    while i < len(extra):
        key = extra[i].lstrip("-")
        val = extra[i + 1]
        if key in cfg:
            default = cfg[key]
            if isinstance(default, bool):
                cfg[key] = val.lower() in ("1", "true", "yes")
            elif isinstance(default, int):
                cfg[key] = int(val)
            elif isinstance(default, float):
                cfg[key] = float(val)
            else:
                cfg[key] = val
        else:
            cfg[key] = val
        i += 2
    return cfg


def main():
    cfg = parse_args()

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])

    run_dir = Path(cfg["log_root"]) / cfg["run_name"]
    ckpt_dir = Path(cfg["ckpt_root"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    wandb_run = None
    if cfg.get("wandb", False) and os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb_run = wandb.init(
            project=cfg.get("wandb_project", "MIRAGE"),
            entity=cfg.get("wandb_entity"),
            name=cfg["run_name"],
            config=cfg,
            dir=str(run_dir),
            sync_tensorboard=True,
            reinit=True,
        )
        print(f"[wandb] run url: {wandb_run.url}")
    elif cfg.get("wandb", False):
        print("[wandb] WANDB_API_KEY not set; skipping wandb init")

    print(f"[train_encoder] device={device}  run_dir={run_dir}")
    print(f"[train_encoder] config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    t0 = time.time()
    data = load_antmaze(cfg["dataset_id"], cfg["datasets_path"])
    train_data, val_data = data.split(cfg["val_frac"], cfg["seed"])
    print(f"[data] loaded in {time.time()-t0:.1f}s  "
          f"train_eps={train_data.n_ep}  val_eps={val_data.n_ep}  "
          f"train_trans={train_data.n_trans}  "
          f"ep_len min/median/max = {train_data.ep_lens.min()}/{int(np.median(train_data.ep_lens))}/{train_data.ep_lens.max()}")
    print(f"[data] input_mode={cfg['input_mode']}  state_dim={data.state_dim(cfg['input_mode'])}")

    state_dim = data.state_dim(cfg["input_mode"])
    act_dim = data.act.shape[-1]

    train_sampler = Sampler(train_data, cfg["input_mode"], cfg["contrastive_window"],
                            seed=cfg["seed"], device=device)
    val_sampler   = Sampler(val_data,   cfg["input_mode"], cfg["contrastive_window"],
                            seed=cfg["seed"] + 1, device=device)

    encoder = StateEncoder(in_dim=state_dim,
                           latent_dim=cfg["latent_dim"],
                           hidden_dim=cfg["hidden_dim"],
                           n_hidden=cfg["n_hidden"],
                           l2_normalize=cfg["l2_normalize"],
                           cold_init_eps=cfg["cold_init_eps"]).to(device)
    fwd = ForwardDynamics(latent_dim=cfg["latent_dim"], act_dim=act_dim,
                          hidden_dim=cfg["dyn_hidden_dim"], n_hidden=cfg["dyn_n_hidden"]).to(device)
    inv = InverseDynamics(latent_dim=cfg["latent_dim"], act_dim=act_dim,
                          hidden_dim=cfg["dyn_hidden_dim"], n_hidden=cfg["dyn_n_hidden"]).to(device)

    n_params = sum(p.numel() for p in list(encoder.parameters()) + list(fwd.parameters()) + list(inv.parameters()))
    print(f"[model] total params: {n_params/1e6:.2f}M")

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(fwd.parameters()) + list(inv.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )

    best_val = float("inf")
    t_start = time.time()
    last_log_t = t_start
    for step in range(1, cfg["total_steps"] + 1):
        encoder.train(); fwd.train(); inv.train()
        pb = train_sampler.pair_batch(cfg["batch_size"])
        z1 = encoder(pb["s1"]); z2 = encoder(pb["s2"])
        nce_loss, nce_log = info_nce(z1, z2, temperature=cfg["nce_temperature"])

        ib = train_sampler.iid_batch(cfg["batch_size"])
        z   = encoder(ib["s"])
        z_n = encoder(ib["s_next"])
        z_n_pred = fwd(z, ib["a"])
        a_pred   = inv(z, z_n)
        fwd_loss, fwd_log = forward_dyn_loss(z_n_pred, z_n.detach())
        inv_loss, inv_log = inverse_dyn_loss(a_pred, ib["a"])

        loss = (cfg["nce_weight"] * nce_loss
                + cfg["fwd_weight"] * fwd_loss
                + cfg["inv_weight"] * inv_loss)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(fwd.parameters()) + list(inv.parameters()),
                cfg["grad_clip"],
            )
        opt.step()

        if step % cfg["log_every"] == 0 or step == 1:
            now = time.time()
            sps = cfg["log_every"] / max(now - last_log_t, 1e-9)
            last_log_t = now
            log = {**nce_log, **fwd_log, **inv_log,
                   "train/total_loss": loss.item(),
                   "train/steps_per_sec": sps}
            for k, v in log.items():
                writer.add_scalar(k, v, step)
            print(f"[step {step:>7d}] loss={loss.item():.4f}  "
                  f"nce={nce_log['info_nce/loss']:.4f}  "
                  f"fwd={fwd_log['forward_dyn/mse']:.5f}  "
                  f"inv={inv_log['inverse_dyn/mse']:.5f}  "
                  f"top1={nce_log['info_nce/top1_acc']:.3f}  "
                  f"sps={sps:.0f}")

        if step % cfg["eval_every"] == 0 or step == cfg["total_steps"]:
            vm = full_eval(encoder, fwd, inv, val_data, val_sampler, device,
                           out_dir=run_dir / f"eval_step{step}",
                           n_batches=cfg["eval_batches"],
                           batch_size=cfg["eval_batch_size"],
                           temperature=cfg["nce_temperature"])
            for k, v in vm.items():
                writer.add_scalar(k, v, step)
            val_metric = vm["val/info_nce_loss"]
            print(f"[eval  {step:>7d}] " + "  ".join(f"{k.split('/',1)[1]}={v:.4f}" for k, v in vm.items()))
            if val_metric < best_val:
                best_val = val_metric
                torch.save({"step": step, "encoder": encoder.state_dict(),
                            "fwd": fwd.state_dict(), "inv": inv.state_dict(),
                            "config": cfg, "val_metrics": vm},
                           ckpt_dir / "best.pt")

        if step % cfg["ckpt_every"] == 0 or step == cfg["total_steps"]:
            torch.save({"step": step, "encoder": encoder.state_dict(),
                        "fwd": fwd.state_dict(), "inv": inv.state_dict(),
                        "config": cfg},
                       ckpt_dir / "last.pt")

    elapsed = time.time() - t_start
    print(f"[done] total wall time: {elapsed/60:.1f} min")

    final_vm = full_eval(encoder, fwd, inv, val_data, val_sampler, device,
                         out_dir=run_dir / "eval_final",
                         n_batches=max(32, cfg["eval_batches"]),
                         batch_size=cfg["eval_batch_size"],
                         temperature=cfg["nce_temperature"])
    final_vm["wall_time_min"] = elapsed / 60.0
    final_vm["total_steps"] = cfg["total_steps"]
    final_vm["run_name"] = cfg["run_name"]
    final_vm["input_mode"] = cfg["input_mode"]
    with open(run_dir / "results.json", "w") as f:
        json.dump(final_vm, f, indent=2)
    print(f"[final]  {json.dumps(final_vm, indent=2)}")
    writer.close()
    if wandb_run is not None:
        for k, v in final_vm.items():
            if isinstance(v, (int, float)):
                wandb_run.summary[k] = v
        wandb_run.finish()


if __name__ == "__main__":
    main()
