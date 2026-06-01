from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from mirage.encoder.load import load_encoder
from mirage.utils.checkpoint import Checkpointer
from mirage.utils.wandb_session import WandbSession

from .data import InverseWindowSampler, load_antmaze
from .eval import full_eval
from .losses import inverse_losses
from .models import InverseWorldModel


class InverseWorldModelTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(cfg["seed"])

        self.run_dir = Path(cfg["log_root"]) / cfg["run_name"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f)

        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        self.checkpointer = Checkpointer(str(self.run_dir), exp_name="inverse_world_model", suffix=".pt")
        self.session = WandbSession(
            run_dir=str(self.run_dir),
            project=cfg.get("wandb_project", "MIRAGE"),
            entity=cfg.get("wandb_entity"),
            run_name=cfg["run_name"],
            config=cfg,
            enabled=bool(cfg.get("wandb", False)),
            sync_tensorboard=True,
        )

    def _save_ckpt(self, path: str, step: int, model, opt, scheduler, best_val, val_metrics):
        self.checkpointer.save({
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_val": float(best_val),
            "val_metrics": val_metrics,
            "config": self.cfg,
            "rng": {
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
            },
        }, path)

    def train(self):
        cfg = self.cfg
        encoder, enc_cfg, _ = load_encoder(cfg["encoder_ckpt"], device=self.device, eval_mode=True)
        if enc_cfg["input_mode"] != cfg["input_mode"]:
            raise ValueError(
                f"input_mode mismatch: encoder ckpt was trained with "
                f"{enc_cfg['input_mode']!r} but inverse-wm cfg says {cfg['input_mode']!r}"
            )
        latent_dim = int(enc_cfg["latent_dim"])

        data = load_antmaze(cfg["dataset_id"], cfg["datasets_path"])
        train_data, val_data = data.split(cfg["val_frac"], cfg["seed"])
        K = int(cfg["k_max"])
        train_sampler = InverseWindowSampler(train_data, cfg["input_mode"], K,
                                             seed=cfg["seed"], device=self.device)
        val_sampler = InverseWindowSampler(val_data, cfg["input_mode"], K,
                                           seed=cfg["seed"] + 1, device=self.device)
        act_dim = int(train_data.act.shape[-1])

        model = InverseWorldModel(
            latent_dim=latent_dim,
            act_dim=act_dim,
            k_max=K,
            hidden_dim=int(cfg["hidden_dim"]),
            n_hidden=int(cfg["n_hidden"]),
            predict_k=bool(cfg.get("predict_k", False)),
        ).to(self.device)

        opt = torch.optim.AdamW(model.parameters(),
                                lr=float(cfg["lr"]),
                                weight_decay=float(cfg["weight_decay"]),
                                betas=(0.9, 0.999))
        total_steps = int(cfg["total_steps"])
        scheduler = None
        if float(cfg.get("cosine_decay_to", 0.0)) > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=total_steps, eta_min=float(cfg["cosine_decay_to"]))

        start_step = 1
        best_val = float("inf")
        if bool(cfg.get("load_default_checkpoint", True)):
            state = self.checkpointer.load(map_location=self.device)
            if state is not None:
                model.load_state_dict(state["model"])
                opt.load_state_dict(state["optimizer"])
                if scheduler is not None and state.get("scheduler") is not None:
                    scheduler.load_state_dict(state["scheduler"])
                start_step = int(state["step"]) + 1
                best_val = float(state.get("best_val", float("inf")))
                if "rng" in state:
                    rng = state["rng"]
                    torch.set_rng_state(rng["torch"].to("cpu", dtype=torch.uint8))
                    cuda_state = rng.get("torch_cuda")
                    if cuda_state is not None and torch.cuda.is_available():
                        torch.cuda.set_rng_state_all([s.to("cpu", dtype=torch.uint8) for s in cuda_state])
                    np.random.set_state(rng["numpy"])
                print(f"[inverse_wm] resumed from step {start_step - 1}  best_val={best_val:.4f}")
            else:
                print("[inverse_wm] no resumable checkpoint; starting fresh")

        self.session.init()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[inverse_wm] device={self.device}  run_dir={self.run_dir}  params={n_params/1e6:.2f}M")
        print(f"[inverse_wm] k_max={K}  latent_dim={latent_dim}  act_dim={act_dim}")

        weights = {
            "action": float(cfg["w_action"]),
            "latent": float(cfg["w_latent"]),
            "reward": float(cfg["w_reward"]),
            "xydist": float(cfg["w_xydist"]),
            "k": float(cfg.get("w_k", 1.0)),
        }
        print(f"[inverse_wm] predict_k={model.predict_k}  loss weights = {weights}")

        grad_clip = float(cfg["grad_clip"])
        t_start = time.time()
        last_log_t = t_start

        for step in range(start_step, total_steps + 1):
            model.train()
            b = train_sampler.batch(int(cfg["batch_size"]))
            states = b["states"]
            B = states.shape[0]
            with torch.no_grad():
                flat = states.reshape(-1, states.shape[-1])
                z_seq = encoder.encode_full(flat).view(B, K + 1, latent_dim)
            z0 = z_seq[:, 0]
            k = b["k"]
            zk = z_seq[torch.arange(B, device=self.device), k]

            if model.predict_k:
                out = model(z0, zk)
            else:
                out = model(z0, zk, k.float() / float(K))
            loss, parts = inverse_losses(out, model, z0, z_seq, b["actions"], k,
                                         b["reward"], b["xydist"], weights)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if scheduler is not None:
                scheduler.step()

            if step % int(cfg["log_every"]) == 0 or step == 1:
                now = time.time()
                sps = int(cfg["log_every"]) / max(now - last_log_t, 1e-9)
                last_log_t = now
                lr = opt.param_groups[0]["lr"]
                log = {
                    "train/total_loss": loss.item(),
                    "train/steps_per_sec": sps,
                    "train/lr": lr,
                }
                for name, v in parts.items():
                    log[f"train/{name}_mse"] = v.item()
                for kk, vv in log.items():
                    self.writer.add_scalar(kk, vv, step)
                psummary = " ".join(f"{n}={v.item():.3f}" for n, v in parts.items())
                print(f"[step {step:>7d}] loss={loss.item():.4f}  {psummary}  lr={lr:.2e}  sps={sps:.0f}")

            if step % int(cfg["eval_every"]) == 0 or step == total_steps:
                vm = full_eval(encoder, model, val_sampler, self.device, latent_dim,
                               n_batches=int(cfg["eval_batches"]),
                               batch_size=int(cfg["eval_batch_size"]),
                               k_breakdown=tuple(cfg["eval_k_breakdown"]))
                for kk, vv in vm.items():
                    self.writer.add_scalar(kk, vv, step)
                val_metric = vm["val/action_mse"]
                pretty = "  ".join(f"{kk.split('/',1)[1]}={vv:.4f}" for kk, vv in vm.items())
                print(f"[eval  {step:>7d}] {pretty}")
                if val_metric < best_val:
                    best_val = val_metric
                    self._save_ckpt(self.checkpointer.best_path, step, model, opt, scheduler, best_val, vm)

            if step % int(cfg["ckpt_every"]) == 0 or step == total_steps:
                self._save_ckpt(self.checkpointer.latest_path, step, model, opt, scheduler, best_val, {})

        elapsed = time.time() - t_start
        print(f"[done] total wall time: {elapsed/60:.1f} min")
        final_vm = full_eval(encoder, model, val_sampler, self.device, latent_dim,
                             n_batches=max(16, int(cfg["eval_batches"])),
                             batch_size=int(cfg["eval_batch_size"]),
                             k_breakdown=tuple(cfg["eval_k_breakdown"]))
        final_vm["wall_time_min"] = elapsed / 60.0
        final_vm["total_steps"] = total_steps
        final_vm["run_name"] = cfg["run_name"]
        with open(self.run_dir / "results.json", "w") as f:
            json.dump(final_vm, f, indent=2)
        print(f"[final]  {json.dumps(final_vm, indent=2)}")
        self.writer.close()
        if self.session.run is not None:
            for kk, vv in final_vm.items():
                if isinstance(vv, (int, float)):
                    self.session.run.summary[kk] = vv
            self.session.finish()
