from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from mirage.encoder.models import StateEncoder
from mirage.utils.checkpoint import Checkpointer
from mirage.utils.wandb_session import WandbSession

from .data import MultiStepSampler, load_antmaze
from .eval import full_eval
from .losses import gaussian_nll
from .models import DynamicsEnsemble


def _horizon_weights(H: int, rho: float) -> list[float]:
    return [rho ** k for k in range(H)]


class WorldModelTrainer:
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
        self.checkpointer = Checkpointer(str(self.run_dir), exp_name="world_model", suffix=".pt")
        self.session = WandbSession(
            run_dir=str(self.run_dir),
            project=cfg.get("wandb_project", "MIRAGE"),
            entity=cfg.get("wandb_entity"),
            run_name=cfg["run_name"],
            config=cfg,
            enabled=bool(cfg.get("wandb", False)),
            sync_tensorboard=True,
        )

    def _load_encoder_and_fwd_init(self):
        cfg = self.cfg
        enc_ckpt_path = cfg["encoder_ckpt"]
        ck = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        enc_cfg = ck["config"]
        if enc_cfg["input_mode"] != cfg["input_mode"]:
            raise ValueError(
                f"input_mode mismatch: encoder ckpt was trained with "
                f"{enc_cfg['input_mode']!r} but world-model cfg says {cfg['input_mode']!r}"
            )
        state_dim = {"xy": 2, "proprio": 27, "full": 29}[enc_cfg["input_mode"]]
        encoder = StateEncoder(
            in_dim=state_dim,
            latent_dim=enc_cfg["latent_dim"],
            hidden_dim=enc_cfg["hidden_dim"],
            n_hidden=enc_cfg["n_hidden"],
            l2_normalize=enc_cfg["l2_normalize"],
            cold_init_eps=enc_cfg["cold_init_eps"],
        )
        encoder.load_state_dict(ck["encoder"])
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
        return encoder.to(self.device), ck["fwd"], enc_cfg

    def _save_ckpt(self, path: str, step: int, ensemble, opt, scheduler, best_val, val_metrics):
        self.checkpointer.save({
            "step": int(step),
            "ensemble": ensemble.state_dict(),
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
        encoder, fwd_state, enc_cfg = self._load_encoder_and_fwd_init()
        latent_dim = int(enc_cfg["latent_dim"])
        l2_norm_targets = bool(enc_cfg["l2_normalize"])

        data = load_antmaze(cfg["dataset_id"], cfg["datasets_path"])
        train_data, val_data = data.split(cfg["val_frac"], cfg["seed"])
        H = int(cfg["horizon"])
        train_sampler = MultiStepSampler(train_data, cfg["input_mode"], H,
                                         seed=cfg["seed"], device=self.device)
        eval_H = max(H, max(cfg["eval_horizons"]))
        val_sampler = MultiStepSampler(val_data, cfg["input_mode"], eval_H,
                                       seed=cfg["seed"] + 1, device=self.device)
        act_dim = int(train_data.act.shape[-1])

        ensemble = DynamicsEnsemble(
            n_members=int(cfg["n_members"]),
            latent_dim=latent_dim,
            act_dim=act_dim,
            hidden_dim=int(cfg["hidden_dim"]),
            n_hidden=int(cfg["n_hidden"]),
        ).to(self.device)
        if bool(cfg.get("init_from_fwd_ckpt", True)):
            ensemble.load_from_forward_dynamics(fwd_state,
                                                noise_std=float(cfg.get("init_noise_std", 1e-3)))

        opt = torch.optim.AdamW(ensemble.parameters(),
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
                ensemble.load_state_dict(state["ensemble"])
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
                print(f"[world_model] resumed from step {start_step - 1}  best_val={best_val:.4f}")
            else:
                print("[world_model] no resumable checkpoint; starting fresh (encoder fwd init used)")

        self.session.init()
        n_params = sum(p.numel() for p in ensemble.parameters())
        print(f"[world_model] device={self.device}  run_dir={self.run_dir}  params={n_params/1e6:.2f}M")
        print(f"[world_model] n_members={cfg['n_members']}  H={H}  latent_dim={latent_dim}  act_dim={act_dim}")

        weights = _horizon_weights(H, float(cfg["horizon_decay"]))
        weight_t = torch.tensor(weights, device=self.device, dtype=torch.float32)
        print(f"[world_model] horizon weights = {[round(w, 4) for w in weights]}")

        grad_clip = float(cfg["grad_clip"])
        t_start = time.time()
        last_log_t = t_start

        for step in range(start_step, total_steps + 1):
            ensemble.train()
            b = train_sampler.batch(int(cfg["batch_size"]))
            with torch.no_grad():
                flat_states = b["states"].reshape(-1, b["states"].shape[-1])
                z_seq = encoder(flat_states).view(b["states"].shape[0], H + 1, latent_dim)

            z = z_seq[:, 0]
            step_losses = []
            loss_type = str(cfg.get("loss_type", "nll"))
            for k in range(H):
                a = b["actions"][:, k]
                mu_all, log_sigma_all = ensemble(z, a)
                target = z_seq[:, k + 1]
                target_e = target.unsqueeze(0).expand_as(mu_all)
                if loss_type == "mse":
                    loss_k = F.mse_loss(mu_all, target_e)
                else:
                    loss_k = gaussian_nll(mu_all, log_sigma_all, target_e)
                step_losses.append(loss_k)
                z = mu_all.mean(dim=0).detach()

            step_loss_t = torch.stack(step_losses)
            loss = (step_loss_t * weight_t).sum() / weight_t.sum()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ensemble.parameters(), grad_clip)
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
                _loss_tag = "mse" if str(cfg.get("loss_type", "nll")) == "mse" else "nll"
                for k, sl in enumerate(step_losses):
                    log[f"train/{_loss_tag}_step{k+1}"] = sl.item()
                for k, v in log.items():
                    self.writer.add_scalar(k, v, step)
                step_summary = " ".join(f"k{k+1}={sl.item():.3f}" for k, sl in enumerate(step_losses))
                print(f"[step {step:>7d}] loss={loss.item():.4f}  {step_summary}  lr={lr:.2e}  sps={sps:.0f}")

            if step % int(cfg["eval_every"]) == 0 or step == total_steps:
                vm = full_eval(encoder, ensemble, val_sampler, self.device,
                               n_batches=int(cfg["eval_batches"]),
                               batch_size=int(cfg["eval_batch_size"]),
                               horizons=tuple(cfg["eval_horizons"]),
                               loss_type=str(cfg.get("loss_type", "nll")))
                for k, v in vm.items():
                    self.writer.add_scalar(k, v, step)
                key = "val/mse_step1" if str(cfg.get("loss_type", "nll")) == "mse" else "val/nll_step1"
                val_metric = vm[key]
                pretty = "  ".join(f"{k.split('/',1)[1]}={v:.4f}" for k, v in vm.items())
                print(f"[eval  {step:>7d}] {pretty}")
                if val_metric < best_val:
                    best_val = val_metric
                    self._save_ckpt(self.checkpointer.best_path, step, ensemble, opt, scheduler, best_val, vm)

            if step % int(cfg["ckpt_every"]) == 0 or step == total_steps:
                self._save_ckpt(self.checkpointer.latest_path, step, ensemble, opt, scheduler, best_val, {})

        elapsed = time.time() - t_start
        print(f"[done] total wall time: {elapsed/60:.1f} min")
        final_vm = full_eval(encoder, ensemble, val_sampler, self.device,
                             n_batches=max(16, int(cfg["eval_batches"])),
                             batch_size=int(cfg["eval_batch_size"]),
                             horizons=tuple(cfg["eval_horizons"]),
                             loss_type=str(cfg.get("loss_type", "nll")))
        final_vm["wall_time_min"] = elapsed / 60.0
        final_vm["total_steps"] = total_steps
        final_vm["run_name"] = cfg["run_name"]
        with open(self.run_dir / "results.json", "w") as f:
            json.dump(final_vm, f, indent=2)
        print(f"[final]  {json.dumps(final_vm, indent=2)}")
        self.writer.close()
        if self.session.run is not None:
            for k, v in final_vm.items():
                if isinstance(v, (int, float)):
                    self.session.run.summary[k] = v
            self.session.finish()
