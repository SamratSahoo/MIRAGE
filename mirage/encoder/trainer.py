from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from mirage.utils.checkpoint import Checkpointer
from mirage.utils.wandb_session import WandbSession

from .data import Sampler, load_antmaze
from .eval import full_eval
from .losses import forward_dyn_loss, info_nce, inverse_dyn_loss
from .models import ForwardDynamics, InverseDynamics, StateEncoder


class EncoderTrainer:
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
        self.checkpointer = Checkpointer(str(self.run_dir), exp_name="encoder", suffix=".pt")
        self.session = WandbSession(
            run_dir=str(self.run_dir),
            project=cfg.get("wandb_project", "MIRAGE"),
            entity=cfg.get("wandb_entity"),
            run_name=cfg["run_name"],
            config=cfg,
            enabled=bool(cfg.get("wandb", False)),
            sync_tensorboard=True,
        )

    def _build_state(self):
        cfg = self.cfg
        data = load_antmaze(cfg["dataset_id"], cfg["datasets_path"])
        train_data, val_data = data.split(cfg["val_frac"], cfg["seed"])
        state_dim = data.state_dim(cfg["input_mode"])
        act_dim = int(data.act.shape[-1])

        train_sampler = Sampler(train_data, cfg["input_mode"], cfg["contrastive_window"],
                                seed=cfg["seed"], device=self.device)
        val_sampler = Sampler(val_data, cfg["input_mode"], cfg["contrastive_window"],
                              seed=cfg["seed"] + 1, device=self.device)

        encoder = StateEncoder(in_dim=state_dim,
                               latent_dim=cfg["latent_dim"],
                               hidden_dim=cfg["hidden_dim"],
                               n_hidden=cfg["n_hidden"],
                               l2_normalize=cfg["l2_normalize"],
                               cold_init_eps=cfg["cold_init_eps"]).to(self.device)
        fwd = ForwardDynamics(latent_dim=cfg["latent_dim"], act_dim=act_dim,
                              hidden_dim=cfg["dyn_hidden_dim"], n_hidden=cfg["dyn_n_hidden"]).to(self.device)
        inv = InverseDynamics(latent_dim=cfg["latent_dim"], act_dim=act_dim,
                              hidden_dim=cfg["dyn_hidden_dim"], n_hidden=cfg["dyn_n_hidden"]).to(self.device)
        opt = torch.optim.Adam(
            list(encoder.parameters()) + list(fwd.parameters()) + list(inv.parameters()),
            lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        )
        return train_data, val_data, train_sampler, val_sampler, encoder, fwd, inv, opt

    def _save_ckpt(self, path: str, step: int, encoder, fwd, inv, opt, best_val, val_metrics):
        self.checkpointer.save({
            "step": int(step),
            "encoder": encoder.state_dict(),
            "fwd": fwd.state_dict(),
            "inv": inv.state_dict(),
            "optimizer": opt.state_dict(),
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
        (train_data, val_data, train_sampler, val_sampler,
         encoder, fwd, inv, opt) = self._build_state()

        load_default = bool(cfg.get("load_default_checkpoint", True))
        start_step = 1
        best_val = float("inf")
        if load_default:
            state = self.checkpointer.load(map_location=self.device)
            if state is not None:
                encoder.load_state_dict(state["encoder"])
                fwd.load_state_dict(state["fwd"])
                inv.load_state_dict(state["inv"])
                opt.load_state_dict(state["optimizer"])
                start_step = int(state["step"]) + 1
                best_val = float(state.get("best_val", float("inf")))
                if "rng" in state:
                    rng = state["rng"]
                    torch.set_rng_state(rng["torch"].to("cpu", dtype=torch.uint8))
                    cuda_state = rng.get("torch_cuda")
                    if cuda_state is not None and torch.cuda.is_available():
                        torch.cuda.set_rng_state_all([s.to("cpu", dtype=torch.uint8) for s in cuda_state])
                    np.random.set_state(rng["numpy"])
                print(f"[encoder] resumed from step {start_step - 1}  best_val={best_val:.4f}")
            else:
                print("[encoder] no checkpoint found; training from scratch")

        self.session.init()
        n_params = sum(p.numel() for p in list(encoder.parameters()) + list(fwd.parameters()) + list(inv.parameters()))
        print(f"[encoder] device={self.device}  run_dir={self.run_dir}  params={n_params/1e6:.2f}M")
        print(f"[encoder] input_mode={cfg['input_mode']}  state_dim={train_data.state_dim(cfg['input_mode'])}")

        t_start = time.time()
        last_log_t = t_start
        total_steps = int(cfg["total_steps"])
        for step in range(start_step, total_steps + 1):
            encoder.train(); fwd.train(); inv.train()
            pb = train_sampler.pair_batch(cfg["batch_size"])
            z1 = encoder(pb["s1"]); z2 = encoder(pb["s2"])
            nce_loss, nce_log = info_nce(z1, z2, temperature=cfg["nce_temperature"])

            ib = train_sampler.iid_batch(cfg["batch_size"])
            z = encoder(ib["s"])
            z_n = encoder(ib["s_next"])
            z_n_pred = fwd(z, ib["a"])
            a_pred = inv(z, z_n)
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
                    self.writer.add_scalar(k, v, step)
                print(f"[step {step:>7d}] loss={loss.item():.4f}  "
                      f"nce={nce_log['info_nce/loss']:.4f}  "
                      f"fwd={fwd_log['forward_dyn/mse']:.5f}  "
                      f"inv={inv_log['inverse_dyn/mse']:.5f}  "
                      f"top1={nce_log['info_nce/top1_acc']:.3f}  "
                      f"sps={sps:.0f}")

            if step % cfg["eval_every"] == 0 or step == total_steps:
                vm = full_eval(encoder, fwd, inv, val_data, val_sampler, self.device,
                               out_dir=self.run_dir / f"eval_step{step}",
                               n_batches=cfg["eval_batches"],
                               batch_size=cfg["eval_batch_size"],
                               temperature=cfg["nce_temperature"])
                for k, v in vm.items():
                    self.writer.add_scalar(k, v, step)
                val_metric = vm["val/info_nce_loss"]
                print(f"[eval  {step:>7d}] " + "  ".join(f"{k.split('/',1)[1]}={v:.4f}" for k, v in vm.items()))
                if val_metric < best_val:
                    best_val = val_metric
                    self._save_ckpt(self.checkpointer.best_path, step, encoder, fwd, inv, opt, best_val, vm)

            if step % cfg["ckpt_every"] == 0 or step == total_steps:
                self._save_ckpt(self.checkpointer.latest_path, step, encoder, fwd, inv, opt, best_val, {})

        elapsed = time.time() - t_start
        print(f"[done] total wall time: {elapsed/60:.1f} min")

        final_vm = full_eval(encoder, fwd, inv, val_data, val_sampler, self.device,
                             out_dir=self.run_dir / "eval_final",
                             n_batches=max(32, cfg["eval_batches"]),
                             batch_size=cfg["eval_batch_size"],
                             temperature=cfg["nce_temperature"])
        final_vm["wall_time_min"] = elapsed / 60.0
        final_vm["total_steps"] = total_steps
        final_vm["run_name"] = cfg["run_name"]
        final_vm["input_mode"] = cfg["input_mode"]
        with open(self.run_dir / "results.json", "w") as f:
            json.dump(final_vm, f, indent=2)
        print(f"[final]  {json.dumps(final_vm, indent=2)}")
        self.writer.close()
        if self.session.run is not None:
            for k, v in final_vm.items():
                if isinstance(v, (int, float)):
                    self.session.run.summary[k] = v
            self.session.finish()
