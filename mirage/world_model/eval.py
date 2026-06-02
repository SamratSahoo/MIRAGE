from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .losses import gaussian_nll


@torch.no_grad()
def rollout_eval(encoder, ensemble, sampler, n_batches: int = 8, batch_size: int = 256,
                 horizons: tuple[int, ...] = (1, 3, 5, 10),
                 loss_type: str = "nll") -> dict:
    encoder.eval(); ensemble.eval()
    H_sampler = sampler.H
    max_h = max(horizons)
    if max_h > H_sampler:
        raise ValueError(f"requested horizon {max_h} > sampler H={H_sampler}")

    step1_metric_vals = []
    mse_per_step = {h: [] for h in horizons}
    disagreement_per_step = {h: [] for h in horizons}

    for _ in range(n_batches):
        b = sampler.batch(batch_size)
        flat_states = b["states"].reshape(-1, b["states"].shape[-1])
        z_flat = encoder(flat_states)
        z_seq = z_flat.view(batch_size, H_sampler + 1, -1)
        z = z_seq[:, 0]
        for step in range(1, max_h + 1):
            a = b["actions"][:, step - 1]
            mu_all, log_sigma_all = ensemble(z, a)
            mu_mean = mu_all.mean(dim=0)
            target = z_seq[:, step]
            if step == 1:
                if loss_type == "mse":
                    per_member = []
                    for e in range(ensemble.n_members):
                        per_member.append(F.mse_loss(mu_all[e], target).item())
                    step1_metric_vals.append(float(np.mean(per_member)))
                else:
                    per_member = []
                    for e in range(ensemble.n_members):
                        per_member.append(gaussian_nll(mu_all[e], log_sigma_all[e], target).item())
                    step1_metric_vals.append(float(np.mean(per_member)))
            mse = F.mse_loss(mu_mean, target).item()
            disag = (mu_all - mu_mean.unsqueeze(0)).norm(dim=-1).mean().item()
            if step in mse_per_step:
                mse_per_step[step].append(mse)
                disagreement_per_step[step].append(disag)
            z = mu_mean

    step1_key = "val/mse_step1" if loss_type == "mse" else "val/nll_step1"
    out = {step1_key: float(np.mean(step1_metric_vals))}
    for h in horizons:
        out[f"val/rollout_mse_h{h}"] = float(np.mean(mse_per_step[h]))
        out[f"val/disagreement_h{h}"] = float(np.mean(disagreement_per_step[h]))
    return out


@torch.no_grad()
def xy_probe_at_h(encoder, ensemble, sampler, horizon: int = 3,
                  n_batches: int = 8, batch_size: int = 256) -> dict:
    encoder.eval(); ensemble.eval()
    Z_real, Z_pred, XY = [], [], []
    H_sampler = sampler.H
    if horizon > H_sampler:
        raise ValueError(f"horizon {horizon} > sampler H={H_sampler}")
    for _ in range(n_batches):
        b = sampler.batch(batch_size)
        flat_states = b["states"].reshape(-1, b["states"].shape[-1])
        z_flat = encoder(flat_states)
        z_seq = z_flat.view(batch_size, H_sampler + 1, -1)
        z = z_seq[:, 0]
        for step in range(1, horizon + 1):
            a = b["actions"][:, step - 1]
            mu_all, _ = ensemble(z, a)
            mu_mean = mu_all.mean(dim=0)
            z = mu_mean
        Z_pred.append(mu_mean.cpu().numpy())
        Z_real.append(z_seq[:, horizon].cpu().numpy())
        XY.append(b["xy"][:, horizon].cpu().numpy())
    Z_real = np.concatenate(Z_real, 0)
    Z_pred = np.concatenate(Z_pred, 0)
    XY = np.concatenate(XY, 0)

    def _r2(Z, Y):
        n = Z.shape[0]
        n_val = max(1, n // 5)
        perm = np.random.default_rng(0).permutation(n)
        va, tr = perm[:n_val], perm[n_val:]
        lam = 1e-3
        A = Z[tr].T @ Z[tr] + lam * np.eye(Z.shape[1], dtype=Z.dtype)
        W = np.linalg.solve(A, Z[tr].T @ Y[tr])
        Yhat = Z[va] @ W
        ss_res = ((Y[va] - Yhat) ** 2).sum()
        ss_tot = ((Y[va] - Y[va].mean(0)) ** 2).sum()
        return 1.0 - ss_res / max(ss_tot, 1e-9)

    return {
        f"val/xy_probe_real_h{horizon}":  float(_r2(Z_real, XY)),
        f"val/xy_probe_wmpred_h{horizon}": float(_r2(Z_pred, XY)),
    }


def full_eval(encoder, ensemble, sampler, device, n_batches: int = 8,
              batch_size: int = 256, horizons: tuple[int, ...] = (1, 3, 5, 10),
              loss_type: str = "nll") -> dict:
    out = {}
    out.update(rollout_eval(encoder, ensemble, sampler, n_batches=n_batches,
                            batch_size=batch_size, horizons=horizons,
                            loss_type=loss_type))
    h_probe = min(3, max(horizons))
    out.update(xy_probe_at_h(encoder, ensemble, sampler, horizon=h_probe,
                              n_batches=n_batches, batch_size=batch_size))
    return out
