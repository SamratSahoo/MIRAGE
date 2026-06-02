from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def full_eval(encoder, model, sampler, device, latent_dim: int,
              n_batches: int = 8, batch_size: int = 256,
              k_breakdown: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    encoder.eval(); model.eval()
    K = model.k_max
    k_norm_div = float(K)

    act_vals, lat_vals, rew_vals, xy_vals, k_mae_vals = [], [], [], [], []
    per_k_act = {k: [] for k in k_breakdown if k <= K}

    for _ in range(n_batches):
        b = sampler.batch(batch_size)
        if "latents" in b:
            z_seq = b["latents"]
            B = z_seq.shape[0]
        else:
            states = b["states"]
            B = states.shape[0]
            flat = states.reshape(-1, states.shape[-1])
            z_seq = encoder.encode_full(flat).view(B, K + 1, latent_dim)
        z0 = z_seq[:, 0]
        k = b["k"]
        zk = z_seq[torch.arange(B, device=device), k]
        k_norm = k.float() / k_norm_div

        out = model(z0, zk) if model.predict_k else model(z0, zk, k_norm)
        if "k_pred" in out:
            k_mae_vals.append(((out["k_pred"] * k_norm_div) - k.float()).abs().mean().item())
        idx = torch.arange(K, device=device).unsqueeze(0)

        act_mask = (idx < k.unsqueeze(1)).float()
        act_sq = ((out["actions"] - b["actions"]) ** 2).sum(dim=-1)
        act_vals.append((act_sq * act_mask).sum().item() / act_mask.sum().clamp_min(1.0).item())

        z_recon = model.reconstruct_latents(z0, out["deltas"])
        lat_mask = ((idx >= 0) & (idx < (k.unsqueeze(1) - 1))).float()
        lat_sq = ((z_recon - z_seq[:, 1:]) ** 2).sum(dim=-1)
        denom = lat_mask.sum().clamp_min(1.0).item()
        lat_vals.append((lat_sq * lat_mask).sum().item() / denom)

        rew_vals.append(((out["reward"] - b["reward"]) ** 2).mean().item())
        xy_vals.append(((out["xydist"] - b["xydist"]) ** 2).mean().item())

    for kb in per_k_act:
        bk = sampler.batch(batch_size, fixed_k=kb)
        if "latents" in bk:
            z_seq = bk["latents"]
            B = z_seq.shape[0]
        else:
            states = bk["states"]
            B = states.shape[0]
            flat = states.reshape(-1, states.shape[-1])
            z_seq = encoder.encode_full(flat).view(B, K + 1, latent_dim)
        z0 = z_seq[:, 0]
        kvec = bk["k"]
        zk = z_seq[torch.arange(B, device=device), kvec]
        k_norm = kvec.float() / k_norm_div
        out = model(z0, zk) if model.predict_k else model(z0, zk, k_norm)
        idx = torch.arange(K, device=device).unsqueeze(0)
        act_mask = (idx < kvec.unsqueeze(1)).float()
        act_sq = ((out["actions"] - bk["actions"]) ** 2).sum(dim=-1)
        per_k_act[kb].append((act_sq * act_mask).sum().item() / act_mask.sum().clamp_min(1.0).item())

    out_d = {
        "val/action_mse": float(np.mean(act_vals)),
        "val/latent_mse": float(np.mean(lat_vals)),
        "val/reward_mse": float(np.mean(rew_vals)),
        "val/xydist_mse": float(np.mean(xy_vals)),
    }
    for kb, vals in per_k_act.items():
        out_d[f"val/action_mse_k{kb}"] = float(np.mean(vals))
    if k_mae_vals:
        out_d["val/k_mae_steps"] = float(np.mean(k_mae_vals))
    return out_d
