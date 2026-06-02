from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_seq_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sq = ((pred - target) ** 2).sum(dim=-1)
    denom = mask.sum().clamp_min(1.0)
    return (sq * mask).sum() / denom


def inverse_losses(out: dict, model, z0: torch.Tensor, z_seq: torch.Tensor,
                   actions: torch.Tensor, k: torch.Tensor,
                   reward: torch.Tensor, xydist: torch.Tensor,
                   weights: dict) -> tuple[torch.Tensor, dict]:
    B, k_max, _ = out["actions"].shape
    device = z0.device
    idx = torch.arange(k_max, device=device).unsqueeze(0)

    act_mask = (idx < k.unsqueeze(1)).float()
    act_loss = masked_seq_mse(out["actions"], actions, act_mask)

    z_recon = model.reconstruct_latents(z0, out["deltas"])
    lat_mask = ((idx >= 0) & (idx < (k.unsqueeze(1) - 1))).float()
    lat_loss = masked_seq_mse(z_recon, z_seq[:, 1:], lat_mask)

    rew_loss = F.mse_loss(out["reward"], reward)
    xy_loss = F.mse_loss(out["xydist"], xydist)

    total = (weights["action"] * act_loss
             + weights["latent"] * lat_loss
             + weights["reward"] * rew_loss
             + weights["xydist"] * xy_loss)
    parts = {
        "action": act_loss.detach(),
        "latent": lat_loss.detach(),
        "reward": rew_loss.detach(),
        "xydist": xy_loss.detach(),
    }
    if "k_pred" in out:
        k_target = k.float() / float(model.k_max)
        k_loss = F.mse_loss(out["k_pred"], k_target)
        total = total + weights.get("k", 1.0) * k_loss
        parts["k"] = k_loss.detach()
    return total, parts
