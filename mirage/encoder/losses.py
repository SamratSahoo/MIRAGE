from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> tuple[torch.Tensor, dict]:
    logits = z1 @ z2.t() / temperature
    B = logits.size(0)
    targets = torch.arange(B, device=logits.device)
    loss_12 = F.cross_entropy(logits, targets)
    loss_21 = F.cross_entropy(logits.t(), targets)
    loss = 0.5 * (loss_12 + loss_21)

    with torch.no_grad():
        top1 = (logits.argmax(dim=1) == targets).float().mean().item()
        top5 = (logits.topk(min(5, B), dim=1).indices == targets[:, None]).any(dim=1).float().mean().item()
    return loss, {"info_nce/loss": loss.item(),
                  "info_nce/top1_acc": top1,
                  "info_nce/top5_acc": top5}


def forward_dyn_loss(z_next_pred: torch.Tensor, z_next_target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    loss = F.mse_loss(z_next_pred, z_next_target)
    return loss, {"forward_dyn/mse": loss.item()}


def inverse_dyn_loss(a_pred: torch.Tensor, a_target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    loss = F.mse_loss(a_pred, a_target)
    return loss, {"inverse_dyn/mse": loss.item()}
