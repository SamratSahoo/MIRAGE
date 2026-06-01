from __future__ import annotations

import torch
import torch.nn as nn

from mirage.encoder.models import build_mlp


class InverseWorldModel(nn.Module):
    def __init__(self, latent_dim: int, act_dim: int, k_max: int,
                 hidden_dim: int = 512, n_hidden: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.act_dim = act_dim
        self.k_max = k_max
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden

        in_dim = 2 * latent_dim + 1
        self.trunk = build_mlp(in_dim, hidden_dim, n_hidden, out_dim=hidden_dim)
        self.act_head = nn.Linear(hidden_dim, k_max * act_dim)
        self.latent_head = nn.Linear(hidden_dim, k_max * latent_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.xydist_head = nn.Linear(hidden_dim, 1)

    def forward(self, z0: torch.Tensor, zk: torch.Tensor, k_norm: torch.Tensor) -> dict:
        if k_norm.dim() == 1:
            k_norm = k_norm.unsqueeze(-1)
        h = self.trunk(torch.cat([z0, zk, k_norm], dim=-1))
        B = z0.shape[0]
        actions = torch.tanh(self.act_head(h)).view(B, self.k_max, self.act_dim)
        deltas = self.latent_head(h).view(B, self.k_max, self.latent_dim)
        reward = self.reward_head(h).squeeze(-1)
        xydist = self.xydist_head(h).squeeze(-1)
        return {
            "actions": actions,
            "deltas": deltas,
            "reward": reward,
            "xydist": xydist,
        }

    def reconstruct_latents(self, z0: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        return z0.unsqueeze(1) + torch.cumsum(deltas, dim=1)
