from __future__ import annotations

import torch
import torch.nn as nn

from .mlp import build_mlp


class ForwardDynamics(nn.Module):
    def __init__(self, latent_dim: int, act_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        self.net = build_mlp(latent_dim + act_dim, hidden_dim, n_hidden, latent_dim)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1))


class InverseDynamics(nn.Module):
    def __init__(self, latent_dim: int, act_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        self.net = build_mlp(2 * latent_dim, hidden_dim, n_hidden, act_dim)

    def forward(self, z: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, z_next], dim=-1))
