from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import build_mlp


class StateEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 16,
                 hidden_dim: int = 512, n_hidden: int = 3,
                 l2_normalize: bool = True, cold_init_eps: float = 1e-12):
        super().__init__()
        self.net = build_mlp(in_dim, hidden_dim, n_hidden, latent_dim,
                             cold_init_eps=cold_init_eps)
        self.l2_normalize = l2_normalize
        self.latent_dim = latent_dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        z = self.net(s)
        if self.l2_normalize:
            z = F.normalize(z, dim=-1, eps=1e-8)
        return z
