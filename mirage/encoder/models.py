from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(in_dim: int, hidden_dim: int, n_hidden: int, out_dim: int,
              cold_init_eps: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_hidden):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        d = hidden_dim
    head = nn.Linear(d, out_dim)
    if cold_init_eps > 0.0:
        nn.init.uniform_(head.weight, -cold_init_eps, cold_init_eps)
        nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


class StateEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 16,
                 hidden_dim: int = 512, n_hidden: int = 3,
                 l2_normalize: bool = True, cold_init_eps: float = 1e-12):
        super().__init__()
        self.in_dim = in_dim
        self.net = build_mlp(in_dim, hidden_dim, n_hidden, latent_dim,
                             cold_init_eps=cold_init_eps)
        self.l2_normalize = l2_normalize
        self.latent_dim = latent_dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        z = self.net(s)
        if self.l2_normalize:
            z = F.normalize(z, dim=-1, eps=1e-8)
        return z


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
