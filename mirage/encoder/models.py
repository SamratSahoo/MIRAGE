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


class MaskedStateEncoder(nn.Module):
    def __init__(self, latent_dim: int = 16,
                 hidden_dim: int = 512, n_hidden: int = 3,
                 l2_normalize: bool = True, cold_init_eps: float = 1e-12,
                 proprio_dim: int = 27, xy_dim: int = 2):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.xy_dim = xy_dim
        self.in_dim = proprio_dim + xy_dim
        self.net = build_mlp(proprio_dim + xy_dim + 1, hidden_dim, n_hidden, latent_dim,
                             cold_init_eps=cold_init_eps)
        self.l2_normalize = l2_normalize
        self.latent_dim = latent_dim

    def forward(self, s: torch.Tensor, mask_prob: float = 0.0,
                prenorm: bool = False) -> torch.Tensor:
        B = s.shape[0]
        if mask_prob > 0:
            flag = (torch.rand(B, 1, device=s.device, dtype=s.dtype) < mask_prob).to(s.dtype)
        else:
            flag = torch.zeros((B, 1), device=s.device, dtype=s.dtype)
        proprio_keep = 1.0 - flag
        masked_proprio = s[:, :self.proprio_dim] * proprio_keep
        xy = s[:, self.proprio_dim:self.proprio_dim + self.xy_dim]
        s_aug = torch.cat([masked_proprio, xy, flag], dim=-1)
        z = self.net(s_aug)
        if self.l2_normalize and not prenorm:
            z = F.normalize(z, dim=-1, eps=1e-8)
        return z

    def encode_full(self, s: torch.Tensor, prenorm: bool = False) -> torch.Tensor:
        return self.forward(s, mask_prob=0.0, prenorm=prenorm)

    def encode_goal(self, s: torch.Tensor, prenorm: bool = False) -> torch.Tensor:
        B = s.shape[0]
        flag = torch.ones((B, 1), device=s.device, dtype=s.dtype)
        masked_proprio = torch.zeros((B, self.proprio_dim), device=s.device, dtype=s.dtype)
        xy = s[:, self.proprio_dim:self.proprio_dim + self.xy_dim]
        s_aug = torch.cat([masked_proprio, xy, flag], dim=-1)
        z = self.net(s_aug)
        if self.l2_normalize and not prenorm:
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


class StateDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int, hidden_dim: int = 512, n_hidden: int = 2):
        super().__init__()
        self.net = build_mlp(latent_dim, hidden_dim, n_hidden, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
