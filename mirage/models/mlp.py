from __future__ import annotations

import torch
import torch.nn as nn


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
