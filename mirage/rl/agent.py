from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


_LOGSTD_MIN = -5.0
_LOGSTD_MAX = 1.0


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 n_layers: int = 2, layernorm: bool = False):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        def _mlp(out_dim: int, out_std: float) -> nn.Sequential:
            layers: list[nn.Module] = []
            d = self.obs_dim
            for _ in range(int(n_layers)):
                layers.append(_layer_init(nn.Linear(d, hidden_dim)))
                if layernorm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                d = hidden_dim
            layers.append(_layer_init(nn.Linear(d, out_dim), std=out_std))
            return nn.Sequential(*layers)

        self.critic = _mlp(1, 1.0)
        self.actor_mean = _mlp(self.act_dim, 0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.act_dim))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.clamp(_LOGSTD_MIN, _LOGSTD_MAX).expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
