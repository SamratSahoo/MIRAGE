from __future__ import annotations

import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):

    def __init__(self, dim: int, epsilon: float = 1e-4, clip: float = 10.0):
        super().__init__()
        self.clip = float(clip)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(float(epsilon)))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count * batch_count / tot_count
        self.var.copy_(m2 / tot_count)
        self.count.copy_(tot_count)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        out = (x - self.mean) / torch.sqrt(self.var + 1e-8)
        return torch.clamp(out, -self.clip, self.clip)
