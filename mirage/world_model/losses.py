from __future__ import annotations

import math

import torch


_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor,
                 reduce: str = "mean") -> torch.Tensor:
    var = (2.0 * log_sigma).exp()
    sq = (target - mu) ** 2
    nll_per_dim = 0.5 * sq / var + log_sigma + _HALF_LOG_2PI
    nll = nll_per_dim.sum(dim=-1)
    if reduce == "mean":
        return nll.mean()
    if reduce == "sum":
        return nll.sum()
    return nll
