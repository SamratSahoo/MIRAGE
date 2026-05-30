from __future__ import annotations

import numpy as np
import torch

from mirage.encoder.data import AntmazeData, load_antmaze, Sampler


class MultiStepSampler:
    def __init__(self, data: AntmazeData, mode: str, horizon: int, seed: int,
                 device: str | torch.device = "cpu"):
        self.data = data
        self.mode = mode
        self.H = int(horizon)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr)).to(self.device, non_blocking=True)

    def batch(self, B: int) -> dict:
        d = self.data
        H = self.H
        ep_idx = self.rng.integers(0, d.n_ep, size=B)
        T_ep = d.ep_lens[ep_idx]
        valid_max = np.maximum(T_ep - H, 1)
        u = self.rng.random(B)
        t0 = (u * valid_max).astype(np.int64)
        state_dim = d.state_dim(self.mode)
        states = np.empty((B, H + 1, state_dim), dtype=np.float32)
        actions = np.empty((B, H, d.act.shape[-1]), dtype=np.float32)
        xys = np.empty((B, H + 1, 2), dtype=np.float32)
        for k in range(H + 1):
            flat = d.state_starts[ep_idx] + t0 + k
            states[:, k] = d.gather_state(self.mode, flat)
            xys[:, k] = d.ach[flat]
        for k in range(H):
            flat = d.trans_starts[ep_idx] + t0 + k
            actions[:, k] = d.act[flat]
        return {
            "states":  self._to_tensor(states),
            "actions": self._to_tensor(actions),
            "xy":      self._to_tensor(xys),
        }
