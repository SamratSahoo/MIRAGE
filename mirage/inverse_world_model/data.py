from __future__ import annotations

import numpy as np
import torch

from mirage.encoder.data import AntmazeData, load_antmaze


class InverseWindowSampler:
    def __init__(self, data: AntmazeData, mode: str, k_max: int, seed: int,
                 device: str | torch.device = "cpu"):
        self.data = data
        self.mode = mode
        self.k_max = int(k_max)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr)).to(self.device, non_blocking=True)

    def batch(self, B: int, fixed_k: int | None = None) -> dict:
        d = self.data
        K = self.k_max
        ok = d.ep_lens >= 1
        ep_pool = np.nonzero(ok)[0]
        ep_idx = ep_pool[self.rng.integers(0, ep_pool.shape[0], size=B)]
        T_ep = d.ep_lens[ep_idx]

        if fixed_k is None:
            k = self.rng.integers(1, K + 1, size=B).astype(np.int64)
        else:
            k = np.full(B, int(fixed_k), dtype=np.int64)
        k = np.minimum(k, T_ep).astype(np.int64)

        valid_max = np.maximum(T_ep - k, 0)
        u = self.rng.random(B)
        t0 = (u * (valid_max + 1)).astype(np.int64)
        t0 = np.minimum(t0, valid_max)

        state_dim = d.state_dim(self.mode)
        states = np.zeros((B, K + 1, state_dim), dtype=np.float32)
        actions = np.zeros((B, K, d.act.shape[-1]), dtype=np.float32)
        xy0 = np.empty((B, 2), dtype=np.float32)
        xyk = np.empty((B, 2), dtype=np.float32)
        rew_sum = np.zeros(B, dtype=np.float32)

        for j in range(K + 1):
            within = j <= k
            flat = d.state_starts[ep_idx] + t0 + np.minimum(j, k)
            s = d.gather_state(self.mode, flat)
            states[:, j] = np.where(within[:, None], s, states[:, j])
        for j in range(K):
            within = j < k
            flat = d.trans_starts[ep_idx] + t0 + np.minimum(j, k - 1)
            a = d.act[flat]
            actions[:, j] = np.where(within[:, None], a, 0.0)
            r = d.rew[flat]
            rew_sum += np.where(within, r, 0.0)

        flat0 = d.state_starts[ep_idx] + t0
        flatk = d.state_starts[ep_idx] + t0 + k
        xy0[:] = d.ach[flat0]
        xyk[:] = d.ach[flatk]
        xydist = -np.linalg.norm(xy0 - xyk, axis=-1).astype(np.float32)

        return {
            "states": self._to_tensor(states),
            "actions": self._to_tensor(actions),
            "k": self._to_tensor(k),
            "reward": self._to_tensor(rew_sum),
            "xydist": self._to_tensor(xydist),
        }
