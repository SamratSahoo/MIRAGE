from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch


OBS_KEY = "observation"
ACH_KEY = "achieved_goal"
DES_KEY = "desired_goal"


@dataclass
class AntmazeData:
    obs: np.ndarray
    ach: np.ndarray
    des: np.ndarray
    act: np.ndarray
    rew: np.ndarray
    ep_lens: np.ndarray
    state_starts: np.ndarray
    trans_starts: np.ndarray
    ep_ids: np.ndarray

    @property
    def n_ep(self) -> int:
        return self.ep_lens.shape[0]

    @property
    def n_trans(self) -> int:
        return int(self.act.shape[0])

    def state_dim(self, mode: str) -> int:
        return {"xy": 2, "proprio": 27, "full": 29}[mode]

    def gather_state(self, mode: str, flat_idx: np.ndarray) -> np.ndarray:
        if mode == "xy":
            return self.ach[flat_idx]
        if mode == "proprio":
            return self.obs[flat_idx]
        if mode == "full":
            return np.concatenate([self.obs[flat_idx], self.ach[flat_idx]], axis=-1)
        raise ValueError(f"unknown input mode: {mode}")

    def split(self, val_frac: float, seed: int) -> tuple["AntmazeData", "AntmazeData"]:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.n_ep)
        n_val = max(1, int(round(val_frac * self.n_ep)))
        val_ids, train_ids = perm[:n_val], perm[n_val:]
        return self._subset(train_ids), self._subset(val_ids)

    def _subset(self, ep_indices: np.ndarray) -> "AntmazeData":
        state_blocks_o, state_blocks_a, state_blocks_d = [], [], []
        trans_blocks_a, trans_blocks_r = [], []
        new_lens = []
        for ei in ep_indices:
            ss, se = self.state_starts[ei], self.state_starts[ei + 1]
            ts, te = self.trans_starts[ei], self.trans_starts[ei + 1]
            state_blocks_o.append(self.obs[ss:se])
            state_blocks_a.append(self.ach[ss:se])
            state_blocks_d.append(self.des[ss:se])
            trans_blocks_a.append(self.act[ts:te])
            trans_blocks_r.append(self.rew[ts:te])
            new_lens.append(self.ep_lens[ei])
        ep_lens = np.asarray(new_lens, dtype=np.int64)
        state_starts = np.concatenate([[0], np.cumsum(ep_lens + 1)]).astype(np.int64)
        trans_starts = np.concatenate([[0], np.cumsum(ep_lens)]).astype(np.int64)
        return AntmazeData(
            obs=np.concatenate(state_blocks_o, axis=0),
            ach=np.concatenate(state_blocks_a, axis=0),
            des=np.concatenate(state_blocks_d, axis=0),
            act=np.concatenate(trans_blocks_a, axis=0),
            rew=np.concatenate(trans_blocks_r, axis=0),
            ep_lens=ep_lens,
            state_starts=state_starts,
            trans_starts=trans_starts,
            ep_ids=self.ep_ids[ep_indices],
        )


def load_antmaze(dataset_id: str = "D4RL/antmaze/umaze-v1",
                 datasets_path: str | None = None,
                 max_episodes: int | None = None) -> AntmazeData:
    if datasets_path is not None:
        os.environ["MINARI_DATASETS_PATH"] = datasets_path
    import minari
    ds = minari.load_dataset(dataset_id)

    obs_b, ach_b, des_b, act_b, rew_b = [], [], [], [], []
    ep_lens, ep_ids = [], []
    for i, ep in enumerate(ds.iterate_episodes()):
        if max_episodes is not None and i >= max_episodes:
            break
        o = ep.observations
        T = ep.actions.shape[0]
        if o[OBS_KEY].shape[0] != T + 1:
            raise RuntimeError(
                f"episode {ep.id}: obs length {o[OBS_KEY].shape[0]} != T+1 ({T+1})"
            )
        obs_b.append(o[OBS_KEY].astype(np.float32))
        ach_b.append(o[ACH_KEY].astype(np.float32))
        des_b.append(o[DES_KEY].astype(np.float32))
        act_b.append(ep.actions.astype(np.float32))
        rew_b.append(ep.rewards.astype(np.float32))
        ep_lens.append(T)
        ep_ids.append(ep.id)

    ep_lens = np.asarray(ep_lens, dtype=np.int64)
    state_starts = np.concatenate([[0], np.cumsum(ep_lens + 1)]).astype(np.int64)
    trans_starts = np.concatenate([[0], np.cumsum(ep_lens)]).astype(np.int64)
    return AntmazeData(
        obs=np.concatenate(obs_b, axis=0),
        ach=np.concatenate(ach_b, axis=0),
        des=np.concatenate(des_b, axis=0),
        act=np.concatenate(act_b, axis=0),
        rew=np.concatenate(rew_b, axis=0),
        ep_lens=ep_lens,
        state_starts=state_starts,
        trans_starts=trans_starts,
        ep_ids=np.asarray(ep_ids, dtype=np.int64),
    )


class Sampler:
    def __init__(self, data: AntmazeData, mode: str, window: int, seed: int,
                 device: str | torch.device = "cpu"):
        self.data = data
        self.mode = mode
        self.window = int(window)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)

    def _to_tensor(self, *arrs):
        return [torch.from_numpy(np.ascontiguousarray(a)).to(self.device, non_blocking=True) for a in arrs]

    def iid_batch(self, B: int) -> dict:
        d = self.data
        ep_idx = self.rng.integers(0, d.n_ep, size=B)
        T_ep = d.ep_lens[ep_idx]
        u = self.rng.random(B)
        t_idx = (u * T_ep).astype(np.int64)
        trans_flat = d.trans_starts[ep_idx] + t_idx
        state_t = d.state_starts[ep_idx] + t_idx
        state_tp1 = state_t + 1
        s = d.gather_state(self.mode, state_t)
        s_n = d.gather_state(self.mode, state_tp1)
        a = d.act[trans_flat]
        xy = d.ach[state_t]
        s_t, s_n_t, a_t, xy_t = self._to_tensor(s, s_n, a, xy)
        return {"s": s_t, "s_next": s_n_t, "a": a_t, "xy": xy_t}

    def pair_batch(self, B: int) -> dict:
        d = self.data
        W = self.window
        ep_idx = self.rng.integers(0, d.n_ep, size=B)
        T_ep = d.ep_lens[ep_idx]
        u1 = self.rng.random(B)
        t1 = (u1 * (T_ep + 1)).astype(np.int64)
        t1 = np.minimum(t1, T_ep)
        lo = np.maximum(0, t1 - W)
        hi = np.minimum(T_ep, t1 + W)
        u2 = self.rng.random(B)
        t2 = lo + (u2 * (hi - lo + 1)).astype(np.int64)
        t2 = np.minimum(t2, hi)

        flat1 = d.state_starts[ep_idx] + t1
        flat2 = d.state_starts[ep_idx] + t2
        s1 = d.gather_state(self.mode, flat1)
        s2 = d.gather_state(self.mode, flat2)
        s1_t, s2_t = self._to_tensor(s1, s2)
        offset = torch.from_numpy(np.abs(t1 - t2).astype(np.int64)).to(self.device)
        return {"s1": s1_t, "s2": s2_t, "offset": offset}
