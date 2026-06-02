from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from mirage.encoder.data import AntmazeData, load_antmaze


def _normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


class DijkstraPathSampler:
    def __init__(self, data: AntmazeData, graph_npz: str, k_max: int, seed: int,
                 device: str | torch.device = "cpu"):
        self.data = data
        self.k_max = int(k_max)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)

        g = np.load(graph_npz, allow_pickle=True)
        self.centroids = _normalize_rows(g["centroids"].astype(np.float32))
        self.latent_dim = self.centroids.shape[1]
        self.K = self.centroids.shape[0]
        edges = g["edges"].astype(np.int64)
        labels_t = g["cluster_labels_t"].astype(np.int64)
        labels_tp1 = g["cluster_labels_tp1"].astype(np.int64)

        self.act_dim = int(data.act.shape[-1])
        self._build_edge_lookups(edges, labels_t, labels_tp1)
        self._build_cluster_xy(labels_t)
        self._build_shortest_paths(edges)

    def _build_edge_lookups(self, edges, labels_t, labels_tp1):
        key = labels_t * self.K + labels_tp1
        uniq, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv, minlength=uniq.shape[0]).astype(np.float64)
        sum_act = np.zeros((uniq.shape[0], self.act_dim), dtype=np.float64)
        np.add.at(sum_act, inv, self.data.act.astype(np.float64))
        sum_rew = np.zeros(uniq.shape[0], dtype=np.float64)
        np.add.at(sum_rew, inv, self.data.rew.astype(np.float64))
        mean_act = (sum_act / cnt[:, None]).astype(np.float32)
        mean_rew = (sum_rew / cnt).astype(np.float32)
        self._edge_key_to_idx = {int(k): i for i, k in enumerate(uniq)}
        self._edge_mean_act = mean_act
        self._edge_mean_rew = mean_rew

    def _build_cluster_xy(self, labels_t):
        d = self.data
        n_ep = d.ep_lens.shape[0]
        trans_to_state = np.empty(d.act.shape[0], dtype=np.int64)
        for e in range(n_ep):
            ts, te = d.trans_starts[e], d.trans_starts[e + 1]
            trans_to_state[ts:te] = d.state_starts[e] + np.arange(te - ts)
        src_xy = d.ach[trans_to_state]
        xy = np.zeros((self.K, 2), dtype=np.float64)
        cnt = np.zeros(self.K, dtype=np.float64)
        np.add.at(xy, labels_t, src_xy.astype(np.float64))
        np.add.at(cnt, labels_t, 1.0)
        cnt = np.maximum(cnt, 1.0)
        self.cluster_xy = (xy / cnt[:, None]).astype(np.float32)

    def _build_shortest_paths(self, edges):
        w = np.ones(edges.shape[0], dtype=np.float64)
        mask = edges[:, 0] != edges[:, 1]
        e = edges[mask]
        graph = csr_matrix((w[mask], (e[:, 0], e[:, 1])), shape=(self.K, self.K))
        dist, pred = shortest_path(graph, method="D", directed=True,
                                   return_predecessors=True)
        self.dist = dist
        self.pred = pred.astype(np.int64)
        finite = np.isfinite(dist)
        within = (dist >= 1) & (dist <= self.k_max) & finite
        ii, jj = np.nonzero(within)
        self.valid_pairs = np.stack([ii, jj], axis=1).astype(np.int64)
        self.valid_dist = dist[ii, jj].astype(np.int64)

    def _reconstruct(self, i: int, j: int) -> list[int]:
        path = [j]
        cur = j
        while cur != i:
            cur = int(self.pred[i, cur])
            if cur < 0:
                return []
            path.append(cur)
        path.reverse()
        return path

    def _edge_action(self, ci: int, cj: int) -> np.ndarray:
        idx = self._edge_key_to_idx.get(ci * self.K + cj, None)
        if idx is None:
            return np.zeros(self.act_dim, dtype=np.float32)
        return self._edge_mean_act[idx]

    def _edge_reward(self, ci: int, cj: int) -> float:
        idx = self._edge_key_to_idx.get(ci * self.K + cj, None)
        if idx is None:
            return 0.0
        return float(self._edge_mean_rew[idx])

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr)).to(self.device, non_blocking=True)

    def batch(self, B: int, fixed_k: int | None = None) -> dict:
        K = self.k_max
        if fixed_k is None:
            sel = self.rng.integers(0, self.valid_pairs.shape[0], size=B)
        else:
            pool = np.nonzero(self.valid_dist == int(fixed_k))[0]
            if pool.shape[0] == 0:
                pool = np.arange(self.valid_pairs.shape[0])
            sel = pool[self.rng.integers(0, pool.shape[0], size=B)]
        pairs = self.valid_pairs[sel]

        latents = np.zeros((B, K + 1, self.latent_dim), dtype=np.float32)
        actions = np.zeros((B, K, self.act_dim), dtype=np.float32)
        k_arr = np.zeros(B, dtype=np.int64)
        rew_sum = np.zeros(B, dtype=np.float32)
        xydist = np.zeros(B, dtype=np.float32)

        for b in range(B):
            i, j = int(pairs[b, 0]), int(pairs[b, 1])
            path = self._reconstruct(i, j)
            if len(path) < 2:
                path = [i, j]
            m = min(len(path) - 1, K)
            path = path[: m + 1]
            k_arr[b] = m
            for s in range(m + 1):
                latents[b, s] = self.centroids[path[s]]
            for s in range(m + 1, K + 1):
                latents[b, s] = self.centroids[path[m]]
            for s in range(m):
                actions[b, s] = self._edge_action(path[s], path[s + 1])
                rew_sum[b] += self._edge_reward(path[s], path[s + 1])
            xy0 = self.cluster_xy[path[0]]
            xyk = self.cluster_xy[path[m]]
            xydist[b] = -float(np.linalg.norm(xy0 - xyk))

        return {
            "latents": self._to_tensor(latents),
            "actions": self._to_tensor(actions),
            "k": self._to_tensor(k_arr),
            "reward": self._to_tensor(rew_sum),
            "xydist": self._to_tensor(xydist),
        }
