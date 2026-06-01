from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from mirage.encoder.data import AntmazeData, load_antmaze
from mirage.encoder.load import load_encoder
from mirage.paths import project_path

DEFAULT_ENCODER = project_path("runs_encoder", "dual_input_masked", "encoder_best.pt")
DEFAULT_DATASET = "D4RL/antmaze/umaze-v1"
DEFAULT_MINARI = project_path("data", "minari")


@torch.no_grad()
def encode_states_full(encoder, states: np.ndarray, batch: int = 32768,
                       device: str = "cpu") -> np.ndarray:
    out = np.empty((states.shape[0], encoder.latent_dim), dtype=np.float32)
    for i in range(0, states.shape[0], batch):
        s = torch.from_numpy(np.ascontiguousarray(states[i:i + batch])).to(device)
        z = encoder.encode_full(s)
        out[i:i + batch] = z.cpu().numpy()
    return out


@torch.no_grad()
def encode_states_goal(encoder, states: np.ndarray, batch: int = 32768,
                       device: str = "cpu") -> np.ndarray:
    out = np.empty((states.shape[0], encoder.latent_dim), dtype=np.float32)
    for i in range(0, states.shape[0], batch):
        s = torch.from_numpy(np.ascontiguousarray(states[i:i + batch])).to(device)
        z = encoder.encode_goal(s)
        out[i:i + batch] = z.cpu().numpy()
    return out


def encode_all_transition_latents(data: AntmazeData, encoder, device: str = "cpu",
                                  verbose: bool = True):
    t0 = time.time()
    n_states = data.obs.shape[0]
    all_states = data.gather_state("full", np.arange(n_states))
    z_all = encode_states_full(encoder, all_states, device=device)
    if verbose:
        print(f"  encoded {n_states} states in {time.time()-t0:.1f}s")

    idx_t = np.empty(data.n_trans, dtype=np.int64)
    idx_tp1 = np.empty(data.n_trans, dtype=np.int64)
    for e in range(data.n_ep):
        ss = data.state_starts[e]
        ts = data.trans_starts[e]
        T = int(data.ep_lens[e])
        local = np.arange(T)
        idx_t[ts:ts + T] = ss + local
        idx_tp1[ts:ts + T] = ss + local + 1
    return z_all[idx_t], z_all[idx_tp1], z_all


def fit_kmeans(latents: np.ndarray, K: int, seed: int = 0, sample: int | None = 300_000):
    from sklearn.cluster import MiniBatchKMeans
    rng = np.random.default_rng(seed)
    fit_X = latents
    if sample is not None and latents.shape[0] > sample:
        sel = rng.choice(latents.shape[0], size=sample, replace=False)
        fit_X = latents[sel]
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=10_000,
                         n_init=3, max_iter=200)
    km.fit(fit_X)
    return km


@dataclass
class LatentGraph:
    K: int
    n_nodes: int
    nodes: np.ndarray
    edges: np.ndarray
    edge_counts: np.ndarray
    centroids: np.ndarray

    def adjacency_csr(self, drop_self: bool = False):
        from scipy.sparse import csr_matrix
        e = self.edges
        if drop_self:
            keep = e[:, 0] != e[:, 1]
            e = e[keep]
        data = np.ones(e.shape[0], dtype=np.float32)
        return csr_matrix((data, (e[:, 0], e[:, 1])), shape=(self.K, self.K))


def build_graph(c_t: np.ndarray, c_tp1: np.ndarray, centroids: np.ndarray,
                K: int) -> LatentGraph:
    pairs = np.stack([c_t, c_tp1], axis=1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    nodes = np.unique(np.concatenate([c_t, c_tp1]))
    return LatentGraph(K=K, n_nodes=nodes.shape[0], nodes=nodes,
                       edges=uniq, edge_counts=counts, centroids=centroids)


def graph_stats(g: LatentGraph) -> dict:
    e = g.edges
    self_mask = e[:, 0] == e[:, 1]
    n_self = int(self_mask.sum())
    e_ns = e[~self_mask]

    out_deg = np.bincount(e_ns[:, 0], minlength=g.K)
    in_deg = np.bincount(e_ns[:, 1], minlength=g.K)
    appear = g.nodes
    mean_out = float(out_deg[appear].mean())
    mean_in = float(in_deg[appear].mean())

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    A = csr_matrix((np.ones(e_ns.shape[0]), (e_ns[:, 0], e_ns[:, 1])), shape=(g.K, g.K))
    n_wcc, wcc_lbl = connected_components(A, directed=True, connection="weak")
    n_scc, scc_lbl = connected_components(A, directed=True, connection="strong")
    appear_mask = np.zeros(g.K, dtype=bool)
    appear_mask[appear] = True
    wcc_sizes = np.bincount(wcc_lbl[appear_mask])
    scc_sizes = np.bincount(scc_lbl[appear_mask])
    n_wcc_real = int((wcc_sizes > 0).sum())
    n_scc_real = int((scc_sizes > 0).sum())

    return {
        "K": g.K,
        "n_nodes": g.n_nodes,
        "n_edges": int(e.shape[0]),
        "n_edges_nonself": int(e_ns.shape[0]),
        "n_self_loops": n_self,
        "mean_out_degree": mean_out,
        "mean_in_degree": mean_in,
        "n_wcc": n_wcc_real,
        "largest_wcc": int(wcc_sizes.max()) if wcc_sizes.size else 0,
        "n_scc": n_scc_real,
        "largest_scc": int(scc_sizes.max()) if scc_sizes.size else 0,
    }


def load_data_and_encoder(encoder_path: str = DEFAULT_ENCODER,
                          dataset_id: str = DEFAULT_DATASET,
                          minari_path: str = DEFAULT_MINARI,
                          device: str = "cpu"):
    data = load_antmaze(dataset_id, minari_path)
    encoder, enc_cfg, _ = load_encoder(encoder_path, device)
    return data, encoder, enc_cfg


if __name__ == "__main__":
    data, encoder, _ = load_data_and_encoder()
    z_t, z_tp1, _ = encode_all_transition_latents(data, encoder)
    for K in (200, 500, 1000):
        km = fit_kmeans(np.concatenate([z_t, z_tp1]), K)
        c_t = km.predict(z_t)
        c_tp1 = km.predict(z_tp1)
        g = build_graph(c_t, c_tp1, km.cluster_centers_, K)
        print(graph_stats(g))
