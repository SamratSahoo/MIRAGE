from __future__ import annotations

import os
import numpy as np
import torch

from mirage.paths import project_path

GRAPH_NPZ = project_path("checkpoints", "graph", "graph_K500.npz")
ENCODER_CKPT = project_path("runs_encoder", "dual_input_masked", "encoder_best.pt")
WM_CKPT = project_path("runs_world_model", "dynamics_dualinput", "world_model_best.pt")
IWM_CKPT = project_path("runs_inverse_world_model", "inverse", "inverse_world_model_best.pt")
DATASET_ID = "D4RL/antmaze/umaze-v1"
MINARI = project_path("data", "minari")
OUT_DIR = project_path("graph_aug")
K = 500
LATENT_DIM = 16
ACT_DIM = 8


def load_full_graph():
    d = np.load(GRAPH_NPZ)
    return {k: d[k] for k in d.keys()}


def edges_to_set(edges: np.ndarray, drop_self: bool = True) -> set:
    s = set()
    for src, dst in edges:
        src, dst = int(src), int(dst)
        if drop_self and src == dst:
            continue
        s.add((src, dst))
    return s


def set_to_edges(edge_set: set) -> np.ndarray:
    if not edge_set:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(sorted(edge_set), dtype=np.int64)


def adjacency_csr(edge_set: set, k: int = K):
    from scipy.sparse import csr_matrix
    if not edge_set:
        return csr_matrix((k, k), dtype=np.float32)
    e = np.array(list(edge_set), dtype=np.int64)
    data = np.ones(e.shape[0], dtype=np.float32)
    return csr_matrix((data, (e[:, 0], e[:, 1])), shape=(k, k))


def connectivity_stats(edge_set: set, nodes: np.ndarray, k: int = K) -> dict:
    from scipy.sparse.csgraph import connected_components
    A = adjacency_csr(edge_set, k)
    n_wcc, wcc = connected_components(A, directed=True, connection="weak")
    n_scc, scc = connected_components(A, directed=True, connection="strong")
    appear = np.zeros(k, dtype=bool)
    appear[nodes] = True
    wcc_sizes = np.bincount(wcc[appear])
    scc_sizes = np.bincount(scc[appear])
    n_app = int(appear.sum())
    return {
        "n_wcc": int((wcc_sizes > 0).sum()),
        "largest_wcc": int(wcc_sizes.max()) if wcc_sizes.size else 0,
        "n_scc": int((scc_sizes > 0).sum()),
        "largest_scc": int(scc_sizes.max()) if scc_sizes.size else 0,
        "largest_scc_frac": float(scc_sizes.max() / n_app) if n_app else 0.0,
    }


def pair_connected(edge_set: set, start_clusters: np.ndarray,
                   goal_clusters: np.ndarray, k: int = K) -> np.ndarray:
    from scipy.sparse.csgraph import dijkstra
    A = adjacency_csr(edge_set, k)
    uniq = np.unique(start_clusters)
    dist = dijkstra(A, directed=True, indices=uniq, unweighted=True)
    row = {int(s): i for i, s in enumerate(uniq)}
    rows = np.array([row[int(s)] for s in start_clusters])
    d = dist[rows, goal_clusters]
    same = start_clusters == goal_clusters
    return np.isfinite(d) | same


def build_trans_to_state_idx(data) -> np.ndarray:
    idx = np.empty(data.n_trans, dtype=np.int64)
    for e in range(data.n_ep):
        ss = int(data.state_starts[e])
        ts = int(data.trans_starts[e])
        T = int(data.ep_lens[e])
        idx[ts:ts + T] = ss + np.arange(T)
    return idx


def bin_latents(z: np.ndarray, centroids: np.ndarray):
    zc = z @ centroids.T
    cn = (centroids * centroids).sum(1)[None, :]
    zn = (z * z).sum(1)[:, None]
    d2 = zn - 2 * zc + cn
    lbl = d2.argmin(1)
    dist = np.sqrt(np.maximum(d2[np.arange(d2.shape[0]), lbl], 0.0))
    return lbl.astype(np.int64), dist


def load_encoder(device="cpu"):
    from mirage.encoder.load import load_encoder as _le
    return _le(ENCODER_CKPT, device)


def load_forward_wm(device="cpu"):
    from mirage.world_model.models import DynamicsEnsemble
    ck = torch.load(WM_CKPT, map_location=device, weights_only=False)
    c = ck["config"]
    fwd = DynamicsEnsemble(n_members=c["n_members"], latent_dim=LATENT_DIM,
                           act_dim=ACT_DIM, hidden_dim=c["hidden_dim"],
                           n_hidden=c["n_hidden"])
    fwd.load_state_dict(ck["ensemble"])
    fwd.eval()
    return fwd.to(device)


def load_inverse_wm(device="cpu"):
    from mirage.inverse_world_model.models import InverseWorldModel
    ck = torch.load(IWM_CKPT, map_location=device, weights_only=False)
    c = ck["config"]
    m = InverseWorldModel(latent_dim=LATENT_DIM, act_dim=ACT_DIM,
                          k_max=c["k_max"], hidden_dim=c["hidden_dim"],
                          n_hidden=c["n_hidden"])
    m.load_state_dict(ck["model"])
    m.eval()
    return m.to(device), c["k_max"]


@torch.no_grad()
def fwd_predict(fwd, z, a):
    mu_all, _ = fwd(z, a)
    z_next = torch.nn.functional.normalize(mu_all.mean(0), dim=-1)
    return z_next


def l2norm_np(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def get_cluster_mean_xy(data, cluster_labels_t, trans_to_state_idx, k: int = K):
    xy = data.ach[trans_to_state_idx]
    sums = np.zeros((k, 2), dtype=np.float64)
    cnts = np.zeros(k, dtype=np.int64)
    np.add.at(sums, cluster_labels_t, xy)
    np.add.at(cnts, cluster_labels_t, 1)
    mean = np.full((k, 2), np.nan, dtype=np.float64)
    nz = cnts > 0
    mean[nz] = sums[nz] / cnts[nz][:, None]
    return mean, cnts
