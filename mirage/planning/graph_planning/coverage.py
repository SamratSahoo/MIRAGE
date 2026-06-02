from __future__ import annotations

import numpy as np

from mirage.encoder.data import AntmazeData

import importlib.util as _ilu
import os as _os
import sys as _sys
if "_mirage_planning_graph" in _sys.modules:
    _graph = _sys.modules["_mirage_planning_graph"]
else:
    _spec = _ilu.spec_from_file_location(
        "_mirage_planning_graph",
        _os.path.join(_os.path.dirname(__file__), "latent_graph.py"))
    _graph = _ilu.module_from_spec(_spec)
    _sys.modules["_mirage_planning_graph"] = _graph
    _spec.loader.exec_module(_graph)
LatentGraph = _graph.LatentGraph
encode_states_full = _graph.encode_states_full
encode_states_goal = _graph.encode_states_goal


def build_goal_states(des_xy: np.ndarray) -> np.ndarray:
    n = des_xy.shape[0]
    out = np.zeros((n, 29), dtype=np.float32)
    out[:, 27:29] = des_xy
    return out


def sample_pairs(val_data: AntmazeData, des_pool: np.ndarray, encoder,
                 km, n_pairs: int = 2000, seed: int = 0, device: str = "cpu"):
    rng = np.random.default_rng(seed)

    n_val_states = val_data.obs.shape[0]
    s_idx = rng.integers(0, n_val_states, size=n_pairs)
    start_states = val_data.gather_state("full", s_idx)
    z_start = encode_states_full(encoder, start_states, device=device)
    start_clusters = km.predict(z_start)

    g_idx = rng.integers(0, des_pool.shape[0], size=n_pairs)
    goal_states = build_goal_states(des_pool[g_idx])
    z_goal = encode_states_goal(encoder, goal_states, device=device)
    goal_clusters = km.predict(z_goal)

    return start_clusters.astype(np.int64), goal_clusters.astype(np.int64)


def measure_coverage(g: LatentGraph, start_clusters: np.ndarray,
                     goal_clusters: np.ndarray) -> dict:
    from scipy.sparse.csgraph import dijkstra

    A = g.adjacency_csr(drop_self=False)

    uniq_starts = np.unique(start_clusters)
    dist = dijkstra(A, directed=True, indices=uniq_starts, unweighted=True)
    src_row = {s: i for i, s in enumerate(uniq_starts)}

    rows = np.array([src_row[s] for s in start_clusters])
    d = dist[rows, goal_clusters]
    connected = np.isfinite(d)
    same = start_clusters == goal_clusters

    path_lens = d[connected & ~same]
    frac = float(connected.mean())
    out = {
        "n_pairs": int(start_clusters.shape[0]),
        "frac_connected": frac,
        "frac_same_cluster": float(same.mean()),
        "n_connected": int(connected.sum()),
    }
    if path_lens.size:
        out.update({
            "path_mean": float(path_lens.mean()),
            "path_median": float(np.median(path_lens)),
            "path_p90": float(np.percentile(path_lens, 90)),
            "path_max": float(path_lens.max()),
        })
    else:
        out.update({"path_mean": float("nan"), "path_median": float("nan"),
                    "path_p90": float("nan"), "path_max": float("nan")})
    return out
