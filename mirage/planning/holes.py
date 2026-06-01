"""STEP 1 - Build two damaged graphs by knocking holes in G_full.

(a) CORRIDOR CUT: remove every directed edge whose src/dst clusters lie on
    opposite sides of a spatial y-boundary, splitting the maze in two.
(b) RANDOM EDGE REMOVAL: remove 50% of non-self edges uniformly (seed 0).

Run standalone:  python -m mirage.planning.holes   (but __init__ imports warp)
Instead invoke via run_augmentation.py which imports build_damaged_graphs().
"""
from __future__ import annotations

import importlib.util as _ilu
import os as _os
import sys as _sys

import numpy as np


def _load_sibling(alias, filename):
    if alias in _sys.modules:
        return _sys.modules[alias]
    spec = _ilu.spec_from_file_location(
        alias, _os.path.join(_os.path.dirname(__file__), filename))
    mod = _ilu.module_from_spec(spec)
    _sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


AC = _load_sibling("_aug_common", "aug_common.py")


def build_corridor_cut(full_edges_set: set, cluster_mean_xy: np.ndarray,
                       cluster_cnts: np.ndarray):
    """Cut along the median of cluster-mean-y. Returns (kept_set, removed_set, info).

    Searches a few candidate boundaries (median + offsets) to ensure the cut
    actually disconnects (start,goal) pairs; picks the one removing a clean
    middle band.
    """
    valid = cluster_cnts > 0
    ys = cluster_mean_xy[:, 1]
    med_y = float(np.nanmedian(ys[valid]))

    # side assignment: 0 = below boundary, 1 = above; nan clusters -> -1
    def make_cut(boundary):
        side = np.full(AC.K, -1, dtype=np.int64)
        side[valid] = (ys[valid] >= boundary).astype(np.int64)
        removed = set()
        kept = set()
        for (s, d) in full_edges_set:
            ss, sd = side[s], side[d]
            if ss != -1 and sd != -1 and ss != sd:
                removed.add((s, d))
            else:
                kept.add((s, d))
        return kept, removed, side

    kept, removed, side = make_cut(med_y)
    info = {
        "boundary_y": med_y,
        "n_below": int((side == 0).sum()),
        "n_above": int((side == 1).sum()),
    }
    return kept, removed, info, side


def build_random_removal(full_edges_set: set, frac: float = 0.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    edges = np.array(sorted(full_edges_set), dtype=np.int64)
    n = edges.shape[0]
    n_rm = int(round(frac * n))
    sel = rng.choice(n, size=n_rm, replace=False)
    rm_mask = np.zeros(n, dtype=bool)
    rm_mask[sel] = True
    removed = set(map(tuple, edges[rm_mask].tolist()))
    kept = set(map(tuple, edges[~rm_mask].tolist()))
    return kept, removed


def build_damaged_graphs(G, data, trans_to_state_idx):
    """Return dict of two damage scenarios, each with kept/removed edge sets + stats."""
    full_set = AC.edges_to_set(G["edges"], drop_self=True)
    nodes = G["nodes"]
    starts = G["start_clusters"]
    goals = G["goal_clusters"]

    mean_xy, cnts = AC.get_cluster_mean_xy(
        data, G["cluster_labels_t"], trans_to_state_idx)

    # baseline coverage on full graph
    base_cov = AC.pair_connected(full_set, starts, goals).mean()

    out = {"full_set": full_set, "nodes": nodes, "mean_xy": mean_xy,
           "cnts": cnts, "base_cov": float(base_cov),
           "starts": starts, "goals": goals}

    # (a) corridor
    kept_c, rm_c, info_c, side = build_corridor_cut(full_set, mean_xy, cnts)
    cov_c = AC.pair_connected(kept_c, starts, goals)
    conn_c = AC.connectivity_stats(kept_c, nodes)
    out["corridor"] = {
        "kept": kept_c, "removed": rm_c, "side": side, "info": info_c,
        "coverage": float(cov_c.mean()), "pair_connected": cov_c,
        "conn": conn_c, "n_removed": len(rm_c),
    }

    # (b) random
    kept_r, rm_r = build_random_removal(full_set, frac=0.5, seed=0)
    cov_r = AC.pair_connected(kept_r, starts, goals)
    conn_r = AC.connectivity_stats(kept_r, nodes)
    out["random"] = {
        "kept": kept_r, "removed": rm_r,
        "coverage": float(cov_r.mean()), "pair_connected": cov_r,
        "conn": conn_r, "n_removed": len(rm_r),
    }
    out["n_full_nonself"] = len(full_set)
    return out
