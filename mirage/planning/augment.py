"""STEP 2 - Four augmentation methods. Each augments a FRESH copy of the damaged
graph (never stacked). Added edges capped at n_removed for comparable budgets.

  1. RANDOM     - random novel directed edges (control).
  2. KNN_LATENT - each node -> k nearest centroids by latent L2 (no world model).
  3. FORWARD_WM - real source latent + random actions -> ensemble z_next -> bin.
  4. INVERSE_WM - for each disconnected (start,goal) pair, query inverse WM,
                  reconstruct intermediate latents, bin to a bridging chain.

Every method returns a set of ADDED edges (novel, not already in damaged graph).
Forward/inverse also return per-edge provenance for the env-executability check.
"""
from __future__ import annotations

import importlib.util as _ilu
import os as _os
import sys as _sys

import numpy as np
import torch


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


# --------------------------------------------------------------------------- #
# 1. RANDOM
# --------------------------------------------------------------------------- #
def aug_random(damaged_set: set, nodes: np.ndarray, n_budget: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    nodes = np.asarray(nodes)
    added = set()
    existing = damaged_set
    tries = 0
    max_tries = n_budget * 200
    while len(added) < n_budget and tries < max_tries:
        s = int(rng.choice(nodes)); d = int(rng.choice(nodes))
        tries += 1
        if s == d:
            continue
        e = (s, d)
        if e in existing or e in added:
            continue
        added.add(e)
    return added


# --------------------------------------------------------------------------- #
# 2. KNN_LATENT
# --------------------------------------------------------------------------- #
def aug_knn_latent(damaged_set: set, nodes: np.ndarray, centroids: np.ndarray,
                   n_budget: int):
    """Add directed edges node -> its nearest centroids. k auto-chosen so total
    novel added ~= n_budget. We rank ALL candidate (node, neighbor) by distance
    and take the closest novel ones until budget filled."""
    nodes = np.asarray(nodes)
    cz = centroids[nodes]                       # (n,16)
    # pairwise dist node-row to all centroids
    cn = (centroids * centroids).sum(1)[None, :]
    zn = (cz * cz).sum(1)[:, None]
    d2 = zn - 2 * (cz @ centroids.T) + cn       # (n, K)
    # build candidate list of (dist, src, dst) excluding self and non-nodes
    node_set = set(nodes.tolist())
    cands = []
    # to keep it bounded, take top-40 neighbors per node then global-sort
    topk = min(40, centroids.shape[0])
    nn = np.argsort(d2, axis=1)[:, :topk]
    for i, s in enumerate(nodes):
        for dst in nn[i]:
            dst = int(dst)
            if dst == int(s) or dst not in node_set:
                continue
            cands.append((d2[i, dst], int(s), dst))
    cands.sort(key=lambda x: x[0])
    added = set()
    for _, s, d in cands:
        if len(added) >= n_budget:
            break
        e = (s, d)
        if e in damaged_set or e in added:
            continue
        added.add(e)
    return added


# --------------------------------------------------------------------------- #
# 3. FORWARD_WM
# --------------------------------------------------------------------------- #
def aug_forward_wm(damaged_set: set, nodes: np.ndarray, centroids: np.ndarray,
                   data, cluster_labels_t, trans_to_state_idx, encoder, fwd,
                   n_budget: int, src_clusters=None, n_actions: int = 16,
                   conf_thresh: float = None, seed: int = 0, device="cpu",
                   max_states_per_cluster: int = 6):
    """For each source cluster, encode several real source states, sample random
    actions, predict z_next via ensemble mean, bin -> candidate edge.

    Returns (added_set, provenance) where provenance maps edge->dict with the
    action and source-state flat index that produced it (for env check)."""
    rng = np.random.default_rng(seed)
    if src_clusters is None:
        src_clusters = nodes
    src_clusters = list(dict.fromkeys(int(c) for c in src_clusters))
    rng.shuffle(src_clusters)

    node_set = set(np.asarray(nodes).tolist())
    cent_t = torch.from_numpy(centroids).float().to(device)

    # precompute per-cluster source-state pools
    added = {}            # edge -> provenance (keep best/first)
    n_gated = 0
    n_proposed = 0

    for c in src_clusters:
        if len(added) >= n_budget:
            break
        # real source states whose z_t is in cluster c
        mask = cluster_labels_t == c
        if not mask.any():
            continue
        st_idx = trans_to_state_idx[mask]
        pick = rng.choice(st_idx, size=min(max_states_per_cluster, st_idx.shape[0]),
                          replace=False)
        states = data.gather_state("full", pick)        # (m,29)
        with torch.no_grad():
            s_t = torch.from_numpy(np.ascontiguousarray(states)).to(device)
            z0 = encoder.encode_full(s_t)               # (m,16) normalized
        m = z0.shape[0]
        # sample actions: (m*n_actions, 8)
        acts = rng.uniform(-1, 1, size=(m * n_actions, AC.ACT_DIM)).astype(np.float32)
        z0_rep = z0.repeat_interleave(n_actions, dim=0)
        a_t = torch.from_numpy(acts).to(device)
        z_next = AC.fwd_predict(fwd, z0_rep, a_t)        # (m*n_actions,16)
        # bin
        zc = z_next @ cent_t.T
        cn = (cent_t * cent_t).sum(1)[None, :]
        zn = (z_next * z_next).sum(1)[:, None]
        d2 = zn - 2 * zc + cn
        lbl = d2.argmin(1)
        dmin = torch.sqrt(torch.clamp(d2.gather(1, lbl[:, None]).squeeze(1), min=0))
        lbl = lbl.cpu().numpy(); dmin = dmin.cpu().numpy()
        src_state_for = pick.repeat(n_actions)           # which source state
        for j in range(lbl.shape[0]):
            n_proposed += 1
            dst = int(lbl[j])
            if dst == c or dst not in node_set:
                continue
            if conf_thresh is not None and dmin[j] > conf_thresh:
                n_gated += 1
                continue
            e = (c, dst)
            if e in damaged_set or e in added:
                continue
            added[e] = {"action": acts[j].copy(),
                        "src_state_idx": int(src_state_for[j]),
                        "src_cluster": c, "dst_cluster": dst,
                        "bin_dist": float(dmin[j])}
            if len(added) >= n_budget:
                break
    return set(added.keys()), added, {"n_gated": n_gated, "n_proposed": n_proposed}


# --------------------------------------------------------------------------- #
# 4. INVERSE_WM
# --------------------------------------------------------------------------- #
def aug_inverse_wm(damaged_set: set, nodes: np.ndarray, centroids: np.ndarray,
                   iwm, k_max: int, disconnected_pairs, n_budget: int,
                   k_sweep=(3, 5, 8), device="cpu"):
    """For each disconnected (start,goal) pair, query inverse WM at several gap
    lengths k, reconstruct intermediate latents, bin -> bridging chain of edges.

    Returns (added_set, provenance). Provenance edge->dict with first action and
    the source CENTROID-derived info (no real state; env-check uses nearest real
    state in src cluster, handled in verify_env)."""
    node_set = set(np.asarray(nodes).tolist())
    cent_norm = AC.l2norm_np(centroids).astype(np.float32)   # WM expects normalized
    cent_t = torch.from_numpy(cent_norm).to(device)

    added = {}
    # dedupe pairs by (start,goal)
    seen_pairs = set()
    pairs = []
    for s, g in disconnected_pairs:
        if (int(s), int(g)) in seen_pairs:
            continue
        seen_pairs.add((int(s), int(g)))
        pairs.append((int(s), int(g)))

    for (cs, cg) in pairs:
        if len(added) >= n_budget:
            break
        z0 = cent_t[cs:cs + 1]
        zk = cent_t[cg:cg + 1]
        for k in k_sweep:
            kn = torch.tensor([[k / k_max]], dtype=torch.float32, device=device)
            with torch.no_grad():
                out = iwm(z0, zk, kn)
                lat = iwm.reconstruct_latents(z0, out["deltas"])[0]   # (k_max,16)
            lat = torch.nn.functional.normalize(lat, dim=-1)
            # take first k intermediate latents, bin them, build chain cs->m1->..->cg
            inter = lat[:k].cpu().numpy()
            lbl, dist = AC.bin_latents(inter, centroids)
            chain = [cs] + [int(x) for x in lbl[:k - 1]] + [cg]  # k edges
            acts = out["actions"][0].cpu().numpy()               # (k_max,8)
            for step in range(len(chain) - 1):
                s_, d_ = chain[step], chain[step + 1]
                if s_ == d_ or s_ not in node_set or d_ not in node_set:
                    continue
                e = (s_, d_)
                if e in damaged_set or e in added:
                    continue
                added[e] = {"action": acts[min(step, k_max - 1)].copy(),
                            "src_cluster": s_, "dst_cluster": d_,
                            "pair": (cs, cg), "k": k}
                if len(added) >= n_budget:
                    break
            if len(added) >= n_budget:
                break
    return set(added.keys()), added
