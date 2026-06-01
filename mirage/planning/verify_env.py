"""STEP 3b - Env-executability (best-effort) for forward/inverse-WM edges.

For a sample of edges added by forward-WM and inverse-WM, set the MuJoCo ant to a
real source state, apply the predicted action, step ONE env.step (== one dataset
transition; frame_skip handled internally), encode the resulting full obs, bin to
a cluster, and check whether it equals / is within top-3 of the target cluster.

Reports executability = fraction of sampled edges whose action moved the ant
into (top-1) or toward (top-3) the target cluster.

Run:  PYTHONPATH=/home/users/asattira/MIRAGE python mirage/planning/verify_env.py
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import os as _os
import pickle
import sys as _sys

import numpy as np
import torch

from mirage.encoder.data import load_antmaze


def _load_sibling(alias, filename):
    if alias in _sys.modules:
        return _sys.modules[alias]
    here = _os.path.dirname(_os.path.abspath(__file__))
    spec = _ilu.spec_from_file_location(alias, _os.path.join(here, filename))
    mod = _ilu.module_from_spec(spec)
    _sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


AC = _load_sibling("_aug_common", "aug_common.py")


def build_obs_full(obs_dict):
    """AntMaze obs dict -> 29-D full state [obs(27), ach_xy(2)]."""
    return np.concatenate([obs_dict["observation"], obs_dict["achieved_goal"]]).astype(np.float32)


def state_to_qpos_qvel(full29):
    ob27 = full29[:27]; ach = full29[27:29]
    qpos = np.concatenate([ach, ob27[:13]]).astype(np.float64)
    qvel = ob27[13:27].astype(np.float64)
    return qpos, qvel


def topk_bin(z, centroids, k=3):
    zc = z @ centroids.T
    cn = (centroids * centroids).sum(1)
    d2 = (z * z).sum() - 2 * zc + cn
    order = np.argsort(d2)
    return order[:k]


def nearest_real_state_in_cluster(cluster, data, cluster_labels_t, trans2state,
                                  centroids, encoder, rng, device="cpu"):
    """Pick a real source state whose z_t bins to `cluster` (closest to centroid)."""
    mask = cluster_labels_t == cluster
    if not mask.any():
        return None
    st_idx = trans2state[mask]
    # subsample for speed
    if st_idx.shape[0] > 64:
        st_idx = rng.choice(st_idx, size=64, replace=False)
    states = data.gather_state("full", st_idx)
    with torch.no_grad():
        z = encoder.encode_full(torch.from_numpy(np.ascontiguousarray(states)).to(device)).cpu().numpy()
    c = centroids[cluster]
    d = np.linalg.norm(z - c[None], axis=1)
    return int(st_idx[d.argmin()])


def run_method(env, ae, prov, data, cluster_labels_t, trans2state, centroids,
               encoder, method_name, n_sample=100, device="cpu", seed=0,
               use_real_src=True):
    rng = np.random.default_rng(seed)
    edges = list(prov.items())
    if len(edges) == 0:
        return {"n": 0}
    if len(edges) > n_sample:
        sel = rng.choice(len(edges), size=n_sample, replace=False)
        edges = [edges[i] for i in sel]

    top1 = 0; top3 = 0; n_ok = 0; xy_moves = []
    for edge, info in edges:
        src, dst = edge
        action = np.asarray(info["action"], dtype=np.float32)
        if "src_state_idx" in info and info["src_state_idx"] is not None:
            src_state_idx = info["src_state_idx"]
        else:
            src_state_idx = nearest_real_state_in_cluster(
                src, data, cluster_labels_t, trans2state, centroids, encoder, rng, device)
            if src_state_idx is None:
                continue
        full = data.gather_state("full", np.array([src_state_idx]))[0]
        qpos, qvel = state_to_qpos_qvel(full)
        env.reset()
        ae.set_state(qpos, qvel)
        ach0 = full[27:29].copy()
        try:
            o, r, te, tr, inf = env.step(np.clip(action, -1, 1))
        except Exception:
            continue
        full2 = build_obs_full(o)
        with torch.no_grad():
            z2 = encoder.encode_full(torch.from_numpy(full2[None]).to(device)).cpu().numpy()[0]
        tk = topk_bin(z2, centroids, k=3)
        n_ok += 1
        if tk[0] == dst:
            top1 += 1
        if dst in tk:
            top3 += 1
        xy_moves.append(float(np.linalg.norm(o["achieved_goal"] - ach0)))
    return {
        "method": method_name,
        "n": n_ok,
        "top1": top1, "top3": top3,
        "exec_top1": top1 / n_ok if n_ok else float("nan"),
        "exec_top3": top3 / n_ok if n_ok else float("nan"),
        "mean_xy_move": float(np.mean(xy_moves)) if xy_moves else float("nan"),
    }


def main():
    device = "cpu"
    import minari
    data = load_antmaze(AC.DATASET_ID, AC.MINARI)
    trans2state = AC.build_trans_to_state_idx(data)
    G = AC.load_full_graph()
    centroids = G["centroids"].astype(np.float32)
    cluster_labels_t = G["cluster_labels_t"]
    encoder, _, _ = AC.load_encoder(device)

    ds = minari.load_dataset(AC.DATASET_ID)
    env = ds.recover_environment()
    ae = env.unwrapped.ant_env.unwrapped
    assert hasattr(ae, "set_state"), "no set_state"

    with open(f"{AC.OUT_DIR}/provenance.pkl", "rb") as f:
        prov_store = pickle.load(f)

    results = {}
    for dname in ("corridor", "random"):
        results[dname] = {}
        for method in ("forward_wm", "inverse_wm"):
            prov = prov_store.get((dname, method), {})
            r = run_method(env, ae, prov, data, cluster_labels_t, trans2state,
                           centroids, encoder, f"{dname}/{method}",
                           n_sample=100, device=device, seed=0)
            results[dname][method] = r
            print(dname, method, r)

    with open(f"{AC.OUT_DIR}/env_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved env_results.json")
    return results


if __name__ == "__main__":
    main()
