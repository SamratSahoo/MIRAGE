from __future__ import annotations

import json
import os
import time

import importlib.util

import numpy as np

from mirage.paths import project_path

_HERE = os.path.dirname(__file__)


def _load(name, fname):
    import sys
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_graph = _load("_mp_graph", "latent_graph.py")
_cov = _load("_mp_coverage", "coverage.py")
build_graph = _graph.build_graph
encode_all_transition_latents = _graph.encode_all_transition_latents
fit_kmeans = _graph.fit_kmeans
graph_stats = _graph.graph_stats
load_data_and_encoder = _graph.load_data_and_encoder
measure_coverage = _cov.measure_coverage
sample_pairs = _cov.sample_pairs

GRAPH_DIR = project_path("checkpoints", "graph")
OUT_JSON = project_path("compare", "phase1_results.json")
KS = (200, 500, 1000)
SEED = 0
N_PAIRS = 2000


def main(encoder_ckpt: str | None = None, out: str | None = None):
    out_path = out or os.path.join(GRAPH_DIR, "graph_K500.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.time()
    kw = {"encoder_path": encoder_ckpt} if encoder_ckpt else {}
    data, encoder, _ = load_data_and_encoder(**kw)
    print(f"loaded data+encoder in {time.time()-t0:.1f}s; "
          f"n_ep={data.n_ep} n_trans={data.n_trans}")

    train, val = data.split(0.05, SEED)
    des_pool = np.unique(data.des, axis=0)
    print(f"val states={val.obs.shape[0]} unique desired goals={des_pool.shape[0]}")

    z_t, z_tp1, _ = encode_all_transition_latents(data, encoder)
    z_fit = np.concatenate([z_t, z_tp1], axis=0)

    results = {"n_ep": int(data.n_ep), "n_trans": int(data.n_trans),
               "n_val_states": int(val.obs.shape[0]),
               "n_unique_goals": int(des_pool.shape[0]),
               "n_pairs": N_PAIRS, "per_K": {}}

    for K in KS:
        tk = time.time()
        km = fit_kmeans(z_fit, K, seed=SEED)
        c_t = km.predict(z_t)
        c_tp1 = km.predict(z_tp1)
        g = build_graph(c_t, c_tp1, km.cluster_centers_, K)
        stats = graph_stats(g)

        start_c, goal_c = sample_pairs(val, des_pool, encoder, km,
                                       n_pairs=N_PAIRS, seed=SEED)
        cov = measure_coverage(g, start_c, goal_c)

        results["per_K"][str(K)] = {"stats": stats, "coverage": cov}
        print(f"K={K} ({time.time()-tk:.1f}s): nodes={stats['n_nodes']} "
              f"edges={stats['n_edges']} self={stats['n_self_loops']} "
              f"largest_wcc={stats['largest_wcc']} largest_scc={stats['largest_scc']} "
              f"| coverage={cov['frac_connected']:.4f} "
              f"path_med={cov['path_median']:.1f}")

        if K == 500:
            np.savez_compressed(
                out_path,
                nodes=g.nodes, edges=g.edges, edge_counts=g.edge_counts,
                centroids=g.centroids, cluster_labels_t=c_t.astype(np.int32),
                cluster_labels_tp1=c_tp1.astype(np.int32),
                start_clusters=start_c.astype(np.int32),
                goal_clusters=goal_c.astype(np.int32),
            )
            print(f"  saved {out_path}")

    best_K = max(KS, key=lambda k: results["per_K"][str(k)]["coverage"]["frac_connected"])
    results["best_K"] = best_K
    results["best_coverage"] = results["per_K"][str(best_K)]["coverage"]["frac_connected"]

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nbest_K={best_K} best_coverage={results['best_coverage']:.4f}")
    print(f"wrote {OUT_JSON}; total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build latent K-means graph (graph_K500.npz)")
    ap.add_argument("--encoder-ckpt", default=None,
                    help="dual-input masked encoder checkpoint (must match the world models / "
                         "planner config). Defaults to runs_encoder/dual_input_masked/encoder_best.pt")
    ap.add_argument("--out", default=None,
                    help="output .npz path (default checkpoints/graph/graph_K500.npz)")
    args = ap.parse_args()
    main(encoder_ckpt=args.encoder_ckpt, out=args.out)
