from __future__ import annotations

import importlib.util as _ilu
import json
import os
import os as _os
import pickle
import sys as _sys
import time

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


AC = _load_sibling("_aug_common", "augment_common.py")
H = _load_sibling("_holes", "graph_damage.py")
A = _load_sibling("_augment", "augment_methods.py")
V = _load_sibling("_verify_graph", "evaluate_augmentation.py")


def main():
    t0 = time.time()
    device = "cpu"
    os.makedirs(AC.OUT_DIR, exist_ok=True)

    print("Loading graph/data/models...")
    G = AC.load_full_graph()
    data = load_antmaze(AC.DATASET_ID, AC.MINARI)
    trans2state = AC.build_trans_to_state_idx(data)
    encoder, _, _ = AC.load_encoder(device)
    fwd = AC.load_forward_wm(device)
    iwm, k_max = AC.load_inverse_wm(device)
    centroids = G["centroids"].astype(np.float32)
    nodes = G["nodes"]
    cluster_labels_t = G["cluster_labels_t"]
    print(f"  done in {time.time()-t0:.1f}s")

    print("Building damaged graphs...")
    dmg = H.build_damaged_graphs(G, data, trans2state)
    full_set = dmg["full_set"]
    starts, goals = dmg["starts"], dmg["goals"]
    base_cov = dmg["base_cov"]
    print(f"  full nonself edges: {dmg['n_full_nonself']}, base coverage: {base_cov:.4f}")

    results = {"base_cov": base_cov, "n_full_nonself": dmg["n_full_nonself"],
               "damage": {}, "methods": {}}

    for dname in ("corridor", "random"):
        d = dmg[dname]
        results["damage"][dname] = {
            "n_removed": d["n_removed"],
            "coverage_after_damage": d["coverage"],
            "conn": d["conn"],
        }
        if dname == "corridor":
            results["damage"][dname]["info"] = d["info"]
        print(f"  [{dname}] removed={d['n_removed']} cov_after={d['coverage']:.4f} "
              f"SCC={d['conn']['n_scc']} largestSCC_frac={d['conn']['largest_scc_frac']:.3f}")

    prov_store = {}
    for dname in ("corridor", "random"):
        d = dmg[dname]
        damaged_set = d["kept"]
        removed_set = d["removed"]
        n_budget = d["n_removed"]
        damaged_cov = d["coverage"]
        pair_conn = d["pair_connected"]

        disc_idx = np.where(~pair_conn)[0]
        disc_pairs = [(int(starts[i]), int(goals[i])) for i in disc_idx]

        if dname == "corridor":
            side = d["side"]
            near = set()
            for (s, t) in removed_set:
                near.add(s); near.add(t)
            src_clusters = np.array(sorted(near), dtype=np.int64)
        else:
            src_clusters = nodes

        print(f"\n=== {dname}: budget={n_budget}, disc_pairs={len(disc_pairs)} ===")
        results["methods"][dname] = {}

        added = A.aug_random(damaged_set, nodes, n_budget, seed=0)
        results["methods"][dname]["random"] = V.evaluate_method(
            damaged_set, added, removed_set, full_set, nodes, starts, goals,
            base_cov, damaged_cov)
        np.save(f"{AC.OUT_DIR}/{dname}_random_added.npy", AC.set_to_edges(added))
        print(f"  random:  {results['methods'][dname]['random']}")

        added = A.aug_knn_latent(damaged_set, nodes, centroids, n_budget)
        results["methods"][dname]["knn"] = V.evaluate_method(
            damaged_set, added, removed_set, full_set, nodes, starts, goals,
            base_cov, damaged_cov)
        np.save(f"{AC.OUT_DIR}/{dname}_knn_added.npy", AC.set_to_edges(added))
        print(f"  knn:     {results['methods'][dname]['knn']}")

        added, prov, gate = A.aug_forward_wm(
            damaged_set, nodes, centroids, data, cluster_labels_t, trans2state,
            encoder, fwd, n_budget, src_clusters=src_clusters, n_actions=16,
            conf_thresh=None, seed=0, device=device)
        res = V.evaluate_method(damaged_set, added, removed_set, full_set,
                                nodes, starts, goals, base_cov, damaged_cov)
        res["gate"] = gate
        results["methods"][dname]["forward_wm"] = res
        np.save(f"{AC.OUT_DIR}/{dname}_forward_wm_added.npy", AC.set_to_edges(added))
        prov_store[(dname, "forward_wm")] = prov
        print(f"  fwd_wm:  {res}")

        added, prov = A.aug_inverse_wm(
            damaged_set, nodes, centroids, iwm, k_max, disc_pairs, n_budget,
            k_sweep=(3, 5, 8), device=device)
        res = V.evaluate_method(damaged_set, added, removed_set, full_set,
                                nodes, starts, goals, base_cov, damaged_cov)
        results["methods"][dname]["inverse_wm"] = res
        np.save(f"{AC.OUT_DIR}/{dname}_inverse_wm_added.npy", AC.set_to_edges(added))
        prov_store[(dname, "inverse_wm")] = prov
        print(f"  inv_wm:  {res}")

    with open(f"{AC.OUT_DIR}/provenance.pkl", "wb") as f:
        pickle.dump(prov_store, f)
    with open(f"{AC.OUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results.json and provenance.pkl to {AC.OUT_DIR}")
    print(f"Total time {time.time()-t0:.1f}s")
    return results


if __name__ == "__main__":
    main()
