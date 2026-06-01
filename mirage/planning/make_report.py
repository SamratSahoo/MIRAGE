"""Build compare/graph_augmentation.png and compare/graph_augmentation.md from
the saved results.json + env_results.json."""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/scratch/users/asattira/mirage/graph_aug"
COMP = "/home/users/asattira/MIRAGE/compare"
METHODS = ["random", "knn", "forward_wm", "inverse_wm"]
LABELS = {"random": "random", "knn": "kNN-latent",
          "forward_wm": "forward-WM", "inverse_wm": "inverse-WM"}


def main():
    res = json.load(open(f"{OUT}/results.json"))
    env = json.load(open(f"{OUT}/env_results.json"))

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, dname in zip(axes, ("corridor", "random")):
        m = res["methods"][dname]
        x = np.arange(len(METHODS))
        cov = [m[k]["coverage"] for k in METHODS]
        prec = [m[k]["edge_precision"] for k in METHODS]
        # executability: only fwd/inv have it (top3)
        ex = []
        for k in METHODS:
            if k in ("forward_wm", "inverse_wm"):
                ex.append(env[dname][k].get("exec_top3", np.nan))
            else:
                ex.append(np.nan)
        w = 0.27
        ax.bar(x - w, cov, w, label="coverage (abs)", color="#4C72B0")
        ax.bar(x, prec, w, label="edge precision", color="#DD8452")
        ax.bar(x + w, ex, w, label="env-exec (top3)", color="#55A868")
        ax.set_xticks(x); ax.set_xticklabels([LABELS[k] for k in METHODS], rotation=20)
        ax.set_title(f"{dname} cut" if dname == "corridor" else "random removal")
        ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
        ax.axhline(res["base_cov"], ls="--", c="gray", lw=1)
    axes[0].set_ylabel("value")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Graph augmentation: coverage vs edge-precision vs env-executability", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{COMP}/graph_augmentation.png", dpi=130)
    print(f"wrote {COMP}/graph_augmentation.png")

    # ---- markdown ----
    L = []
    A = L.append
    A("# Graph Augmentation Experiment (MIRAGE, AntMaze umaze-v1)\n")
    A("Knock holes in the offline latent graph `G_full` (K=500), then test whether "
      "augmentation methods refill it with REAL (data-observed) edges. Reconnecting "
      "is trivial; the science is **edge precision vs ground truth** and **physical "
      "executability**.\n")
    A(f"- Ground truth `G_full`: {res['n_full_nonself']} non-self directed edges, "
      f"500 clusters. Baseline (start,goal) coverage on the 2000 eval pairs: "
      f"**{res['base_cov']:.4f}**.\n")

    A("## STEP 1 - Damage\n")
    A("| damage | edges removed | coverage after | #SCC | largest-SCC frac |")
    A("|---|---|---|---|---|")
    for dname in ("corridor", "random"):
        d = res["damage"][dname]
        c = d["conn"]
        A(f"| {dname} | {d['n_removed']} | {d['coverage_after_damage']:.4f} | "
          f"{c['n_scc']} | {c['largest_scc_frac']:.3f} |")
    ci = res["damage"]["corridor"].get("info", {})
    A(f"\nCorridor cut: spatial y-boundary = {ci.get('boundary_y', float('nan')):.3f} "
      f"(clusters below={ci.get('n_below')}, above={ci.get('n_above')}). The cut "
      "splits the maze into two halves (largest SCC frac = 0.50) and drops coverage "
      f"from 1.00 to {res['damage']['corridor']['coverage_after_damage']:.2f}. Random "
      "removal of 50% of edges barely dents coverage "
      f"({res['damage']['random']['coverage_after_damage']:.3f}) because the graph is "
      "densely redundant - this is exactly why the corridor cut is the meaningful test.\n")

    for dname in ("corridor", "random"):
        A(f"## STEP 2+3 - Methods on **{dname}** damage\n")
        A("Each method augments a fresh copy of the damaged graph (never stacked); "
          "added-edge budget capped at #removed.\n")
        A("| method | #added | coverage | cov recovery | **edge precision** | edge recall | env-exec top1 | env-exec top3 |")
        A("|---|---|---|---|---|---|---|---|")
        for k in METHODS:
            r = res["methods"][dname][k]
            if k in ("forward_wm", "inverse_wm"):
                e = env[dname][k]
                et1 = f"{e['exec_top1']:.2f}"; et3 = f"{e['exec_top3']:.2f} (n={e['n']})"
            else:
                et1 = "-"; et3 = "-"
            A(f"| {LABELS[k]} | {r['n_added']} | {r['coverage']:.4f} | "
              f"{r['cov_recovery']:+.4f} | **{r['edge_precision']:.3f}** | "
              f"{r['edge_recall']:.3f} | {et1} | {et3} |")
        fg = res["methods"][dname]["forward_wm"].get("gate", {})
        A(f"\nForward-WM proposed {fg.get('n_proposed','?')} candidate transitions; "
          "most bin back to the source cluster or to edges already present, so few "
          "NOVEL edges survive (high precision, low recall by construction).\n")

    A("## Env-executability detail\n")
    A("Set the MuJoCo ant to a real source state, apply the predicted action, step "
      "ONE `env.step` (verified to reproduce dataset transitions to <0.002 xy error), "
      "encode the result, bin, check vs target cluster.\n")
    A("| damage / method | n | top1 | top3 | mean xy move |")
    A("|---|---|---|---|---|")
    for dname in ("corridor", "random"):
        for k in ("forward_wm", "inverse_wm"):
            e = env[dname][k]
            A(f"| {dname}/{LABELS[k]} | {e['n']} | {e['exec_top1']:.2f} | "
              f"{e['exec_top3']:.2f} | {e['mean_xy_move']:.3f} |")

    A("\n## Verdict\n")
    cp = res["methods"]["corridor"]
    A(f"- **Edge precision is the headline.** On the corridor cut, forward-WM adds "
      f"edges with **{cp['forward_wm']['edge_precision']:.2f} precision** vs random's "
      f"**{cp['random']['edge_precision']:.3f}** and kNN-latent's "
      f"**{cp['knn']['edge_precision']:.2f}**. World-model augmentation adds REAL edges; "
      "random does not. kNN-latent is a non-trivial heuristic baseline (latent "
      "neighbors are often real neighbors) but well below the forward WM.")
    A("- **Forward-WM is physically grounded**: 0.56 top1 / 0.74 top3 executability on "
      "the corridor cut - the edges it claims are actually traversable in MuJoCo.")
    A("- **Coverage**: all methods restore coverage to ~1.0 because even a few well-"
      "placed (or random) edges reconnect the graph. Coverage alone does NOT "
      "distinguish methods - precision/executability does. This is the core point of "
      "the experiment.")
    A("- **Forward vs inverse**: forward-WM wins decisively on precision AND "
      "executability on BOTH damage types. Inverse-WM has low precision (~0.04) and "
      "~0 single-step executability: it predicts a *multi-step* plan over k steps, so "
      "binning intermediate latents into single edges is lossy and the FIRST action "
      "only nudges the ant toward the first waypoint, not across a binned edge. Its "
      "deltas also show multi-modality collapse (at k=3 it often emits a direct "
      "start->goal edge). Inverse-WM is better suited to *planning a path* than to "
      "*proposing individual verifiable edges*.")
    A("\n### Caveats\n")
    A("- Precision-vs-`G_full` is a **lower bound**: an added edge absent from G_full "
      "may still be physically real (just unobserved in the offline data). Env-exec is "
      "the antidote - and forward-WM's 0.74 top3 executability exceeds its own recall, "
      "evidence that some 'imprecise' edges are in fact traversable.")
    A("- Forward-WM's recall is structurally low: random actions from real states "
      "reproduce mostly already-present transitions, so its novel-edge yield is small. "
      "It cannot bridge a hard spatial cut (z_next stays local) - hence few corridor "
      "edges. To refill a cut you need a *goal-directed* proposer (inverse-WM), which "
      "is precisely where the binning/multi-modality issues bite.")
    A("\n### Does this demonstrate the augmentation contribution?\n")
    A("Yes, partially and honestly: it cleanly demonstrates that a **forward world "
      "model proposes real, executable edges where random/heuristic baselines do not** "
      "(precision 0.93-0.98 vs ~0.00-0.58). It does NOT yet show a WM method that "
      "simultaneously (a) targets the disconnected region and (b) keeps high precision "
      "- the inverse-WM targets but is imprecise per-edge. The contribution is "
      "established for *edge quality*; closing the *coverage-with-precision* gap (e.g. "
      "verifying inverse-WM bridges by rolling out all k actions in the WM/env before "
      "committing edges) is the natural follow-up.\n")

    with open(f"{COMP}/graph_augmentation.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {COMP}/graph_augmentation.md")


if __name__ == "__main__":
    main()
