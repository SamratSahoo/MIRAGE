# Graph Coverage Experiment — AntMaze umaze-v1 (MIRAGE)

**Goal:** Build a latent-state graph from offline data using the frozen dual-input
masked encoder, then measure what fraction of realistic (start, goal) pairs can be
connected by shortest-path search on the directed transition graph. This decides
whether we need world-model graph augmentation.

**Encoder:** `/scratch/users/asattira/mirage/runs_encoder/dual_input_masked/encoder_best.pt`
(`MaskedStateEncoder`, 16-D L2-normalized latents).
**Dataset:** `D4RL/antmaze/umaze-v1` — 1430 episodes, 1,001,430 states, 1,000,000 transitions.
**Compute:** CPU only (no GPU needed). Full 1M-state encode in ~26s; whole sweep ~90s.
**No subsampling of transitions** — all 1M transitions used to build the graph.
K-means is fit on a 300k-latent random subsample (MiniBatchKMeans) for speed; assignment
(`predict`) is run on all latents.

## Method

1. Encode every state once with `encode_full`; index out `z_t`, `z_{t+1}` per transition
   using episode offsets (`state_starts[e]+t`, `+t+1`).
2. K-means cluster the pooled `{z_t, z_{t+1}}` latents (K ∈ {200, 500, 1000}).
3. Directed graph: nodes = clusters; edges = unique `(c_t, c_{t+1})` pairs (unit weight).
   Self-loops kept in graph but excluded from "real" connectivity stats.
4. **Realistic pairs (2000):** start = a full state sampled uniformly from the val split
   (`encode_full` → start cluster); goal = a desired-goal xy sampled from `data.des`
   (29-D vector: zeroed proprio + xy, `encode_goal` → goal cluster). This mirrors planning
   time: start is a full state, goal is xy-only.
5. Coverage = fraction of pairs with a directed path start→goal (multi-source BFS via
   `scipy.sparse.csgraph.dijkstra`, unweighted).

## Graph stats (per K)

| K | #nodes | #edges | #self-loops | mean out-deg | mean in-deg | #WCC | largest WCC | #SCC | largest SCC |
|------|--------|--------|-------------|--------------|-------------|------|-------------|------|-------------|
| 200  | 200    | 2788   | 200         | 12.94        | 12.94       | 1    | 200         | 1    | 200         |
| 500  | 500    | 6696   | 500         | 12.39        | 12.39       | 1    | 500         | 4    | 497         |
| 1000 | 1000   | 13763  | 1000        | 12.76        | 12.76       | 1    | 1000        | 14   | 985         |

(Degrees computed over non-self edges; WCC/SCC over the non-self adjacency.)

All K's produce a **single weakly-connected component** and a **near-total strongly-connected
component** (SCC covers 100% / 99.4% / 98.5% of nodes for K=200/500/1000). The handful of
nodes outside the largest SCC at higher K are a few dead-end clusters (e.g. 3 nodes with
out-degree 0 at K=500) that do not break start→goal reachability.

## Coverage (per K, un-augmented)

| K | frac connected | same-cluster frac | path mean | path median | path p90 | path max |
|------|----------------|-------------------|-----------|-------------|----------|----------|
| 200  | **1.000**      | 0.006             | 2.60      | 3.0         | 4.0      | 4        |
| 500  | **1.000**      | 0.001             | 3.43      | 3.0         | 5.0      | 8        |
| 1000 | **1.000**      | 0.001             | 4.46      | 4.0         | 7.0      | 11       |

**Headline: 100.0% of realistic (start, goal) pairs are connected at every K** (best K = 200,
coverage = 1.000). Path lengths are short (median 3–4 hops; p90 ≤ 7), as expected for the
small umaze layout. The 2000-pair sample touched 245 distinct start clusters and 336 distinct
goal clusters (out of 500 at K=500), with only 0.1% of pairs landing trivially in the same
cluster — so the 100% is genuine reachability, not a same-node artifact. Every sampled goal
cluster has in-degree ≥ 1.

## Phase 2 (world-model augmentation): NOT TRIGGERED

Phase 2 is conditional on best-K coverage < 90%. Best-K coverage is **100%**, far above the
threshold, so the forward-WM retrain + graph-augmentation experiment was **not run**. No GPU
job was submitted. (Per the interpretation guide: >90% connected → augmentation barely
matters; the offline graph already covers the task.)

## Verdict

**Augmentation is not needed for umaze.** The offline transition graph on the dual-input
encoder latent is already a single connected component with a near-complete SCC, and every
realistic start→goal pair is reachable in a few hops. This holds robustly across K = 200, 500,
and 1000. World-model graph augmentation would add edges that the offline data already covers;
it cannot improve a coverage that is already saturated at 100%.

This is expected for umaze specifically — it is a small maze and the 1M-transition offline
dataset densely covers the reachable state space. The result does **not** generalize to larger
mazes (medium/large/diverse), where offline coverage is sparser and the SCC may fragment; the
augmentation question should be re-evaluated per-environment when we move beyond umaze.

## Artifacts

- `compare/phase1_results.json` — full stats + coverage per K (machine-readable).
- `/scratch/users/asattira/mirage/graph/graph_K500.npz` — K=500 graph: `nodes`, `edges`,
  `edge_counts`, `centroids`, per-transition `cluster_labels_t`/`cluster_labels_tp1`, and the
  sampled `start_clusters`/`goal_clusters`.
- Code: `mirage/planning/graph_planning/latent_graph.py`, `mirage/planning/graph_planning/coverage.py`,
  `mirage/planning/graph_planning/run_build_graph.py`.
