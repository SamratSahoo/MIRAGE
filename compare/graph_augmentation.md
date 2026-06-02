# Graph Augmentation Experiment (MIRAGE, AntMaze umaze-v1)

Knock holes in the offline latent graph `G_full` (K=500), then test whether augmentation methods refill it with REAL (data-observed) edges. Reconnecting is trivial; the science is **edge precision vs ground truth** and **physical executability**.

- Ground truth `G_full`: 6196 non-self directed edges, 500 clusters. Baseline (start,goal) coverage on the 2000 eval pairs: **1.0000**.

## STEP 1 - Damage

| damage | edges removed | coverage after | #SCC | largest-SCC frac |
|---|---|---|---|---|
| corridor | 630 | 0.4895 | 6 | 0.500 |
| random | 3098 | 0.9925 | 25 | 0.948 |

Corridor cut: spatial y-boundary = 3.829 (clusters below=250, above=250). The cut splits the maze into two halves (largest SCC frac = 0.50) and drops coverage from 1.00 to 0.49. Random removal of 50% of edges barely dents coverage (0.993) because the graph is densely redundant - this is exactly why the corridor cut is the meaningful test.

## STEP 2+3 - Methods on **corridor** damage

Each method augments a fresh copy of the damaged graph (never stacked); added-edge budget capped at #removed.

| method | #added | coverage | cov recovery | **edge precision** | edge recall | env-exec top1 | env-exec top3 |
|---|---|---|---|---|---|---|---|
| random | 630 | 1.0000 | +0.5105 | **0.003** | 0.003 | - | - |
| kNN-latent | 630 | 1.0000 | +0.5105 | **0.392** | 0.392 | - | - |
| forward-WM | 27 | 1.0000 | +0.5105 | **0.926** | 0.040 | 0.56 | 0.74 (n=27) |
| inverse-WM | 630 | 1.0000 | +0.5105 | **0.037** | 0.037 | 0.00 | 0.00 (n=100) |

Forward-WM proposed 17472 candidate transitions; most bin back to the source cluster or to edges already present, so few NOVEL edges survive (high precision, low recall by construction).

## STEP 2+3 - Methods on **random** damage

Each method augments a fresh copy of the damaged graph (never stacked); added-edge budget capped at #removed.

| method | #added | coverage | cov recovery | **edge precision** | edge recall | env-exec top1 | env-exec top3 |
|---|---|---|---|---|---|---|---|
| random | 3098 | 1.0000 | +0.0075 | **0.012** | 0.012 | - | - |
| kNN-latent | 3098 | 1.0000 | +0.0075 | **0.576** | 0.576 | - | - |
| forward-WM | 162 | 0.9925 | +0.0000 | **0.981** | 0.051 | 0.30 | 0.63 (n=100) |
| inverse-WM | 27 | 1.0000 | +0.0075 | **0.111** | 0.001 | 0.00 | 0.11 (n=27) |

Forward-WM proposed 48000 candidate transitions; most bin back to the source cluster or to edges already present, so few NOVEL edges survive (high precision, low recall by construction).

## Env-executability detail

Set the MuJoCo ant to a real source state, apply the predicted action, step ONE `env.step` (verified to reproduce dataset transitions to <0.002 xy error), encode the result, bin, check vs target cluster.

| damage / method | n | top1 | top3 | mean xy move |
|---|---|---|---|---|
| corridor/forward-WM | 27 | 0.56 | 0.74 | 0.076 |
| corridor/inverse-WM | 100 | 0.00 | 0.00 | 0.052 |
| random/forward-WM | 100 | 0.30 | 0.63 | 0.060 |
| random/inverse-WM | 27 | 0.00 | 0.11 | 0.009 |

## Verdict

- **Edge precision is the headline.** On the corridor cut, forward-WM adds edges with **0.93 precision** vs random's **0.003** and kNN-latent's **0.39**. World-model augmentation adds REAL edges; random does not. kNN-latent is a non-trivial heuristic baseline (latent neighbors are often real neighbors) but well below the forward WM.
- **Forward-WM is physically grounded**: 0.56 top1 / 0.74 top3 executability on the corridor cut - the edges it claims are actually traversable in MuJoCo.
- **Coverage**: all methods restore coverage to ~1.0 because even a few well-placed (or random) edges reconnect the graph. Coverage alone does NOT distinguish methods - precision/executability does. This is the core point of the experiment.
- **Forward vs inverse**: forward-WM wins decisively on precision AND executability on BOTH damage types. Inverse-WM has low precision (~0.04) and ~0 single-step executability: it predicts a *multi-step* plan over k steps, so binning intermediate latents into single edges is lossy and the FIRST action only nudges the ant toward the first waypoint, not across a binned edge. Its deltas also show multi-modality collapse (at k=3 it often emits a direct start->goal edge). Inverse-WM is better suited to *planning a path* than to *proposing individual verifiable edges*.

### Caveats

- Precision-vs-`G_full` is a **lower bound**: an added edge absent from G_full may still be physically real (just unobserved in the offline data). Env-exec is the antidote - and forward-WM's 0.74 top3 executability exceeds its own recall, evidence that some 'imprecise' edges are in fact traversable.
- Forward-WM's recall is structurally low: random actions from real states reproduce mostly already-present transitions, so its novel-edge yield is small. It cannot bridge a hard spatial cut (z_next stays local) - hence few corridor edges. To refill a cut you need a *goal-directed* proposer (inverse-WM), which is precisely where the binning/multi-modality issues bite.

### Does this demonstrate the augmentation contribution?

Yes, partially and honestly: it cleanly demonstrates that a **forward world model proposes real, executable edges where random/heuristic baselines do not** (precision 0.93-0.98 vs ~0.00-0.58). It does NOT yet show a WM method that simultaneously (a) targets the disconnected region and (b) keeps high precision - the inverse-WM targets but is imprecise per-edge. The contribution is established for *edge quality*; closing the *coverage-with-precision* gap (e.g. verifying inverse-WM bridges by rolling out all k actions in the WM/env before committing edges) is the natural follow-up.

