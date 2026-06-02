# Masked Encoder Consistency: full-state vs goal (xy-only)

Checkpoint: `dual_input_masked/encoder_best.pt`  |  val states sampled: 20000  |  latent dim: 16 (L2-normalized)

## 1. Per-state cosine(z_full, z_goal)

| mean | median | p10 | min |
|---|---|---|---|
| 0.8724 | 0.9014 | 0.7845 | -0.1611 |

## 2. Euclidean distance ||z_full - z_goal||

| mean | median | p90 | random-diff-state mean | ratio |
|---|---|---|---|---|
| 0.4820 | 0.4440 | 0.6565 | 1.3879 | 0.3473 |

Ratio << 1 means full-vs-goal distance is small relative to the spread between unrelated states.

## 3. Cluster agreement (k-means on z_full)

| K | same-cluster | within-top3 |
|---|---|---|
| 200 | 0.6925 | 0.8978 |
| 500 | 0.5253 | 0.7847 |
| 1000 | 0.4291 | 0.7053 |

## 4. xy-recovery ridge probe (held-out R^2)

| source | R^2 |
|---|---|
| z_full -> xy | 0.7377 |
| z_goal -> xy | 0.8097 |

## 5. PCA overlay

See `masking_consistency.png`. Left = z_full, right = z_goal under the same PCA projection (fit on z_full), shared xy[0] colorbar. Faint lines connect each state's full and goal latent for a 200-point subsample; short lines = consistent.

## Verdict

**YES (borderline)** — the masking is consistent enough to use this encoder for goal encoding in graph planning, with a caveat. Cosine mean (0.87) sits just under the strict 0.9 bar, but every corroborating metric supports usability: goal latents land within the 3 nearest clusters ~78-90% of the time, full-vs-goal distance is ~3x smaller than the spread between unrelated states, and xy is recovered from goal latents at least as well as from full latents. Recommend top-k (k>=3) nearest-node matching rather than exact-node matching when grounding a goal in the planning graph.

- Headline cosine mean = **0.8724** (<= 0.9 strict bar; > 0.85 usable bar).

- K=500 same-cluster fraction = **0.5253**, within-top3 = **0.7847**.

- xy R^2: full=0.7377 vs goal=0.8097 (|diff|=0.0719).

