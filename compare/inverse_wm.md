# Inverse World Model (Approach B) — Results

Trained 200k steps on AntMaze umaze-v1 offline data, frozen dual-input masked encoder
(`runs_encoder/dual_input_masked/encoder_best.pt`), one NVIDIA L40S, 30.5 min wall time.
wandb: https://wandb.ai/samratsahoo-stanford-university/MIRAGE/runs/2mjap0nt

## What it does

Given `(z_t, z_{t+k}, k)` — current latent, k-steps-ahead latent, and the horizon k fed
as an input — a single MLP (3×512, LayerNorm+ReLU) predicts, for a fixed max horizon
K_max=10:
- the action sequence `a_t, ..., a_{t+k-1}` (loss masked beyond the actual k)
- the intermediate latents `z_{t+1}, ..., z_{t+k-1}` (masked beyond k)
- cumulative reward over the window
- negative xy-distance between endpoints

Design choices (from the sub-agent): **k is fed as an input, not predicted** (simpler, and
at planning time we choose k); latents predicted directly (not deltas). Loss = sum of the
four MSE terms, equally weighted, with per-step masking.

## Final validation metrics

| Metric | Value | Read |
|---|---|---|
| action_mse (overall) | 1.095 | mean over all k and steps |
| action_mse @ k=1 | **0.652** | 1-step plans are the most accurate |
| action_mse @ k=3 | 0.991 | |
| action_mse @ k=5 | 1.077 | |
| action_mse @ k=10 | 1.161 | long-horizon plans degrade, as expected |
| latent_mse | 0.034 | intermediate-latent prediction is accurate (latents are unit-norm) |
| reward_mse | 0.067 | reward head learned well |
| xydist_mse | 0.006 | xy-distance head very accurate |

## Interpretation

- **The auxiliary heads (latent, reward, xy-distance) learned cleanly.** Low MSE on all
  three. The model has a good sense of "how far apart are these two latents and what reward
  lies between them" — useful as a *scorer* for candidate subgoal nodes.
- **Action prediction has a floor (~1.1 MSE) and degrades with horizon** exactly as
  predicted. Actions are 8-D in [-1,1]; an MSE of ~0.65 at k=1 means roughly ±0.28/dim,
  rising to ±0.38/dim at k=10.
- **The known caveat is real here.** Action MSE measures imitation of the demonstrator's
  action, not whether executing the predicted actions actually reaches `z_{t+k}`. Because
  there are many valid action sequences from A to B (multi-modality), a regression model
  averages over modes — and the average action may not itself be a valid path. The plateau
  at ~1.1 is consistent with mode-averaging rather than the model failing to learn. We have
  NOT verified executability (would require rolling the predicted actions through the env or
  a forward model and checking arrival).

## How this fits the project

Per the coverage experiment (`graph_coverage.md`), **umaze never needs this** — the
un-augmented offline graph already connects 100% of (start, goal) pairs, so Dijkstra never
fails and the inverse-WM fallback is never invoked. The inverse WM matters only on harder
mazes (medium/large/diverse) where offline coverage is sparse and Dijkstra hits gaps.

**Usable as a node-scorer now; not yet validated as a path-generator.** The reward and
xy-distance heads are reliable enough to *rank* candidate reachable nodes (the "which
intermediate node best bridges the gap" step). Using its predicted *action sequence* as an
actual executable plan would need an executability check first, because of the
multi-modality concern above.

## Recommendation

- Keep the checkpoint as a capability for the harder-maze experiments.
- If/when we use it as a planner, add a forward-rollout validation: execute the predicted
  actions (in env or through the forward WM) and measure actual arrival distance to the
  target latent. That's the metric that catches mode-averaging; action MSE alone does not.
- For umaze writeup: report it as "trained and available, but coverage made it unnecessary."
