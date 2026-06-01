import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs_world_model"
CONFIG_ROOT = PROJECT_ROOT / "config" / "world_model"
OUT = PROJECT_ROOT / "compare" / "ablation_compare.md"

RUNS = [
    ("dynamics", CONFIG_ROOT / "dynamics.yaml", "main (NLL, n=5, H=3)"),
    ("ablation_single_net", CONFIG_ROOT / "ablation_single_net.yaml", "single-net (n=1)"),
    ("ablation_h1", CONFIG_ROOT / "ablation_h1.yaml", "horizon=1"),
    ("ablation_h5", CONFIG_ROOT / "ablation_h5.yaml", "horizon=5"),
    ("ablation_init_noise_high", CONFIG_ROOT / "ablation_init_noise_high.yaml", "init_noise=0.1"),
    ("ablation_deterministic_mse", CONFIG_ROOT / "ablation_deterministic_mse.yaml", "deterministic MSE"),
]

COLS = [
    ("run", lambda cfg, r: r.get("run_name", "?")),
    ("n_members", lambda cfg, r: cfg.get("n_members", "?")),
    ("horizon", lambda cfg, r: cfg.get("horizon", "?")),
    ("loss_type", lambda cfg, r: cfg.get("loss_type", "nll")),
    ("init_noise_std", lambda cfg, r: cfg.get("init_noise_std", "?")),
    ("val/step1", lambda cfg, r: r.get("val/nll_step1", r.get("val/mse_step1", float("nan")))),
    ("rollout_mse_h1", lambda cfg, r: r.get("val/rollout_mse_h1", float("nan"))),
    ("rollout_mse_h3", lambda cfg, r: r.get("val/rollout_mse_h3", float("nan"))),
    ("rollout_mse_h5", lambda cfg, r: r.get("val/rollout_mse_h5", float("nan"))),
    ("rollout_mse_h10", lambda cfg, r: r.get("val/rollout_mse_h10", float("nan"))),
    ("disagreement_h1", lambda cfg, r: r.get("val/disagreement_h1", float("nan"))),
    ("disagreement_h3", lambda cfg, r: r.get("val/disagreement_h3", float("nan"))),
    ("wall_time_min", lambda cfg, r: r.get("wall_time_min", float("nan"))),
]


def fmt(v):
    if isinstance(v, float):
        if v != v:
            return "—"
        if abs(v) >= 100 or (abs(v) < 1e-3 and v != 0):
            return f"{v:.4g}"
        return f"{v:.4f}"
    return str(v)


def load_run(run_name, cfg_path):
    rj = RUNS_ROOT / run_name / "results.json"
    if not rj.exists():
        return None, None
    with open(rj) as f:
        results = json.load(f)
    run_cfg = RUNS_ROOT / run_name / "config.yaml"
    if run_cfg.exists():
        with open(run_cfg) as f:
            cfg = yaml.safe_load(f)
    else:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    return cfg, results


def main():
    lines = []
    lines.append("# World-Model Ablation Comparison\n")
    lines.append("Each ablation changes exactly one field vs the main run "
                 "(`config/world_model/dynamics.yaml`); all other hyperparameters are identical "
                 "(200k steps, AdamW, cosine LR 3e-4 → 3e-5, batch 256, encoder=full29d, "
                 "AntMaze umaze-v1).\n")
    lines.append("`val/step1` is `val/nll_step1` for NLL runs and `val/mse_step1` for the "
                 "deterministic-MSE run. They are NOT directly comparable; use `rollout_mse_h*` "
                 "for cross-loss comparison.\n")
    header = "| " + " | ".join(c[0] for c in COLS) + " |"
    sep = "|" + "|".join(["---"] * len(COLS)) + "|"
    lines.append(header)
    lines.append(sep)

    by_run = {}
    for run_name, cfg_path, _label in RUNS:
        cfg, results = load_run(run_name, cfg_path)
        if results is None:
            row = "| " + " | ".join([run_name] + ["—"] * (len(COLS) - 1)) + " |"
            lines.append(row)
            by_run[run_name] = None
            continue
        by_run[run_name] = (cfg, results)
        row = "| " + " | ".join(fmt(c[1](cfg, results)) for c in COLS) + " |"
        lines.append(row)
    lines.append("")

    def get(run, key, default=float("nan")):
        ent = by_run.get(run)
        if ent is None:
            return default
        return ent[1].get(key, default)

    main_h10 = get("dynamics", "val/rollout_mse_h10")
    single_h10 = get("ablation_single_net", "val/rollout_mse_h10")
    h1_h10 = get("ablation_h1", "val/rollout_mse_h10")
    h5_h10 = get("ablation_h5", "val/rollout_mse_h10")
    noise_h3_disag = get("ablation_init_noise_high", "val/disagreement_h3")
    main_h3_disag = get("dynamics", "val/disagreement_h3")
    mse_h10 = get("ablation_deterministic_mse", "val/rollout_mse_h10")

    lines.append("## Interpretation\n")
    bullets = []
    if main_h10 == main_h10 and h1_h10 == h1_h10:
        rel = (h1_h10 - main_h10) / max(abs(main_h10), 1e-9) * 100
        bullets.append(
            f"Multi-step training matters for compounding error: shrinking the training "
            f"horizon to 1 changed `rollout_mse_h10` by {rel:+.1f}% vs the H=3 main run "
            f"({h1_h10:.4f} vs {main_h10:.4f}), and extending to H=5 gave "
            f"{h5_h10:.4f}."
        )
    if single_h10 == single_h10:
        rel = (single_h10 - main_h10) / max(abs(main_h10), 1e-9) * 100
        bullets.append(
            f"Single-net ablation: collapsing the 5-member ensemble to 1 changed "
            f"`rollout_mse_h10` by {rel:+.1f}% ({single_h10:.4f}); since the prediction is "
            f"the same mean either way, this isolates the regularization effect of the "
            f"ensemble loss rather than the test-time averaging."
        )
    if noise_h3_disag == noise_h3_disag and main_h3_disag == main_h3_disag:
        ratio = noise_h3_disag / max(main_h3_disag, 1e-9)
        bullets.append(
            f"Disagreement-collapse hypothesis: the main run finished with "
            f"`disagreement_h3 ≈ {main_h3_disag:.4f}`; bumping `init_noise_std` from 1e-3 to "
            f"0.1 brought it to {noise_h3_disag:.4f} ({ratio:.1f}× the main-run level), "
            f"{'addressing' if ratio > 2 else 'not meaningfully addressing'} the collapse."
        )
    if mse_h10 == mse_h10:
        rel = (mse_h10 - main_h10) / max(abs(main_h10), 1e-9) * 100
        bullets.append(
            f"Deterministic MSE: the loss-agnostic `rollout_mse_h10` was {mse_h10:.4f} "
            f"({rel:+.1f}% vs NLL main); this isolates whether the Gaussian-NLL framework "
            f"(learnable log_sigma) buys us accuracy beyond plain MSE. Loss values across "
            f"loss types are not directly comparable."
        )
    for b in bullets:
        lines.append(f"- {b}")
    lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
