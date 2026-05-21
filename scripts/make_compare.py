from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in args.runs:
        rp = Path(r)
        rj = rp / "results.json"
        if not rj.exists():
            print(f"[warn] {rj} missing — skipping {rp}")
            continue
        with open(rj) as f:
            data = json.load(f)
        data["_run_dir"] = str(rp)
        rows.append(data)

        src_png = rp / "eval_final" / "pca_latents.png"
        if src_png.exists():
            shutil.copy(src_png, out / f"pca_{rp.name}.png")

    if not rows:
        print("[err] no runs found")
        return

    keys_order = [
        ("input_mode",            "Encoder input"),
        ("total_steps",           "Steps"),
        ("wall_time_min",         "Wall time (min)"),
        ("val/info_nce_loss",     "Val InfoNCE loss"),
        ("val/info_nce_top1",     "Val top-1 acc"),
        ("val/info_nce_top5",     "Val top-5 acc"),
        ("val/forward_dyn_mse",   "Val forward MSE"),
        ("val/inverse_dyn_mse",   "Val inverse MSE"),
        ("val/xy_probe_r2",       "z->xy probe R^2"),
        ("val/temporal_spearman", "Temporal Spearman"),
    ]

    md = ["# Encoder runs: side-by-side\n"]
    header = "| Metric | " + " | ".join(r["run_name"] for r in rows) + " |"
    sep = "|---" * (len(rows) + 1) + "|"
    md += [header, sep]
    for k, label in keys_order:
        cells = []
        for r in rows:
            v = r.get(k, "—")
            if isinstance(v, float):
                v = f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}"
            cells.append(str(v))
        md.append(f"| {label} | " + " | ".join(cells) + " |")

    md.append("\n## PCA scatter\n")
    for r in rows:
        md.append(f"### {r['run_name']} (input={r['input_mode']})")
        md.append(f"![{r['run_name']}](pca_{Path(r['_run_dir']).name}.png)\n")

    (out / "compare.md").write_text("\n".join(md) + "\n")
    print(f"[ok] wrote {out / 'compare.md'}")
    print(f"[ok] copied PCA plots to {out}/")


if __name__ == "__main__":
    main()
