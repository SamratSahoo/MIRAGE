from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def encode_all_flat(encoder, data, mode: str, device, batch_size: int = 8192) -> np.ndarray:
    encoder.eval()
    n = data.obs.shape[0]
    out = np.empty((n, encoder.latent_dim), dtype=np.float32)
    for i in range(0, n, batch_size):
        flat_idx = np.arange(i, min(i + batch_size, n), dtype=np.int64)
        s = data.gather_state(mode, flat_idx)
        s_t = torch.from_numpy(s).to(device)
        out[flat_idx] = encoder(s_t).cpu().numpy()
    return out


def info_nce_eval(sampler, encoder, n_batches: int = 16, batch_size: int = 1024, temperature: float = 0.1) -> dict:
    encoder.eval()
    tops1, tops5, losses = [], [], []
    with torch.no_grad():
        for _ in range(n_batches):
            b = sampler.pair_batch(batch_size)
            z1 = encoder(b["s1"]); z2 = encoder(b["s2"])
            logits = z1 @ z2.t() / temperature
            B = logits.size(0)
            tgt = torch.arange(B, device=logits.device)
            loss = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))
            losses.append(loss.item())
            tops1.append((logits.argmax(1) == tgt).float().mean().item())
            tops5.append((logits.topk(min(5, B), dim=1).indices == tgt[:, None]).any(1).float().mean().item())
    return {"val/info_nce_loss": float(np.mean(losses)),
            "val/info_nce_top1": float(np.mean(tops1)),
            "val/info_nce_top5": float(np.mean(tops5))}


def dyn_eval(sampler, encoder, fwd, inv, n_batches: int = 16, batch_size: int = 1024) -> dict:
    encoder.eval(); fwd.eval(); inv.eval()
    fwd_mses, inv_mses = [], []
    with torch.no_grad():
        for _ in range(n_batches):
            b = sampler.iid_batch(batch_size)
            z   = encoder(b["s"])
            z_n = encoder(b["s_next"])
            zp  = fwd(z, b["a"])
            ap  = inv(z, z_n)
            fwd_mses.append(F.mse_loss(zp, z_n).item())
            inv_mses.append(F.mse_loss(ap, b["a"]).item())
    return {"val/forward_dyn_mse": float(np.mean(fwd_mses)),
            "val/inverse_dyn_mse": float(np.mean(inv_mses))}


def xy_probe(Z: np.ndarray, XY: np.ndarray, val_frac: float = 0.2) -> dict:
    n = Z.shape[0]
    n_val = max(1, int(val_frac * n))
    perm = np.random.default_rng(0).permutation(n)
    va, tr = perm[:n_val], perm[n_val:]
    Ztr, Zva = Z[tr], Z[va]
    Ytr, Yva = XY[tr], XY[va]
    lam = 1e-3
    A = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1], dtype=Ztr.dtype)
    B = Ztr.T @ Ytr
    W = np.linalg.solve(A, B)
    Yhat = Zva @ W
    ss_res = ((Yva - Yhat) ** 2).sum()
    ss_tot = ((Yva - Yva.mean(0)) ** 2).sum()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    return {"val/xy_probe_r2": float(r2)}


def temporal_coherence(Z: np.ndarray, data, n_pairs: int = 50_000, seed: int = 0) -> dict:
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    ep = rng.integers(0, data.n_ep, size=n_pairs)
    T_ep = data.ep_lens[ep]
    u1 = rng.random(n_pairs); u2 = rng.random(n_pairs)
    t1 = np.minimum((u1 * (T_ep + 1)).astype(np.int64), T_ep)
    t2 = np.minimum((u2 * (T_ep + 1)).astype(np.int64), T_ep)
    flat1 = data.state_starts[ep] + t1
    flat2 = data.state_starts[ep] + t2
    z1 = Z[flat1]; z2 = Z[flat2]
    d = np.linalg.norm(z1 - z2, axis=-1)
    dt = np.abs(t1 - t2).astype(np.float32)
    rho, _ = spearmanr(d, dt)
    return {"val/temporal_spearman": float(rho)}


def pca_scatter(Z: np.ndarray, XY: np.ndarray, out_path: Path, max_points: int = 20_000) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = Z.shape[0]
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, max_points, replace=False)
        Z = Z[idx]; XY = XY[idx]
    Zc = Z - Z.mean(0, keepdims=True)
    if Zc.shape[1] < 2:
        Z2 = np.concatenate([Zc, np.zeros_like(Zc)], axis=1)
    else:
        U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        Z2 = Zc @ Vt[:2].T

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc0 = axes[0].scatter(Z2[:, 0], Z2[:, 1], c=XY[:, 0], cmap="viridis", s=2, alpha=0.6)
    axes[0].set_title("PCA(z), colored by x")
    plt.colorbar(sc0, ax=axes[0])
    sc1 = axes[1].scatter(Z2[:, 0], Z2[:, 1], c=XY[:, 1], cmap="viridis", s=2, alpha=0.6)
    axes[1].set_title("PCA(z), colored by y")
    plt.colorbar(sc1, ax=axes[1])
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def full_eval(encoder, fwd, inv, val_data, val_sampler, device, out_dir: Path,
              n_batches: int = 16, batch_size: int = 1024, temperature: float = 0.1) -> dict:
    metrics: dict = {}
    metrics.update(info_nce_eval(val_sampler, encoder, n_batches=n_batches,
                                 batch_size=batch_size, temperature=temperature))
    metrics.update(dyn_eval(val_sampler, encoder, fwd, inv,
                            n_batches=n_batches, batch_size=batch_size))

    Z = encode_all_flat(encoder, val_data, val_sampler.mode, device)
    XY = val_data.ach
    metrics.update(xy_probe(Z, XY))
    metrics.update(temporal_coherence(Z, val_data))
    pca_scatter(Z, XY, out_dir / "pca_latents.png")
    return metrics
