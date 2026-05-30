from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mirage.encoder.models import build_mlp


LOG_SIGMA_MIN = -10.0
LOG_SIGMA_MAX = 2.0


class ProbabilisticDynamics(nn.Module):
    def __init__(self, latent_dim: int, act_dim: int,
                 hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.net = build_mlp(latent_dim + act_dim, hidden_dim, n_hidden,
                             out_dim=2 * latent_dim)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([z, a], dim=-1))
        mu_delta, log_sigma = out.split(self.latent_dim, dim=-1)
        log_sigma = log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        mu = z + mu_delta
        return mu, log_sigma

    @torch.no_grad()
    def predict(self, z: torch.Tensor, a: torch.Tensor, l2_normalize: bool = True) -> torch.Tensor:
        mu, _ = self.forward(z, a)
        if l2_normalize:
            mu = F.normalize(mu, dim=-1, eps=1e-8)
        return mu


class DynamicsEnsemble(nn.Module):
    def __init__(self, n_members: int, latent_dim: int, act_dim: int,
                 hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        self.n_members = n_members
        self.latent_dim = latent_dim
        self.act_dim = act_dim
        self.members = nn.ModuleList([
            ProbabilisticDynamics(latent_dim, act_dim, hidden_dim, n_hidden)
            for _ in range(n_members)
        ])

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mus, lss = [], []
        for m in self.members:
            mu, ls = m(z, a)
            mus.append(mu); lss.append(ls)
        return torch.stack(mus, dim=0), torch.stack(lss, dim=0)

    @torch.no_grad()
    def ensemble_stats(self, z: torch.Tensor, a: torch.Tensor) -> dict:
        mu_all, _ = self.forward(z, a)
        mu_mean = mu_all.mean(dim=0)
        disagreement = (mu_all - mu_mean.unsqueeze(0)).norm(dim=-1)
        return {
            "mu_mean": mu_mean,
            "ensemble_disagreement_mean": disagreement.mean(),
            "ensemble_disagreement_max": disagreement.max(),
        }

    def load_from_forward_dynamics(self, fwd_state_dict: dict, noise_std: float = 0.0,
                                   verbose: bool = True) -> None:
        loaded_keys, skipped_keys = [], []
        for i, m in enumerate(self.members):
            dst_state = dict(m.state_dict())
            for k, v in fwd_state_dict.items():
                if k in dst_state and dst_state[k].shape == v.shape:
                    new_v = v.clone()
                    if noise_std > 0.0:
                        new_v = new_v + noise_std * torch.randn_like(new_v)
                    dst_state[k] = new_v
                    if i == 0:
                        loaded_keys.append(k)
                else:
                    if i == 0:
                        target_shape = dst_state.get(k, None)
                        target_shape = tuple(target_shape.shape) if target_shape is not None else "MISSING"
                        skipped_keys.append((k, tuple(v.shape), target_shape))
            m.load_state_dict(dst_state, strict=True)
        if verbose:
            print(f"[ensemble init] loaded {len(loaded_keys)} tensors per member from fwd ckpt: "
                  f"{loaded_keys}")
            if skipped_keys:
                print(f"[ensemble init] skipped (shape mismatch, kept fresh init):")
                for k, src, dst in skipped_keys:
                    print(f"   {k}  src{src} -> dst{dst}")
            print(f"[ensemble init] {len(self.members)} members, noise_std={noise_std}")
