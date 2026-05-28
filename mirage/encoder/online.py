from __future__ import annotations

import torch
import torch.optim as optim

from .losses import forward_dyn_loss, info_nce, inverse_dyn_loss
from .models import ForwardDynamics, InverseDynamics, StateEncoder


_PROPRIO_27 = 27


class OnlineEncoderUpdater:
    def __init__(
        self,
        encoder: StateEncoder,
        fwd: ForwardDynamics,
        inv: InverseDynamics,
        input_mode: str,
        lr: float = 3e-4,
        nce_weight: float = 1.0,
        fwd_weight: float = 0.1,
        inv_weight: float = 1.0,
        nce_temperature: float = 0.1,
        contrastive_window: int = 8,
        steps_per_iter: int = 4,
        batch_size: int = 4096,
        grad_clip: float = 10.0,
    ):
        self.encoder = encoder
        self.fwd = fwd
        self.inv = inv
        self.input_mode = input_mode
        self.nce_weight = float(nce_weight)
        self.fwd_weight = float(fwd_weight)
        self.inv_weight = float(inv_weight)
        self.nce_temperature = float(nce_temperature)
        self.contrastive_window = int(contrastive_window)
        self.steps_per_iter = int(steps_per_iter)
        self.batch_size = int(batch_size)
        self.grad_clip = float(grad_clip)

        params = list(encoder.parameters()) + list(fwd.parameters()) + list(inv.parameters())
        self.optimizer = optim.Adam(params, lr=float(lr))

    def slice_state(self, obs_TN_107: torch.Tensor, achieved_TN_2: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "xy":
            return achieved_TN_2
        if self.input_mode == "proprio":
            return obs_TN_107[..., :_PROPRIO_27]
        if self.input_mode == "full":
            return torch.cat([obs_TN_107[..., :_PROPRIO_27], achieved_TN_2], dim=-1)
        raise ValueError(f"unknown input_mode {self.input_mode!r}")

    def step(self, obs_TN: torch.Tensor, actions_TN: torch.Tensor, achieved_TN: torch.Tensor,
             dones_TN: torch.Tensor) -> dict:
        T, N = obs_TN.shape[:2]
        device = obs_TN.device
        s_TN = self.slice_state(obs_TN, achieved_TN)
        W = self.contrastive_window
        B = self.batch_size

        metrics: dict = {}
        self.encoder.train(); self.fwd.train(); self.inv.train()
        for _ in range(self.steps_per_iter):
            env_idx = torch.randint(0, N, (B,), device=device)
            t1 = torch.randint(0, T, (B,), device=device)
            lo = torch.clamp(t1 - W, min=0)
            hi = torch.clamp(t1 + W, max=T - 1)
            t2 = lo + (torch.rand(B, device=device) * (hi - lo + 1).float()).long()
            t2 = torch.minimum(t2, hi)

            t_iid = torch.randint(0, T - 1, (B,), device=device)
            env_iid = torch.randint(0, N, (B,), device=device)

            s1 = s_TN[t1, env_idx]
            s2 = s_TN[t2, env_idx]
            s_cur = s_TN[t_iid, env_iid]
            s_next = s_TN[t_iid + 1, env_iid]
            a_cur = actions_TN[t_iid, env_iid]

            z1 = self.encoder(s1)
            z2 = self.encoder(s2)
            nce_loss, nce_log = info_nce(z1, z2, temperature=self.nce_temperature)

            z_cur = self.encoder(s_cur)
            z_next = self.encoder(s_next)
            z_next_pred = self.fwd(z_cur, a_cur)
            a_pred = self.inv(z_cur, z_next)
            fwd_loss, fwd_log = forward_dyn_loss(z_next_pred, z_next.detach())
            inv_loss, inv_log = inverse_dyn_loss(a_pred, a_cur)

            loss = (self.nce_weight * nce_loss
                    + self.fwd_weight * fwd_loss
                    + self.inv_weight * inv_loss)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters())
                    + list(self.fwd.parameters())
                    + list(self.inv.parameters()),
                    self.grad_clip,
                )
            self.optimizer.step()

            metrics = {**nce_log, **fwd_log, **inv_log, "encoder_train/total_loss": loss.item()}
        return metrics

    def state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "fwd": self.fwd.state_dict(),
            "inv": self.inv.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.fwd.load_state_dict(state["fwd"])
        self.inv.load_state_dict(state["inv"])
        self.optimizer.load_state_dict(state["optimizer"])
