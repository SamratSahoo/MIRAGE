from __future__ import annotations

import torch
import torch.nn as nn

from mirage.encoder.models import StateEncoder


_VALID_GOAL_ITEMS = {"goal", "latent_goal", "subgoal", "latent_subgoal"}
_PROPRIO_DIM = 105
_GOAL_DIM = 2
_PROPRIO_27 = 27


def _slice_proprio(obs_107: torch.Tensor) -> torch.Tensor:
    return obs_107[:, :_PROPRIO_27]


def _slice_proprio_contact(obs_107: torch.Tensor) -> torch.Tensor:
    return obs_107[:, :_PROPRIO_DIM]


def _slice_desired_goal(obs_107: torch.Tensor) -> torch.Tensor:
    return obs_107[:, _PROPRIO_DIM:_PROPRIO_DIM + _GOAL_DIM]


class PolicyInputBuilder(nn.Module):
    def __init__(self, state_mode: str, goal_conditioning: list[str],
                 state_encoder: StateEncoder | None, state_encoder_input_mode: str | None,
                 goal_encoder: StateEncoder | None, goal_encoder_input_mode: str | None):
        super().__init__()
        if state_mode not in ("raw", "latent"):
            raise ValueError(f"unknown state_mode {state_mode!r}")
        for item in goal_conditioning:
            if item not in _VALID_GOAL_ITEMS:
                raise ValueError(f"unknown goal_conditioning item {item!r}")
        if state_mode == "latent" and state_encoder is None:
            raise ValueError("state_mode='latent' requires a state encoder")
        needs_goal_encoder = any(c in ("latent_goal", "latent_subgoal") for c in goal_conditioning)
        if needs_goal_encoder and goal_encoder is None:
            raise ValueError("goal_conditioning includes latent_* but no goal_encoder was provided")
        if needs_goal_encoder and goal_encoder_input_mode != "xy":
            raise ValueError(
                f"goal encoder must have input_mode='xy' for latent_goal/latent_subgoal; "
                f"got {goal_encoder_input_mode!r}"
            )

        self.state_mode = state_mode
        self.goal_conditioning = list(goal_conditioning)
        self.state_encoder = state_encoder
        self.state_encoder_input_mode = state_encoder_input_mode
        self.goal_encoder = goal_encoder
        self.goal_encoder_input_mode = goal_encoder_input_mode

        if state_mode == "latent":
            self.state_dim = int(state_encoder.latent_dim)
        else:
            self.state_dim = _PROPRIO_DIM

        per_item_dim = {
            "goal": _GOAL_DIM,
            "subgoal": _GOAL_DIM,
            "latent_goal": int(goal_encoder.latent_dim) if goal_encoder is not None else 0,
            "latent_subgoal": int(goal_encoder.latent_dim) if goal_encoder is not None else 0,
        }
        self.goal_dim = sum(per_item_dim[c] for c in self.goal_conditioning)
        self.obs_dim = self.state_dim + self.goal_dim

    def _state_input(self, obs_107: torch.Tensor, achieved_xy: torch.Tensor) -> torch.Tensor:
        mode = self.state_encoder_input_mode
        if mode == "xy":
            return achieved_xy
        if mode == "proprio":
            return _slice_proprio(obs_107)
        if mode == "full":
            return torch.cat([_slice_proprio(obs_107), achieved_xy], dim=-1)
        raise ValueError(f"unknown state encoder input_mode {mode!r}")

    @torch.no_grad()
    def assemble(self, obs_107: torch.Tensor, achieved_xy: torch.Tensor,
                 subgoal_xy: torch.Tensor | None) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.state_mode == "latent":
            parts.append(self.state_encoder(self._state_input(obs_107, achieved_xy)))
        else:
            parts.append(_slice_proprio_contact(obs_107))

        for c in self.goal_conditioning:
            if c == "goal":
                parts.append(_slice_desired_goal(obs_107))
            elif c == "subgoal":
                parts.append(subgoal_xy)
            elif c == "latent_goal":
                parts.append(self.goal_encoder(_slice_desired_goal(obs_107)))
            elif c == "latent_subgoal":
                parts.append(self.goal_encoder(subgoal_xy))
        return torch.cat(parts, dim=-1)
