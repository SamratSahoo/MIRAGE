from __future__ import annotations

import torch
import torch.nn as nn


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
                 unified_encoder=None, graph_planning: bool = False,
                 planning_latent_dim: int | None = None):
        super().__init__()
        if state_mode not in ("raw", "latent"):
            raise ValueError(f"unknown state_mode {state_mode!r}")
        for item in goal_conditioning:
            if item not in _VALID_GOAL_ITEMS:
                raise ValueError(f"unknown goal_conditioning item {item!r}")
        needs_latent = state_mode == "latent" or any(
            c in ("latent_goal", "latent_subgoal") for c in goal_conditioning)
        self.graph_planning = bool(graph_planning)
        self.unified_encoder = unified_encoder

        if self.graph_planning:
            if needs_latent and planning_latent_dim is None:
                raise ValueError("graph_planning requires planning_latent_dim")
            latent_dim = int(planning_latent_dim) if planning_latent_dim is not None else 0
        else:
            if needs_latent and unified_encoder is None:
                raise ValueError("latent state/goal conditioning requires a unified_encoder")
            latent_dim = int(unified_encoder.latent_dim) if unified_encoder is not None else 0

        self.state_mode = state_mode
        self.goal_conditioning = list(goal_conditioning)
        self.state_dim = latent_dim if state_mode == "latent" else _PROPRIO_DIM
        per_item_dim = {
            "goal": _GOAL_DIM,
            "subgoal": _GOAL_DIM,
            "latent_goal": latent_dim,
            "latent_subgoal": latent_dim,
        }
        self.goal_dim = sum(per_item_dim[c] for c in self.goal_conditioning)
        self.obs_dim = self.state_dim + self.goal_dim

    def _pad_xy(self, xy: torch.Tensor) -> torch.Tensor:
        pad = torch.zeros(xy.shape[0], _PROPRIO_27, device=xy.device, dtype=xy.dtype)
        return torch.cat([pad, xy], dim=-1)

    @torch.no_grad()
    def assemble(self, obs_107: torch.Tensor, achieved_xy: torch.Tensor,
                 subgoal_xy: torch.Tensor | None = None, *,
                 z_state: torch.Tensor | None = None,
                 z_goal: torch.Tensor | None = None,
                 z_subgoal: torch.Tensor | None = None) -> torch.Tensor:
        if self.graph_planning:
            parts: list[torch.Tensor] = []
            if self.state_mode == "latent":
                parts.append(z_state)
            else:
                parts.append(_slice_proprio_contact(obs_107))
            for c in self.goal_conditioning:
                if c == "goal":
                    parts.append(_slice_desired_goal(obs_107))
                elif c == "latent_goal":
                    parts.append(z_goal)
                elif c == "latent_subgoal":
                    parts.append(z_subgoal)
                else:
                    raise ValueError(
                        f"goal_conditioning item {c!r} not supported in graph_planning mode")
            return torch.cat(parts, dim=-1)

        parts: list[torch.Tensor] = []
        if self.state_mode == "latent":
            parts.append(self.unified_encoder.encode_full(
                torch.cat([_slice_proprio(obs_107), achieved_xy], dim=-1)))
        else:
            parts.append(_slice_proprio_contact(obs_107))
        for c in self.goal_conditioning:
            if c == "goal":
                parts.append(_slice_desired_goal(obs_107))
            elif c == "subgoal":
                parts.append(subgoal_xy)
            elif c == "latent_goal":
                parts.append(self.unified_encoder.encode_goal(self._pad_xy(_slice_desired_goal(obs_107))))
            elif c == "latent_subgoal":
                parts.append(self.unified_encoder.encode_goal(self._pad_xy(subgoal_xy)))
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def assemble_from_latents(self, z_state: torch.Tensor, z_goal: torch.Tensor,
                              z_subgoal: torch.Tensor,
                              goal_xy: torch.Tensor | None = None) -> torch.Tensor:
        if self.state_mode != "latent":
            raise ValueError("assemble_from_latents requires state_mode='latent'")
        parts: list[torch.Tensor] = [z_state]
        for c in self.goal_conditioning:
            if c == "goal":
                if goal_xy is None:
                    raise ValueError("assemble_from_latents needs goal_xy for raw 'goal' conditioning")
                parts.append(goal_xy)
            elif c == "latent_goal":
                parts.append(z_goal)
            elif c == "latent_subgoal":
                parts.append(z_subgoal)
            else:
                raise ValueError(
                    f"goal_conditioning item {c!r} not supported in graph_planning mode")
        return torch.cat(parts, dim=-1)
