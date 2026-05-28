from __future__ import annotations

import os
import torch


class Checkpointer:
    def __init__(self, run_dir: str, exp_name: str, suffix: str = ".cleanrl_model"):
        self.run_dir = run_dir
        self.exp_name = exp_name
        self.suffix = suffix
        os.makedirs(self.run_dir, exist_ok=True)
        self.latest_path = os.path.join(run_dir, f"{exp_name}{suffix}")
        self.best_path = os.path.join(run_dir, f"{exp_name}_best{suffix}")

    def step_path(self, global_step: int) -> str:
        return os.path.join(self.run_dir, f"{self.exp_name}_step{global_step:09d}{self.suffix}")

    def save(self, state: dict, path: str) -> None:
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)

    def save_latest(self, state: dict) -> None:
        self.save(state, self.latest_path)

    def save_best(self, state: dict) -> None:
        self.save(state, self.best_path)

    def save_step(self, state: dict, global_step: int) -> str:
        path = self.step_path(global_step)
        self.save(state, path)
        return path

    def load(self, path: str | None = None, map_location=None) -> dict | None:
        target = path or self.latest_resumable()
        if target is None or not os.path.exists(target):
            return None
        return torch.load(target, map_location=map_location, weights_only=False)

    def latest_resumable(self) -> str | None:
        for p in (self.latest_path, self.best_path):
            if os.path.exists(p):
                return p
        return None
