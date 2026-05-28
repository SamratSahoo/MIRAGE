from __future__ import annotations

import os


class WandbSession:
    def __init__(self, run_dir: str, project: str, entity: str | None,
                 run_name: str, config: dict, enabled: bool = True,
                 sync_tensorboard: bool = True):
        self.run_dir = run_dir
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.config = config
        self.enabled = bool(enabled)
        self.sync_tensorboard = bool(sync_tensorboard)
        self.id_path = os.path.join(run_dir, "wandb_run_id.txt")
        self.run = None
        self.resumed = False

    def _read_id(self) -> str | None:
        if not os.path.exists(self.id_path):
            return None
        with open(self.id_path) as f:
            rid = f.read().strip()
        return rid or None

    def _write_id(self, rid: str) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        with open(self.id_path, "w") as f:
            f.write(rid)

    def init(self):
        if not self.enabled:
            return None
        import wandb

        existing = self._read_id()
        kwargs = dict(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            config=self.config,
            sync_tensorboard=self.sync_tensorboard,
            save_code=True,
        )
        if existing is not None:
            kwargs["id"] = existing
            kwargs["resume"] = "allow"
        self.run = wandb.init(**kwargs)
        self.resumed = existing is not None
        if not self.resumed:
            self._write_id(self.run.id)
        return self.run

    def log(self, data: dict, step: int | None = None) -> None:
        if self.run is None:
            return
        self.run.log(data, step=step)

    def finish(self) -> None:
        if self.run is None:
            return
        import wandb
        wandb.finish()
        self.run = None
