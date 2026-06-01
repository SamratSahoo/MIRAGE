from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path, root: Path = PROJECT_ROOT):
    if path is None:
        return None
    p = Path(os.path.expanduser(str(path)))
    if not p.is_absolute():
        p = root / p
    return str(p)


def project_path(*parts) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))
