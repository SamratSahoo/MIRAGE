import argparse
import os
import sys

import yaml

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mirage.world_model.trainer import WorldModelTrainer


def _coerce(default, val: str):
    if isinstance(default, bool):
        return val.lower() in ("1", "true", "yes")
    if isinstance(default, int):
        return int(val)
    if isinstance(default, float):
        return float(val)
    if isinstance(default, list):
        import json as _json
        return _json.loads(val)
    return val


def parse_args() -> tuple[dict, str]:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=os.path.join(_PROJECT_ROOT, "config", "world_model", "dynamics.yaml"))
    args, extra = p.parse_known_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    i = 0
    while i < len(extra):
        key = extra[i].lstrip("-")
        val = extra[i + 1]
        if key in cfg:
            cfg[key] = _coerce(cfg[key], val)
        else:
            cfg[key] = val
        i += 2
    return cfg, args.config


def main():
    cfg, config_path = parse_args()
    cfg["run_name"] = os.path.splitext(os.path.basename(config_path))[0]
    WorldModelTrainer(cfg).train()


if __name__ == "__main__":
    main()
