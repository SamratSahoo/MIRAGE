from __future__ import annotations

import importlib.util as _ilu
import os as _os
import sys as _sys

import numpy as np


def _load_sibling(alias, filename):
    if alias in _sys.modules:
        return _sys.modules[alias]
    spec = _ilu.spec_from_file_location(
        alias, _os.path.join(_os.path.dirname(__file__), filename))
    mod = _ilu.module_from_spec(spec)
    _sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


AC = _load_sibling("_aug_common", "augment_common.py")


def evaluate_method(damaged_set: set, added_set: set, removed_set: set,
                    full_set: set, nodes, starts, goals, base_cov, damaged_cov):
    aug_set = damaged_set | added_set
    cov = AC.pair_connected(aug_set, starts, goals).mean()
    n_added = len(added_set)
    n_in_full = len(added_set & full_set)
    precision = (n_in_full / n_added) if n_added else float("nan")
    n_re = len(added_set & removed_set)
    recall = (n_re / len(removed_set)) if removed_set else float("nan")
    return {
        "n_added": n_added,
        "coverage": float(cov),
        "cov_recovery": float(cov - damaged_cov),
        "cov_vs_base": float(cov - base_cov),
        "edge_precision": float(precision),
        "edges_in_full": n_in_full,
        "edge_recall": float(recall),
        "removed_readded": n_re,
    }
