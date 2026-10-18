"""Load config.yaml once and share it."""
from __future__ import annotations

import os
from functools import lru_cache

ROOT = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=1)
def load_config(path: str = "") -> dict:
    import yaml

    path = path or os.environ.get("CRIME_RECON_CONFIG", os.path.join(ROOT, "config.yaml"))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("scenes_dir", "outputs_dir"):
        p = cfg["paths"][key]
        if not os.path.isabs(p):
            cfg["paths"][key] = os.path.join(ROOT, p)
    return cfg
