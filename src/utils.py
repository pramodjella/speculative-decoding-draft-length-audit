"""Shared utilities: config loading, seeding, logging, timing.

These helpers are intentionally dependency-light so the repo runs on day one
(``python -m src.run_experiments --smoke``) without a GPU or model download.
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import time


def set_seed(seed: int = 0) -> None:
    """Seed python / numpy / torch (whichever are installed)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_config(path: str) -> dict:
    """Load a YAML config (falls back to JSON if PyYAML is missing)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml
        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def get_logger(name: str = "exp"):
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger(name)


@contextlib.contextmanager
def timer(name: str = "block"):
    t0 = time.perf_counter()
    yield
    print(f"[timer] {name}: {time.perf_counter() - t0:.3f}s")


def save_jsonl(rows, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
