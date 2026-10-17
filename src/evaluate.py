"""Evaluation entry point: score model outputs against a benchmark.

Wire your benchmark loaders here (Spec-Bench, MT-Bench, LongBench, GSM8K, ...).
The ``--smoke`` path scores a tiny in-memory set so CI / day-one runs pass.
"""
from __future__ import annotations

import argparse

from .metrics import exact_match
from .utils import get_logger, save_jsonl

log = get_logger("evaluate")

SMOKE_SET = [
    {"id": 0, "prompt": "2+2=", "gold": "4"},
    {"id": 1, "prompt": "capital of France?", "gold": "Paris"},
]


def score(predictions: list[dict]) -> dict:
    """predictions: [{id, pred, gold}] -> aggregate metrics."""
    em = [exact_match(p["pred"], p["gold"]) for p in predictions]
    return {"n": len(predictions), "exact_match": sum(em) / len(em) if em else 0.0}


def run_smoke() -> dict:
    preds = [{"id": x["id"], "pred": x["gold"], "gold": x["gold"]} for x in SMOKE_SET]
    rep = score(preds)
    save_jsonl(preds, "results/smoke_predictions.jsonl")
    log.info("smoke evaluate: %s", rep)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run a tiny offline check")
    args = ap.parse_args()
    if args.smoke:
        run_smoke()
    else:
        log.info("Plug in your benchmark loader + model here (see README milestones).")
