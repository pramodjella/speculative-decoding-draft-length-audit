"""Experiment driver: iterate the experiment matrix of controllers and workloads.

Runs the simulated benchmark sweeps and saves the resulting metrics.
"""
from __future__ import annotations

import argparse
import csv
import os
import numpy as np

from src.utils import get_logger, set_seed, timer
from src.bench.harness import run_experiment_stream
from src.controllers import EntropyThreshold, EpsilonGreedy, UCB, AcceptanceHistoryController, OracleController, LinUCBController

log = get_logger("run")

def get_controller_fn(policy_name: str):
    if policy_name.startswith("fixed_"):
        k = int(policy_name.split("_")[1])
        return lambda: k
    elif policy_name == "entropy_threshold":
        return lambda: EntropyThreshold(tau=1.0, max_len=8)
    elif policy_name == "epsilon_greedy":
        return lambda: EpsilonGreedy(eps=0.1, seed=42)
    elif policy_name == "ucb":
        return lambda: UCB(c=0.5)
    elif policy_name == "history":
        return lambda: AcceptanceHistoryController()
    elif policy_name == "oracle":
        return lambda: OracleController()
    elif policy_name == "linucb":
        return lambda: LinUCBController(arms=(1, 4, 8), alpha=1.0)
    elif policy_name == "linucb_explore":
        return lambda: LinUCBController(arms=(1, 4, 8), alpha=2.0)
    elif policy_name == "linucb_fine":
        return lambda: LinUCBController(arms=(1, 2, 4, 8), alpha=1.0)
    elif policy_name == "ucb_coarse":
        return lambda: UCB(c=2.0, arms=(1, 4, 8))
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

def run_matrix() -> list[dict]:
    """Run all policies across all workloads and batch sizes."""
    workloads = ["humaneval", "gsm8k", "mt_bench", "spec_bench", "mixed"]
    policies = (
        [f"fixed_{k}" for k in [1, 2, 3, 4, 6, 8]] +
        ["entropy_threshold", "epsilon_greedy", "ucb", "history", "oracle",
         "ucb_coarse", "linucb", "linucb_explore", "linucb_fine"]
    )
    batch_sizes = [1, 8, 32, 64]
    
    rows = []
    total_runs = len(workloads) * len(policies) * len(batch_sizes)
    current_run = 0
    
    for w in workloads:
        for p in policies:
            for b in batch_sizes:
                current_run += 1
                if current_run % 20 == 0:
                    log.info(f"Progress: run {current_run}/{total_runs} (workload={w}, policy={p}, batch={b})")
                    
                # Run the experiment stream
                records = run_experiment_stream(
                    workload=w,
                    policy_name=p,
                    controller_fn=get_controller_fn(p),
                    num_steps=500, # 500 steps per run is highly stable and fast
                    batch_size=b,
                    seed=42
                )
                
                # Aggregate metrics over all steps in the run
                net_speedups = [r["net_speedup"] for r in records]
                mean_lengths = [r["mean_accepted_length"] for r in records]
                wasted_tokens = [r["wasted_tokens_per_accepted"] for r in records]
                
                rows.append({
                    "workload": w,
                    "policy": p,
                    "batch_size": b,
                    "mean_accepted_length": round(np.mean(mean_lengths), 3),
                    "net_speedup": round(np.mean(net_speedups), 3),
                    "wasted_tokens_per_accepted": round(np.mean(wasted_tokens), 3)
                })
                
    return rows

def _write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = ["workload", "policy", "batch_size", "mean_accepted_length", "net_speedup", "wasted_tokens_per_accepted"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true", help="offline sanity run")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    set_seed(args.seed)
    
    output_path = "results/metrics_all.csv"
    
    log.info("Running simulated speculative decoding experiments matrix...")
    with timer("experiments_sweep"):
        rows = run_matrix()
    _write_csv(rows, output_path)
    log.info(f"Successfully completed all experiments. Stored results in {output_path}.")

if __name__ == "__main__":
    main()
