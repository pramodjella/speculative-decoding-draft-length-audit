"""Run the full simulation matrix with LinUCB policies included.

Saves to results/metrics_all_v2.csv with all policies including LinUCB variants.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import csv
import numpy as np
from src.bench.harness import run_experiment_stream
from controllers import (
    EntropyThreshold, EpsilonGreedy, UCB,
    AcceptanceHistoryController, OracleController, LinUCBController
)

def get_controller_fn(policy_name):
    if policy_name.startswith("fixed_"):
        k = int(policy_name.split("_")[1])
        return lambda: k
    return {
        "entropy_threshold": lambda: EntropyThreshold(tau=1.0, max_len=8),
        "epsilon_greedy": lambda: EpsilonGreedy(eps=0.1, seed=42),
        "ucb": lambda: UCB(c=0.5),
        "ucb_coarse": lambda: UCB(c=2.0, arms=(1, 4, 8)),
        "history": lambda: AcceptanceHistoryController(),
        "oracle": lambda: OracleController(),
        "linucb": lambda: LinUCBController(arms=(1, 4, 8), alpha=1.0),
        "linucb_explore": lambda: LinUCBController(arms=(1, 4, 8), alpha=2.0),
        "linucb_fine": lambda: LinUCBController(arms=(1, 2, 4, 8), alpha=1.0),
    }[policy_name]

def main():
    workloads = ["humaneval", "gsm8k", "mt_bench", "spec_bench", "mixed"]
    policies = (
        [f"fixed_{k}" for k in [1, 2, 3, 4, 6, 8]] +
        ["entropy_threshold", "epsilon_greedy", "ucb", "history", "oracle",
         "ucb_coarse", "linucb", "linucb_explore", "linucb_fine"]
    )
    batch_sizes = [1, 8, 32, 64]

    output_path = "results/metrics_all_v2.csv"
    os.makedirs("results", exist_ok=True)

    rows = []
    total = len(workloads) * len(policies) * len(batch_sizes)
    current = 0

    for w in workloads:
        for p in policies:
            for b in batch_sizes:
                current += 1
                if current % 50 == 0:
                    print(f"  Progress: {current}/{total} ({w}, {p}, B={b})")

                records = run_experiment_stream(
                    workload=w, policy_name=p,
                    controller_fn=get_controller_fn(p),
                    num_steps=500, batch_size=b, seed=42
                )

                speedups = [r["net_speedup"] for r in records]
                lengths = [r["mean_accepted_length"] for r in records]
                wasted = [r["wasted_tokens_per_accepted"] for r in records]

                rows.append({
                    "workload": w, "policy": p, "batch_size": b,
                    "mean_accepted_length": round(np.mean(lengths), 3),
                    "net_speedup": round(np.mean(speedups), 3),
                    "wasted_tokens_per_accepted": round(np.mean(wasted), 3),
                })

    # Write CSV
    keys = ["workload", "policy", "batch_size", "mean_accepted_length", "net_speedup", "wasted_tokens_per_accepted"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! {len(rows)} rows written to {output_path}")

    # Print summary for LinUCB vs baselines at B=1
    print("\n" + "=" * 70)
    print("  LinUCB vs Baselines (B=1 summary)")
    print("=" * 70)
    for w in workloads:
        print(f"\n  {w}:")
        for p in ["fixed_4", "ucb", "ucb_coarse", "linucb", "linucb_explore", "linucb_fine"]:
            match = [r for r in rows if r["workload"] == w and r["policy"] == p and r["batch_size"] == 1]
            if match:
                r = match[0]
                print(f"    {p:20s}  speedup={r['net_speedup']:.3f}  wasted={r['wasted_tokens_per_accepted']:.3f}")


if __name__ == "__main__":
    main()
