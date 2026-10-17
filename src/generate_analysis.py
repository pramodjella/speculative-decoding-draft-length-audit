#!/usr/bin/env python
"""
generate_analysis.py
====================
Comprehensive analysis of adaptive draft-length speculative decoding results.

Outputs:
  1. results/a5_generalisation.csv        – cross-workload generalisation table
  2. results/best_fixed_comparison.csv    – gap (%) vs best fixed-K per workload
  3. results/controller_overhead.csv      – wall-clock choose()/update() micro-benchmarks
  4. Formatted summary to stdout
"""

from __future__ import annotations

import os
import sys
import timeit
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

PHYSICAL_METRICS = RESULTS / "physical_roadmap_metrics.csv"
ERROR_TAXONOMY   = RESULTS / "physical_error_taxonomy.csv"
METRICS_ALL      = RESULTS / "metrics_all.csv"

# Policy name mappings (physical CSV names → canonical names used in outputs)
ADAPTIVE_POLICIES = ["entropy", "epsilon_greedy", "ucb", "history", "ucb_coarse", "linucb", "linucb_explore", "linucb_fine"]
FIXED_POLICIES    = ["fixed_1", "fixed_2", "fixed_4", "fixed_8"]

# ===========================================================================
# 1.  Cross-workload generalisation table  (a5_generalisation.csv)
# ===========================================================================
def build_generalisation_table(phys: pd.DataFrame) -> pd.DataFrame:
    """
    For each adaptive controller compute mean net_speedup and mean
    wasted_tokens_per_accepted on EACH workload.  The controllers were run
    with FIXED hyper-params (tau=0.8, eps=0.1, c=2.0) across all workloads,
    so the table IS the generalisation test.
    """
    adaptive = phys[phys["policy"].isin(ADAPTIVE_POLICIES)].copy()
    gen = (
        adaptive
        .groupby(["policy", "workload"], as_index=False)
        .agg(
            mean_net_speedup=("net_speedup", "mean"),
            mean_wasted_tokens=("wasted_tokens_per_accepted", "mean"),
        )
    )
    # Sort nicely
    gen = gen.sort_values(["policy", "workload"]).reset_index(drop=True)
    return gen


# ===========================================================================
# 2.  Best fixed-K comparison  (best_fixed_comparison.csv)
# ===========================================================================
def build_best_fixed_comparison(phys: pd.DataFrame) -> pd.DataFrame:
    """
    For each workload, find the best fixed-K policy (highest mean
    net_speedup among fixed_1 / fixed_2 / fixed_4 / fixed_8).  Then compute
    the gap (%) of every adaptive controller versus that baseline, plus the
    wasted-tokens savings.
    """
    fixed = phys[phys["policy"].isin(FIXED_POLICIES)].copy()
    adaptive = phys[phys["policy"].isin(ADAPTIVE_POLICIES)].copy()

    # Mean speedup per (workload, policy)
    fixed_means = (
        fixed.groupby(["workload", "policy"], as_index=False)
        .agg(mean_speedup=("net_speedup", "mean"), mean_wasted=("wasted_tokens_per_accepted", "mean"))
    )
    adaptive_means = (
        adaptive.groupby(["workload", "policy"], as_index=False)
        .agg(mean_speedup=("net_speedup", "mean"), mean_wasted=("wasted_tokens_per_accepted", "mean"))
    )

    # Best fixed per workload
    idx_best = fixed_means.groupby("workload")["mean_speedup"].idxmax()
    best_fixed = fixed_means.loc[idx_best, ["workload", "policy", "mean_speedup", "mean_wasted"]].rename(
        columns={"policy": "best_fixed_policy", "mean_speedup": "best_fixed_speedup", "mean_wasted": "best_fixed_wasted"}
    )

    # Merge adaptive with best-fixed
    merged = adaptive_means.merge(best_fixed, on="workload", how="left")
    merged["gap_pct"] = (
        (merged["mean_speedup"] - merged["best_fixed_speedup"])
        / merged["best_fixed_speedup"]
        * 100
    )
    merged["wasted_saved_pct"] = (
        (merged["best_fixed_wasted"] - merged["mean_wasted"])
        / merged["best_fixed_wasted"]
        * 100
    )

    out = merged[
        ["workload", "best_fixed_policy", "best_fixed_speedup",
         "policy", "mean_speedup", "gap_pct", "wasted_saved_pct"]
    ].rename(columns={"mean_speedup": "policy_speedup"})
    out = out.sort_values(["workload", "policy"]).reset_index(drop=True)
    return out


# ===========================================================================
# 3.  Controller overhead micro-benchmark  (controller_overhead.csv)
# ===========================================================================
def benchmark_controllers(n_iters: int = 10_000) -> pd.DataFrame:
    """
    Measure wall-clock time of choose() and update() for each controller.
    Reports mean time in microseconds.
    """
    from controllers import (
        EntropyThreshold,
        EpsilonGreedy,
        UCB,
        AcceptanceHistoryController,
        LinUCBController,
    )

    # Synthetic inputs for choose() / update()
    dummy_entropies = [0.3, 0.5, 0.8, 1.2, 0.4, 0.6, 0.9, 1.1]

    controllers = {
        "EntropyThreshold": {
            "instance": EntropyThreshold(tau=0.8, max_len=8),
            "choose_args": (dummy_entropies,),
            "update_args": None,          # No update method
        },
        "EpsilonGreedy": {
            "instance": EpsilonGreedy(eps=0.1),
            "choose_args": (),
            "update_args": (4, 0.75),     # arm, reward
        },
        "UCB": {
            "instance": UCB(c=2.0),
            "choose_args": (),
            "update_args": (4, 0.75),
        },
        "AcceptanceHistoryController": {
            "instance": AcceptanceHistoryController(),
            "choose_args": (),
            "update_args": (4, 3),        # arm, accepted
        },
        "LinUCBController": {
            "instance": LinUCBController(),
            "choose_args": (0.5,),
            "update_args": (4, 0.75),
        },
    }

    rows = []
    for name, spec in controllers.items():
        ctrl = spec["instance"]

        # --- choose() ---
        c_args = spec["choose_args"]
        t_choose = timeit.timeit(lambda: ctrl.choose(*c_args), number=n_iters)
        choose_us = t_choose / n_iters * 1e6

        # --- update() ---
        if spec["update_args"] is not None and hasattr(ctrl, "update"):
            u_args = spec["update_args"]
            t_update = timeit.timeit(lambda: ctrl.update(*u_args), number=n_iters)
            update_us = t_update / n_iters * 1e6
        else:
            update_us = np.nan

        rows.append({
            "controller": name,
            "choose_us": round(choose_us, 3),
            "update_us": round(update_us, 3) if not np.isnan(update_us) else np.nan,
            "total_us": round(choose_us + (update_us if not np.isnan(update_us) else 0), 3),
            "under_100us": (choose_us + (update_us if not np.isnan(update_us) else 0)) < 100,
        })

    return pd.DataFrame(rows)


# ===========================================================================
# 4.  Pretty-print summary
# ===========================================================================
def print_summary(gen_df: pd.DataFrame, comp_df: pd.DataFrame, overhead_df: pd.DataFrame) -> None:
    sep = "=" * 80

    # --- Section 1: Generalisation ---
    print(f"\n{sep}")
    print("  TABLE A5 – Cross-Workload Generalisation (Physical Hardware)")
    print(f"{sep}")
    print("  Controllers run with FIXED hyper-params across ALL workloads")
    print("  (tau=0.8, eps=0.1, c=2.0)\n")

    pivot_speed = gen_df.pivot(index="policy", columns="workload", values="mean_net_speedup")
    pivot_waste = gen_df.pivot(index="policy", columns="workload", values="mean_wasted_tokens")

    print("  -- Mean Net Speedup --")
    print(pivot_speed.round(4).to_string(index=True))
    print()
    print("  -- Mean Wasted Tokens / Accepted --")
    print(pivot_waste.round(4).to_string(index=True))
    print()

    # Per-controller cross-workload stats
    overall = gen_df.groupby("policy").agg(
        overall_mean_speedup=("mean_net_speedup", "mean"),
        speedup_std=("mean_net_speedup", "std"),
        overall_mean_wasted=("mean_wasted_tokens", "mean"),
    ).round(4)
    print("  -- Per-Controller Overall (mean +/- std across workloads) --")
    print(overall.to_string())

    # --- Section 2: Best-Fixed comparison ---
    print(f"\n{sep}")
    print("  TABLE - Adaptive vs Best Fixed-K (Physical Hardware)")
    print(f"{sep}\n")

    for wl in sorted(comp_df["workload"].unique()):
        subset = comp_df[comp_df["workload"] == wl]
        bf_policy = subset["best_fixed_policy"].iloc[0]
        bf_speed  = subset["best_fixed_speedup"].iloc[0]
        print(f"  Workload: {wl}  |  Best Fixed: {bf_policy} (speedup={bf_speed:.4f})")
        print(f"  {'Policy':<20s} {'Speedup':>9s} {'Gap%':>8s} {'Wasted Saved%':>14s}")
        print(f"  {'-'*20} {'-'*9} {'-'*8} {'-'*14}")
        for _, r in subset.iterrows():
            gap_str = f"{r['gap_pct']:+.2f}%"
            ws_str  = f"{r['wasted_saved_pct']:+.2f}%"
            print(f"  {r['policy']:<20s} {r['policy_speedup']:9.4f} {gap_str:>8s} {ws_str:>14s}")
        print()

    # --- Section 3: Controller overhead ---
    print(f"{sep}")
    print(f"  TABLE - Controller Overhead (us per call, averaged over 10,000 iters)")
    print(f"{sep}\n")
    print(f"  {'Controller':<30s} {'choose(us)':>12s} {'update(us)':>12s} {'total(us)':>12s} {'<100us':>8s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for _, r in overhead_df.iterrows():
        update_str = f"{r['update_us']:12.3f}" if not np.isnan(r["update_us"]) else f"{'N/A':>12s}"
        ok = "  Y" if r["under_100us"] else "  N"
        print(f"  {r['controller']:<30s} {r['choose_us']:12.3f} {update_str} {r['total_us']:12.3f} {ok:>8s}")
    print()
    all_pass = overhead_df["under_100us"].all()
    print(f"  All controllers < 100 us overhead? {'YES' if all_pass else 'NO'}")
    print(f"\n{sep}\n")


# ===========================================================================
# Main
# ===========================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, default=str(PHYSICAL_METRICS))
    parser.add_argument("--taxonomy", type=str, default=str(ERROR_TAXONOMY))
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    suffix = args.suffix
    metrics_path = Path(args.metrics)
    taxonomy_path = Path(args.taxonomy)

    print(f"Loading data from {metrics_path}...")
    phys = pd.read_csv(metrics_path)

    print(f"  physical_roadmap_metrics: {len(phys):,} rows")
    print(f"  Workloads : {sorted(phys['workload'].unique())}")
    print(f"  Policies  : {sorted(phys['policy'].unique())}")

    # --- 1. Generalisation table ---
    print("\n>> Building generalisation table...")
    gen_df = build_generalisation_table(phys)
    gen_file = RESULTS / f"a5_generalisation{suffix}.csv"
    gen_df.to_csv(gen_file, index=False)
    print(f"  Saved -> {gen_file}  ({len(gen_df)} rows)")

    # --- 2. Best-fixed comparison ---
    print(">> Building best-fixed comparison...")
    comp_df = build_best_fixed_comparison(phys)
    comp_file = RESULTS / f"best_fixed_comparison{suffix}.csv"
    comp_df.to_csv(comp_file, index=False)
    print(f"  Saved -> {comp_file}  ({len(comp_df)} rows)")

    # --- 3. Controller overhead ---
    print(">> Benchmarking controller overhead...")
    overhead_df = benchmark_controllers(n_iters=10_000)
    overhead_file = RESULTS / f"controller_overhead{suffix}.csv"
    overhead_df.to_csv(overhead_file, index=False)
    print(f"  Saved -> {overhead_file}  ({len(overhead_df)} rows)")

    # --- 4. Summary ---
    print_summary(gen_df, comp_df, overhead_df)

    print("Done. All output files written to results/")


if __name__ == "__main__":
    main()
