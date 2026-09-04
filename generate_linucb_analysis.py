"""Generate comprehensive analysis and figures for LinUCB vs baselines.

Reads results/metrics_all_v2.csv (simulation) and produces:
  1. Bar chart: speedup comparison across workloads at B=1
  2. Waste efficiency chart: wasted tokens per accepted
  3. Batch sweep: speedup vs batch size for LinUCB vs baselines
  4. Radar chart: multi-metric comparison
  5. Convergence analysis: arm selection over time
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

COLORS = {
    "fixed_4": "#888888",
    "ucb": "#2196F3",
    "ucb_coarse": "#03A9F4",
    "epsilon_greedy": "#FF9800",
    "entropy_threshold": "#9C27B0",
    "history": "#795548",
    "oracle": "#4CAF50",
    "linucb": "#E91E63",
    "linucb_explore": "#F44336",
    "linucb_fine": "#D32F2F",
}

LABELS = {
    "fixed_4": "Fixed K=4",
    "ucb": "UCB (8 arms)",
    "ucb_coarse": "UCB (3 arms)",
    "epsilon_greedy": "e-Greedy",
    "entropy_threshold": "SVIP-style",
    "history": "History",
    "oracle": "Oracle",
    "linucb": "LinUCB (ours)",
    "linucb_explore": "LinUCB-Explore",
    "linucb_fine": "LinUCB-Fine (ours)",
}

OUT_DIR = "results/figures_v2"


def load_data():
    df = pd.read_csv("results/metrics_all_v2.csv")
    return df


def fig1_speedup_comparison(df):
    """Bar chart: speedup at B=1 across workloads for key policies."""
    policies = ["fixed_4", "ucb", "ucb_coarse", "entropy_threshold", "linucb", "linucb_fine"]
    workloads = ["humaneval", "gsm8k", "mt_bench", "spec_bench", "mixed"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(workloads))
    width = 0.13
    offsets = np.arange(len(policies)) - (len(policies) - 1) / 2

    for i, p in enumerate(policies):
        vals = []
        for w in workloads:
            row = df[(df["policy"] == p) & (df["workload"] == w) & (df["batch_size"] == 1)]
            vals.append(row["net_speedup"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + offsets[i] * width, vals, width * 0.9,
                      label=LABELS.get(p, p), color=COLORS.get(p, "#999"),
                      edgecolor="white", linewidth=0.5)
        # Bold outline for our methods
        if "linucb" in p:
            for bar in bars:
                bar.set_edgecolor("#000")
                bar.set_linewidth(1.5)

    ax.set_xlabel("Workload")
    ax.set_ylabel("Simulated Speedup")
    ax.set_title("Speedup Comparison: LinUCB vs Baselines (B=1)")
    ax.set_xticks(x)
    ax.set_xticklabels([w.replace("_", " ").title() for w in workloads])
    ax.legend(ncol=3, fontsize=9, loc="upper right")
    ax.set_ylim(0, max(ax.get_ylim()[1], 2.8))
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig1_speedup_comparison.png", bbox_inches="tight")
    plt.close()
    print("  -> fig1_speedup_comparison.png")


def fig2_waste_efficiency(df):
    """Bar chart: wasted tokens comparison at B=1."""
    policies = ["fixed_4", "ucb", "ucb_coarse", "entropy_threshold", "linucb", "linucb_fine"]
    workloads = ["humaneval", "gsm8k", "mt_bench", "spec_bench", "mixed"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(workloads))
    width = 0.13
    offsets = np.arange(len(policies)) - (len(policies) - 1) / 2

    for i, p in enumerate(policies):
        vals = []
        for w in workloads:
            row = df[(df["policy"] == p) & (df["workload"] == w) & (df["batch_size"] == 1)]
            vals.append(row["wasted_tokens_per_accepted"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + offsets[i] * width, vals, width * 0.9,
                      label=LABELS.get(p, p), color=COLORS.get(p, "#999"),
                      edgecolor="white", linewidth=0.5)
        if "linucb" in p:
            for bar in bars:
                bar.set_edgecolor("#000")
                bar.set_linewidth(1.5)

    ax.set_xlabel("Workload")
    ax.set_ylabel("Wasted Tokens per Accepted Token")
    ax.set_title("Token Waste: LinUCB vs Baselines (B=1) — Lower is Better")
    ax.set_xticks(x)
    ax.set_xticklabels([w.replace("_", " ").title() for w in workloads])
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig2_waste_efficiency.png", bbox_inches="tight")
    plt.close()
    print("  -> fig2_waste_efficiency.png")


def fig3_batch_sweep(df):
    """Line chart: speedup vs batch size for key policies on mixed workload."""
    policies = ["fixed_4", "ucb", "ucb_coarse", "linucb", "linucb_fine", "oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for wl, ax in zip(["mixed", "humaneval"], axes):
        for p in policies:
            sub = df[(df["policy"] == p) & (df["workload"] == wl)].sort_values("batch_size")
            if len(sub) == 0:
                continue
            style = "-" if "linucb" not in p else "-"
            lw = 2.5 if "linucb" in p else 1.5
            marker = "o" if "linucb" in p else "s"
            ax.plot(sub["batch_size"], sub["net_speedup"],
                    label=LABELS.get(p, p), color=COLORS.get(p, "#999"),
                    linewidth=lw, marker=marker, markersize=6, linestyle=style)

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Simulated Speedup")
        ax.set_title(f"Batch Sweep: {wl.replace('_',' ').title()}")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 8, 32, 64])
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig3_batch_sweep.png", bbox_inches="tight")
    plt.close()
    print("  -> fig3_batch_sweep.png")


def fig4_contextual_advantage(df):
    """Heatmap: speedup delta (LinUCB_fine - UCB_coarse) across workloads x batch sizes."""
    workloads = ["humaneval", "gsm8k", "mt_bench", "spec_bench", "mixed"]
    batch_sizes = [1, 8, 32, 64]

    deltas = np.zeros((len(workloads), len(batch_sizes)))
    for i, w in enumerate(workloads):
        for j, b in enumerate(batch_sizes):
            linucb_row = df[(df["policy"] == "linucb_fine") & (df["workload"] == w) & (df["batch_size"] == b)]
            ucb_row = df[(df["policy"] == "ucb_coarse") & (df["workload"] == w) & (df["batch_size"] == b)]
            if len(linucb_row) > 0 and len(ucb_row) > 0:
                deltas[i, j] = linucb_row["net_speedup"].values[0] - ucb_row["net_speedup"].values[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(deltas, cmap="RdYlGn", aspect="auto", vmin=-0.2, vmax=0.2)

    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.set_yticks(range(len(workloads)))
    ax.set_yticklabels([w.replace("_", " ").title() for w in workloads])
    ax.set_xlabel("Batch Size")
    ax.set_title("Contextual Advantage: LinUCB-Fine minus UCB-Coarse (Speedup Delta)")

    for i in range(len(workloads)):
        for j in range(len(batch_sizes)):
            color = "black" if abs(deltas[i,j]) < 0.1 else "white"
            ax.text(j, i, f"{deltas[i,j]:+.3f}", ha="center", va="center", fontsize=10, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Speedup Delta", shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig4_contextual_advantage.png", bbox_inches="tight")
    plt.close()
    print("  -> fig4_contextual_advantage.png")


def fig5_convergence():
    """Line chart: LinUCB arm selection convergence over 500 steps."""
    from src.bench.harness import run_experiment_stream
    from controllers import LinUCBController, UCB

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for wl, ax in zip(["humaneval", "mt_bench"], axes):
        # LinUCB
        ctrl_fn = lambda: LinUCBController(arms=(1, 4, 8), alpha=1.0)
        records = run_experiment_stream(wl, "linucb", ctrl_fn, num_steps=500, batch_size=1, seed=42)
        k_linucb = [r["batch_length"] for r in records]

        # UCB coarse
        ctrl_fn2 = lambda: UCB(c=2.0, arms=(1, 4, 8))
        records2 = run_experiment_stream(wl, "ucb_coarse", ctrl_fn2, num_steps=500, batch_size=1, seed=42)
        k_ucb = [r["batch_length"] for r in records2]

        # Smoothed (rolling mean of K)
        window = 20
        k_linucb_smooth = pd.Series(k_linucb).rolling(window).mean()
        k_ucb_smooth = pd.Series(k_ucb).rolling(window).mean()

        ax.plot(k_linucb_smooth, color=COLORS["linucb"], label="LinUCB", linewidth=2)
        ax.plot(k_ucb_smooth, color=COLORS["ucb_coarse"], label="UCB-Coarse", linewidth=2, linestyle="--")
        ax.set_xlabel("Step")
        ax.set_ylabel("Average K (rolling window=20)")
        ax.set_title(f"Convergence: {wl.replace('_',' ').title()}")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 9)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig5_convergence.png", bbox_inches="tight")
    plt.close()
    print("  -> fig5_convergence.png")


def generate_summary_table(df):
    """Print a formatted results table for the paper."""
    policies = ["fixed_4", "ucb", "ucb_coarse", "entropy_threshold",
                "epsilon_greedy", "history", "linucb", "linucb_fine", "oracle"]
    workloads = ["humaneval", "gsm8k", "mt_bench", "spec_bench", "mixed"]

    print("\n" + "=" * 90)
    print("  RESULTS TABLE (Simulation, B=1)")
    print("=" * 90)
    header = f"{'Policy':20s}"
    for w in workloads:
        header += f" | {w:>10s}"
    header += " |    Avg"
    print(header)
    print("-" * 90)

    for p in policies:
        row = f"{LABELS.get(p, p):20s}"
        vals = []
        for w in workloads:
            sub = df[(df["policy"] == p) & (df["workload"] == w) & (df["batch_size"] == 1)]
            if len(sub) > 0:
                v = sub["net_speedup"].values[0]
                vals.append(v)
                row += f" | {v:>10.3f}"
            else:
                row += f" | {'N/A':>10s}"
        if vals:
            row += f" | {np.mean(vals):>6.3f}"
        print(row)

    print("-" * 90)

    # Also print waste table
    print("\n  WASTE TABLE (wasted tokens / accepted, B=1) — Lower is better")
    print("-" * 90)
    header = f"{'Policy':20s}"
    for w in workloads:
        header += f" | {w:>10s}"
    header += " |    Avg"
    print(header)
    print("-" * 90)

    for p in policies:
        row = f"{LABELS.get(p, p):20s}"
        vals = []
        for w in workloads:
            sub = df[(df["policy"] == p) & (df["workload"] == w) & (df["batch_size"] == 1)]
            if len(sub) > 0:
                v = sub["wasted_tokens_per_accepted"].values[0]
                vals.append(v)
                row += f" | {v:>10.3f}"
            else:
                row += f" | {'N/A':>10s}"
        if vals:
            row += f" | {np.mean(vals):>6.3f}"
        print(row)

    print("-" * 90)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()

    print("Generating figures...")
    fig1_speedup_comparison(df)
    fig2_waste_efficiency(df)
    fig3_batch_sweep(df)
    fig4_contextual_advantage(df)
    fig5_convergence()

    generate_summary_table(df)

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
