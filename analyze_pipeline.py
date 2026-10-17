"""Analyze M5 pipeline output: mean +/- 95% CI tables and figures.

Reads results/m5_metrics.csv and produces:
  results/m5_summary.csv            per (workload, policy): mean speedup + 95% CI
  results/m5_beat_best_fixed.csv    best-adaptive vs best-fixed per workload (with CI)
  results/figures_m5/*.png          speedup-with-CI bars, waste, contextual advantage
  results/m5_insight.md             refreshed numbers for the manuscript
"""
import os
import numpy as np
import pandas as pd

METRICS = "results/m5_metrics.csv"
OUTDIR = "results/figures_m5"

FIXED = ["fixed_1", "fixed_2", "fixed_4", "fixed_8"]
ADAPTIVE = ["entropy", "epsilon_greedy", "ucb", "history", "ucb_coarse",
            "linucb", "linucb_explore", "linucb_fine"]
BASELINES = ["baseline_honest", "baseline_hf"]


def _md(df):
    """Markdown table if `tabulate` is available, else plain text."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan"), 0.0)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(m), float(1.96 * se)


def per_prompt_mean_over_seeds(df, value="net_speedup"):
    """Average each prompt across seeds first, then we have one value per prompt."""
    return df.groupby(["workload", "policy", "prompt_idx"])[value].mean().reset_index()


def build_summary(df):
    spec = df[~df.policy.isin(BASELINES)].copy()
    spec["net_speedup"] = pd.to_numeric(spec["net_speedup"], errors="coerce")
    spec["wasted_tokens_per_accepted"] = pd.to_numeric(spec["wasted_tokens_per_accepted"], errors="coerce")
    # Headline metric: accepted tokens per target step (deterministic under greedy
    # decoding, so stable across seeds). Falls back gracefully on older CSVs.
    has_aps = "accepted_tokens_per_step" in spec.columns
    if has_aps:
        spec["accepted_tokens_per_step"] = pd.to_numeric(spec["accepted_tokens_per_step"], errors="coerce")

    rows = []
    for (w, p), g in spec.groupby(["workload", "policy"]):
        m, h = ci95(g["net_speedup"])
        wm, _ = ci95(g["wasted_tokens_per_accepted"])
        row = {
            "workload": w, "policy": p,
            "acc_per_step_mean": float("nan"), "acc_per_step_ci95": float("nan"),
            "speedup_mean": round(m, 4), "speedup_ci95": round(h, 4),
            "wasted_mean": round(wm, 4),
            "n_obs": len(g),
        }
        if has_aps:
            am, ah = ci95(g["accepted_tokens_per_step"])
            row["acc_per_step_mean"] = round(am, 4)
            row["acc_per_step_ci95"] = round(ah, 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["workload", "acc_per_step_mean"], ascending=[True, False])


def beat_best_fixed(summary, mean_col="acc_per_step_mean", ci_col="acc_per_step_ci95"):
    """Best adaptive vs best fixed-K per workload, on the given metric.

    Defaults to the headline accepted-tokens-per-step metric; pass speedup columns
    for the secondary wall-clock view.
    """
    if mean_col not in summary.columns or summary[mean_col].isna().all():
        return pd.DataFrame()
    rows = []
    for w, g in summary.groupby("workload"):
        gf = g[g.policy.isin(FIXED)]
        ga = g[g.policy.isin(ADAPTIVE)]
        if gf.empty or ga.empty:
            continue
        bf = gf.loc[gf[mean_col].idxmax()]
        ba = ga.loc[ga[mean_col].idxmax()]
        if bf[mean_col] == 0 or pd.isna(bf[mean_col]):
            continue
        gap = (ba[mean_col] - bf[mean_col]) / bf[mean_col] * 100
        # CIs overlap? -> not significant
        overlap = (ba[mean_col] - ba[ci_col]) <= (bf[mean_col] + bf[ci_col])
        rows.append({
            "workload": w,
            "best_fixed": bf.policy, "best_fixed_metric": bf[mean_col], "bf_ci95": bf[ci_col],
            "best_adaptive": ba.policy, "best_adaptive_metric": ba[mean_col], "ba_ci95": ba[ci_col],
            "gap_pct": round(gap, 2),
            "significant": "no (CIs overlap)" if overlap else "yes",
        })
    return pd.DataFrame(rows)


def make_figures(summary, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUTDIR, exist_ok=True)
    workloads = sorted(summary.workload.unique())

    # Fig 1: speedup bars with 95% CI, per workload
    for w in workloads:
        g = summary[summary.workload == w].sort_values("speedup_mean")
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["#4C72B0" if p in FIXED else "#C44E52" if p.startswith("linucb")
                  else "#55A868" for p in g.policy]
        ax.barh(g.policy, g.speedup_mean, xerr=g.speedup_ci95, color=colors, capsize=3)
        ax.axvline(1.0, ls="--", c="gray", lw=1, label="parity (no speedup)")
        ax.set_xlabel("Net speedup vs honest baseline (mean ± 95% CI)")
        ax.set_title(f"Net speedup by policy — {w}")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(f"{OUTDIR}/speedup_{w}.png", dpi=200)
        plt.close(fig)

    # Fig 2: contextual advantage — linucb vs ucb across workloads
    fig, ax = plt.subplots(figsize=(9, 5))
    piv = summary.pivot_table(index="workload", columns="policy", values="speedup_mean")
    for p in ["ucb", "ucb_coarse", "linucb", "linucb_fine", "history"]:
        if p in piv.columns:
            ax.plot(piv.index, piv[p], marker="o", label=p)
    ax.set_ylabel("Net speedup")
    ax.set_title("Contextual (LinUCB) vs context-free controllers across workloads")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/contextual_advantage.png", dpi=200)
    plt.close(fig)

    # Fig 3: speedup vs waste scatter (efficiency frontier)
    fig, ax = plt.subplots(figsize=(8, 6))
    agg = summary.groupby("policy").agg(s=("speedup_mean", "mean"), w=("wasted_mean", "mean")).reset_index()
    for _, r in agg.iterrows():
        c = "#4C72B0" if r.policy in FIXED else "#C44E52" if r.policy.startswith("linucb") else "#55A868"
        ax.scatter(r.w, r.s, c=c, s=80)
        ax.annotate(r.policy, (r.w, r.s), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Wasted tokens per accepted (lower better)")
    ax.set_ylabel("Net speedup (higher better)")
    ax.set_title("Speedup vs waste — efficiency frontier")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/efficiency_frontier.png", dpi=200)
    plt.close(fig)
    print(f"Figures -> {OUTDIR}/")


def write_insight(summary, beat_aps, beat_speedup, df):
    lines = ["# M5 Refreshed Results (honest baseline, 3 seeds, 95% CI)\n"]
    lines.append("_Headline metric: **accepted tokens per target step** (deterministic "
                 "under greedy decoding, hardware-independent — the quantity a draft-length "
                 "controller directly optimizes). Wall-clock speedup is reported as a "
                 "secondary, regime-dependent view._\n")
    hf = df[df.policy == "baseline_hf"]
    if not hf.empty:
        lines.append(f"_HF .generate() reference latency recorded for {hf.workload.nunique()} workloads "
                     "(appendix only — net speedup is measured vs the same-backend Python baseline)._\n")
    lines.append("## Beat-the-best-fixed — accepted tokens / target step (headline)\n")
    lines.append(_md(beat_aps) if not beat_aps.empty else "_(metric not present in CSV)_")
    lines.append("\n\n## Beat-the-best-fixed — wall-clock speedup (secondary)\n")
    lines.append(_md(beat_speedup) if not beat_speedup.empty else "_(no data)_")
    lines.append("\n\n## Full summary (mean ± 95% CI)\n")
    lines.append(_md(summary))
    rank_col = "acc_per_step_mean" if summary["acc_per_step_mean"].notna().any() else "speedup_mean"
    overall = (summary[summary.policy.isin(ADAPTIVE + FIXED)]
               .groupby("policy")[rank_col].mean().sort_values(ascending=False)
               .round(4).reset_index())
    lines.append(f"\n\n## Overall controller ranking (mean {rank_col} across workloads)\n")
    lines.append(_md(overall))
    with open("results/m5_insight.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Insight -> results/m5_insight.md")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=METRICS,
                    help="Path to the metrics CSV (use the tagged file for a "
                         "specific model pair, e.g. results/m5_metrics_qwen3b.csv)")
    args = ap.parse_args()
    if not os.path.exists(args.metrics):
        raise SystemExit(f"No metrics at {args.metrics}; run run_full_pipeline.py first.")
    df = pd.read_csv(args.metrics)
    summary = build_summary(df)
    summary.to_csv("results/m5_summary.csv", index=False)
    beat_aps = beat_best_fixed(summary, "acc_per_step_mean", "acc_per_step_ci95")
    beat_speedup = beat_best_fixed(summary, "speedup_mean", "speedup_ci95")
    (beat_aps if not beat_aps.empty else beat_speedup).to_csv(
        "results/m5_beat_best_fixed.csv", index=False)

    print("\n=== Beat-the-best-fixed (accepted tokens / target step — headline) ===")
    print(beat_aps.to_string(index=False) if not beat_aps.empty else "(metric not in CSV)")
    print("\n=== Beat-the-best-fixed (wall-clock speedup — secondary) ===")
    print(beat_speedup.to_string(index=False) if not beat_speedup.empty else "(no data)")

    make_figures(summary, df)
    write_insight(summary, beat_aps, beat_speedup, df)
    print("\nDone. Summary -> results/m5_summary.csv")


if __name__ == "__main__":
    main()
