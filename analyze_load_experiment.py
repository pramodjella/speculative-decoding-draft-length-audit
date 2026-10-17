"""Analyze the load+content thesis experiment.

Reads results/load_dynamic_metrics.csv (headline) and results/load_metrics.csv
(motivation), prints tables with mean +/- 95% CI, tests whether the combined
controller significantly beats the published single-sided baselines (Nightjar,
TapOut), reports oracle-headroom capture, and writes results/load_findings.md.
"""
import os
import numpy as np
import pandas as pd

DYN = "results/load_dynamic_metrics.csv"
SWEEP = "results/load_metrics.csv"
OUT = "results/load_findings.md"
FIXED = ["fixed_1", "fixed_2", "fixed_4", "fixed_8"]


def ci95(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan"), 0.0)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def _md(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def paired_gain(df, w, metric, a, b):
    """Per-seed paired % gain of policy a over policy b on workload w (same streams)."""
    sub = df[df.workload == w]
    pa = sub[sub.policy == a].set_index("seed")[metric]
    pb = sub[sub.policy == b].set_index("seed")[metric]
    seeds = pa.index.intersection(pb.index)
    g = (pa.loc[seeds].values / pb.loc[seeds].values - 1.0) * 100.0
    m, h = ci95(g)
    sig = "yes" if (m - h) > 0 else "no"
    return m, h, sig


def analyze_dynamic():
    df = pd.read_csv(DYN)
    lines = ["# Load+Content Thesis — Dynamic-Load Results\n",
             "Headline metric = per-step wall-clock speedup (and accepted tokens/step), "
             "under a time-varying batch/load schedule with autocorrelated content difficulty. "
             f"Seeds={df.seed.nunique()}.\n"]
    metric = "mean_speedup"

    # Per-workload mean +/- CI table
    rows = []
    policies = ["best_fixed", "tapout", "nightjar", "content_only", "load_only",
                "combined", "oracle"]
    for w in sorted(df.workload.unique()):
        sub = df[df.workload == w]
        bf = sub[sub.policy.isin(FIXED)].groupby("seed")[metric].max()  # best fixed per seed
        rec = {"workload": w}
        for p in policies:
            if p == "best_fixed":
                m, h = ci95(bf.values)
            else:
                m, h = ci95(sub[sub.policy == p][metric].values)
            rec[p] = f"{m:.3f}±{h:.3f}"
        rows.append(rec)
    tbl = pd.DataFrame(rows)
    lines.append("## Speedup by policy (mean ± 95% CI)\n")
    lines.append(_md(tbl))

    # Headline significance: combined vs published baselines + vs best ablation
    lines.append("\n\n## Does combined beat the published single-sided baselines? (paired % gain)\n")
    grows = []
    for w in sorted(df.workload.unique()):
        r = {"workload": w}
        for label, base in [("vs TapOut(content-only)", "tapout"),
                            ("vs Nightjar(load-only)", "nightjar"),
                            ("vs content_only(ablation)", "content_only"),
                            ("vs load_only(ablation)", "load_only")]:
            m, h, sig = paired_gain(df, w, metric, "combined", base)
            r[label] = f"{m:+.1f}%±{h:.1f} ({sig})"
        grows.append(r)
    lines.append(_md(pd.DataFrame(grows)))

    # Oracle-headroom capture over best single-sided ablation
    lines.append("\n\n## Oracle-headroom capture (combined vs best single-sided, "
                 "toward clairvoyant oracle)\n")
    orows = []
    for w in sorted(df.workload.unique()):
        sub = df[df.workload == w]
        cm = sub[sub.policy == "combined"][metric].mean()
        orc = sub[sub.policy == "oracle"][metric].mean()
        base = max(sub[sub.policy == "content_only"][metric].mean(),
                   sub[sub.policy == "load_only"][metric].mean())
        cap = (cm - base) / (orc - base) * 100 if orc > base else float("nan")
        orows.append({"workload": w, "best_single_sided": round(base, 3),
                      "combined": round(cm, 3), "oracle(clairvoyant)": round(orc, 3),
                      "headroom_captured_%": round(cap, 0)})
    lines.append(_md(pd.DataFrame(orows)))

    return df, "\n".join(lines)


def analyze_sweep():
    if not os.path.exists(SWEEP):
        return ""
    df = pd.read_csv(SWEEP)
    lines = ["\n\n# Motivation — fixed-B sweep: the best fixed K shifts with batch\n"]
    for w in sorted(df.workload.unique()):
        sub = df[(df.workload == w) & (df.policy.isin(FIXED))]
        piv = sub.groupby(["batch", "policy"]).mean_speedup.mean().unstack()
        bestK = piv.idxmax(axis=1)
        lines.append(f"\n**{w}** — best fixed K by batch: " +
                     ", ".join(f"B={b}:{bestK[b].replace('fixed_','K=')}" for b in piv.index))
    return "\n".join(lines)


def make_figures(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("results/figures_load", exist_ok=True)
    order = ["tapout", "nightjar", "content_only", "load_only", "combined", "oracle"]
    workloads = sorted(df.workload.unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.13
    x = np.arange(len(workloads))
    for i, p in enumerate(order):
        means = [df[(df.workload == w) & (df.policy == p)].mean_speedup.mean() for w in workloads]
        errs = [ci95(df[(df.workload == w) & (df.policy == p)].mean_speedup.values)[1] for w in workloads]
        ax.bar(x + i * width, means, width, yerr=errs, capsize=2, label=p)
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    ax.set_xticks(x + 2.5 * width); ax.set_xticklabels(workloads)
    ax.set_ylabel("Speedup (dynamic load, mean ± 95% CI)")
    ax.set_title("Combined load+content controller vs single-sided baselines")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig("results/figures_load/dynamic_speedup.png", dpi=200)
    plt.close(fig)
    print("Figure -> results/figures_load/dynamic_speedup.png")


def main():
    if not os.path.exists(DYN):
        raise SystemExit(f"No {DYN}; run run_load_experiment.py first.")
    df, dyn_md = analyze_dynamic()
    sweep_md = analyze_sweep()
    make_figures(df)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(dyn_md + "\n" + sweep_md + "\n")
    print(dyn_md)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
