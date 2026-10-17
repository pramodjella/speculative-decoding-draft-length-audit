"""Analyze long-context EAGLE-3 results (both vLLM-patched and SGLang).

Reads:
  results/eagle3_longctx_patched/{model}_b{batch}_{tag}.json
  results/eagle3_longctx_sglang/{model}_b{batch}_{tag}.json

Outputs:
  results/eagle3_longctx_full/gap_summary.md    -- per-model batch-collapse table
  results/eagle3_longctx_full/approach_compare.md -- vLLM-patch vs SGLang speedup
  results/eagle3_longctx_full/curves.csv        -- all rows for plotting
  results/eagle3_longctx_full/figures/          -- speedup-vs-K plots per model

Usage:
  python analyze_eagle3_longctx_full.py
"""
import os, json, glob, sys
import csv

BASE    = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(BASE, "results")
OUT_DIR = os.path.join(RES_DIR, "eagle3_longctx_full")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

APPROACH_DIRS = {
    "vllm_patched": os.path.join(RES_DIR, "eagle3_longctx_patched"),
    "sglang":       os.path.join(RES_DIR, "eagle3_longctx_sglang"),
}

MODEL_LABELS = {
    "llama8b":  "Llama-3.1-8B (instruct)",
    "qwen14b":  "Qwen3-14B (instruct)",
    "deepseek": "DeepSeek-R1-Distill-LLaMA-8B (reasoning)",
}


def load_all():
    rows = []
    for approach, d in APPROACH_DIRS.items():
        if not os.path.isdir(d):
            continue
        for fpath in glob.glob(os.path.join(d, "*.json")):
            try:
                with open(fpath) as f:
                    data = json.load(f)
                model_key = data.get("model", "unknown")
                batch     = data.get("batch", 1)
                ctx_len   = data.get("ctx_len", 8192)
                for r in data.get("results", []):
                    rows.append({
                        "approach":   approach,
                        "model":      model_key,
                        "batch":      batch,
                        "ctx_len":    ctx_len,
                        "method":     r["method"],
                        "K":          r["K"],
                        "workload":   r["workload"],
                        "tok_per_s":  r.get("tok_per_s", 0),
                        "net_speedup": r.get("net_speedup", 1.0),
                        "aps":        r.get("accepted_tokens_per_step"),
                    })
            except Exception as e:
                print(f"  [skip {os.path.basename(fpath)}]: {e}")
    return rows


def best_per_workload_batch(rows, approach=None):
    """For each (model, batch, workload): find best K and gap vs K=1."""
    filtered = [r for r in rows if r["method"] != "baseline"]
    if approach:
        filtered = [r for r in filtered if r["approach"] == approach]

    # group by (model, batch, workload)
    groups = {}
    for r in filtered:
        key = (r["model"], r["batch"], r["workload"])
        groups.setdefault(key, []).append(r)

    summary = {}
    for key, rs in groups.items():
        best = max(rs, key=lambda x: x["net_speedup"])
        k1   = next((x for x in rs if x["K"] == 1), None)
        summary[key] = {
            "best_K":      best["K"],
            "best_speedup": best["net_speedup"],
            "k1_speedup":  k1["net_speedup"] if k1 else None,
            "gap_vs_k1":   (best["net_speedup"] - k1["net_speedup"]) if k1 else None,
        }
    return summary


def write_gap_summary(rows):
    out_path = os.path.join(OUT_DIR, "gap_summary.md")
    lines = ["# Long-context EAGLE-3 — batch collapse (ctx=8192)", ""]

    for approach, label in [("vllm_patched", "vLLM-patched"), ("sglang", "SGLang")]:
        approach_rows = [r for r in rows if r["approach"] == approach]
        if not approach_rows:
            lines.append(f"## {label}: no results found\n")
            continue
        summary = best_per_workload_batch(approach_rows)
        if not summary:
            lines.append(f"## {label}: no eagle3 results (all baselines only)\n")
            continue

        lines.append(f"## {label}")
        lines.append("")
        lines.append("| model | workload | batch | best K | best speedup | K=1 speedup | gap vs K=1 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for (model, batch, workload), s in sorted(summary.items()):
            m_label = MODEL_LABELS.get(model, model)
            k1  = f"{s['k1_speedup']:.3f}" if s["k1_speedup"] is not None else "—"
            gap = f"{s['gap_vs_k1']:+.3f}" if s["gap_vs_k1"] is not None else "—"
            lines.append(f"| {m_label} | {workload} | {batch} | {s['best_K']} "
                         f"| {s['best_speedup']:.3f} | {k1} | {gap} |")
        lines.append("")

    lines += [
        "## Comparison with short-context batch sweep (ctx=2048)",
        "",
        "Short-context collapse thresholds (from results/eagle3_batch/gap_summary.md):",
        "- B=16: still +5-11% gap vs K=1  (headroom exists)",
        "- B=32: gap = exactly 0% on ALL workloads  (complete collapse)",
        "",
        "**MagicDec hypothesis:** at long context (8K tokens), KV bandwidth bottleneck",
        "dominates even at B≥32, so K>1 should remain beneficial.",
        "",
        "If the table above shows gap>0 at B=32 for ctx=8192 -> MagicDec confirmed.",
        "If the table shows gap=0 at B=32 for ctx=8192 -> collapse is universal.",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def write_approach_compare(rows):
    """Side-by-side vLLM-patch vs SGLang speedup for the same (model, batch, K, workload)."""
    out_path = os.path.join(OUT_DIR, "approach_compare.md")
    lines = [
        "# Approach comparison: vLLM-patched vs SGLang (ctx=8192)",
        "",
        "Identical (model, batch, K, workload) measured by both approaches.",
        "",
        "| model | workload | batch | K | vLLM-patch speedup | SGLang speedup | delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    # Index by (model, batch, workload, K)
    idx = {}
    for r in rows:
        if r["method"] == "baseline":
            continue
        key = (r["model"], r["batch"], r["workload"], r["K"])
        idx.setdefault(key, {})[r["approach"]] = r["net_speedup"]

    for key in sorted(idx):
        model, batch, workload, k = key
        v = idx[key].get("vllm_patched", "—")
        s = idx[key].get("sglang", "—")
        if v == "—" or s == "—":
            continue
        delta = f"{s - v:+.3f}" if isinstance(v, float) and isinstance(s, float) else "—"
        v_str = f"{v:.3f}" if isinstance(v, float) else str(v)
        s_str = f"{s:.3f}" if isinstance(s, float) else str(s)
        m_label = MODEL_LABELS.get(model, model)
        lines.append(f"| {m_label} | {workload} | {batch} | {k} | {v_str} | {s_str} | {delta} |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def write_curves_csv(rows):
    out_path = os.path.join(OUT_DIR, "curves.csv")
    fields = ["approach", "model", "batch", "ctx_len", "workload",
              "method", "K", "tok_per_s", "net_speedup", "aps"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return out_path


def plot_curves(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skip plots")
        return

    models = sorted({r["model"] for r in rows})
    approaches = ["vllm_patched", "sglang"]
    approach_labels = {"vllm_patched": "vLLM-patched", "sglang": "SGLang"}
    colors = {"vllm_patched": "#1f77b4", "sglang": "#ff7f0e"}
    linestyles = {1: "-", 8: "--", 16: ":", 32: "-."}
    batches = sorted({r["batch"] for r in rows})

    for model in models:
        mrows = [r for r in rows if r["model"] == model and r["method"] != "baseline"]
        if not mrows:
            continue
        workloads = sorted({r["workload"] for r in mrows})
        fig, axes = plt.subplots(1, len(workloads), figsize=(5 * len(workloads), 4), squeeze=False)
        for wi, wl in enumerate(workloads):
            ax = axes[0][wi]
            for approach in approaches:
                for b in batches:
                    pts = [(r["K"], r["net_speedup"]) for r in mrows
                           if r["approach"] == approach and r["batch"] == b
                           and r["workload"] == wl]
                    if pts:
                        pts.sort()
                        xs, ys = zip(*pts)
                        ax.plot(xs, ys,
                                color=colors.get(approach, "gray"),
                                linestyle=linestyles.get(b, "-"),
                                marker="o", markersize=4,
                                label=f"{approach_labels.get(approach, approach)} B={b}")
            ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_xlabel("K (draft tokens)")
            ax.set_ylabel("Net speedup")
            ax.set_title(f"{MODEL_LABELS.get(model, model)}\n{wl} (ctx=8192)")
            ax.legend(fontsize=7)
        fig.suptitle("Long-context EAGLE-3: vLLM-patch vs SGLang", fontsize=10)
        plt.tight_layout()
        out_fig = os.path.join(FIG_DIR, f"longctx_{model}.png")
        plt.savefig(out_fig, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  wrote {out_fig}")


def main():
    rows = load_all()
    if not rows:
        print("No results found in:")
        for k, v in APPROACH_DIRS.items():
            print(f"  {k}: {v}")
        print("\nRun modal scripts first, then download results:")
        print("  modal volume get spec-dec-m5-results eagle3_longctx_patched/ results/eagle3_longctx_patched/")
        print("  modal volume get spec-dec-m5-results eagle3_longctx_sglang/  results/eagle3_longctx_sglang/")
        sys.exit(0)

    approaches  = {r["approach"] for r in rows}
    models      = {r["model"] for r in rows if r["method"] != "baseline"}
    batches     = sorted({r["batch"] for r in rows})
    eagle3_rows = [r for r in rows if r["method"] not in ("baseline",)]

    print(f"Loaded {len(rows)} rows")
    print(f"  approaches : {approaches}")
    print(f"  models     : {models}")
    print(f"  batches    : {batches}")
    print(f"  eagle3 rows: {len(eagle3_rows)}")

    gap_md      = write_gap_summary(rows)
    compare_md  = write_approach_compare(rows)
    curves_csv  = write_curves_csv(rows)
    plot_curves(rows)

    print(f"\nOutputs:")
    print(f"  {gap_md}")
    print(f"  {compare_md}")
    print(f"  {curves_csv}")
    print(f"  {FIG_DIR}/longctx_*.png")

    # Print headline for quick sanity-check
    print("\n--- Headline (best K per model/batch/workload, all approaches) ---")
    summary = best_per_workload_batch(rows)
    for (model, batch, workload), s in sorted(summary.items()):
        gap_str = f"+{s['gap_vs_k1']:.1%}" if s["gap_vs_k1"] is not None else "—"
        print(f"  {MODEL_LABELS.get(model,model):42s} B={batch:2d} {workload:12s} "
              f"best K={s['best_K']}  {s['best_speedup']:.3f}x  gap={gap_str}")


if __name__ == "__main__":
    main()
