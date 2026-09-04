"""Analyze long-context EAGLE-3 batch sweep results.

Compares best-K vs batch size at ctx_len=8192 (new) against the short-ctx
(ctx_len=2048) results from analyze_eagle3_batch.py.

The central question: does the K->1 collapse at B>=32 (observed for short context)
persist at long context, or does MagicDec's memory-bandwidth argument save K>1?

Usage: python analyze_eagle3_longctx.py
Outputs: results/eagle3_longctx/{summary.md, bestK_vs_batch.csv,
          figures/bestK_longctx_vs_shortctx.png}
"""
import json, os, csv, glob
import statistics as st

ROOT    = os.path.dirname(os.path.abspath(__file__))
LC_DIR  = os.path.join(ROOT, "results", "eagle3_longctx")
SC_DIR  = os.path.join(ROOT, "results", "eagle3_batch")       # short-ctx data
OUT_DIR = LC_DIR
os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

C = 0.15   # draft cost ratio (same as all other analyses)


def cost_model_speedup(acc_per_step, mean_k):
    """MAT / (1 + c * mean_k)"""
    return acc_per_step / (1.0 + C * mean_k)


def load_longctx_jsons():
    """Load all vllm_eagle3_longctx_b*.json from the results volume."""
    rows = []
    pattern = os.path.join(LC_DIR, "vllm_eagle3_longctx_b*.json")
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        d = json.load(open(path))
        batch   = d.get("batch", 1)
        ctx_len = d.get("ctx_len", 8192)
        for r in d["results"]:
            r["batch"]   = batch
            r["ctx_len"] = ctx_len
            r["source"]  = fname
            rows.append(r)
    print(f"Loaded {len(rows)} rows from {len(glob.glob(pattern))} long-ctx JSON files")
    return rows


def load_shortctx_jsons():
    """Load short-ctx batch sweep results for comparison."""
    rows = []
    for fname in ["vllm_eagle3_b8_llama8b.json",
                  "vllm_eagle3_b8_llama8b_lowk.json",
                  "vllm_eagle3_b32_llama8b.json",
                  "vllm_eagle3_b64_llama8b.json"]:
        path = os.path.join(SC_DIR, fname)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        batch = d.get("batch", 1)
        for r in d["results"]:
            r["batch"]   = batch
            r["ctx_len"] = 2048
            r["source"]  = fname
            rows.append(r)
    # also try to load B=1 from the main deliverable CSV
    b1_csv = os.path.join(ROOT, "results", "eagle3_8b", "fixedK_by_workload.csv")
    if os.path.exists(b1_csv):
        import csv as csv_mod
        with open(b1_csv) as f:
            for row in csv_mod.DictReader(f):
                # CSV uses 'scope' for workload, 'accept_len' for accepted_tokens_per_step
                rows.append({
                    "method": "eagle3", "K": int(row["K"]),
                    "workload": row.get("workload", row.get("scope", "unknown")),
                    "net_speedup": float(row.get("net_speedup", row.get("speedup", 0))),
                    "accepted_tokens_per_step": float(row.get("accepted_tokens_per_step",
                                                    row.get("accept_len", "nan"))),
                    "batch": 1, "ctx_len": 2048, "source": "fixedK_by_workload.csv",
                })
    print(f"Loaded {len(rows)} short-ctx rows")
    return rows


def best_k_table(rows, ctx_label):
    """Return {batch: {workload: {bestK, speedup}}} for eagle3 rows only."""
    spec = [r for r in rows if r.get("method") == "eagle3"]
    batches = sorted({r["batch"] for r in spec})
    workloads = sorted({r["workload"] for r in spec})
    result = {}
    for b in batches:
        result[b] = {}
        for w in workloads:
            sub = [r for r in spec if r["batch"] == b and r["workload"] == w]
            if not sub:
                continue
            best = max(sub, key=lambda r: r["net_speedup"])
            result[b][w] = {"K": best["K"], "speedup": best["net_speedup"],
                            "aps": best.get("accepted_tokens_per_step")}
    return result, batches, workloads


def main():
    lc_rows = load_longctx_jsons()
    sc_rows = load_shortctx_jsons()

    if not lc_rows:
        print("\nNo long-ctx results yet. Run modal_eagle3_longctx.py first:")
        print("  modal run modal_eagle3_longctx.py --batch 1  --tag longctx_b1")
        print("  modal run modal_eagle3_longctx.py --batch 8  --tag longctx_b8  --gpu-mem 0.85")
        print("  modal run modal_eagle3_longctx.py --batch 32 --tag longctx_b32 --gpu-mem 0.90")
        print("  modal run modal_eagle3_longctx.py --batch 64 --tag longctx_b64 --gpu-mem 0.92")
        return

    lc_tbl, lc_batches, lc_wl = best_k_table(lc_rows, "long-ctx")
    sc_tbl, sc_batches, sc_wl = best_k_table(sc_rows, "short-ctx")

    # ── print comparison ────────────────────────────────────────────────────
    print("\n=== Best K by batch size: LONG ctx (8K) vs SHORT ctx (2K) ===")
    all_batches = sorted(set(lc_batches + sc_batches))
    for b in all_batches:
        print(f"\n  B={b}")
        for w in sorted(set(lc_wl + sc_wl)):
            lc = lc_tbl.get(b, {}).get(w)
            sc = sc_tbl.get(b, {}).get(w)
            lc_str = f"K={lc['K']} ({lc['speedup']:.2f}x)" if lc else "—"
            sc_str = f"K={sc['K']} ({sc['speedup']:.2f}x)" if sc else "—"
            collapse = ""
            if lc and sc:
                if lc["K"] > sc["K"]:
                    collapse = " <-- K HIGHER at long ctx (MagicDec regime)"
                elif lc["K"] == sc["K"]:
                    collapse = " <-- same K"
                else:
                    collapse = " <-- K LOWER at long ctx (unusual)"
            print(f"    {w:14s}: long={lc_str:20s}  short={sc_str:20s}{collapse}")

    # ── write CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "bestK_vs_batch.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ctx_len", "batch", "workload", "best_K", "speedup", "aps"])
        for ctx_label, tbl in [("long_8k", lc_tbl), ("short_2k", sc_tbl)]:
            ctx_val = 8192 if "long" in ctx_label else 2048
            for b, wls in tbl.items():
                for wl, v in wls.items():
                    w.writerow([ctx_val, b, wl, v["K"], v["speedup"], v.get("aps", "")])
    print(f"\nWrote {csv_path}")

    # ── markdown summary ────────────────────────────────────────────────────
    summary_path = os.path.join(OUT_DIR, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Long-context vs short-context batch sweep: best K comparison\n\n")
        f.write("**Context lengths compared:** long = 8192, short = 2048\n")
        f.write("**Target/draft:** Llama-3.1-8B-Instruct + EAGLE-3\n\n")

        f.write("## Best-K by batch size\n\n")
        for ctx_label, tbl, ctx_val in [("Long (8K)", lc_tbl, 8192),
                                         ("Short (2K)", sc_tbl, 2048)]:
            f.write(f"\n### {ctx_label}\n\n")
            all_wl = sorted({w for wls in tbl.values() for w in wls})
            header = "| batch | " + " | ".join(all_wl) + " |\n"
            f.write(header + "|" + "---|" * (len(all_wl) + 1) + "\n")
            for b in sorted(tbl.keys()):
                cells = []
                for w in all_wl:
                    v = tbl[b].get(w)
                    cells.append(f"K={v['K']} ({v['speedup']:.2f}x)" if v else "—")
                f.write(f"| {b} | " + " | ".join(cells) + " |\n")

        f.write("\n## Interpretation\n\n")
        # auto-detect finding
        collapse_persists = []
        collapse_reverses = []
        for b in lc_batches:
            if b < 8:
                continue
            for w in lc_wl:
                lc = lc_tbl.get(b, {}).get(w)
                sc = sc_tbl.get(b, {}).get(w)
                if lc and sc:
                    if lc["K"] > sc["K"]:
                        collapse_reverses.append((b, w))
                    elif lc["K"] == sc["K"] == 1:
                        collapse_persists.append((b, w))

        if collapse_reverses:
            f.write(
                f"**K->1 collapse REVERSES at long context** for: "
                f"{collapse_reverses}. MagicDec's memory-bandwidth argument is confirmed: "
                f"at 8K context the KV cache restores memory-boundedness even at large batch, "
                f"so K>1 is still optimal. This is WHERE adaptive-K headroom survives.\n\n"
            )
        elif collapse_persists:
            f.write(
                f"**K->1 collapse PERSISTS at long context** for: "
                f"{collapse_persists}. Contrary to MagicDec, the memory-bandwidth argument "
                f"does not rescue larger K even at 8K context on H100 with this model. "
                f"This STRENGTHENS the negative result — even long context doesn't save "
                f"adaptive-K headroom at large batch.\n\n"
            )
        else:
            f.write("Mixed results — see table.\n\n")

        f.write("### Paper implication\n\n")
        f.write(
            "Short-context collapse: per our batch sweep, K->1 at B>=32 "
            "(verified at ctx=2048). Long-context batch sweep: "
            f"this experiment fills the 'open limitation' in the oracle-study paper (§8) "
            f"and determines whether MagicDec's long-context regime rescues adaptive-K.\n"
        )

    print(f"Wrote {summary_path}")

    # ── optional plot ────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(lc_wl), figsize=(5*len(lc_wl), 4), sharey=False)
        if len(lc_wl) == 1:
            axes = [axes]
        for ax, w in zip(axes, sorted(lc_wl)):
            for ctx_label, tbl, color, ls in [
                ("long-8K", lc_tbl, "steelblue", "-"),
                ("short-2K", sc_tbl, "tomato", "--"),
            ]:
                bs = sorted(b for b in tbl if w in tbl[b])
                ks = [tbl[b][w]["K"] for b in bs]
                ax.plot(bs, ks, marker="o", linestyle=ls, color=color, label=ctx_label)
            ax.set_title(w)
            ax.set_xlabel("batch size")
            ax.set_ylabel("best K")
            ax.legend()
            ax.set_xticks(sorted({b for tbl in [lc_tbl, sc_tbl] for b in tbl}))

        plt.suptitle("Best K vs batch: long (8K) vs short (2K) context\n"
                     "Llama-3.1-8B + EAGLE-3", fontsize=11)
        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, "bestK_longctx_vs_shortctx.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {fig_path}")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
