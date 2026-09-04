"""Stitch the EAGLE-3 batch-size sweep into the per-step deliverable picture.

Reads (all optional — uses whatever is present):
  results/eagle3_batch/vllm_eagle3_b*_*.json   one per batch size (the new sweep)
  results/eagle3_8b/fixedK_by_workload.csv     existing B=1 deliverable curve

Writes:
  results/eagle3_batch/curves.csv              long format: workload,K,batch,net_speedup,accept_len
  results/eagle3_batch/gap_vs_batch.csv        per (workload,batch): best K, max speedup, K=1 speedup
  results/eagle3_batch/figures/speedup_vs_K_by_batch.png
  results/eagle3_batch/gap_summary.md          one-page read of the result

The question this answers (Yash's open ask): does the best fixed K shift down as
batch grows? If yes, per-step adaptive K has real headroom in the batched regime
even though B=1 oracle ceiling is only +2%.
"""
import csv, glob, json, os, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
BATCH_DIR = os.path.join(RES, "eagle3_batch")
B1_CSV = os.path.join(RES, "eagle3_8b", "fixedK_by_workload.csv")
FIG_DIR = os.path.join(BATCH_DIR, "figures")
os.makedirs(BATCH_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_batch_jsons():
    """Returns rows: (workload, K, batch, net_speedup, acc_per_step) — eagle3 only."""
    rows = []
    paths = sorted(glob.glob(os.path.join(BATCH_DIR, "vllm_eagle3_b*_*.json")))
    for p in paths:
        d = json.load(open(p))
        b = int(d.get("batch", 0))
        for r in d["results"]:
            if r["method"] != "eagle3":
                continue
            rows.append((r["workload"], int(r["K"]), b,
                         float(r["net_speedup"]),
                         r.get("accepted_tokens_per_step")))
    return rows


def load_b1_csv():
    """Returns rows from the B=1 deliverable CSV (skips the MIXED aggregate row)."""
    rows = []
    if not os.path.exists(B1_CSV):
        return rows
    with open(B1_CSV) as f:
        for row in csv.DictReader(f):
            w = row["scope"]
            if w == "MIXED":
                continue
            rows.append((w, int(row["K"]), 1,
                         float(row["net_speedup"]),
                         float(row["accept_len"])))
    return rows


def write_curves_csv(rows, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["workload", "K", "batch", "net_speedup", "accept_len"])
        for r in sorted(rows):
            ap = "" if r[4] is None else f"{r[4]:.4f}"
            wr.writerow([r[0], r[1], r[2], f"{r[3]:.4f}", ap])


def best_per_workload_batch(rows):
    """(workload, batch) -> {best_K, best_speedup, k1_speedup, gap_vs_k1}."""
    by_wb = defaultdict(list)
    for w, K, b, spd, _ in rows:
        by_wb[(w, b)].append((K, spd))
    out = {}
    for (w, b), lst in by_wb.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        best_K, best_spd = lst[0]
        k1 = next((spd for K, spd in lst if K == 1), None)
        out[(w, b)] = {"best_K": best_K, "best_speedup": best_spd,
                       "k1_speedup": k1,
                       "gap_vs_k1": (best_spd - k1) if k1 is not None else None}
    return out


def write_gap_csv(summary, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["workload", "batch", "best_K", "best_speedup",
                     "k1_speedup", "gap_vs_k1"])
        for (w, b), s in sorted(summary.items()):
            wr.writerow([w, b, s["best_K"], f"{s['best_speedup']:.4f}",
                         f"{s['k1_speedup']:.4f}" if s["k1_speedup"] else "",
                         f"{s['gap_vs_k1']:.4f}" if s["gap_vs_k1"] is not None else ""])


def plot_curves(rows, fig_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [skip plot — matplotlib not available: {e}]")
        return
    workloads = sorted({r[0] for r in rows})
    batches = sorted({r[2] for r in rows})
    if not workloads or not batches:
        print("  [skip plot — no rows]")
        return
    fig, axes = plt.subplots(1, len(workloads), figsize=(5 * len(workloads), 4),
                             sharey=True, squeeze=False)
    for ax, w in zip(axes[0], workloads):
        for b in batches:
            pts = sorted([(K, spd) for ww, K, bb, spd, _ in rows
                          if ww == w and bb == b])
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=f"B={b}")
        ax.set_title(w)
        ax.set_xlabel("K (num_speculative_tokens)")
        ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("net speedup vs no-spec")
    axes[0][-1].legend(loc="best", fontsize=9)
    fig.suptitle("EAGLE-3 / Llama-3.1-8B — fixed-K curve shifts with batch size")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def write_summary_md(summary, out_path):
    """One-page read of the result — point at it in the paper write-up."""
    lines = ["# EAGLE-3 batch-size sweep — gap-vs-batch read", ""]
    by_b = defaultdict(list)
    for (w, b), s in summary.items():
        by_b[b].append((w, s))
    if not by_b:
        lines.append("_No batch-sweep data yet. Run modal_vllm_eagle3.py --batch N first._")
        open(out_path, "w", encoding="utf-8").write("\n".join(lines))
        return
    lines.append("| workload | batch | best K | best speedup | K=1 speedup | gap vs K=1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for b in sorted(by_b):
        for w, s in sorted(by_b[b]):
            k1 = f"{s['k1_speedup']:.3f}" if s["k1_speedup"] else "—"
            g = f"{s['gap_vs_k1']:+.3f}" if s["gap_vs_k1"] is not None else "—"
            lines.append(f"| {w} | {b} | {s['best_K']} | "
                         f"{s['best_speedup']:.3f} | {k1} | {g} |")
    lines.append("")
    lines.append("## What to look for")
    lines.append("- Best K **drops** as batch grows -> verification cost dominates ->"
                 " per-step adaptive K has real headroom in the batched regime.")
    lines.append("- Curves **flatten** at high B -> fixed K leaves money on the table"
                 " precisely where a controller could pick K per step.")
    lines.append("- If best K stays put and curves keep their shape -> Yash's"
                 " hypothesis is rejected; lock in the negative result.")
    open(out_path, "w", encoding="utf-8").write("\n".join(lines))


def main():
    rows = load_b1_csv() + load_batch_jsons()
    if not rows:
        print(f"no inputs found:")
        print(f"  expected JSONs at {BATCH_DIR}/vllm_eagle3_b*_*.json")
        print(f"  or B=1 CSV at  {B1_CSV}")
        sys.exit(0)
    batches = sorted({r[2] for r in rows})
    print(f"loaded {len(rows)} rows; batches present: {batches}")

    curves_csv = os.path.join(BATCH_DIR, "curves.csv")
    gap_csv = os.path.join(BATCH_DIR, "gap_vs_batch.csv")
    fig_path = os.path.join(FIG_DIR, "speedup_vs_K_by_batch.png")
    summary_md = os.path.join(BATCH_DIR, "gap_summary.md")

    write_curves_csv(rows, curves_csv)
    summary = best_per_workload_batch(rows)
    write_gap_csv(summary, gap_csv)
    plot_curves(rows, fig_path)
    write_summary_md(summary, summary_md)

    print(f"  wrote {curves_csv}")
    print(f"  wrote {gap_csv}")
    print(f"  wrote {fig_path}")
    print(f"  wrote {summary_md}")


if __name__ == "__main__":
    main()
