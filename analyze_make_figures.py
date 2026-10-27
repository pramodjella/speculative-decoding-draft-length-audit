"""Generate the five paper figures from result JSONs (>=200 dpi, paper/figures/)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "paper", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ── Fig 2 (paper): Bayes-ceiling decomposition rungs (Llama-3.1-8B) ─────────────────── #
# Source is the CORRECTED paired-within-fold ladder (analyze_bayes_ceiling_paired.py).
# The superseded un-paired bayes_ceiling.json scored the deployable and ceiling rungs on
# DIFFERENT data, so the ceiling did not bound the probe it is supposed to bound; plotting
# it put the "probe (deploy)" bar above "probe (oracle thr.)" and contradicted the decomposition table.
# Rung naming follows the paper's decomposition table: the fourth rung is "probe (oracle thr.)". The retired
# name Bayes-(hidden) overclaimed it: it removes threshold-selection error only, not
# hypothesis-class error, so it bounds this probe family and not any reader of the hidden state.
bcp = json.load(open(os.path.join(RES, "perstep_signal", "bayes_ceiling_paired.json")))
folds = bcp["llama8b"]["folds"]
mean = lambda k: sum(f[k] for f in folds) / len(folds)
rungs = ["best fixed K", "Bayes(position)", "probe (deploy)", "probe (oracle thr.)",
         "per-step oracle"]
vals = [mean("fixed"), mean("bayes_position"), mean("deployable"), mean("ceiling"),
        mean("oracle")]
assert vals[3] >= vals[2], "nesting control violated: ceiling must bound the deployable probe"
colors = ["#666666", "#666666", "#4878A8", "#4878A8", "#B04030"]
fig, ax = plt.subplots(figsize=(7, 3.6))
bars = ax.bar(rungs, vals, color=colors, width=0.62)
ax.set_ylim(min(vals) - 0.05, max(vals) + 0.075)
ax.set_ylabel("cost-model speedup")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=10)
# Quote the split in words, not digits: the exact per-fold percentages live in the decomposition table, and
# a rounded number here that disagrees with the table by a tenth is a referee magnet.
# Aim inside the bar, not at its top edge: an arrow tip landing on the bar-top value label
# renders as a strike-through of the number.
ax.annotate("reached by the\nhidden-state probe\n(~1/5 of span)",
            xy=(3, vals[3] - 0.022), xytext=(1.55, vals[-1] - 0.035),
            arrowprops=dict(arrowstyle="->", lw=1), fontsize=9)
ax.annotate("not reached by the\nsignals we tested\n(~4/5 of span)",
            xy=(4, vals[-1] - 0.012), xytext=(2.55, vals[-1] + 0.012),
            arrowprops=dict(arrowstyle="->", lw=1), fontsize=9)
ax.set_title("Bayes-ceiling decomposition of the per-step oracle (Llama-3.1-8B)")
plt.xticks(rotation=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_decomposition.png"), dpi=220)
plt.close()
print("fig_decomposition.png")

# ── Fig 3 (paper): survivor across settings (paired gains vs strongest tuned fixed) ──── #
settings = ["Llama-8B\nvLLM chain", "Llama-8B\nEAGLE tree", "Qwen3-14B\nvLLM chain",
            "DeepSeek (weak head)\nEAGLE tree"]
# per-workload paired gains (he, gsm8k, mtb)
gains = [
    [5.62, 5.74, 1.99],       # vllm llama cum0.2, fair baseline (Table 5)
    [2.90, 3.58, 4.77],       # tree llama paired
    [1.26, 0.32, 0.68],       # vllm qwen cum0.2
    [-0.73, -0.86, 1.36],     # tree deepseek
]
ses = [
    [1.17, 2.02, 1.12],
    [1.83, 0.32, 0.50],
    [0.27, 1.38, 0.50],
    [1.19, 2.26, 3.52],
]
fig, ax = plt.subplots(figsize=(7, 3.6))
x = np.arange(len(settings))
w = 0.24
wl_names = ["HumanEval", "GSM8K", "MT-Bench"]
wl_colors = ["#4878A8", "#6AA84F", "#B04030"]
for i in range(3):
    ax.bar(x + (i - 1) * w, [g[i] for g in gains], width=w,
           yerr=[s[i] for s in ses], capsize=3, color=wl_colors[i], label=wl_names[i])
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(settings, fontsize=9)
ax.set_ylabel("paired gain vs strongest tuned fixed (%)")
ax.set_title("Saturation tail-pruning at batch size 1: ties or beats tuned fixed depth")
ax.legend(fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_survivor.png"), dpi=220)
plt.close()
print("fig_survivor.png")

# ── Fig 4 (paper): why pairing is necessary (per-cycle raw tok/s, vLLM Llama) ─────────── #
tp = json.load(open(os.path.join(RES, "vllm_tailprune2_llama8b.json")))
w = "gsm8k"
fig, ax = plt.subplots(figsize=(7, 3.4))
styles = {"fixed2": ("#888888", "--"), "fixed3": ("#666666", "-."),
          "fixed4": ("#444444", ":"), "fixed7": ("#222222", "-"),
          "cum0.05": ("#4878A8", "-"), "cum0.2": ("#B04030", "-")}
for nm, runs in tp["tps"][w].items():
    c, ls = styles.get(nm, ("#999999", "-"))
    ax.plot(range(1, len(runs) + 1), runs, marker="o", ms=4, color=c, ls=ls, label=nm)
ax.set_xticks(range(1, 5))
ax.set_xlabel("cycle (round-robin, order rotated per cycle)")
ax.set_ylabel("tok/s (GSM8K)")
# Keep the title short enough to fit the rendered figure width: the previous one was clipped
# mid-word ("...drift is visible and cance") inside the PNG itself, where no LaTeX-side
# change could fix it. Legend goes above the axes so it cannot sit on the data.
ax.set_title("Paired round-robin: arms separate, drift cancels", pad=26)
ax.legend(fontsize=8, ncol=6, frameon=False, loc="lower center",
          bbox_to_anchor=(0.5, 1.005), columnspacing=1.4, handletextpad=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_paired_protocol.png"), dpi=220)
plt.close()
print("fig_paired_protocol.png")


# ── Fig 5 (paper): batch decay of the survivor (paired vs strongest NATIVE engine) ───── #
# The profit window is the headline scoping claim (batch size 1) and previously appeared in prose
# only. Baseline is the strongest native arm (native2/native3, each drafting its true K) --
# never the padded native7 -- and "strongest" means the arm that minimises the reported gain.
WL = ["humaneval", "gsm8k", "mt_bench"]
NATIVES = ["native2", "native3"]


def _paired_vs_strongest_native(tps, w, adaptive="cum0.2"):
    """Paired per-cycle % gains vs each native arm; return those of the strongest (worst-case)."""
    cand = {}
    for nat in NATIVES:
        if nat not in tps[w]:
            continue
        a, b = tps[w][adaptive], tps[w][nat]
        n = min(len(a), len(b))
        cand[nat] = [100 * (a[i] - b[i]) / b[i] for i in range(n)]
    best = min(cand, key=lambda k: float(np.mean(cand[k])))
    return cand[best]


series = {}
for w in WL:
    b1 = []
    for tag in ("r1", "r2", "r3"):
        d = json.load(open(os.path.join(RES, "vllm_fairbase_llama8b_%s.json" % tag)))
        b1 += _paired_vs_strongest_native(d["tps"], w)
    pts = [(1, b1)]
    for tag, b in (("b4", 4), ("b8", 8)):
        d = json.load(open(os.path.join(RES, "vllm_fairbase2_llama8b_%s.json" % tag)))
        pts.append((b, _paired_vs_strongest_native(d["tps"], w)))
    series[w] = [(b, float(np.mean(g)), float(np.std(g, ddof=1) / len(g) ** 0.5)) for b, g in pts]

fig, ax = plt.subplots(figsize=(7, 3.4))
wl_names = {"humaneval": "HumanEval", "gsm8k": "GSM8K", "mt_bench": "MT-Bench"}
wl_colors = {"humaneval": "#4878A8", "gsm8k": "#6AA84F", "mt_bench": "#B04030"}
for w, pts in series.items():
    xs = [p[0] for p in pts]
    ax.errorbar(xs, [p[1] for p in pts], yerr=[p[2] for p in pts], marker="o", ms=5,
                capsize=3, color=wl_colors[w], label=wl_names[w])
ax.axhline(0, color="black", lw=0.8)
ax.axvspan(0.85, 1.25, color="#6AA84F", alpha=0.10)
ax.text(1.3, ax.get_ylim()[0] + 0.4, "profit window: B=1", fontsize=8, color="#3A6B2A")
ax.set_xscale("log", base=2)
ax.set_xticks([1, 4, 8])
ax.set_xticklabels(["1", "4", "8"])
ax.set_xlabel("batch size")
ax.set_ylabel("paired gain vs strongest\nnative engine (%)")
ax.set_title("The survivor's gain decays with batch, not with verification cost", pad=8)
ax.legend(fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_batch_decay.png"), dpi=220)
plt.close()
print("fig_batch_decay.png")
print("done ->", OUT)

# ── Fig 1 (paper): the audit as a waterfall (oracle -> survivor, gate by gate) ───────── #
# Bars 1-2 are cost-model figures on the Llama-3.1-8B ladder; bars 3-4 are measured
# wall-clock. They sit in two separate panels with their own headers so the two currencies
# never share a plot area. The y-range is shared so heights stay comparable; gate 3 is the
# arrow that crosses the gap from the cost model to the clock.
from matplotlib.patches import ConnectionPatch
bcp2 = json.load(open(os.path.join(RES, "perstep_signal", "bayes_ceiling_paired.json")))
fl = bcp2["llama8b"]["folds"]
_m = lambda k: sum(f[k] for f in fl) / len(fl)
oracle_pct = 100 * (_m("oracle") - _m("fixed")) / _m("fixed")
reach_pct = 100 * (_m("deployable") - _m("fixed")) / _m("fixed")

realized_pct = -5.1          # in-loop probe, HumanEval (the in-loop probe table)
# The survivor bar is the Spec-Bench result: cum(0.2) in vLLM, all 480 prompts at 512 tokens,
# each category against its own best fixed depth, mean over the three fresh containers.
_sb = json.load(open(os.path.join(RES, "specbench", "summary.json")))
_tuned = [t for t in _sb["main"]
          if any(int(a.replace("native", "").replace("fixed", "")) >= 4
                 for a in _sb["main_baselines"][t])]
survivor_pct = sum(_sb["main"][t]["all_best_per_category"]["gain"] for t in _tuned) / len(_tuned)

# Drawn for a SINGLE IEEE column (~3.4in wide), so it is sized small with proportionally large
# type. A 7in-wide figure scaled into one column shrinks every label by half.
fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.2, 3.4), sharey=True,
                               gridspec_kw={"wspace": 0.42})
panels = [
    (axl, ["per-step\noracle", "reached by\nthe probe"], [oracle_pct, reach_pct],
     ["#B04030", "#4878A8"], "cost model"),
    (axr, ["in-loop probe\n(tree engine)", "signal-free\npruning (vLLM,\nSpec-Bench)"],
     [realized_pct, survivor_pct],
     ["#8C8C8C", "#4E9A3A"], "measured wall-clock"),
]
lo = min(realized_pct, 0) - 4.0
hi = max(oracle_pct, survivor_pct) + 4.5
for ax, labels, vals, cols, header in panels:
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, color=cols, width=0.62)
    ax.axhline(0, color="black", lw=0.9)
    # Whole percentages only: the paired ladder gives 18.0 where the decomposition table quotes the pooled
    # 18.1, an aggregation convention; exact values with error bars live in the tables.
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.9 if v >= 0 else -2.4),
                "%+.0f%%" % v, ha="center", fontsize=11.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_title(header, fontsize=10, style="italic", color="#333333", pad=6)
axl.set_ylim(lo, hi)
axl.set_ylabel("gain over tuned fixed $K$ (%)", fontsize=10)
axr.tick_params(axis="y", labelleft=False, left=False)
axr.spines["left"].set_visible(False)
# gate 1: inside the cost-model panel
axl.annotate("", xy=(0.66, reach_pct + 0.4), xytext=(0.33, oracle_pct - 0.8),
             arrowprops=dict(arrowstyle="->", lw=1.3, color="#555555"))
axl.text(0.78, oracle_pct * 0.62, "gate 1\n~82% not\nrecovered", ha="left", va="center",
         fontsize=8.5, color="#333333")
# gate 3: from the last cost-model bar to the first wall-clock bar, across the panel gap
con = ConnectionPatch(xyA=(1.31, reach_pct), coordsA=axl.transData,
                      xyB=(-0.31, realized_pct * 0.5), coordsB=axr.transData,
                      arrowstyle="->", lw=1.3, color="#555555")
fig.add_artist(con)
axr.text(-0.55, hi - 3.0, "gate 3\nrun inside\nthe engine", ha="left", va="top",
         fontsize=8.5, color="#333333")
fig.suptitle("Following the promised speedup", fontsize=11.5, y=0.98)
fig.subplots_adjust(left=0.15, right=0.98, bottom=0.17, top=0.82)
# the break between the two currencies: a rule down the middle of the gap
_xl, _xr = axl.get_position().x1, axr.get_position().x0
fig.add_artist(plt.Line2D([(_xl + _xr) / 2] * 2, [0.06, 0.86], transform=fig.transFigure,
                          color="#9A9A9A", lw=1.0, ls=(0, (4, 3))))
plt.savefig(os.path.join(OUT, "fig_waterfall.png"), dpi=220)
plt.close()
print("fig_waterfall.png")
