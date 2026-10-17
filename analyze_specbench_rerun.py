"""Analyse the Spec-Bench rerun (modal_vllm_specbench_rerun.py).

Reads   results/specbench/specbench_reference_*.json   no-spec greedy reference
        results/specbench/specbench_spec_*.json        bracket / main / batch phases
Writes  results/specbench/summary.md, summary.json
        paper/tables/specbench_{main,identity,batch,bracket}.tex   (tabular bodies)
        paper/figures/fig_specbench_batch.png

Statistics
  gain      throughput of cum(0.2) over the faster of native K=2 and K=3, per category, as a
            ratio of pooled tokens/second. The faster native is picked after the fact inside
            every bootstrap resample, which favours the baseline.
  CI        95% paired bootstrap over prompts (B=1) or chunks (B>1), 4000 resamples.
  containers  each main run is one container; the last column of the main table is the mean
            over containers with its standard error, as in the fair-baseline table.
  identity  each arm's tokens against the no-spec reference: share of prompts with an identical
            output, position of the first differing token, and the reference's own top-1 minus
            top-2 logprob gap at that position (a small gap means a numerical near-tie).

Run: python analyze_specbench_rerun.py [--in DIR] [--out DIR] [--tables DIR] [--fig DIR]
"""
import argparse
import glob
import json
import os

import numpy as np

CATS = ["mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag"]
CAT_LABEL = {"mt_bench": "MT-Bench", "translation": "Translation",
             "summarization": "Summarization", "qa": "QA",
             "math_reasoning": "Math", "rag": "RAG", "all": "All"}
NATIVES = ["native2", "native3"]
ADAPT = "cum0.2"
NBOOT = 4000
RNG = np.random.default_rng(20261012)


# ─────────────────────────────── loading ─────────────────────────────── #

def load(in_dir):
    refs, specs = [], []
    for p in sorted(glob.glob(os.path.join(in_dir, "specbench_reference_*.json"))):
        refs.append(json.load(open(p)))
    for p in sorted(glob.glob(os.path.join(in_dir, "specbench_spec_*.json"))):
        specs.append(json.load(open(p)))
    return refs, specs


def units(records, arms, bnom=None):
    """Group records into paired units (one prompt, or one chunk) that every arm completed.

    Returns {cat: [ {arm: (ntok, dt, depth, tokens|None, qids)} ]}.
    """
    by = {}
    for r in records:
        if bnom is not None and r.get("Bnom", r.get("B", 1)) != bnom:
            continue
        key = (r["cat"], r["chunk"])
        by.setdefault(key, {})[r["arm"]] = (r["ntok"], r["dt"], r.get("depth"),
                                            r.get("tokens"), r["qids"])
    out = {}
    for (cat, _), d in sorted(by.items(), key=lambda kv: kv[0][1]):
        if all(a in d for a in arms):
            out.setdefault(cat, []).append(d)
    return out


# ───────────────────────────── statistics ───────────────────────────── #

def _tps(rows, arm, idx):
    tok = sum(rows[i][arm][0] for i in idx)
    dt = sum(rows[i][arm][1] for i in idx)
    return tok / dt if dt > 0 else float("nan")


def gain_stats(rows, natives=NATIVES, adapt=ADAPT):
    """Point gain (%) of adapt over the faster native, bootstrap CI, which native, depth."""
    n = len(rows)
    idx = np.arange(n)
    nat = {a: _tps(rows, a, idx) for a in natives}
    best = max(nat, key=nat.get)
    point = 100 * (_tps(rows, adapt, idx) / nat[best] - 1)
    boots = []
    for _ in range(NBOOT):
        s = RNG.integers(0, n, n)
        nb = max(_tps(rows, a, s) for a in natives)
        boots.append(100 * (_tps(rows, adapt, s) / nb - 1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    depths = [r[adapt][2] for r in rows if r[adapt][2] is not None]
    return {"n": n, "gain": round(point, 2), "ci": [round(lo, 2), round(hi, 2)],
            "best_native": best, "tps": {a: round(v, 2) for a, v in nat.items()},
            "tps_adapt": round(_tps(rows, adapt, idx), 2),
            "depth": round(float(np.mean(depths)), 2) if depths else None}


def by_category(unit_map, baselines=NATIVES):
    res = {c: gain_stats(unit_map[c], baselines) for c in CATS if unit_map.get(c)}
    pooled = [u for c in CATS for u in unit_map.get(c, [])]
    if pooled:
        res["all"] = gain_stats(pooled, baselines)
    return res


def run_arms(records):
    """Arms present in a run's records, and the baseline arms (everything except cum(0.2))."""
    arms = sorted({r["arm"] for r in records})
    return arms, [a for a in arms if a != ADAPT]


def is_tuned(baselines):
    """A run whose baselines reach past K=3, added after the bracket check."""
    return any(int(a.replace("native", "").replace("fixed", "")) >= 4 for a in baselines)


# ────────────────────────────── identity ────────────────────────────── #

def identity(ref_records, spec_records, arms):
    """Token-by-token comparison of each arm with the no-spec reference."""
    ref = {r["qid"]: r for r in ref_records}
    out = {}
    for arm in arms:
        n = same = 0
        first_div, margins, per_cat = [], [], {}
        for r in spec_records:
            if r["arm"] != arm or r.get("tokens") is None:
                continue
            for qid, toks in zip(r["qids"], r["tokens"]):
                if qid not in ref:
                    continue
                rt, rm = ref[qid]["tokens"], ref[qid]["margins"]
                n += 1
                c = per_cat.setdefault(r["cat"], [0, 0])
                c[0] += 1
                if toks == rt:
                    same += 1
                    c[1] += 1
                    continue
                d = next((i for i, (a, b) in enumerate(zip(toks, rt)) if a != b),
                         min(len(toks), len(rt)))
                first_div.append(d)
                if d < len(rm) and rm[d] is not None:
                    margins.append(rm[d])
        if n == 0:
            continue
        m = np.array(margins) if margins else np.array([np.nan])
        out[arm] = {
            "n": n, "identical": same, "identical_pct": round(100 * same / n, 1),
            "per_cat_identical_pct": {k: round(100 * v[1] / v[0], 1) for k, v in per_cat.items()},
            "first_div_median": float(np.median(first_div)) if first_div else None,
            "margin_at_div": ({"median": round(float(np.nanmedian(m)), 4),
                               "p95": round(float(np.nanpercentile(m, 95)), 4),
                               "max": round(float(np.nanmax(m)), 4),
                               "share_below_0.1": round(float(np.mean(m < 0.1)), 3),
                               "share_below_0.5": round(float(np.mean(m < 0.5)), 3)}
                              if margins else None),
        }
    return out


def reference_margin_baseline(ref_records):
    """Distribution of the reference's own top-2 gap over all positions, for contrast."""
    m = np.array([x for r in ref_records for x in r["margins"] if x is not None])
    if m.size == 0:
        return None
    return {"positions": int(m.size), "median": round(float(np.median(m)), 4),
            "share_below_0.1": round(float(np.mean(m < 0.1)), 4)}


# ─────────────────────────────── output ─────────────────────────────── #

def fmt_gain(s):
    return f"${s['gain']:+.1f}$ [{s['ci'][0]:+.1f}, {s['ci'][1]:+.1f}]"


def tex_main(tuned, untuned):
    """Headline columns from the tuned-baseline run, replication columns from the K<=3 runs."""
    tt, ut = sorted(tuned), sorted(untuned)
    cols = "l" + "c" + "r" * (len(tt) + len(ut) + 1)
    head = ["Category", "$K^{\\ast}$"]
    head += [r"gain [95\% CI]" if len(tt) == 1 else f"tuned {i + 1}" for i in range(len(tt))]
    head += [f"$K{{\\le}}3$, c{i + 1}" for i in range(len(ut))]
    head += ["depth"]
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for c in CATS + ["all"]:
        if not all(c in tuned[t] for t in tt) or not all(c in untuned[t] for t in ut):
            continue
        if c == "all":
            lines.append(r"\midrule")
        ref = tuned[tt[0]][c] if tt else untuned[ut[0]][c]
        best = ref["best_native"].replace("native", "").replace("fixed", "")
        best += "" if ref["best_native"].startswith("native") else "$^\\dagger$"
        cells = [CAT_LABEL[c], best]
        for t in tt:
            g = tuned[t][c]
            cells.append(f"${g['gain']:+.1f}$ [{g['ci'][0]:+.1f}, {g['ci'][1]:+.1f}]")
        for t in ut:
            cells.append(f"${untuned[t][c]['gain']:+.1f}$")
        cells.append(f"{ref['depth']:.2f}" if ref["depth"] is not None else "--")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"% dagger: best fixed depth was the padded K=2 arm inside the K=7 engine"]
    return "\n".join(lines)


def tex_identity(ident, ref_base):
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Arm & identical & first diff. & gap at diff. (median) & gap $<0.1$ \\",
             r"\midrule"]
    for arm, s in ident.items():
        mg = s["margin_at_div"]
        name = arm.replace("cum0.2", "cum(0.2)")
        fd = s["first_div_median"]
        lines.append(f"{name} & {s['identical_pct']:.1f}\\% & "
                     + (f"{fd:.0f} & {mg['median']:.3f} & {100 * mg['share_below_0.1']:.0f}\\%"
                        if mg and fd is not None else "-- & -- & --") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if ref_base:
        lines.append(f"% reference top-2 gap over all {ref_base['positions']} positions: "
                     f"median {ref_base['median']}, share below 0.1 = {ref_base['share_below_0.1']}")
    return "\n".join(lines)


def tex_batch(batch_by_b):
    bs = sorted(batch_by_b)
    lines = [r"\begin{tabular}{l" + "r" * len(bs) + "}", r"\toprule",
             "Category & " + " & ".join(f"$B{{=}}{b}$" for b in bs) + r" \\", r"\midrule"]
    for c in CATS + ["all"]:
        if not all(c in batch_by_b[b] for b in bs):
            continue
        if c == "all":
            lines.append(r"\midrule")
        lines.append(f"{CAT_LABEL[c]} & " + " & ".join(
            f"${batch_by_b[b][c]['gain']:+.1f}$" for b in bs) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def bracket_table(records):
    arms = sorted({r["arm"] for r in records})
    fixed = [a for a in arms if a.startswith("fixed")]
    um = units(records, arms)
    res = {}
    for c in CATS:
        rows = um.get(c)
        if not rows:
            continue
        idx = np.arange(len(rows))
        tps = {a: _tps(rows, a, idx) for a in arms}
        base = tps["native2"]
        best_fixed = max(fixed, key=tps.get) if fixed else None
        res[c] = {"n": len(rows), "rel_to_native2_pct":
                  {a: round(100 * (v / base - 1), 2) for a, v in tps.items()},
                  "best_fixed": best_fixed,
                  "bracketed": best_fixed in ("fixed2", "fixed3") if best_fixed else None}
    return res


def figure(main_all, batch_by_b, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    pts = {}
    if main_all:
        pts[1] = main_all
    for b, d in batch_by_b.items():
        pts[b] = d
    bs = sorted(pts)
    if not bs:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x = np.arange(len(bs))
    for i, c in enumerate(CATS):
        ys = [pts[b][c]["gain"] if c in pts[b] else np.nan for b in bs]
        ax.plot(x, ys, color="#B8B8B8", lw=1.0, marker="o", ms=3, zorder=1,
                label="each category" if i == 0 else None)
    ys = [pts[b]["all"]["gain"] for b in bs]
    lo = [pts[b]["all"]["gain"] - pts[b]["all"]["ci"][0] for b in bs]
    hi = [pts[b]["all"]["ci"][1] - pts[b]["all"]["gain"] for b in bs]
    ax.errorbar(x, ys, yerr=[lo, hi], color="#4878A8", lw=2, marker="o", ms=5,
                capsize=3, zorder=3, label="all categories pooled (95% CI)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"B={b}" for b in bs])
    ax.set_ylabel("gain over best\nfixed depth (%)")
    ax.legend(frameon=False, fontsize=7.5, loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--in", dest="inp", default=os.path.join(base, "results", "specbench"))
    ap.add_argument("--out", default=os.path.join(base, "results", "specbench"))
    ap.add_argument("--tables", default=os.path.join(base, "paper", "tables"))
    ap.add_argument("--fig", default=os.path.join(base, "paper", "figures"))
    a = ap.parse_args()
    refs, specs = load(a.inp)
    if not specs:
        print(f"no specbench_spec_*.json in {a.inp} yet; nothing to analyse")
        return
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.tables, exist_ok=True)
    os.makedirs(a.fig, exist_ok=True)

    summary = {"runs": [s["meta"] for s in specs] + [r["meta"] for r in refs]}
    smoke = [s for s in specs if s["meta"].get("tag") == "smoke"]
    specs = [s for s in specs if s["meta"].get("tag") != "smoke"]
    refs = [r for r in refs if r["meta"].get("tag") != "smoke"] or refs
    if smoke and not specs:
        # Smoke-only: still exercise every table, but never into the paper.
        specs = smoke
        a.tables = os.path.join(a.out, "smoke_check")
        a.fig = a.tables
        os.makedirs(a.tables, exist_ok=True)
        print(f"smoke data only: tables and figure go to {a.tables}, not the paper")

    main_runs = {s["meta"]["tag"]: s for s in specs if s.get("main")}
    per_container, baselines_of = {}, {}
    for t, s_ in main_runs.items():
        arms, base = run_arms(s_["main"])
        baselines_of[t] = base
        per_container[t] = by_category(units(s_["main"], arms), base)
    tuned = {t: r for t, r in per_container.items() if is_tuned(baselines_of[t])}
    untuned = {t: r for t, r in per_container.items() if not is_tuned(baselines_of[t])}
    summary["main"] = per_container
    summary["main_baselines"] = baselines_of

    batch_runs = {s["meta"]["tag"]: s for s in specs if s.get("batch")}
    batch_by_run = {}
    for t, s_ in batch_runs.items():
        arms, base = run_arms(s_["batch"])
        baselines_of[t] = base
        batch_by_run[t] = {b: by_category(units(s_["batch"], arms, bnom=b), base)
                           for b in sorted({r.get("Bnom", r["B"]) for r in s_["batch"]})}
    summary["batch"] = {t: {str(b): v for b, v in d.items()} for t, d in batch_by_run.items()}
    tuned_batch = [t for t in batch_by_run if is_tuned(baselines_of[t])]
    head_batch = batch_by_run[tuned_batch[0]] if tuned_batch else (
        batch_by_run[sorted(batch_by_run)[0]] if batch_by_run else {})

    bracket_recs = [r for s_ in specs for r in s_.get("bracket", [])]
    summary["bracket"] = bracket_table(bracket_recs) if bracket_recs else {}

    ident, ref_base = {}, None
    if refs:
        ref_records = [r for ref in refs for r in ref["records"]]
        all_main = [r for s_ in main_runs.values() for r in s_["main"]]
        main_arms = sorted({r["arm"] for r in all_main})
        ident = identity(ref_records, all_main, main_arms)
        batch_recs = [r for s_ in batch_runs.values() for r in s_["batch"]]
        ident_batch = identity(ref_records, batch_recs,
                               sorted({r["arm"] for r in batch_recs})) if batch_recs else {}
        ref_base = reference_margin_baseline(ref_records)
        ntok = sum(r["ntok"] for r in ref_records)
        dt = sum(r["dt"] for r in ref_records)
        summary["reference"] = {"prompts": len(ref_records),
                                "nospec_tps": round(ntok / dt, 2) if dt else None,
                                "margin_baseline": ref_base}
        summary["identity_main"] = ident
        summary["identity_batch"] = ident_batch
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)

    if per_container:
        open(os.path.join(a.tables, "specbench_main.tex"), "w").write(tex_main(tuned, untuned))
    if ident:
        open(os.path.join(a.tables, "specbench_identity.tex"), "w").write(
            tex_identity(ident, ref_base))
    if head_batch:
        open(os.path.join(a.tables, "specbench_batch.tex"), "w").write(tex_batch(head_batch))
    main_all = None
    if per_container:
        main_all = (tuned or untuned)[sorted(tuned or untuned)[0]]
    figure(main_all, head_batch, os.path.join(a.fig, "fig_specbench_batch.png"))

    md = ["# Spec-Bench rerun: summary", "",
          "Generated by `analyze_specbench_rerun.py`. Numbers only; no interpretation.", ""]
    for t, res in sorted(per_container.items()):
        base = baselines_of[t]
        label = "tuned baseline" if t in tuned else "K<=3 baseline (fair-baseline table configuration)"
        md += [f"## Main, B=1, run `{t}` ({label}; baselines: {', '.join(base)})", "",
               "| category | n | " + " | ".join(f"{a} tok/s" for a in base)
               + " | cum(0.2) tok/s | gain % [95% CI] | vs | depth |",
               "|---|---|" + "---|" * len(base) + "---|---|---|---|"]
        for c in CATS + ["all"]:
            if c in res:
                s_ = res[c]
                md.append(f"| {CAT_LABEL[c]} | {s_['n']} | "
                          + " | ".join(str(s_['tps'][a]) for a in base)
                          + f" | {s_['tps_adapt']} | "
                          f"{s_['gain']:+.2f} [{s_['ci'][0]:+.2f}, {s_['ci'][1]:+.2f}] | "
                          f"{s_['best_native']} | {s_['depth']} |")
        md.append("")
    for t, d in sorted(batch_by_run.items()):
        for b, res in sorted(d.items()):
            md += [f"## Batch B={b}, run `{t}` (baselines: {', '.join(baselines_of[t])})", "",
                   "| category | chunks | gain % [95% CI] | vs | depth |", "|---|---|---|---|---|"]
            for c in CATS + ["all"]:
                if c in res:
                    s_ = res[c]
                    md.append(f"| {CAT_LABEL[c]} | {s_['n']} | {s_['gain']:+.2f} "
                              f"[{s_['ci'][0]:+.2f}, {s_['ci'][1]:+.2f}] | {s_['best_native']} | "
                              f"{s_['depth']} |")
            md.append("")
    if bracket_recs:
        arms, _ = run_arms(bracket_recs)
        fixed = [a for a in arms if a.startswith("fixed")]
        bb = by_category(units(bracket_recs, arms), fixed)
        md += ["## Bracket subsample: cum(0.2) against the best padded fixed depth (2-5)", "",
               "| category | n | best | gain % [95% CI] |", "|---|---|---|---|"]
        for c in CATS + ["all"]:
            if c in bb:
                s_ = bb[c]
                md.append(f"| {CAT_LABEL[c]} | {s_['n']} | {s_['best_native']} | "
                          f"{s_['gain']:+.2f} [{s_['ci'][0]:+.2f}, {s_['ci'][1]:+.2f}] |")
        md.append("")
        summary["bracket_vs_best_padded"] = bb
        json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    if summary["bracket"]:
        md += ["## Bracket check (padded fixed depths, subsample)", "",
               "| category | n | best fixed | bracketed by K=2/3 |", "|---|---|---|---|"]
        for c, s_ in summary["bracket"].items():
            md.append(f"| {CAT_LABEL[c]} | {s_['n']} | {s_['best_fixed']} | {s_['bracketed']} |")
        md.append("")
    if ident:
        md += ["## Token identity vs no-spec reference (B=1)", "",
               "| arm | prompts | identical % | median first diff | gap at diff median / p95 / max "
               "| share gap<0.1 |", "|---|---|---|---|---|---|"]
        for arm, s_ in ident.items():
            mg = s_["margin_at_div"] or {}
            md.append(f"| {arm} | {s_['n']} | {s_['identical_pct']} | {s_['first_div_median']} | "
                      f"{mg.get('median')} / {mg.get('p95')} / {mg.get('max')} | "
                      f"{mg.get('share_below_0.1')} |")
        if ref_base:
            md += ["", f"Reference top-2 gap over all {ref_base['positions']} positions: median "
                   f"{ref_base['median']}, share below 0.1 = {ref_base['share_below_0.1']}."]
        md.append("")
    md += ["## Runs", ""] + [f"- `{m.get('mode')}` `{m.get('tag')}`: vLLM {m.get('vllm')}, "
                             f"{m.get('gpu')}, {round(m.get('elapsed_s', 0) / 60, 1)} min, "
                             f"~${m.get('est_usd')}, truncated={m.get('truncated')}"
                             for m in summary["runs"]]
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
