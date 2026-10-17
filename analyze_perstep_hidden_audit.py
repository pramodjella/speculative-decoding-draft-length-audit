"""Gate 2: does the EAGLE draft HIDDEN STATE recover per-step headroom that cheap
logit features cannot? Compares base (entropy+margin+position+history) vs
base+hidden through the identical, leak-free protocol of analyze_perstep_signal_audit.

Input: results/eagle3_perstep_hidden_llama8b.json (from modal_eagle3_hidden_capture.py),
where each step has h = [[norm, proj_0..proj_15] per drafted position].

Reuses the validated sim functions (sim_fixed/sim_oracle/sim_predictor) from the base
audit; only the feature matrix is richer. Reports % of the per-step oracle ceiling
recovered by each feature set, mean±std over 8 generation splits.

Usage: python analyze_perstep_hidden_audit.py
"""
import os, csv
import numpy as np
import statistics as st

import analyze_perstep_signal_audit as A   # C, MIN_K, SEED, sim_fixed, sim_oracle, sim_predictor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(ROOT, "results", "eagle3_perstep_hidden_llama8b.json")
OUT = os.path.join(ROOT, "results", "perstep_signal")
os.makedirs(OUT, exist_ok=True)
EMA_ALPHA = A.EMA_ALPHA


def load_steps_hidden(path):
    import json
    d = json.load(open(path))
    mk = d["max_k"]; hdim = d.get("hdim", 16)
    hlen = hdim + 1                              # [norm, proj_0..proj_{hdim-1}]
    steps = []
    for gi, g in enumerate(d["gens"]):
        ema = None
        for s in g["steps"]:
            ent, mar, h = s.get("ent", []), s.get("margin", []), s.get("h", [])
            if len(ent) != mk or len(mar) != mk or len(h) != mk:
                continue
            if any(len(hp) != hlen for hp in h):
                continue
            acc = int(s["acc"])
            steps.append({"ent": ent, "margin": mar, "h": h, "acc": acc, "gen": gi,
                          "prev_acc_ema": (ema if ema is not None else 0.0)})
            ema = acc if ema is None else (1 - EMA_ALPHA) * ema + EMA_ALPHA * acc
    return steps, mk, hlen


def feat(s, j):
    """Full feature vector for position j: base(4) + hidden(hlen)."""
    return [s["ent"][j], s["margin"][j], float(j), s["prev_acc_ema"]] + list(s["h"][j])


def build_labeled(steps, mk, idx):
    X, y = [], []
    for s in steps:
        acc = s["acc"]; n_obs = acc if acc >= mk else acc + 1
        for j in range(min(n_obs, mk)):
            X.append([feat(s, j)[i] for i in idx]); y.append(1 if j < acc else 0)
    return np.array(X), np.array(y)


def predict_all_positions(steps, mk, model, idx):
    out = []
    for s in steps:
        M = np.array([[feat(s, j)[i] for i in idx] for j in range(mk)])
        out.append(model.predict_proba(M)[:, 1])
    return out


def run_predictor(tr, te, mk, idx, seed):
    Xtr, ytr = build_labeled(tr, mk, idx)
    if len(set(ytr)) < 2:
        return None
    m = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=seed)
    m.fit(Xtr, ytr)
    p_tr = predict_all_positions(tr, mk, m, idx)
    p_te = predict_all_positions(te, mk, m, idx)
    best = (None, -1)
    for thr in [round(t, 2) for t in np.arange(0.20, 0.96, 0.05)]:
        sp = A.sim_predictor(tr, mk, p_tr, thr)[0]
        if sp > best[1]:
            best = (thr, sp)
    return A.sim_predictor(te, mk, p_te, best[0])[0]


def main():
    if not os.path.exists(PATH):
        print(f"!! {PATH} not found — run modal_eagle3_hidden_capture.py and download first.")
        return
    steps, mk, hlen = load_steps_hidden(PATH)
    print(f"clean steps {len(steps)}  max_k {mk}  hidden-len {hlen}")

    BASE = [0, 1, 2, 3]
    HID = list(range(4, 4 + hlen))
    FULL = BASE + HID

    # descriptive: hidden-norm AUC vs accept (all labeled positions)
    Xall, yall = build_labeled(steps, mk, FULL)
    print(f"labeled positions {len(yall)} (accept rate {yall.mean()*100:.1f}%)")
    auc_norm = roc_auc_score(yall, Xall[:, 4])     # column 4 = hidden norm
    print(f"hidden-norm single-feature AUC = {auc_norm:.3f}")

    gens0 = sorted({s["gen"] for s in steps})
    rows = {"base": [], "full": [], "ceil": []}
    for seed in range(8):
        g = list(gens0); np.random.RandomState(seed).shuffle(g)
        cut = len(g) // 2
        tr = [s for s in steps if s["gen"] in set(g[:cut])]
        te = [s for s in steps if s["gen"] in set(g[cut:])]
        bestK = max(range(1, mk + 1), key=lambda K: A.sim_fixed(tr, K)[0])
        bf = A.sim_fixed(te, bestK)[0]; orc = A.sim_oracle(te)[0]; gp = orc - bf
        rc = lambda sp: (sp - bf) / gp * 100 if gp > 0 else float("nan")
        b = run_predictor(tr, te, mk, BASE, seed)
        f = run_predictor(tr, te, mk, FULL, seed)
        if b is not None: rows["base"].append(rc(b))
        if f is not None: rows["full"].append(rc(f))
        rows["ceil"].append((orc / bf - 1) * 100)

    def ms(a): return (st.mean(a), st.stdev(a) if len(a) > 1 else 0.0)
    bm, bs = ms(rows["base"]); fm, fs = ms(rows["full"]); cm, _ = ms(rows["ceil"])
    delta = fm - bm

    print(f"\nTEST headroom recovered, mean±std / 8 splits (c={A.C}, min K={A.MIN_K}):")
    print(f"   base (entropy+margin+position+history)   {bm:5.1f}% ± {bs:.1f}")
    print(f"   base + EAGLE hidden state                {fm:5.1f}% ± {fs:.1f}")
    print(f"   per-step oracle ceiling = +{cm:.1f}% over fixed (=100%)")
    print(f"\n   >>> hidden state adds {delta:+.1f} pts of ceiling over cheap features <<<")

    verdict = ("BREAKTHROUGH — EAGLE hidden state recovers materially more than cheap "
               "features; a real per-step controller has room. Build it."
               if delta > 8 and fm > 18 else
               "DEAD END — even the EAGLE hidden state does not crack the per-step "
               "ceiling. Adaptive draft length has no exploitable signal. Lock the "
               "negative result; pivot to a different axis (quality/batch).")
    with open(os.path.join(OUT, "hidden_audit.md"), "w", encoding="utf-8") as fh:
        fh.write("# Per-step HIDDEN-STATE audit (EAGLE-3 / Llama-3.1-8B)\n\n")
        fh.write(f"- clean steps {len(steps)}; labeled positions {len(yall)} "
                 f"(accept rate {yall.mean()*100:.1f}%); hidden-norm AUC {auc_norm:.3f}\n\n")
        fh.write("| feature set | % of oracle ceiling recovered |\n|---|---:|\n")
        fh.write(f"| base (entropy+margin+position+history) | {bm:.1f}% ± {bs:.1f} |\n")
        fh.write(f"| base + EAGLE hidden state | {fm:.1f}% ± {fs:.1f} |\n")
        fh.write(f"| per-step oracle | 100% (= +{cm:.1f}% over fixed) |\n\n")
        fh.write(f"**Hidden state over cheap features: {delta:+.1f} pts of ceiling.**\n\n")
        fh.write(f"## Verdict\n\n{verdict}\n")
    print(f"\nwrote {OUT}/hidden_audit.md")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
