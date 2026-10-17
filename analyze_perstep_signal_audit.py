"""Predictability audit (Gate 1): can any cheap pre-verification signal recover the
per-step oracle headroom on EAGLE-3 / Llama-3.1-8B?

The per-step oracle sets K_t = (tokens that WILL be accepted), giving a large ceiling
over best-fixed-K. A real controller must predict acceptance BEFORE verifying, from
features available at draft time. This script measures how much of that ceiling is
recoverable, and — crucially — whether per-position DRAFT features add anything over a
history-only predictor (the DSDE / SGLang-EMA prior-art lever).

Label: chain drafting accepts a prefix. For a step with `acc` accepted and max_k
drafted positions, position j<acc is ACCEPTED (1), position j==acc (if acc<max_k) is
the REJECTION (0); positions j>acc are unobserved (no label — but we STILL predict on
them at sim time, because a real drafter produces all max_k positions before verifying).

Honesty controls:
  * All max_k positions are scored at sim time (no peeking at where the chain stopped).
  * Generations are split TRAIN/TEST; the classifier is fit and the decision threshold
    is chosen on TRAIN only; every speedup is reported on TEST.
  * Two predictors run through the identical pipeline:
      history-only  = GBM on [prev_acc_ema]            (prior-art ceiling)
      full          = GBM on [ent, margin, position, prev_acc_ema]
    The delta between them is exactly what per-position draft signal buys.

Policy simulation (offline-exact): for any per-step draft length K_t, accepted =
min(K_t, acc_t) and cost = K_t, because the acceptance prefix is independent of how
many tokens we draft. Speedup uses the same cost model as analyze_eagle3_8b.py:
    speedup = MAT / (1 + c*mean_k),   MAT = mean(accepted)+1,  c = 0.15.
All policies draft at least 1 token (min K=1) to match the deliverable's oracle.

Outputs: results/perstep_signal/{audit.md, feature_auc.csv, policy_pareto.csv}
Usage: python analyze_perstep_signal_audit.py
"""
import json, os, csv
import numpy as np
import statistics as st

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(ROOT, "results", "eagle3_perstep_target_llama8b.json")
OUT = os.path.join(ROOT, "results", "perstep_signal")
os.makedirs(OUT, exist_ok=True)
C = 0.15            # draft/verify cost ratio (same as analyze_eagle3_8b.py)
EMA_ALPHA = 0.3     # for the history feature
MIN_K = 1           # always draft at least one token (matches oracle definition)
SEED = 0


# ----------------------------- load + clean ------------------------------- #
def load_steps():
    """Return list of clean steps with per-position features + step group id.

    Each step: {ent:[7], margin:[7], acc:int, gen:int, prev_acc_ema:float}.
    Keep only steps with exactly max_k entropy/margin entries (drop prefill/odd ones).
    """
    d = json.load(open(PATH))
    mk = d["max_k"]
    steps = []
    for gi, g in enumerate(d["gens"]):
        ema = None
        for s in g["steps"]:
            ent, mar = s.get("ent", []), s.get("margin", [])
            if len(ent) != mk or len(mar) != mk:
                continue
            acc = int(s["acc"])
            steps.append({"ent": ent, "margin": mar, "acc": acc, "gen": gi,
                          "prev_acc_ema": (ema if ema is not None else 0.0)})
            ema = acc if ema is None else (1 - EMA_ALPHA) * ema + EMA_ALPHA * acc
    return steps, mk


# --------------------------- per-position dataset ------------------------- #
FEATURE_NAMES = ["draft_entropy", "top1_margin", "position", "prev_acc_ema"]
FULL_IDX = [0, 1, 2, 3]            # full predictor uses all features
HIST_IDX = [3]                     # history-only predictor uses prev_acc_ema only


def step_feature_matrix(s, mk):
    """Features for ALL mk drafted positions of a step (scored at sim time)."""
    return [[s["ent"][j], s["margin"][j], float(j), s["prev_acc_ema"]] for j in range(mk)]


def build_labeled(steps, mk):
    """Observed (position, label) rows for TRAINING the classifier."""
    X, y, posidx = [], [], []
    for s in steps:
        acc = s["acc"]
        n_obs = acc if acc >= mk else acc + 1     # 0..acc-1 accepted, acc = rejection
        for j in range(min(n_obs, mk)):
            X.append([s["ent"][j], s["margin"][j], float(j), s["prev_acc_ema"]])
            y.append(1 if j < acc else 0); posidx.append(j)
    return np.array(X), np.array(y), np.array(posidx)


def per_feature_auc(X, y):
    out = {}
    for i, name in enumerate(FEATURE_NAMES):
        try:
            out[name] = roc_auc_score(y, X[:, i])
        except ValueError:
            out[name] = float("nan")
    return out


# ------------------------------ policies ---------------------------------- #
def speedup_from(accepted_list, k_list):
    mat = st.mean(accepted_list) + 1.0
    mean_k = st.mean(k_list)
    return mat / (1.0 + C * mean_k), mat, mean_k


def sim_fixed(steps, K):
    return speedup_from([min(K, s["acc"]) for s in steps], [K] * len(steps))


def sim_oracle(steps):
    # K_t = max(acc_t, MIN_K): draft exactly the accepted run (>=1)
    acc_l = [s["acc"] if s["acc"] >= MIN_K else 0 for s in steps]  # accepted tokens
    k_l = [max(s["acc"], MIN_K) for s in steps]
    return speedup_from(acc_l, k_l)


def sim_predictor(steps, mk, probs_by_step, thr):
    """Draft position j while P(accept_j) >= thr; always draft >= MIN_K.
    probs_by_step[i] is a length-mk array of P(accept) for ALL positions of step i."""
    acc_l, k_l = [], []
    for i, s in enumerate(steps):
        p = probs_by_step[i]
        K = MIN_K
        for j in range(MIN_K, mk):
            if p[j] >= thr:
                K = j + 1
            else:
                break
        acc_l.append(min(K, s["acc"])); k_l.append(K)
    return speedup_from(acc_l, k_l)


def predict_all_positions(steps, mk, model, idx):
    """Score every position of every step with a fitted model (feature subset `idx`)."""
    out = []
    for s in steps:
        M = np.array(step_feature_matrix(s, mk))[:, idx]
        out.append(model.predict_proba(M)[:, 1])
    return out


def run_predictor(train_steps, test_steps, mk, idx, name):
    """Fit on train observed positions, choose threshold on train, eval on test."""
    Xtr, ytr, _ = build_labeled(train_steps, mk)
    model = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=SEED)
    model.fit(Xtr[:, idx], ytr)

    probs_tr = predict_all_positions(train_steps, mk, model, idx)
    probs_te = predict_all_positions(test_steps, mk, model, idx)

    # choose threshold on TRAIN
    best = (None, -1)
    for thr in [round(t, 2) for t in np.arange(0.20, 0.96, 0.05)]:
        sp = sim_predictor(train_steps, mk, probs_tr, thr)[0]
        if sp > best[1]:
            best = (thr, sp)
    thr = best[0]
    test_sp = sim_predictor(test_steps, mk, probs_te, thr)[0]
    return {"name": name, "thr": thr, "test_speedup": test_sp,
            "train_speedup": best[1], "model": model}


def main():
    steps, mk = load_steps()
    pos0_reject = sum(1 for s in steps if s["acc"] == 0) / len(steps)
    print(f"clean steps: {len(steps)}  max_k: {mk}")
    print(f"steps with acc==0 (position-0 rejected): {pos0_reject*100:.1f}%")

    # ---- (A) descriptive single-feature AUC (full labeled set) ----
    Xall, yall, posidx = build_labeled(steps, mk)
    aucs = per_feature_auc(Xall, yall)
    print(f"\nlabeled positions: {len(yall)}  (accept rate {yall.mean()*100:.1f}%)")
    print("[A] single-feature AUC (predict per-position accept):")
    for n, a in aucs.items():
        print(f"   {n:16s} AUC={a:.3f}")
    pos0 = posidx == 0
    auc0 = {}
    print("[A2] position-0 only ('should we speculate at all?'):")
    for i, n in enumerate(FEATURE_NAMES[:3]):
        try:
            auc0[n] = roc_auc_score(yall[pos0], Xall[pos0, i])
        except ValueError:
            auc0[n] = float("nan")
        print(f"   {n:16s} AUC={auc0[n]:.3f}")

    # ---- multi-seed TRAIN/TEST split by generation (robustness) ----
    gens0 = sorted({s["gen"] for s in steps})
    rows = {"hist": [], "full": [], "oracle_gain": [], "bestK": []}
    for seed in range(8):
        gens = list(gens0); np.random.RandomState(seed).shuffle(gens)
        cut = len(gens) // 2
        train_g, test_g = set(gens[:cut]), set(gens[cut:])
        tr = [s for s in steps if s["gen"] in train_g]
        te = [s for s in steps if s["gen"] in test_g]
        bestK = max(range(1, mk + 1), key=lambda K: sim_fixed(tr, K)[0])
        bf = sim_fixed(te, bestK)[0]
        orc = sim_oracle(te)[0]
        gp = orc - bf
        rc = lambda sp: (sp - bf) / gp * 100 if gp > 0 else float("nan")
        h = run_predictor(tr, te, mk, HIST_IDX, "hist")
        fl = run_predictor(tr, te, mk, FULL_IDX, "full")
        rows["hist"].append(rc(h["test_speedup"]))
        rows["full"].append(rc(fl["test_speedup"]))
        rows["oracle_gain"].append((orc / bf - 1) * 100)
        rows["bestK"].append(bestK)

    def ms(a): return st.mean(a), (st.stdev(a) if len(a) > 1 else 0.0)
    h_m, h_s = ms(rows["hist"]); f_m, f_s = ms(rows["full"])
    og_m, _ = ms(rows["oracle_gain"]); delta = f_m - h_m
    best_K = max(set(rows["bestK"]), key=rows["bestK"].count)

    print(f"\n[C] TEST-set headroom recovered, mean±std over 8 gen-splits "
          f"(c={C}, min K={MIN_K}):")
    print(f"   best fixed K={best_K}            0% (baseline)")
    print(f"   history-only predictor      recovers {h_m:5.1f}% ± {h_s:.1f}")
    print(f"   full draft+history pred     recovers {f_m:5.1f}% ± {f_s:.1f}")
    print(f"   per-step ORACLE             ceiling = +{og_m:.1f}% over fixed (=100%)")
    print(f"\n   >>> per-position DRAFT features add {delta:+.1f} pts of ceiling over history "
          f"(full {f_m:.1f}% vs hist {h_m:.1f}%) <<<")

    # ---- artifacts ----
    with open(os.path.join(OUT, "feature_auc.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["feature", "auc_allpos", "auc_pos0"])
        for n in FEATURE_NAMES:
            w.writerow([n, f"{aucs[n]:.4f}", f"{auc0.get(n, float('nan')):.4f}"])

    verdict = ("ALIVE — per-position draft features add real headroom over history "
               "(promising; next step: add EAGLE hidden-state features)"
               if delta > 5 and f_m > 12 else
               "WEAK — cheap logit-level features do NOT meaningfully beat history; the "
               "per-step ceiling is largely irreducible from logit features. The only "
               "untested lever is EAGLE hidden-state features (needs new capture).")
    with open(os.path.join(OUT, "audit.md"), "w", encoding="utf-8") as f:
        f.write("# Per-step signal predictability audit (EAGLE-3 / Llama-3.1-8B)\n\n")
        f.write(f"- clean steps {len(steps)}; labeled positions {len(yall)} "
                f"(accept rate {yall.mean()*100:.1f}%); pos-0 immediate-reject {pos0_reject*100:.1f}%\n")
        f.write("- protocol: GBM fit on train gens, stop-threshold chosen on train, "
                "all positions scored at sim time, reported on held-out test gens, "
                "mean±std over 8 generation splits.\n\n")
        f.write("## Single-feature AUC (predict per-position accept)\n\n")
        f.write("| feature | AUC all-pos | AUC pos-0 |\n|---|---:|---:|\n")
        for n in FEATURE_NAMES:
            f.write(f"| {n} | {aucs[n]:.3f} | {auc0.get(n, float('nan')):.3f} |\n")
        f.write("\n## Headroom recovered on TEST (cost model c=%.2f, min K=%d)\n\n" % (C, MIN_K))
        f.write("| policy | % of oracle ceiling recovered |\n|---|---:|\n")
        f.write(f"| best fixed K={best_K} | 0% (baseline) |\n")
        f.write(f"| history-only predictor | {h_m:.1f}% ± {h_s:.1f} |\n")
        f.write(f"| full draft+history predictor | {f_m:.1f}% ± {f_s:.1f} |\n")
        f.write(f"| per-step oracle | 100% (= +{og_m:.1f}% over fixed) |\n\n")
        f.write(f"**Per-position draft features over history alone: {delta:+.1f} pts of ceiling.**\n\n")
        f.write(f"## Verdict\n\n{verdict}\n")
    print(f"\nwrote {OUT}/audit.md, feature_auc.csv")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
