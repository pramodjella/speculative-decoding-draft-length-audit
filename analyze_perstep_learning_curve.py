"""Is the cheap-feature verdict limited by DATA SIZE? Learning curve test.

Holds out a fixed TEST set of generations, then trains the full-feature GBM on an
increasing fraction of the remaining generations and measures % of the per-step
oracle ceiling recovered on TEST. If recovery plateaus before 100% of the data is
used, more rows would not help -> the weak result is a feature/signal limit, not a
sample-size limit. If it is still rising at the end, the audit was data-starved.

Reuses the validated pieces from analyze_perstep_signal_audit.py.
Usage: python analyze_perstep_learning_curve.py
"""
import os, csv
import numpy as np
import statistics as st

import analyze_perstep_signal_audit as A
from sklearn.ensemble import GradientBoostingClassifier

OUT = A.OUT
os.makedirs(OUT, exist_ok=True)


def recovery_on_test(train_steps, test_steps, mk, idx, seed):
    Xtr, ytr, _ = A.build_labeled(train_steps, mk)
    if len(set(ytr)) < 2:
        return None, len(ytr)
    m = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=seed)
    m.fit(Xtr[:, idx], ytr)
    probs_tr = A.predict_all_positions(train_steps, mk, m, idx)
    probs_te = A.predict_all_positions(test_steps, mk, m, idx)
    best = (None, -1)
    for thr in [round(t, 2) for t in np.arange(0.20, 0.96, 0.05)]:
        sp = A.sim_predictor(train_steps, mk, probs_tr, thr)[0]
        if sp > best[1]:
            best = (thr, sp)
    test_sp = A.sim_predictor(test_steps, mk, probs_te, best[0])[0]
    return test_sp, len(ytr)


def main():
    steps, mk = A.load_steps()
    gens = sorted({s["gen"] for s in steps})
    print(f"clean steps {len(steps)}, generations {len(gens)}, max_k {mk}")

    fracs = [0.15, 0.3, 0.5, 0.7, 1.0]
    SEEDS = 6
    curve = {f: [] for f in fracs}
    rows_used = {f: [] for f in fracs}

    for seed in range(SEEDS):
        g = list(gens); np.random.RandomState(seed).shuffle(g)
        test_g = set(g[: len(g) // 3])                  # fixed 1/3 held out as TEST
        pool = g[len(g) // 3:]                           # 2/3 available to train from
        test_steps = [s for s in steps if s["gen"] in test_g]
        bestK = max(range(1, mk + 1), key=lambda K: A.sim_fixed(test_steps, K)[0])
        bf = A.sim_fixed(test_steps, bestK)[0]
        orc = A.sim_oracle(test_steps)[0]
        gap = orc - bf
        for f in fracs:
            ntr = max(2, int(round(f * len(pool))))
            train_g = set(pool[:ntr])
            train_steps = [s for s in steps if s["gen"] in train_g]
            sp, nrows = recovery_on_test(train_steps, test_steps, mk, A.FULL_IDX, seed)
            if sp is not None and gap > 0:
                curve[f].append((sp - bf) / gap * 100)
                rows_used[f].append(nrows)

    print(f"\nLearning curve — % of +{(orc/bf-1)*100:.0f}%-ceiling recovered on held-out TEST")
    print("(full feature set; mean±std over %d seeds)\n" % SEEDS)
    print(f"  {'train gens %':>12} {'~labeled rows':>14} {'recovered %':>14}")
    out_rows = []
    for f in fracs:
        m, s = st.mean(curve[f]), (st.stdev(curve[f]) if len(curve[f]) > 1 else 0.0)
        nr = int(st.mean(rows_used[f]))
        print(f"  {int(f*100):>11}% {nr:>14} {m:>9.1f} ± {s:.1f}")
        out_rows.append((int(f * 100), nr, round(m, 2), round(s, 2)))

    # plateau check: last two points within noise of each other?
    a, b = curve[fracs[-2]], curve[fracs[-1]]
    rise = st.mean(b) - st.mean(a)
    pooled = (st.pstdev(a + b) or 1e-9)
    plateau = abs(rise) < pooled
    print(f"\n  last-step rise {rise:+.1f} pts vs noise {pooled:.1f} -> "
          f"{'PLATEAUED (more data would NOT help; signal/feature limit)' if plateau else 'STILL RISING (more data might help)'}")

    with open(os.path.join(OUT, "learning_curve.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["train_gens_pct", "labeled_rows", "recovered_pct_mean", "recovered_pct_std"])
        for r in out_rows:
            w.writerow(r)
    print(f"\nwrote {OUT}/learning_curve.csv")


if __name__ == "__main__":
    main()
