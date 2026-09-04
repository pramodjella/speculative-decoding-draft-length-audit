"""Bayes-ceiling ladder, PAIRED WITHIN FOLD (corrects a comparison flaw in the original).

THE FLAW. `analyze_bayes_ceiling.py` computes the two hidden-probe rungs on DIFFERENT data:
  probe_deployable : 8-fold CV — lambda from each TRAIN fold, scored on that fold's TEST rows,
                     then averaged over folds;
  bayes_hidden     : lambda swept over the ENTIRE dataset and scored on the entire dataset.
They are therefore not nested, neither bounds the other, and the "ceiling" can be beaten by
the thing it supposedly bounds. It is: Llama-8B 18.9% (deployable) vs 17.8% (bayes_hidden);
Llama-8B reasoning 18.3% vs 15.2%. Those inversions are fold noise, but they make the word
"ceiling" unsupportable and invite a reviewer to distrust the whole ladder.

THE FIX. Compute BOTH rungs inside the SAME fold, on the SAME rows:
  for each fold i:
      lam_train  = argmax speedup over the TRAIN rows  -> score on TEST rows  = deployable_i
      lam_oracle = argmax speedup over the TEST rows   -> score on TEST rows  = ceiling_i
Now ceiling_i >= deployable_i holds BY CONSTRUCTION in every fold (same probe scores, same
evaluation rows, one merely has the better threshold). The quantity of interest becomes a
PAIRED per-fold difference with a standard error:

    threshold-selection loss = mean_i (ceiling_i - deployable_i)  +- SE

which answers the actual scientific question ("is threshold choice the binding constraint?")
with an uncertainty attached, instead of eyeballing two numbers computed on different data.

This also gives the ladder its own positive control, matching the discipline used everywhere
else in this project: if ANY fold reports deployable_i > ceiling_i, the code is broken.

Reuses the loader, probe and cost model from analyze_bayes_ceiling.py unchanged (C=0.15).
Out: results/perstep_signal/bayes_ceiling_paired.json + .md
Usage: python analyze_bayes_ceiling_paired.py [--model llama8b]
"""
import os, json, argparse
import numpy as np

import analyze_bayes_ceiling as B   # loader, probe, cost model, threshold helpers

OUT_DIR = B.OUT_DIR


def paired_ladder(model_key, n_splits=8):
    import pandas as pd
    pq = os.path.join(B.IN_DIR, f"hidden_full_{model_key}.parquet")
    if not os.path.exists(pq):
        print(f"  [skip] {pq} not found")
        return None
    df = pd.read_parquet(pq, engine="pyarrow")
    df["uid"] = df["workload"].astype(str) + ":" + df["gen_i"].astype(str)
    hdim_cols = [c for c in df.columns if c.startswith("h")]
    maxk = int(df["ndraft"].iloc[0])
    print(f"  rows={len(df)} maxk={maxk} gens={df['uid'].nunique()}")

    surv = df.groupby("position")["accept"].mean().to_dict()
    df["p_position"] = df["position"].map(surv)
    print("  training out-of-fold PCA-50+LR probe...")
    df["p_hidden"] = B.oof_probe_predictions(df, hdim_cols, n_splits=n_splits)
    df = df[~df["p_hidden"].isna()].copy()

    grid = np.linspace(0.0, 1.0, 101)
    p_pos_steps, acc_arr, step_uid = B.build_step_arrays(df, "p_position")
    p_hid_steps, _, _ = B.build_step_arrays(df, "p_hidden")
    n_steps = len(acc_arr)

    uids = sorted(set(step_uid.tolist()))
    rng = np.random.RandomState(7); rng.shuffle(uids)
    gsplits = np.array_split(uids, n_splits)

    rows = []
    for i in range(n_splits):
        test_g = set(gsplits[i].tolist())
        te = [k for k in range(n_steps) if step_uid[k] in test_g]
        tr = [k for k in range(n_steps) if step_uid[k] not in test_g]
        if not te or not tr:
            continue
        acc_te = acc_arr[te]
        p_hid_te = [p_hid_steps[k] for k in te]
        p_hid_tr = [p_hid_steps[k] for k in tr]
        p_pos_te = [p_pos_steps[k] for k in te]

        # every rung scored on the SAME test rows
        fixed_i, fixed_k_i = B.best_fixed_speedup(acc_te, maxk)
        oracle_i = B.oracle_speedup(acc_te, maxk)
        pos_i, _ = B.best_threshold_speedup(p_pos_te, acc_te, maxk, grid)   # self-test rung
        _, lam_tr = B.best_threshold_speedup(p_hid_tr, acc_arr[tr], maxk, grid)
        deploy_i = B.speedup(B.threshold_K(p_hid_te, lam_tr, maxk), acc_te)
        ceil_i, lam_te = B.best_threshold_speedup(p_hid_te, acc_te, maxk, grid)

        span = oracle_i - fixed_i
        rows.append({
            "fold": i, "n_test_steps": len(te), "best_fixed_k": int(fixed_k_i),
            "fixed": fixed_i, "bayes_position": pos_i, "deployable": deploy_i,
            "ceiling": ceil_i, "oracle": oracle_i, "span": span,
            "lam_train": float(lam_tr), "lam_oracle": float(lam_te),
            "rec_deployable_pct": 100 * (deploy_i - fixed_i) / span if span > 0 else 0.0,
            "rec_ceiling_pct": 100 * (ceil_i - fixed_i) / span if span > 0 else 0.0,
            "threshold_loss": ceil_i - deploy_i,
            "position_selftest_gap": pos_i - fixed_i})

    def ms(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return float(v.mean()), float(v.std() / max(len(v) - 1, 1) ** 0.5)

    # ---- POSITIVE CONTROLS (must hold by construction) ----
    viol_nest = [r["fold"] for r in rows if r["threshold_loss"] < -1e-12]
    viol_pos = [r["fold"] for r in rows if r["position_selftest_gap"] > 1e-9]
    tl_m, tl_se = ms("threshold_loss")
    dep_m, dep_se = ms("rec_deployable_pct")
    cei_m, cei_se = ms("rec_ceiling_pct")

    print(f"  folds={len(rows)}")
    print(f"  CONTROL nesting (ceiling >= deployable in every fold): "
          f"{'PASS' if not viol_nest else f'FAIL {viol_nest}'}")
    print(f"  CONTROL Bayes(position) == best fixed K:               "
          f"{'PASS' if not viol_pos else f'FAIL {viol_pos}'}")
    print(f"  deployable recovery : {dep_m:+.1f} +- {dep_se:.1f}% of the oracle span")
    print(f"  ceiling    recovery : {cei_m:+.1f} +- {cei_se:.1f}%")
    print(f"  THRESHOLD-SELECTION LOSS (paired): {tl_m:+.5f} +- {tl_se:.5f} speedup pts "
          f"({100*tl_m/np.mean([r['span'] for r in rows]):+.1f}% of span)")
    return {"model": model_key, "n_folds": len(rows), "folds": rows,
            "controls": {"nesting_violations": viol_nest,
                         "position_selftest_violations": viol_pos},
            "deployable_recovery_pct": [round(dep_m, 2), round(dep_se, 2)],
            "ceiling_recovery_pct": [round(cei_m, 2), round(cei_se, 2)],
            "threshold_selection_loss_pts": [round(tl_m, 5), round(tl_se, 5)],
            "threshold_selection_loss_pct_of_span":
                round(100 * tl_m / float(np.mean([r["span"] for r in rows])), 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    ALL = ["llama8b", "qwen14b", "llama8b_reasoning", "deepseek",
           "deepseek_reasoning"]
    keys = [a.model] if a.model else ALL
    out = {}
    for k in keys:
        print(f"\n=== {k} ===")
        r = paired_ladder(k)
        if r:
            out[k] = r
    dst = os.path.join(OUT_DIR, "bayes_ceiling_paired.json")
    merged = {}
    if os.path.exists(dst):
        try:
            merged = json.load(open(dst))
        except Exception:
            merged = {}
    merged.update(out)
    out = merged
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)

    lines = ["# Bayes ladder — PAIRED WITHIN FOLD (corrected comparison)", "",
             "Both hidden-probe rungs scored on the SAME test rows in each fold, so the",
             "ceiling genuinely bounds the deployable probe and their difference is a paired",
             "per-fold statistic with a standard error.", "",
             "| model | deployable recovery | ceiling recovery | threshold-selection loss | nesting control |",
             "|---|---:|---:|---:|---|"]
    for k, r in out.items():
        d, ds = r["deployable_recovery_pct"]; c, cs = r["ceiling_recovery_pct"]
        t, ts = r["threshold_selection_loss_pts"]
        ok = "PASS" if not r["controls"]["nesting_violations"] else "FAIL"
        lines.append(f"| {k} | {d:+.1f} ±{ds:.1f}% | {c:+.1f} ±{cs:.1f}% | "
                     f"{t:+.4f} ±{ts:.4f} pts ({r['threshold_selection_loss_pct_of_span']:+.1f}% of span) | {ok} |")
    with open(os.path.join(OUT_DIR, "bayes_ceiling_paired.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote results/perstep_signal/bayes_ceiling_paired.{json,md}")
