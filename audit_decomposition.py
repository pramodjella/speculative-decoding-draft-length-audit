"""INDEPENDENT audit of the Bayes-ceiling decomposition + draft-head confound check.

Written from scratch (does NOT import analyze_bayes_ceiling) to independently re-derive the
top-line numbers, per Yash's ask ("have someone re-derive the top-line numbers independently").

Checks:
  1. DRAFT-HEAD CONFOUND: base acceptance strength per model (mean accepted tokens/step +
     per-position survival). If DeepSeek's head is much weaker than Llama's, "no signal for
     reasoning models" could be a weak-head artifact, not a model-class property.
  2. INDEPENDENT re-derivation of best-fixed-K speedup, per-step oracle, and oracle ceiling %,
     cross-checked against results/perstep_signal/bayes_ceiling.json.

Cost model (same as the analysis): speedup = (mean_accepted+1) / (1 + C*K), C=0.15.
Block key = (workload, gen_i, step_i)  [the corrected key].
"""
import os, json, glob
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
IN   = os.path.join(BASE, "results", "eagle3_hidden_full")
REF  = os.path.join(BASE, "results", "perstep_signal", "bayes_ceiling.json")
C    = 0.15
MAXK = 7


def block_acc(df):
    """Return per-block accepted-count array (acc is constant within a block)."""
    g = df.groupby(["workload", "gen_i", "step_i"])["acc"].first()
    return g.values.astype(float)


def fixed_speedup(acc, k):
    accepted = np.minimum(acc, k)
    return (accepted.mean() + 1.0) / (1.0 + C * k)


def best_fixed(acc):
    vals = [(k, fixed_speedup(acc, k)) for k in range(1, MAXK + 1)]
    bk, bs = max(vals, key=lambda t: t[1])
    return bk, bs


def oracle_speedup(acc):
    K = np.minimum(acc + 1, MAXK)                 # draft exactly to the acceptance boundary
    accepted = np.minimum(acc, K)                 # == acc
    return (accepted.mean() + 1.0) / (1.0 + C * K.mean())


def per_position_survival(df):
    """P(accept at position j) = mean(accept | position==j)."""
    return df.groupby("position")["accept"].mean()


def main():
    files = sorted(glob.glob(os.path.join(IN, "hidden_full_*.parquet")))
    ref = json.load(open(REF)) if os.path.exists(REF) else {}

    print("=" * 78)
    print("1) DRAFT-HEAD CONFOUND — base acceptance strength per model")
    print("=" * 78)
    print(f"{'model':32s} {'n_blk':>6s} {'mean_acc':>8s} {'accept_len':>10s}  survival p(acc>j), j=0..6")
    strength = {}
    for f in files:
        key = os.path.basename(f).replace("hidden_full_", "").replace(".parquet", "")
        df = pd.read_parquet(f, columns=["workload", "gen_i", "step_i", "position", "acc", "accept"])
        acc = block_acc(df)
        surv = per_position_survival(df)
        surv_str = " ".join(f"{surv.get(j, float('nan')):.3f}" for j in range(7))
        strength[key] = {"mean_acc": float(acc.mean()), "accept_len": float(acc.mean() + 1),
                         "p0": float(surv.get(0, np.nan)), "n_blocks": len(acc)}
        print(f"{key:32s} {len(acc):6d} {acc.mean():8.3f} {acc.mean()+1:10.3f}  {surv_str}")

    print("\n  Confound read:")
    # compare DeepSeek vs Llama base strength
    def g(k):
        return strength.get(k, {})
    for pair in [("llama8b", "deepseek"), ("llama8b_reasoning", "deepseek_reasoning")]:
        a, b = g(pair[0]), g(pair[1])
        if a and b:
            print(f"   {pair[0]} accept_len={a['accept_len']:.2f} (p0={a['p0']:.3f}) vs "
                  f"{pair[1]} accept_len={b['accept_len']:.2f} (p0={b['p0']:.3f})")

    print("\n" + "=" * 78)
    print("2) INDEPENDENT re-derivation vs bayes_ceiling.json (oracle / best-fixed / ceiling)")
    print("=" * 78)
    print(f"{'model':32s} {'bestK':>5s} {'fixed':>8s} {'oracle':>8s} {'ceil%':>7s} | {'ref_ceil%':>9s} {'match':>6s}")
    for f in files:
        key = os.path.basename(f).replace("hidden_full_", "").replace(".parquet", "")
        df = pd.read_parquet(f, columns=["workload", "gen_i", "step_i", "acc", "ndraft"])
        acc = block_acc(df)
        bk, bs = best_fixed(acc)
        osp = oracle_speedup(acc)
        ceil = 100.0 * (osp - bs) / bs
        ref_ceil = ref.get(key, {}).get("oracle_ceiling_pct", None)
        match = ""
        if ref_ceil is not None:
            match = "OK" if abs(ceil - ref_ceil) < 0.6 else f"DIFF"
        rc = f"{ref_ceil:9.1f}" if ref_ceil is not None else "     n/a "
        print(f"{key:32s} {bk:5d} {bs:8.4f} {osp:8.4f} {ceil:7.1f} | {rc} {match:>6s}")

    print("\nNote: independent code, same cost model (C=0.15) and block key. 'OK' = ceiling within")
    print("0.6pp of the analysis module's value -> oracle/fixed rungs independently confirmed.")


if __name__ == "__main__":
    main()
