"""Ground the per-step cost model in MEASURED throughput, then re-run the Bayes-ceiling
recovery with the calibrated cost constant.

The proxy cost model speedup = MAT/(1+C*K) used C=0.15 by assumption. But the EAGLE-3
fixed-K sweep (results/eagle3_8b/fixedK_by_workload.csv) gives, per K, the MEASURED
net_speedup and mean acceptance length on H100/vLLM. Since a fixed-K run's net speedup
already embeds the real draft+verify+CUDA-graph cost, we can fit C from it:

    net_speedup(K) ~= accept_len(K) / (1 + C * K)      =>   C = (accept_len/net_speedup - 1)/K

We fit one C (least squares over all K) and re-run the decomposition with it.

IMPORTANT (state in paper): this grounds the *cost per drafted token* in measured fixed-K
throughput, but still ASSUMES the adaptive policy pays no extra per-step overhead beyond
fixed-K drafting. A true in-loop probe can only be SLOWER (CUDA-graph switching), so this
is an upper bound on the deployable gain — the honest ceiling.
"""
import os, json
import numpy as np
import pandas as pd
import analyze_bayes_ceiling as A

BASE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(BASE, "results", "eagle3_8b", "fixedK_by_workload.csv")


def fit_C(df, scope="MIXED"):
    """Fit net_speedup(K) = accept_len(K)/(1+C*K) for one scope; return C and per-K residuals."""
    sub = df[df["scope"] == scope].sort_values("K")
    K   = sub["K"].values.astype(float)
    acc = sub["accept_len"].values.astype(float)
    net = sub["net_speedup"].values.astype(float)
    # per-point C, then LS over grid for the single best constant
    grid = np.linspace(0.01, 0.30, 2901)
    best_C, best_err = grid[0], 1e9
    for C in grid:
        pred = acc / (1 + C * K)
        err  = float(np.mean((pred - net) ** 2))
        if err < best_err:
            best_err, best_C = err, C
    pred = acc / (1 + best_C * K)
    return best_C, K, net, pred, np.sqrt(best_err)


def main():
    df = pd.read_csv(CSV)
    print("scopes:", sorted(df["scope"].unique()))
    print("\n=== Fit measured cost constant C (net = accept_len/(1+C*K)) ===")
    per_scope = {}
    for scope in ["gsm8k", "humaneval", "mt_bench", "MIXED"]:
        if scope not in df["scope"].values:
            continue
        C, K, net, pred, rmse = fit_C(df, scope)
        per_scope[scope] = C
        print(f"  {scope:10s}: C={C:.4f}  rmse={rmse:.4f}")
        for k, n, p in zip(K, net, pred):
            print(f"      K={int(k)}: measured {n:.3f}  fit {p:.3f}")
    C_mixed = per_scope.get("MIXED", np.mean(list(per_scope.values())))
    print(f"\n>>> Using MEASURED C = {C_mixed:.4f} (was 0.15 proxy) for llama8b re-grounding")

    # re-run llama8b decomposition with measured C
    A.C = float(C_mixed)
    print("\n=== Bayes-ceiling recovery with MEASURED cost constant (llama8b) ===")
    out = A.analyze_model("llama8b")
    if out:
        sp = out["speedups"]; rec = out["recovery_pct"]
        print(json.dumps({"C_measured": round(C_mixed, 4),
                          "speedups": sp, "recovery_pct": rec,
                          "oracle_ceiling_pct": out["oracle_ceiling_pct"]}, indent=2))
        os.makedirs(os.path.join(BASE, "results", "perstep_signal"), exist_ok=True)
        with open(os.path.join(BASE, "results", "perstep_signal",
                               "measured_cost_ground_llama8b.json"), "w") as f:
            json.dump({"C_measured": C_mixed, "per_scope_C": per_scope,
                       "result": out}, f, indent=2)


if __name__ == "__main__":
    main()
