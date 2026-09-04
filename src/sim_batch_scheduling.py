"""H1 gate for the acceptance-aware batch-scheduling project: does grouping requests
by predicted acceptance (each batch runs its own optimal speculation length K) beat
acceptance-agnostic batching with a single global K? Measures compute-per-output-token
(lower = faster); the speedup of AWARE vs AGNOSTIC = the headroom acceptance-aware
scheduling can capture from batch interference.

Model (separate-draft SD, per request with per-token acceptance a, draft length K):
  MAT(a,K) = sum_{k=1..K} a^k + 1   (expected accepted run capped at K, + bonus token)
  compute/token(a,K) = (K*r + 1) / MAT(a,K)     [K draft fwds + 1 verify, target-fwd units]
  K*(a) = argmin_K compute/token   (per-request optimum; K=0 = no speculation = AR)
Policies: AGNOSTIC (one global K for the pool) | AWARE-oracle (true K*(a) per request) |
AWARE-tiered (K from a small tier set) | AWARE-estimated (tier from a NOISY acceptance estimate
calibrated to AUC~0.88, modeling the cheap predictor). Run: python src/sim_batch_scheduling.py
"""
import numpy as np

KS = [0, 1, 2, 3, 4, 6, 8]
TIERS = [0, 2, 4, 8]


def mat(a, K):
    if K == 0:
        return 1.0
    if a >= 1.0:
        return K + 1.0
    return a * (1 - a ** K) / (1 - a) + 1.0


def cpt(a, K, r):  # compute per output token (lower = faster)
    return (K * r + 1.0) / mat(a, K)


def best_K(a, r, ks):
    return min(ks, key=lambda K: cpt(a, K, r))


def best_global_K(accs, r, ks):
    return min(ks, key=lambda K: np.mean([cpt(a, K, r) for a in accs]))


def noisy_estimate(accs, target_auc=0.88, seed=0):
    """Add Gaussian noise to the rank-signal so that AUC(estimate, high/low split) ~ target.
    Calibrated empirically: AUC≈0.88 corresponds to noise sigma≈0.12 on [0,1] acceptance."""
    rng = np.random.default_rng(seed)
    sigma = 0.12
    return np.clip(np.array(accs) + rng.normal(0, sigma, len(accs)), 0.01, 0.99)


def evaluate(accs, r, label):
    accs = np.asarray(accs)
    Kg = best_global_K(accs, r, KS)
    agnostic = np.mean([cpt(a, Kg, r) for a in accs])
    oracle = np.mean([cpt(a, best_K(a, r, KS), r) for a in accs])
    tiered = np.mean([cpt(a, best_K(a, r, TIERS), r) for a in accs])
    est = noisy_estimate(accs)
    tier_est = np.mean([cpt(a, best_K(ahat, r, TIERS), r) for a, ahat in zip(accs, est)])
    print(f"\n[{label}]  r={r}  mean_a={accs.mean():.2f} std_a={accs.std():.2f}  global K*={Kg}")
    print(f"  agnostic (1 global K)     compute/tok={agnostic:.4f}")
    print(f"  AWARE oracle (K* per req) compute/tok={oracle:.4f}   gain={agnostic/oracle-1:+.1%}")
    print(f"  AWARE tiered {TIERS}        compute/tok={tiered:.4f}   gain={agnostic/tiered-1:+.1%}")
    print(f"  AWARE tiered + noisy est  compute/tok={tier_est:.4f}   gain={agnostic/tier_est-1:+.1%}")
    return agnostic / oracle - 1, agnostic / tier_est - 1


def main():
    rng = np.random.default_rng(0)
    n = 4000
    dists = {
        "narrow ~0.8 (homogeneous)": np.clip(rng.normal(0.80, 0.06, n), 0.3, 0.98),
        "wide (heterogeneous)": np.clip(rng.beta(2, 2, n) * 0.7 + 0.25, 0.25, 0.98),
        "bimodal code/chat (0.9 / 0.55)": np.where(rng.random(n) < 0.5,
                                                   rng.normal(0.90, 0.04, n),
                                                   rng.normal(0.55, 0.07, n)).clip(0.3, 0.98),
    }
    print("=" * 64)
    print("H1: acceptance-aware batch scheduling headroom (compute/token; speedup vs agnostic)")
    print("=" * 64)
    summary = {}
    for r in (0.1, 0.15, 0.3):
        for name, accs in dists.items():
            o, e = evaluate(accs, r, name)
            summary[(name, r)] = (o, e)

    print("\n" + "=" * 64)
    print("GATE (representative r=0.15):")
    for name in dists:
        o, e = summary[(name, 0.15)]
        print(f"  {name:32s} oracle +{o:.1%}  | tiered+noisy +{e:.1%}")
    best = max(summary[(n, 0.15)][1] for n in dists)
    print(f"\n  -> heterogeneous workloads: realistic (tiered+noisy) gain up to +{best:.1%}")
    print("     GO if >~10% on heterogeneous traffic; WEAK if homogeneous-only / small.")


if __name__ == "__main__":
    main()
