"""VERIFY the EAGLE-3 per-step traces, then EXPLORE controllers that might beat
the +2.5% entropy/margin threshold rules toward the +21.5% per-step oracle ceiling.

Verify:  per-step ent-list length vs max_k, acc<=ndraft, acc<=len(ent), and that
         the MAT matches the counter-based acceptance (~2.0) -> trace integrity.
Explore: (1) Persistence/EMA of recent accept-run (exploits within-stream
         autocorrelation -- a signal entropy/margin can't see); (2) lag-1 oracle
         (K=prev acc) to bound what autocorrelation alone can give; (3) combined
         entropy-OR-margin stop. Reports lag-1 autocorrelation of accept-run.

Usage: python src/verify_explore_perstep.py results/eagle3_perstep_llama8b.json
"""
import json, sys, os
import statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controllers import EntropyThreshold

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/eagle3_perstep_llama8b.json"
C = float(os.environ.get("EAGLE_C", "0.15"))


def load():
    d = json.load(open(PATH)); return d["gens"], d["max_k"]


def verify(gens, mk):
    print("=== VERIFY ===")
    lens, bad_len, bad_acc_ndraft, bad_acc_ent, accs = [], 0, 0, 0, []
    for g in gens:
        for s in g["steps"]:
            L = len(s["ent"]); lens.append(L); accs.append(s["acc"])
            if L != mk:
                bad_len += 1
            if "ndraft" in s and s["acc"] > s["ndraft"]:
                bad_acc_ndraft += 1
            if s["acc"] > L:
                bad_acc_ent += 1
    n = len(lens)
    from collections import Counter
    print(f"  steps={n}  ent-list len: {dict(Counter(lens).most_common(6))}")
    print(f"  steps with len(ent)!=max_k({mk}): {bad_len} ({100*bad_len/n:.1f}%)  "
          f"[expected: ~1 per gen = prefill leak]")
    print(f"  acc>ndraft anomalies: {bad_acc_ndraft}   acc>len(ent) anomalies: {bad_acc_ent}")
    mat = st.mean(accs) + 1.0
    print(f"  mean accept_len (acc/step + 1) = {mat:.3f}  [counter-based earlier: ~2.0]")
    # acc distribution
    from collections import Counter as C2
    print(f"  accept-run dist: {dict(sorted(C2(accs).items()))}")
    return mat


def autocorr(gens):
    pairs = []
    for g in gens:
        a = [s["acc"] for s in g["steps"]]
        for i in range(1, len(a)):
            pairs.append((a[i-1], a[i]))
    if len(pairs) < 3:
        return float("nan")
    xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x-mx)*(y-my) for x, y in pairs)
    den = (sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys)) ** 0.5
    return num/den if den else float("nan")


def report(name, acc, draft, steps):
    mat = acc/steps + 1.0; mean_k = draft/steps
    sp = mat/(1.0 + C*mean_k)
    print(f"  {name:26s} MAT={mat:.3f}  mean_K={mean_k:.2f}  speedup~={sp:.3f}")
    return sp


def run_stateful(gens, make):
    """Controller has choose()->K and update(acc); reset per generation."""
    acc = draft = steps = 0
    for g in gens:
        ctrl = make()
        for s in g["steps"]:
            K = max(1, min(len(s["ent"]) or 1, ctrl.choose()))
            r = s["acc"]
            acc += min(K, r); draft += K; steps += 1
            ctrl.update(r)
    return acc, draft, steps


def run_stateless(gens, pick):
    acc = draft = steps = 0
    for g in gens:
        for s in g["steps"]:
            K = pick(s); r = s["acc"]
            acc += min(K, r); draft += K; steps += 1
    return acc, draft, steps


class Persistence:
    """K_t from EMA of recent accept-runs (+1 lookahead). Exploits autocorrelation."""
    def __init__(self, maxk, alpha, bias=1):
        self.maxk = maxk; self.alpha = alpha; self.ema = 2.0; self.bias = bias
    def choose(self):
        return max(1, min(self.maxk, round(self.ema) + self.bias))
    def update(self, acc):
        self.ema = (1-self.alpha)*self.ema + self.alpha*acc


def learned_stopper(gens, mk):
    """Logistic P(accept at pos j | entropy_j, margin_j, j, prev_acc), trained on
    half the gens, evaluated on the other half. Stop drafting when P < threshold.
    Reported against best-fixed computed on the SAME held-out test gens (fair)."""
    import numpy as np
    half = len(gens) // 2
    train, test = gens[:half], gens[half:]

    def rows(split):
        X, y = [], []
        for g in split:
            prev = 2
            for s in g["steps"]:
                L = min(mk, len(s["ent"]))
                for j in range(L):
                    e, m = s["ent"][j], s["margin"][j]
                    if e is None or m is None:
                        continue
                    X.append([e, m, j, prev]); y.append(1.0 if s["acc"] > j else 0.0)
                prev = s["acc"]
        return np.array(X, float), np.array(y, float)

    Xtr, ytr = rows(train)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xn = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    w = np.zeros(Xn.shape[1])
    for _ in range(400):                       # batch GD
        p = 1.0 / (1.0 + np.exp(-Xn @ w))
        w -= 0.2 * (Xn.T @ (p - ytr)) / len(ytr)

    def pcont(e, m, j, prev):
        x = np.append((np.array([e, m, j, prev]) - mu) / sd, 1.0)
        return 1.0 / (1.0 + np.exp(-x @ w))

    # best fixed on the SAME test gens (fair baseline)
    def fixed_sp(k):
        a, d, st_ = run_stateless(test, lambda s, k=k: k)
        mat = a / st_ + 1.0; mk_ = d / st_; return mat / (1.0 + C * mk_)
    bf = max(fixed_sp(k) for k in range(1, mk + 1))

    best = None
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        acc = draft = steps = 0
        for g in test:
            prev = 2
            for s in g["steps"]:
                K = 0
                for j in range(min(mk, len(s["ent"]))):
                    e, m = s["ent"][j], s["margin"][j]
                    if e is None or m is None:
                        break
                    if pcont(e, m, j, prev) < thr:
                        break
                    K += 1
                K = max(1, K); r = s["acc"]
                acc += min(K, r); draft += K; steps += 1
                prev = r
        mat = acc / steps + 1.0; mk_ = draft / steps; sp = mat / (1.0 + C * mk_)
        print(f"  learned thr={thr}          MAT={mat:.3f}  mean_K={mk_:.2f}  speedup~={sp:.3f}")
        if best is None or sp > best:
            best = sp
    print(f"  [held-out test: best fixed speedup={bf:.3f}; learned best={best:.3f} "
          f"-> {(best/bf-1)*100:+.1f}% on test]")
    return best, bf


def main():
    gens, mk = load()
    verify(gens, mk)

    ac = autocorr(gens)
    print(f"\n=== EXPLORE ===  lag-1 autocorrelation of accept-run = {ac:.3f}")

    # baselines
    best_fixed = max((report(f"fixed K={k}", *run_stateless(gens, lambda s, k=k: k))
                      for k in range(1, mk+1)))
    print()

    # (1) persistence / EMA of accept-run (sweep alpha + lookahead bias)
    best_pers = None
    for alpha in (0.3, 0.5, 0.8):
        for bias in (0, 1):
            sp = report(f"persist a={alpha} b={bias}",
                        *run_stateful(gens, lambda alpha=alpha, bias=bias: Persistence(mk, alpha, bias)))
            best_pers = sp if best_pers is None else max(best_pers, sp)

    # (2) lag-1 oracle (K = previous step's acc) -- bound of pure autocorrelation
    class Lag1:
        def __init__(self): self.prev = 2
        def choose(self): return max(1, self.prev)
        def update(self, acc): self.prev = acc
    sp_lag = report("lag1-oracle (K=prev acc)", *run_stateful(gens, lambda: Lag1()))

    # (3) combined entropy-OR-margin stop (stop if entropy>te OR margin<tm)
    def combined(s, te=6, tm=0.3):
        n = 0
        for e, m in zip(s["ent"][:mk], s["margin"][:mk]):
            if (e is not None and e > te) or (m is not None and m < tm):
                break
            n += 1
        return max(1, n)
    sp_comb = report("entropy-OR-margin", *run_stateless(gens, combined))

    # (4) LEARNED per-position stopper: logistic(entropy, margin, position, prev_acc)
    #     -> P(accept at position j); stop when it drops. Train/test split by gen.
    sp_learned, fixed_on_test = learned_stopper(gens, mk)

    # per-step oracle ceiling
    sp_oracle = report("ORACLE per-step", *run_stateless(gens, lambda s: s["acc"] or 1))

    print("\n=== VERDICT (vs best fixed) ===")
    for nm, sp in [("persistence(best)", best_pers), ("lag1-oracle", sp_lag),
                   ("entropy-OR-margin", sp_comb), ("per-step ORACLE", sp_oracle)]:
        print(f"  {nm:20s} {(sp/best_fixed-1)*100:+.1f}%")
    print("  (entropy alone +2.4%, margin alone +2.5% from simulate_eagle_perstep.py)")


if __name__ == "__main__":
    main()
