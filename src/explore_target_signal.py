"""Does the TARGET model's verification entropy predict per-step acceptance
better than DRAFT-side signals? If yes -> a novel target-aware gating controller.
If no -> reinforces 'signal is the bottleneck'.

Input: eagle3_perstep_target_llama8b.json (steps carry ent[], margin[], ent_t[], acc).
  ent_t[j] = TARGET entropy at draft position j (verifier's own uncertainty).
  ent_t[-1] = target entropy at the bonus position = predicts NEXT step's 1st accept
              (this is CAUSAL: known before next step drafts).

Reports: correlation of draft-ent[0] vs acc, target-ent[0] vs acc (informativeness),
and a CAUSAL target-gate controller using the previous step's bonus entropy.
"""
import json, sys, os
import statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/eagle3_perstep_target_llama8b.json"
C = float(os.environ.get("EAGLE_C", "0.15"))


def pearson(xy):
    xs = [a for a, b in xy]; ys = [b for a, b in xy]
    if len(xs) < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in xy)
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


def quartile_table(name, pairs):
    """mean acc within quartiles of the signal."""
    pairs = sorted(pairs)
    n = len(pairs); q = n // 4
    print(f"  {name}: mean accept-run by signal quartile (low->high signal):")
    for i, lbl in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        seg = pairs[i*q:(i+1)*q] if i < 3 else pairs[3*q:]
        accs = [b for a, b in seg]
        sig = [a for a, b in seg]
        print(f"     {lbl}: signal[{min(sig):.2f}..{max(sig):.2f}]  mean_acc={st.mean(accs):.3f}")


def main():
    d = json.load(open(PATH)); gens = d["gens"]; mk = d["max_k"]
    print(f"gens={len(gens)} max_k={mk}  n_tlogits hook fired: "
          f"{d.get('n_tlogits')}  (target signal present={d.get('n_tlogits',0)>0})\n")

    # gather per-step signals
    d0_acc, t0_acc, prevtlast_acc = [], [], []
    coverage = 0; total = 0
    for g in gens:
        prev_tlast = None
        for s in g["steps"]:
            total += 1
            acc = s["acc"]
            if s["ent"] and s["ent"][0] is not None:
                d0_acc.append((s["ent"][0], acc))
            et = s.get("ent_t") or []
            if et:
                coverage += 1
                if et[0] is not None:
                    t0_acc.append((et[0], acc))
                if prev_tlast is not None:
                    prevtlast_acc.append((prev_tlast, acc))
                prev_tlast = et[-1]
    print(f"target-entropy coverage: {coverage}/{total} steps have ent_t\n")

    print("=== CORRELATION with accept-run (more negative = signal predicts acceptance) ===")
    print(f"  draft entropy[0]      vs acc:  r = {pearson(d0_acc):+.3f}")
    print(f"  target entropy[0]     vs acc:  r = {pearson(t0_acc):+.3f}   (non-causal informativeness)")
    print(f"  prev target ent[-1]   vs acc:  r = {pearson(prevtlast_acc):+.3f}   (CAUSAL: usable signal)\n")

    quartile_table("draft  entropy[0]", d0_acc)
    quartile_table("target entropy[0]", t0_acc)

    # CAUSAL target-gate controller: K from previous step's bonus entropy
    def run(make):
        acc = draft = steps = 0
        for g in gens:
            ctrl = make()
            for s in g["steps"]:
                K = max(1, min(len(s["ent"]) or 1, ctrl.choose()))
                r = s["acc"]; acc += min(K, r); draft += K; steps += 1
                et = s.get("ent_t") or []
                ctrl.update(et[-1] if et else None)
        mat = acc / steps + 1.0; mk_ = draft / steps
        return mat, mk_, mat / (1.0 + C * mk_)

    def fixed_sp(k):
        a = d_ = s_ = 0
        for g in gens:
            for s in g["steps"]:
                a += min(k, s["acc"]); d_ += k; s_ += 1
        mat = a / s_ + 1.0; return mat / (1.0 + C * (d_ / s_))
    bf = max(fixed_sp(k) for k in range(1, mk + 1))

    class TargetGate:
        """Low prev-bonus target entropy (confident next token) -> draft deep."""
        def __init__(self, tau, deep, shallow=1):
            self.tau = tau; self.deep = deep; self.shallow = shallow; self.pte = None
        def choose(self):
            if self.pte is None: return 2
            return self.deep if self.pte < self.tau else self.shallow
        def update(self, te):
            if te is not None: self.pte = te

    print("\n=== CAUSAL target-gate controller (prev bonus entropy -> K) ===")
    best = None
    for tau in (2, 4, 6, 8, 10):
        for deep in (3, 5, 7):
            mat, mk_, sp = run(lambda tau=tau, deep=deep: TargetGate(tau, deep))
            if best is None or sp > best[2]:
                best = (tau, deep, sp, mat, mk_)
    tau, deep, sp, mat, mk_ = best
    print(f"  best target-gate: tau={tau} deep={deep}  MAT={mat:.3f} mean_K={mk_:.2f} "
          f"speedup~={sp:.3f}  -> {(sp/bf-1)*100:+.1f}% vs best fixed")
    print(f"  (draft-side cheap controllers all capped ~+3%; per-step oracle +21.5%)")


if __name__ == "__main__":
    main()
