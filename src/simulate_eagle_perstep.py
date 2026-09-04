"""Offline PER-STEP controller simulation on EAGLE-3 traces.

Answers the one surviving controller question: does SVIP-style per-step entropy
stopping beat fixed K on EAGLE-3? (Per-request bandits already shown null, +2%
oracle ceiling -- see simulate_eagle_controllers.py.) Uses the REAL roadmap
EntropyThreshold (SVIP) controller from the codebase.

Chain-mode (eagle-topk=1) drafting is a prefix, so drafting K'<=max_k accepts
min(K', acc) -- this makes the offline sim EXACT without re-running.

Input: eagle3_perstep_llama8b.json
  {"max_k": 7,
   "gens": [ {"workload": w,
              "steps": [ {"ent": [e0, e1, ...], "acc": accepted_run_len}, ... ]},
             ... ]}
  ent[j] = draft-head entropy at draft position j; acc = accepted run length @ max_k.

Metric: Mean Accepted Tokens/step (MAT), wasted draft tokens per accepted, mean K,
and a cost-model net speedup MAT/(1 + c*mean_K) -- framework-independent, exactly
what SVIP/BanditSpec report (MAT + wall-time ratio). Set EAGLE_C to the measured
draft/target per-step cost ratio (EAGLE head is cheap, ~0.1-0.2).
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controllers import EntropyThreshold

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/eagle3_perstep_llama8b.json"


def _accumulate(gens, pick_K):
    """pick_K(step) -> K drafted that step. Returns (accepted, drafted, steps)."""
    acc = draft = steps = 0
    for g in gens:
        for s in g["steps"]:
            K = pick_K(s)
            r = s["acc"]
            acc += min(K, r); draft += K; steps += 1
    return acc, draft, steps


def report(name, acc, draft, steps, c):
    mat = acc / steps + 1.0
    waste = (draft - acc) / max(1, acc)
    mean_k = draft / steps
    speed = mat / (1.0 + c * mean_k)         # cost-model net speedup
    print(f"  {name:20s} MAT={mat:.3f}  wasted/acc={waste:.2f}  mean_K={mean_k:.2f}  "
          f"speedup~={speed:.3f}")
    return speed, mat


class MarginThreshold:
    """Per-step stop on the top1-top2 probability margin: draft while the draft
    head is CONFIDENT (margin high), stop when margin drops below tau (uncertain
    -> likely rejection). Complementary signal to entropy; we logged both."""
    def __init__(self, tau, max_len):
        self.tau = tau; self.max_len = max_len
    def choose(self, margins):
        n = 0
        for m in margins[:self.max_len]:
            if m is None or m < self.tau:
                break
            n += 1
        return max(1, n)


def main():
    d = json.load(open(PATH)); gens = d["gens"]; mk = d["max_k"]
    c = float(os.environ.get("EAGLE_C", "0.15"))
    nsteps = sum(len(g["steps"]) for g in gens)
    print(f"gens={len(gens)}  steps={nsteps}  max_k={mk}  draft/target cost c={c}\n")

    print("fixed K:")
    best = None
    for K in range(1, mk + 1):
        sp, _ = report(f"fixed K={K}", *_accumulate(gens, lambda s, K=K: K), c)
        if best is None or sp > best[1]:
            best = (K, sp)
    bk, bsp = best
    print(f"  -> best fixed K={bk} (speedup~{bsp:.3f})\n")

    # entropies are in BITS (vocab up to ~128k -> up to ~17). Sweep the full range
    # plus data-driven percentiles so we bracket the real operating point.
    import statistics as _st
    allent = [e for g in gens for s in g["steps"] for e in s["ent"] if e is not None]
    qs = sorted(allent)
    pct = [qs[int(p * (len(qs) - 1))] for p in (0.25, 0.5, 0.75, 0.9)] if qs else []
    print(f"entropy(bits): min={min(allent):.2f} med={_st.median(allent):.2f} "
          f"max={max(allent):.2f}  p25/50/75/90={[round(x,1) for x in pct]}\n")

    print("SVIP EntropyThreshold (tau sweep -- per-STEP entropy stop):")
    best_svip = None
    for tau in (2, 4, 6, 8, 9, 10, 11, 12, 13, 15):
        ctrl = EntropyThreshold(tau=tau, max_len=mk)
        sp, mat = report(f"SVIP tau={tau}", *_accumulate(gens, lambda s: ctrl.choose(s["ent"])), c)
        if best_svip is None or sp > best_svip[1]:
            best_svip = (tau, sp)
    # MARGIN signal (top1-top2 prob diff, range ~0..1) -- also captured per step.
    allmar = [m for g in gens for s in g["steps"] for m in s["margin"] if m is not None]
    mqs = sorted(allmar)
    mpct = [mqs[int(p * (len(mqs) - 1))] for p in (0.25, 0.5, 0.75, 0.9)] if mqs else []
    print(f"\nmargin: min={min(allmar):.3f} med={_st.median(allmar):.3f} "
          f"max={max(allmar):.3f}  p25/50/75/90={[round(x, 3) for x in mpct]}")
    print("MarginThreshold (per-STEP margin stop):")
    best_margin = None
    for tau in (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
        ctrl = MarginThreshold(tau=tau, max_len=mk)
        sp, mat = report(f"margin tau={tau}", *_accumulate(gens, lambda s: ctrl.choose(s["margin"])), c)
        if best_margin is None or sp > best_margin[1]:
            best_margin = (tau, sp)

    print()
    osp, _ = report("ORACLE per-step", *_accumulate(gens, lambda s: s["acc"] or 1), c)

    print("\nVERDICT:")
    gsvip = (best_svip[1] / bsp - 1) * 100
    gmar = (best_margin[1] / bsp - 1) * 100
    ceil = (osp / bsp - 1) * 100
    print(f"  best SVIP-entropy (tau={best_svip[0]}) vs best fixed K={bk}:  {gsvip:+.1f}%")
    print(f"  best MARGIN       (tau={best_margin[0]}) vs best fixed K={bk}:  {gmar:+.1f}%")
    print(f"  per-step oracle ceiling vs best fixed:  +{ceil:.1f}%")
    print(f"  (SVIP paper got ~+13% on EAGLE-2 from this within-stream lever.)")


if __name__ == "__main__":
    main()
