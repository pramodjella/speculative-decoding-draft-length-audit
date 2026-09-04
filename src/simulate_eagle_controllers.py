"""Offline per-request controller simulation on EAGLE-3 / Llama-3.1-8B traces.

vLLM 0.23.0 has no in-engine hook to vary EAGLE-3's K per step, so we captured
EAGLE-3 acceptance + latency at each fixed K per prompt (modal_eagle3_traces.py)
and ask OFFLINE: does a per-request adaptive-K controller beat fixed K=3?

Input : eagle3_multik_llama8b.json  (rows: {workload, idx, K, accept_len, latency})
Usage : python src/simulate_eagle_controllers.py [path_to_json]

Each controller processes prompts in arrival order, picks K from prior-prompt
reward history (per-REQUEST, reactive), then "realizes" that prompt's captured
accept_len + latency at the chosen K. Reported vs every fixed K and a per-request
ORACLE (picks the best K per prompt = cheating upper bound).
"""
import json, sys, os, random
from collections import defaultdict
import statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Real roadmap controllers from the codebase (acceptance-driven ones are testable
# on EAGLE-3 traces; EntropyThreshold/LinUCB need per-step draft entropy that vLLM
# EAGLE-3 does not expose, so they are NOT included here).
from controllers import UCB, EpsilonGreedy, AcceptanceHistoryController

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/eagle3_multik_llama8b.json"


def load(path):
    d = json.load(open(path))
    ks = d["ks"]
    per = defaultdict(dict)   # (w,idx) -> {K: {"al":accept_len, "lat":latency}}
    base = {}                 # (w,idx) -> baseline latency
    for r in d["rows"]:
        key = (r["workload"], r["idx"])
        if r["K"] == 0:
            base[key] = r["latency"]
        elif r["accept_len"] is not None:
            per[key][r["K"]] = {"al": r["accept_len"], "lat": r["latency"]}
    keys = [k for k in per if k in base and all(K in per[k] for K in ks)]
    return ks, base, per, keys


# ---------------- policies ----------------
# Real codebase controllers (UCB/EpsilonGreedy/AcceptanceHistoryController) are
# imported above. Fixed is a trivial baseline. They run PER-REQUEST here (one K
# per prompt) because vLLM EAGLE-3 traces are per-prompt aggregates, not per-step.
class Fixed:
    def __init__(self, k): self.k = k
    def choose(self): return self.k
    def update(self, *a): pass

def speedup(base, per, key, k):
    return base[key] / per[key][k]["lat"]

def run_policy(make, base, per, keys, update_kind="reward", seeds=5):
    """Per-request sim: choose K per prompt, realize that prompt's captured
    speedup + accept_len at chosen K. Averaged over random arrival orders.
    update_kind: 'reward' (bandits) | 'accepted' (history) | 'none' (fixed)."""
    sp_runs, al_runs = [], []
    for s in range(seeds):
        order = list(keys); random.Random(s).shuffle(order)
        ctrl = make()
        tot_base = tot_spec = 0.0; als = []
        for key in order:
            k = ctrl.choose()
            cell = per[key][k]
            if update_kind == "reward":
                ctrl.update(k, base[key] / cell["lat"])     # bandits maximize net speedup
            elif update_kind == "accepted":
                ctrl.update(k, cell["al"] - 1.0)            # accepted draft tokens/step
            tot_base += base[key]; tot_spec += cell["lat"]; als.append(cell["al"])
        sp_runs.append(tot_base / tot_spec); al_runs.append(st.mean(als))
    return st.mean(sp_runs), st.mean(al_runs)

def oracle(ks, base, per, keys):
    tot_base = tot_spec = 0.0; als = []
    for key in keys:
        k = max(ks, key=lambda k: speedup(base, per, key, k))
        tot_base += base[key]; tot_spec += per[key][k]["lat"]; als.append(per[key][k]["al"])
    return tot_base / tot_spec, st.mean(als)


def main():
    ks, base, per, keys = load(PATH)
    print(f"loaded {len(keys)} prompts with full K-coverage; K={ks}\n")

    # per-K acceptance + aggregate speedup (sanity / the fixed-K curve)
    print("fixed-K reference (aggregate over all prompts):")
    best_fixed = None
    for k in ks:
        sp, al = run_policy(lambda k=k: Fixed(k), base, per, keys, update_kind="none", seeds=1)
        tag = ""
        if best_fixed is None or sp > best_fixed[1]:
            best_fixed = (k, sp)
        print(f"  K={k}:  net_speedup={sp:.3f}  accept_len={al:.3f}")
    bk, bsp = best_fixed
    print(f"  -> best fixed K={bk} at {bsp:.3f}x\n")

    print("controllers (mean over 5 arrival orders):")
    rows = [
        ("Fixed K=3", lambda: Fixed(3), "none"),
        (f"Fixed K={bk} (best)", lambda: Fixed(bk), "none"),
        ("AcceptanceHistory", lambda: AcceptanceHistoryController(arms=tuple(ks)), "accepted"),
        ("UCB (BanditSpec)", lambda: UCB(arms=tuple(ks), c=1.0), "reward"),
        ("EpsilonGreedy", lambda: EpsilonGreedy(arms=tuple(ks), eps=0.1), "reward"),
    ]
    for name, make, kind in rows:
        sp, al = run_policy(make, base, per, keys, update_kind=kind)
        gap = (sp / bsp - 1) * 100
        print(f"  {name:18s} net_speedup={sp:.3f}  accept_len={al:.3f}  "
              f"gap_vs_bestfixed={gap:+.1f}%")
    osp, oal = oracle(ks, base, per, keys)
    print(f"  {'ORACLE (per-req)':18s} net_speedup={osp:.3f}  accept_len={oal:.3f}  "
          f"ceiling=+{(osp/bsp-1)*100:.1f}%")

    print("\nVERDICT:")
    print(f"  controller headroom = oracle - best_fixed = +{(osp/bsp-1)*100:.1f}% (the MOST a")
    print(f"  per-request controller could win). If small, fixed K={bk} is the recipe on EAGLE-3.")


if __name__ == "__main__":
    main()
