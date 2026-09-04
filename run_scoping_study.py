"""Scoping study: WHEN does adaptive (and joint load+content) draft-length control pay?

The honest reframed contribution. Rather than claim a winning controller, we characterize
the headroom for adaptive draft length across regimes, using the oracle decomposition:

  Oracle-Task        best fixed K per workload         (the honest fixed bar)
  Oracle-Load        best K per batch  (content-blind)  (ceiling of load-only methods)
  Oracle-Content     best K per entropy bin (batch-blind)(ceiling of content-only methods)
  Oracle-Joint       best K per (batch x entropy), expected (achievable joint ceiling)
  Oracle-Token       best K per step, realized          (absolute ceiling)

Derived per-regime metrics:
  adaptive_headroom = Oracle-Joint / Oracle-Task - 1     (is adaptation worth anything?)
  load_value        = Oracle-Load / Oracle-Task - 1      (does load-awareness help?)
  content_value     = Oracle-Content / Oracle-Task - 1   (does content-awareness help?)
  complementarity   = Oracle-Joint / max(Load,Content)-1 (are the axes complementary?)
  online_capture    = (combined - Task) / (Joint - Task) (what an online learner realizes)

Swept axes: draft cost r=M_D/M_T (cheap EAGLE-ish .. expensive small-pair), load regime
(static low / static high / dynamic), and workload (easy code .. hard chat).
No GPU. Output: results/scoping_metrics.csv + results/scoping_findings.md.
"""
import os, sys, csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import serve.load_simulator as LS
from controllers import ContextLinUCB, NightjarStyle, LinUCBController
from controllers.oracle_baselines import calibrate
import run_load_experiment as R

WORKLOADS = ["humaneval", "gsm8k", "mt_bench", "spec_bench"]
ARMS = (1, 2, 4, 8)
R_VALUES = [0.1, 0.2, 0.5, 0.85]      # cheap EAGLE-ish ... expensive small-pair (measured 0.85)
LOAD_REGIMES = {
    "static_low": [1, 1, 1],          # batch 1 (latency regime)
    "static_high": [32, 32, 32],      # heavy load
    "dynamic": None,                  # time-varying 1..64
}
SEEDS = 4
STEPS = 3000


def eval_policy(workload, policy, schedule, seed):
    rng = np.random.default_rng(seed * 13 + hash(workload) % 1000)
    return R.run_dynamic_cell(workload, policy, schedule, rng)["mean_speedup"]


def schedule_for(regime, seed):
    if regime == "dynamic":
        return R.gen_load_schedule(STEPS, np.random.default_rng(seed * 7919 + 7))
    base = LOAD_REGIMES[regime][0]
    return [base] * STEPS


def batches_for(regime):
    return R.BATCHES if regime == "dynamic" else [LOAD_REGIMES[regime][0]]


def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    for w in WORKLOADS:
        for r in R_VALUES:
            LS.M_D = r * LS.M_T                      # set draft cost
            for regime in LOAD_REGIMES:
                batches = batches_for(regime)
                # calibrate regime-specific oracles (depend on r and batch set)
                ol, oc, olc = calibrate(w, batches, ARMS,
                                        np.random.default_rng(hash((w, r, regime)) % 99991), n=4000)
                per_seed = {k: [] for k in
                            ["fixed_1", "fixed_2", "fixed_4", "fixed_8",
                             "oracle_load", "oracle_content", "oracle_joint", "oracle_token",
                             "combined", "nightjar", "linucb_content"]}
                for seed in range(SEEDS):
                    sched = schedule_for(regime, seed)
                    pols = {
                        "fixed_1": 1, "fixed_2": 2, "fixed_4": 4, "fixed_8": 8,
                        "oracle_load": ol, "oracle_content": oc, "oracle_joint": olc,
                        "oracle_token": "oracle",
                        "combined": ContextLinUCB(arms=ARMS,
                            features=("entropy", "accept", "batch", "load"),
                            interactions=(("batch", "accept"), ("batch", "entropy"), ("load", "accept"))),
                        "nightjar": NightjarStyle(arms=ARMS, c=0.5),
                        "linucb_content": LinUCBController(arms=ARMS, alpha=1.0),
                    }
                    for name, pol in pols.items():
                        per_seed[name].append(eval_policy(w, pol, sched, seed))
                m = {k: float(np.mean(v)) for k, v in per_seed.items()}
                task = max(m["fixed_1"], m["fixed_2"], m["fixed_4"], m["fixed_8"])
                single = max(m["oracle_load"], m["oracle_content"])
                row = {
                    "workload": w, "r": r, "load_regime": regime,
                    "oracle_task": round(task, 4),
                    "oracle_load": round(m["oracle_load"], 4),
                    "oracle_content": round(m["oracle_content"], 4),
                    "oracle_joint": round(m["oracle_joint"], 4),
                    "oracle_token": round(m["oracle_token"], 4),
                    "combined": round(m["combined"], 4),
                    "best_single_learned": round(max(m["nightjar"], m["linucb_content"]), 4),
                    "adaptive_headroom_%": round((m["oracle_joint"] / task - 1) * 100, 1),
                    "load_value_%": round((m["oracle_load"] / task - 1) * 100, 1),
                    "content_value_%": round((m["oracle_content"] / task - 1) * 100, 1),
                    "complementarity_%": round((m["oracle_joint"] / single - 1) * 100, 1),
                    "online_capture_%": round((m["combined"] - task) /
                                              (m["oracle_joint"] - task + 1e-9) * 100, 0),
                }
                rows.append(row)
                print(f"{w:11s} r={r:<4} {regime:11s} | adapt_headroom={row['adaptive_headroom_%']:>5}% "
                      f"load={row['load_value_%']:>5}% content={row['content_value_%']:>5}% "
                      f"complement={row['complementarity_%']:>4}% online_cap={row['online_capture_%']:>4}%")
            LS.M_D = 5.0  # restore

    with open("results/scoping_metrics.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    print("\nWrote results/scoping_metrics.csv")


if __name__ == "__main__":
    main()
