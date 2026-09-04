"""Thesis experiment: load+content contextual draft-length control (simulator).

Runs the head-to-head that the novelty memo motivates, entirely in the load-aware
simulator (no GPU -> fits the no-cloud-budget constraint). For each
(seed, workload, batch B, policy) it streams N decode steps and records the headline
metric (accepted tokens / target step) and per-step wall-clock speedup.

Policies (the ablation is the SAME ContextLinUCB algorithm, differing only in features):
  fixed_1/2/4/8   raw fixed draft lengths (best-per-(workload,B) = honest baseline)
  content_only    ContextLinUCB(entropy, accept)         -- TapOut-family side
  load_only       ContextLinUCB(batch, load)             -- Nightjar-family side
  combined        ContextLinUCB(entropy, accept, batch, load)  -- THE THESIS
  nightjar        NightjarStyle (per-batch-bucket UCB)    -- published load-only baseline
  tapout          TapOutStyle (UCB over stop-rules)       -- published content-only baseline
  oracle          per-step argmax K over true step speedup -- ceiling

Output: results/load_metrics.csv  (one row per seed x workload x batch x policy)
        plus a printed summary. Run analyze_load_experiment.py for tables/CIs.
"""
import os
import sys
import csv
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from serve.load_simulator import (
    MAX_LEN, ContentStream, accepted_given_K, step_speedup, oracle_K,
    accepted_tokens_per_step,
)
from controllers import (ContextLinUCB, NightjarStyle, UCB, EpsilonGreedy,
                         LinUCBController)
from controllers.baselines_2025 import TapOutStyle, EXP3Spec
from controllers.oracle_baselines import calibrate

WORKLOADS = ["humaneval", "gsm8k", "mt_bench", "spec_bench"]
BATCHES = [1, 4, 8, 16, 32, 64]
ARMS = (1, 2, 4, 8)
METRICS_FILE = "results/load_metrics.csv"
DYN_FILE = "results/load_dynamic_metrics.csv"
HEADER = ["seed", "workload", "batch", "policy",
          "mean_speedup", "accepted_tokens_per_step", "mean_K", "n_steps"]
DYN_HEADER = ["seed", "workload", "policy",
              "mean_speedup", "accepted_tokens_per_step", "mean_K", "n_steps"]


from controllers import EntropyThreshold
from serve.load_simulator import spec_step_latency

_ORACLE_CACHE = {}   # workload -> (OracleLoad, OracleContent); calibration is B-independent


def _oracles(workload, rng):
    if workload not in _ORACLE_CACHE:
        _ORACLE_CACHE[workload] = calibrate(workload, BATCHES, ARMS,
                                            np.random.default_rng(hash(workload) % 99991), n=5000)
    return _ORACLE_CACHE[workload]


def make_policies(workload, rng):
    """Fresh policy instances. Includes the field-standard baseline set so the
    head-to-head matches what Nightjar/BanditSpec/TapOut/SVIP report."""
    oracle_load, oracle_content, oracle_loadcontent = _oracles(workload, rng)
    return {
        # fixed (Oracle-Task = best of these per workload)
        "fixed_1": 1, "fixed_2": 2, "fixed_4": 4, "fixed_8": 8,
        # content-only published baselines
        "svip": EntropyThreshold(tau=1.5, max_len=MAX_LEN),
        "tapout": TapOutStyle(max_len=MAX_LEN),
        # context-free bandits (BanditSpec) + classics
        "banditspec_ucb": UCB(c=0.5, arms=ARMS),
        "banditspec_exp3": EXP3Spec(arms=ARMS, seed=0),
        "eps_greedy": EpsilonGreedy(eps=0.1, arms=ARMS, seed=0),
        # plain content-only contextual bandit (Nightjar's own baseline)
        "linucb_content": LinUCBController(arms=ARMS, alpha=1.0),
        # load-only published baseline (tuned to converge)
        "nightjar": NightjarStyle(arms=ARMS, c=0.5),
        # strawman-proof single-sided CEILINGS + achievable joint ceiling
        "oracle_load": oracle_load,                # best fixed K per batch (content-blind)
        "oracle_content": oracle_content,          # best fixed K per entropy bin (batch-blind)
        "oracle_loadcontent": oracle_loadcontent,  # best K per (batch x entropy) — joint ceiling
        # capacity-matched ablations (each side gets its own interaction)
        "content_only": ContextLinUCB(arms=ARMS, features=("entropy", "accept"),
                                      interactions=(("entropy", "accept"),)),
        "load_only": ContextLinUCB(arms=ARMS, features=("batch", "load"),
                                   interactions=(("batch", "load"),)),
        # THE THESIS
        "combined": ContextLinUCB(
            arms=ARMS, features=("entropy", "accept", "batch", "load"),
            interactions=(("batch", "accept"), ("batch", "entropy"), ("load", "accept"))),
        # per-step ceiling (Oracle-Token)
        "oracle": "oracle",
    }


def choose_K(policy, entropies, accept_booleans, B):
    """Uniform K selection across the heterogeneous policy interfaces."""
    if isinstance(policy, int):
        return policy
    if policy == "oracle":
        return oracle_K(B, accept_booleans, ARMS)
    if hasattr(policy, "should_stop"):          # TapOut: draft to max, stop per token
        kmax = min(policy.choose(), MAX_LEN)
        K = 1
        for i in range(kmax):
            K = i + 1
            margin = max(0.0, 1.0 - entropies[i] / 4.0)
            if policy.should_stop(i, entropies[i], margin):
                break
        return K
    if hasattr(policy, "tau"):                   # SVIP: stop when entropy crosses tau
        return policy.choose(entropies)
    if hasattr(policy, "choose_k"):              # ContextLinUCB / Nightjar / Oracles
        return policy.choose_k(entropy=entropies[0], batch=B)
    if isinstance(policy, LinUCBController):      # plain content-only contextual
        return policy.choose(entropy=entropies[0])
    return policy.choose()                        # UCB / EXP3 / eps-greedy (context-free)


def update_policy(policy, K, accepted, B):
    if isinstance(policy, int) or policy == "oracle":
        return
    if not hasattr(policy, "update"):             # SVIP threshold has no update
        return
    spd = step_speedup(B, K, accepted)
    if hasattr(policy, "should_stop") or isinstance(policy, EXP3Spec):
        policy.update(K, accepted, cycle_time=spec_step_latency(B, K), K=K)
    elif isinstance(policy, (UCB, EpsilonGreedy)):
        policy.update(K, min(spd / 3.0, 1.0))     # 2-arg update(arm, reward)
    else:                                         # ContextLinUCB / Nightjar / LinUCB / Oracles
        policy.update(K, min(spd / 3.0, 1.0), accepted=accepted, K=K)  # reward ~[0,1]


def run_cell(workload, policy, B, n_steps, rng):
    if hasattr(policy, "set_max_steps"):
        policy.set_max_steps(n_steps)
    if hasattr(policy, "reset_episode"):
        policy.reset_episode()
    stream = ContentStream(workload)
    tot_acc, tot_spd, tot_K = 0, 0.0, 0
    for t in range(n_steps):
        entropies, accepts = stream.step(rng)
        K = min(MAX_LEN, max(1, choose_K(policy, entropies, accepts, B)))
        accepted = accepted_given_K(accepts, K)
        update_policy(policy, K, accepted, B)
        tot_acc += accepted
        tot_spd += step_speedup(B, K, accepted)
        tot_K += K
    return {
        "mean_speedup": tot_spd / n_steps,
        "accepted_tokens_per_step": accepted_tokens_per_step(tot_acc, n_steps),
        "mean_K": tot_K / n_steps,
        "n_steps": n_steps,
    }


def gen_load_schedule(n_steps, rng):
    """Piecewise-constant batch schedule (emulates time-varying serving load/QPS).

    Segments of random length carry a random batch level from BATCHES, so batch
    size varies WITHIN a single run -- the regime where a controller must read both
    load (which batch level we're in) and content (per-step difficulty).
    """
    sched, t = [], 0
    while t < n_steps:
        L = int(rng.integers(20, 60))
        B = int(rng.choice(BATCHES))
        sched.extend([B] * min(L, n_steps - t))
        t += L
    return sched[:n_steps]


def run_dynamic_cell(workload, policy, schedule, rng):
    if hasattr(policy, "set_max_steps"):
        policy.set_max_steps(len(schedule))
    if hasattr(policy, "reset_episode"):
        policy.reset_episode()
    stream = ContentStream(workload)
    tot_acc, tot_spd, tot_K = 0, 0.0, 0
    for B in schedule:
        entropies, accepts = stream.step(rng)
        K = min(MAX_LEN, max(1, choose_K(policy, entropies, accepts, B)))
        accepted = accepted_given_K(accepts, K)
        update_policy(policy, K, accepted, B)
        tot_acc += accepted
        tot_spd += step_speedup(B, K, accepted)
        tot_K += K
    n = len(schedule)
    return {"mean_speedup": tot_spd / n,
            "accepted_tokens_per_step": accepted_tokens_per_step(tot_acc, n),
            "mean_K": tot_K / n, "n_steps": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=400, help="steps per fixed-B sweep cell")
    ap.add_argument("--dyn-steps", type=int, default=3000, help="steps per dynamic-load run")
    ap.add_argument("--mode", choices=["sweep", "dynamic", "both"], default="both")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    # ── Experiment A: fixed-B sweep (MOTIVATION: optimal K shifts with B) ──
    if args.mode in ("sweep", "both"):
        with open(METRICS_FILE, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)
        for seed in range(args.seeds):
            for workload in WORKLOADS:
                for B in BATCHES:
                    policies = make_policies(workload, None)
                    for pname, policy in policies.items():
                        rng = np.random.default_rng(seed * 100003 + B * 17 + hash(workload) % 1000)
                        res = run_cell(workload, policy, B, args.steps, rng)
                        with open(METRICS_FILE, "a", newline="") as f:
                            csv.writer(f).writerow([seed, workload, B, pname,
                                                    round(res["mean_speedup"], 4),
                                                    round(res["accepted_tokens_per_step"], 4),
                                                    round(res["mean_K"], 3), res["n_steps"]])
            print(f"  [sweep] seed {seed} done")
        print(f"Wrote {METRICS_FILE}")

    # ── Experiment B: dynamic-load stream (HEADLINE: combined beats single-sided) ──
    if args.mode in ("dynamic", "both"):
        with open(DYN_FILE, "w", newline="") as f:
            csv.writer(f).writerow(DYN_HEADER)
        for seed in range(args.seeds):
            for workload in WORKLOADS:
                # one schedule per (seed,workload); all policies see the SAME load+content
                sched_rng = np.random.default_rng(seed * 7919 + hash(workload) % 1000)
                schedule = gen_load_schedule(args.dyn_steps, sched_rng)
                policies = make_policies(workload, None)
                for pname, policy in policies.items():
                    rng = np.random.default_rng(seed * 13 + hash(workload) % 1000)
                    res = run_dynamic_cell(workload, policy, schedule, rng)
                    with open(DYN_FILE, "a", newline="") as f:
                        csv.writer(f).writerow([seed, workload, pname,
                                                round(res["mean_speedup"], 4),
                                                round(res["accepted_tokens_per_step"], 4),
                                                round(res["mean_K"], 3), res["n_steps"]])
            print(f"  [dynamic] seed {seed} done")
        print(f"Wrote {DYN_FILE}")

    print("\nRun: python analyze_load_experiment.py")


if __name__ == "__main__":
    main()
