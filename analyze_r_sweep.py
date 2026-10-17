"""Offline r-sensitivity analysis of adaptive-draft-length headroom.

Consumes oracle_ceiling.json (with per-prompt match-run traces captured at
max_k=32) and answers, WITHOUT any GPU: at what draft/verify cost ratio r and
action-space width max_k does a perfect per-step K controller beat the best
constant K?

Method (exact, since greedy acceptance is position-local):
  1. Reconstruct the per-position accept-bit sequence {a_i} from the traces.
  2. Cost model: a step that drafts K and accepts `acc` leading bits commits
     acc+1 tokens for cost (K*r + 1) verify-units. Plain AR => throughput 1.0,
     so throughput T(policy, r) IS the speedup vs autoregressive.
  3. Fixed-K: simulate each constant K, take the best.
  4. Oracle: shortest-path DP over positions => provably optimal per-step K
     (true ceiling; resolves the earlier wall-clock replay's step-misalignment).
  5. Headroom = T_oracle / T_bestfixed - 1, swept over r x max_k.

Run:  python analyze_r_sweep.py            # expects ./oracle_ceiling.json
"""
import json, sys, math

PATH = sys.argv[1] if len(sys.argv) > 1 else "oracle_ceiling.json"
R_SWEEP = [0.50, 0.30, 0.20, 0.10, 0.05, 0.02]
MAXK_SWEEP = [8, 16, 32]
PROBE_K = 32


def bits_from_trace(trace):
    """Reconstruct accept-bit sequence from per-step match-run lengths (max_k=PROBE_K).
    m < PROBE_K  -> m accepts then a reject (0).
    m == PROBE_K -> >=PROBE_K run; bonus position approximated as accept (1)."""
    bits = []
    for m in trace:
        bits.extend([1] * m)
        bits.append(0 if m < PROBE_K else 1)
    return bits


def throughput_fixed(bits, K, r):
    n = len(bits); committed = 0; cost = 0.0; p = 0
    step_cost = K * r + 1.0
    while p < n:
        acc = 0
        while acc < K and p + acc < n and bits[p + acc] == 1:
            acc += 1
        committed += acc + 1
        cost += step_cost
        p += acc + 1
    return committed / cost if cost else 0.0


def throughput_oracle(bits, arms, r):
    """Optimal per-step K via shortest-path DP: min total cost to consume all
    positions, then throughput = total_committed / min_cost."""
    n = len(bits)
    INF = float("inf")
    cost = [INF] * (n + 1); cost[0] = 0.0
    # precompute run length of consecutive 1s starting at each p
    run = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        run[i] = run[i + 1] + 1 if bits[i] == 1 else 0
    for p in range(n):
        if cost[p] == INF:
            continue
        for K in arms:
            acc = min(K, run[p], n - p)          # leading accepts within window
            gained = acc + 1
            np_ = min(p + gained, n)
            c = cost[p] + (K * r + 1.0)
            if c < cost[np_]:
                cost[np_] = c
    total_committed = n                          # every position committed once
    return total_committed / cost[n] if cost[n] < INF else 0.0


def main():
    data = json.load(open(PATH, encoding="utf-8"))
    r_measured = data.get("r")
    print(f"Loaded {PATH}  (measured r at batch=1 with 1B draft = {r_measured})\n")
    wl = data["workloads"]
    for w, oc in wl.items():
        traces = oc.get("m_traces")
        if not traces:
            print(f"--- {w}: no m_traces in JSON (re-run capture) ---"); continue
        allbits = [bits_from_trace(t) for t in traces]
        print(f"===== {w}  ({len(allbits)} prompts, {sum(len(b) for b in allbits)} tokens) =====")
        print(f"  {'r':>5} {'maxK':>5} {'bestfix(K)':>14} {'T_fixed':>8} {'T_oracle':>9} {'headroom':>9}")
        for maxk in MAXK_SWEEP:
            arms = [k for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32) if k <= maxk]
            for r in R_SWEEP:
                # aggregate throughput = total committed / total cost across prompts
                def agg_fixed(K):
                    tc = sum(len(b) for b in allbits)
                    cc = 0.0
                    for b in allbits:
                        n = len(b); p = 0
                        while p < n:
                            acc = 0
                            while acc < K and p + acc < n and b[p + acc] == 1: acc += 1
                            cc += K * r + 1.0; p += acc + 1
                    return tc / cc
                fixed = {K: agg_fixed(K) for K in arms}
                bestK = max(fixed, key=fixed.get)
                # oracle aggregate
                tc = sum(len(b) for b in allbits)
                oc_cost = 0.0
                for b in allbits:
                    t = throughput_oracle(b, arms, r)
                    oc_cost += len(b) / t if t else 0.0
                T_or = tc / oc_cost
                head = T_or / fixed[bestK] - 1
                flag = "  <==" if head > 0.10 else ""
                print(f"  {r:>5.2f} {maxk:>5} {bestK:>11d}    {fixed[bestK]:>8.3f} "
                      f"{T_or:>9.3f} {head*100:>+7.1f}%{flag}")
        print()


if __name__ == "__main__":
    main()
