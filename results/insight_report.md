# Research Insight Report: Adaptive Draft-Length Controllers for Speculative Decoding

**Date:** June 15, 2026  
**Author:** Pramod Jella, Vizuara Inference Engineering Team  
**Milestone 4 Deliverable**  

---

## Executive Summary

This report compiles and analyzes the empirical outcomes of our adaptive draft-length study across **two evaluation modes**:

1. **Simulation** (high-fidelity statistical speculative decoding simulator): 5 workloads × 11 policies × 4 batch sizes = **220 configurations**, providing batch-sweep data and oracle upper bounds.
2. **Physical** (Qwen2.5-1.5B target / Qwen2.5-0.5B draft on RTX 5070): 4 workloads × 9 policies × 1,663 prompts = **16,630 per-prompt evaluations**, providing real-hardware measurements.

We compare three active controller families—**EntropyThreshold**, **EpsilonGreedy**, and **UCB**—against tuned static fixed-length baselines and a theoretical **Oracle** ceiling.

### Key Takeaway

On small model pairs (0.5B/1.5B), HuggingFace `.generate()` with optimized C++/CUDA internals outperforms our pure-Python speculative loop, yielding absolute net speedups < 1.0×. However, the **relative ordering** between controllers is consistent across both simulation and physical evaluation, validating the controller comparison. The physical evaluation reveals that the **History controller** uniquely beats the best fixed baseline on SpecBench (+2.0%), while **ε-Greedy** and **UCB-Coarse** are the most efficient in terms of token waste.

---

## I1. Net Speedup of Best Controller over Per-Workload-Optimal Fixed Length

### Simulation Results (Mixed Traffic, batch=1)

| Controller | Speedup | Gap to Best Fixed | Wasted Tokens |
|:---|:---:|:---:|:---:|
| Best Fixed (per workload, K=3) | 1.845× | — | 0.950 |
| ε-Greedy (ε=0.1) | 1.806× | −2.1% | 1.043 |
| UCB (c=0.5) | 1.780× | −3.5% | 0.900 |
| Entropy Threshold (τ=1.0) | 1.780× | −3.5% | 1.618 |
| Oracle (upper bound) | 2.160× | +17.1% | 0.172 |

### Physical Results (Qwen 1.5B/0.5B, RTX 5070)

| Workload | Best Fixed | Best Adaptive | Gap | Winner |
|:---|:---:|:---:|:---:|:---|
| HumanEval | 0.601× (K=2) | 0.598× (Entropy) | −0.5% | Near-parity |
| GSM8K | 0.501× (K=2) | 0.497× (ε-Greedy) | −1.0% | Near-parity |
| MT-Bench | 0.460× (K=2) | 0.437× (ε-Greedy) | −5.2% | Fixed wins |
| SpecBench | 0.519× (K=4) | 0.529× (History) | **+2.0%** | **Adaptive wins** |

**Interpretation:** On the physical hardware, the History controller is the only adaptive policy to beat the best fixed baseline, doing so on the most diverse workload (SpecBench). The ε-Greedy controller comes closest to matching best-fixed on GSM8K (within 1%). The gap on other workloads comes from the Python-loop overhead dominating on small models — on larger model pairs where the target verification is the true bottleneck, the adaptive advantage will be larger.

---

## I2. Which Signal and Controller is Best, and How Does the Optimum Shift?

### Signal Analysis

The entropy threshold controller shows the clearest signal-based behaviour but incurs the **highest token waste** across all workloads (8–22× wasted tokens per accepted token), because with τ=0.8 and max_len=8, it always drafts up to 8 tokens before checking entropy, wasting massive compute on hard spans. This contrasts with the simulation where a well-tuned τ provides excellent results.

### Controller Ranking (Physical, by Mean Speedup Across Workloads)

| Rank | Controller | Mean Speedup | Mean Wasted Tokens |
|:---:|:---|:---:|:---:|
| 1 | History | 0.508 | 1.217 |
| 2 | ε-Greedy | 0.486 | 0.638 |
| 3 | UCB-Coarse | 0.489 | 0.677 |
| 4 | UCB | 0.486 | 1.100 |
| 5 | Entropy | 0.430 | 14.642 |

### Best Controller per Workload (Physical)

| Workload | Best Controller | Speedup | Why |
|:---|:---|:---:|:---|
| HumanEval (Code) | Entropy | 0.598× | Predictable boilerplate — long drafts pay off |
| GSM8K (Math) | ε-Greedy | 0.497× | Mixed reasoning — online learning adapts well |
| MT-Bench (Chat) | ε-Greedy | 0.437× | High entropy variance — balanced exploration helps |
| SpecBench (Mixed) | History | 0.529× | Multi-domain — per-request adaptation excels |

### Batch-Size Shift (Simulation)

As batch size increases from 1 to 64 on mixed traffic:
- At B=1: UCB (1.780×) is highly competitive with best fixed
- At B=64: Adaptive controllers drop to 1.32× due to **batch interference** (the batch-level latency is determined by max(K) across all requests), while fixed K=3 maintains 1.62×
- **Conclusion:** Large-batch serving requires global/coordinated draft-length policies

---

## I3. Generalisation and Convergence (A5 Ablation)

### Cross-Workload Generalisation

Controllers were tuned with **fixed hyperparameters** (τ=0.8, ε=0.1, c=2.0) and evaluated across all 4 workloads without retuning:

| Controller | HumanEval | GSM8K | MT-Bench | SpecBench | Std Dev |
|:---|:---:|:---:|:---:|:---:|:---:|
| ε-Greedy | 0.578 | 0.497 | 0.437 | 0.434 | 0.058 |
| UCB-Coarse | 0.584 | 0.485 | 0.425 | 0.461 | 0.059 |
| UCB | 0.578 | 0.480 | 0.432 | 0.454 | 0.055 |
| History | 0.595 | 0.482 | 0.428 | 0.529 | 0.061 |
| Entropy | 0.598 | 0.432 | 0.392 | 0.300 | 0.114 |

**Finding:** ε-Greedy and UCB generalise most robustly (lowest std dev across workloads). The Entropy controller has the highest cross-workload variance (σ=0.114), suggesting it is sensitive to the distribution of entropy values in each workload.

### Online Bandit Convergence (Simulation)

- **UCB** converges to its steady-state optimal speedup within **150–200 steps**
- **ε-Greedy** converges faster (~75 steps) but maintains a steady-state loss due to continuous 10% exploration
- **History** is slowest to converge (~200+ steps) but achieves competitive final performance
- Implication: Bandits are viable for long-lived sessions (multi-turn chat, code completion), but may suffer cold-start regret on short single-turn completions

---

## I4. Controller Overhead

All controller overheads are negligible and well below the 100μs requirement:

| Controller | choose() μs | update() μs | Total μs |
|:---|:---:|:---:|:---:|
| EntropyThreshold | 0.247 | — | 0.247 |
| EpsilonGreedy | 0.507 | 0.169 | 0.675 |
| UCB | 0.158 | 0.188 | 0.347 |
| AcceptanceHistory | 0.069 | 0.315 | 0.385 |

All controllers operate at **sub-microsecond** overhead, confirming they are fully deployable inside the decode loop without measurable impact on latency.

---

## I5. Practitioner Decision Table

| Serving Scenario | Recommended Controller | Rationale |
|:---|:---|:---|
| **Low-latency, batch=1, single workload** | **Tuned Fixed K** (K=2 for most) | No adaptation needed; simplest and fastest |
| **Diverse/mixed traffic, batch=1** | **History Controller** | Adapts per-request; beats fixed on SpecBench (+2%); moderate waste |
| **Unknown traffic, long sessions** | **UCB Bandit (c=2.0)** | Principled exploration; converges in ~200 steps; low regret |
| **Fast deployment, zero training** | **ε-Greedy (ε=0.1)** | Good generalisation; lowest token waste; near-parity with best fixed |
| **High-throughput, large batch (B≥32)** | **Tuned Fixed K (K=3)** | Avoids batch interference; stable under load |
| **Research / ceiling analysis** | **Oracle** | Shows the 17% speedup headroom still available |

---

## Error Taxonomy Summary

### Distribution by Error Type (Physical, All Workloads)

| Error Type | Description | Frequency |
|:---|:---|:---|
| **Over-Drafting** | Controller drafts past rejection point; wasted compute | Higher in Entropy, fixed_8 |
| **Under-Drafting** | Controller stops too early on easy spans; speedup left unused | Higher in fixed_1, conservative policies |

### Key Observations
- **Entropy controller** has the highest over-drafting rate (drafts max_len=8 on every step)
- **ε-Greedy** and **UCB-Coarse** show the best balance between over- and under-drafting
- **History** exhibits moderate under-drafting on HumanEval but compensates with strong performance on diverse workloads

---

## A4 Ablation: Candidate Set Size

Coarse candidate set {1, 4, 8} vs fine candidate set {1, 2, 3, 4, 6, 8}:

| Controller | Fine (UCB) | Coarse (UCB-Coarse) | Winner |
|:---|:---:|:---:|:---|
| Mean Speedup | 0.486 | 0.489 | **Coarse** (slightly better) |
| Mean Wasted | 1.100 | 0.677 | **Coarse** (38% less waste) |

**Finding:** The coarse candidate set outperforms the fine set on both speedup and wasted tokens. Fewer arms mean faster convergence and less exploration overhead. This suggests that for practical deployment, a simple 3-arm set {1, 4, 8} is sufficient.

---

## Conclusion

The study demonstrates that adaptive draft-length controllers can match or beat per-workload-tuned fixed baselines on diverse traffic, with the History controller achieving a **+2% speedup over best-fixed on SpecBench**. While Python-loop overhead on small model pairs prevents net speedup > 1.0× in physical evaluation, the relative controller rankings are consistent between simulation and physical runs, validating the experimental protocol. Controller overhead is < 1μs, confirming deployability. For production systems, the ε-Greedy controller offers the best generalisation-to-waste ratio, while the History controller excels on genuinely mixed workloads.
