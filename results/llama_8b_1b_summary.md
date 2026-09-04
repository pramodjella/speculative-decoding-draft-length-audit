# Llama-3.1-8B / Llama-3.2-1B speculative decoding stats (for Yash)

Matched-vocab pair (both vocab_size=128256), so HF built-in assisted generation
runs (unlike the Qwen 7B pairs). Config follows Yash's checklist exactly:
**bf16, batch=1, greedy (both models), KV cache on, SDPA, both models on one
A100-40GB, timing excludes model load + a warmup pass.** 24 prompts/workload,
max_new_tokens=96. Baseline = standard `target.generate()` (≈ lean Python AR loop,
so no baseline trick).

## Acceptance rate (fraction of drafted tokens accepted)

| K | HumanEval (code) | GSM8K (math) | MT-Bench (chat) |
|---|---|---|---|
| 1 | 89% | 88% | 77% |
| 2 | 85% | 81% | 67% |
| 4 | 77% | 71% | 55% |
| 8 | 63% | 58% | 38% |

Healthy and ≥ Yash's expected 50–70% (except deep K=8). Not a sampling-mismatch problem.

## Net speedup vs `generate()` (batch=1)

| Method | HumanEval | GSM8K | MT-Bench |
|---|---|---|---|
| ours fixed K=2 | 1.02× | 0.99× | 0.89× |
| **ours fixed K=4** | **1.11×** | **1.04×** | 0.88× |
| ours fixed K=8 | 1.06× | 0.98× | 0.71× |
| **HF built-in assisted K=2** | **1.16×** | 0.95× | 0.95× |
| HF built-in assisted K=4 | 1.15× | 0.95× | 0.95× |

## Reading (using Yash's decision tree)

- **Code:** HF built-in clears 1× (1.16×) AND our harness clears 1× (1.11×). Per
  Yash's tree, that confirms the algorithm + config are sound. Our harness tracks
  HF's within ~4%.
- **Math:** marginal — ours 1.04×, HF built-in 0.95× (our harness actually *beats*
  HF's built-in here).
- **Chat:** even HF's built-in stays <1× (0.95×). By Yash's tree, that's the
  hardware/regime branch, NOT our harness: a fast A100 + an 8B target + a 1B draft
  + lower acceptance (55–67%) means draft overhead isn't repaid.
- **Net:** the harness is validated (matches/beats HF's own assisted generation).
  The remaining sub-1× cases are inherent to this fast-GPU / 8B-target / moderate-
  acceptance regime, exactly as Yash predicted — not a harness bug.

## Implication

Robust >1× across *all* workloads needs a bigger target/draft cost ratio
(e.g. 70B target, or a slower-memory GPU) or an optimized engine. Cross-check:
on vLLM (Qwen2.5-7B), the same kind of sweep gave 1.44× (code) / 1.23× (math),
because vLLM's batched verification beats naive `generate()` by more than a
hand-rolled loop can. The benefit is workload-dependent (code > math > chat) and
best-K is workload-dependent (K=4 code/math, K=2 chat) — which is the motivation
for an adaptive controller, on the workloads (code/math) that have headroom.
