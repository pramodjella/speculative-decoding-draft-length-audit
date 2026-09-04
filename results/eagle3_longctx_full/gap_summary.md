# Long-context EAGLE-3 — batch collapse (ctx=8192)

## vLLM-patched

| model | workload | batch | best K | best speedup | K=1 speedup | gap vs K=1 |
|---|---|---:|---:|---:|---:|---:|
| DeepSeek-R1-Distill-LLaMA-8B (reasoning) | quality | 1 | 1 | 1.005 | 1.005 | +0.000 |
| DeepSeek-R1-Distill-LLaMA-8B (reasoning) | quality | 32 | 1 | 0.944 | 0.944 | +0.000 |
| Llama-3.1-8B (instruct) | quality | 1 | 1 | 0.988 | 0.988 | +0.000 |
| Llama-3.1-8B (instruct) | quality | 32 | 1 | 1.003 | 1.003 | +0.000 |
| Qwen3-14B (instruct) | quality | 1 | 3 | 1.387 | 1.320 | +0.067 |
| Qwen3-14B (instruct) | quality | 32 | 2 | 1.308 | 1.278 | +0.030 |

## SGLang: no eagle3 results (all baselines only)

## Comparison with short-context batch sweep (ctx=2048)

Short-context collapse thresholds (from results/eagle3_batch/gap_summary.md):
- B=16: still +5-11% gap vs K=1  (headroom exists)
- B=32: gap = exactly 0% on ALL workloads  (complete collapse)

**MagicDec hypothesis:** at long context (8K tokens), KV bandwidth bottleneck
dominates even at B≥32, so K>1 should remain beneficial.

If the table above shows gap>0 at B=32 for ctx=8192 -> MagicDec confirmed.
If the table shows gap=0 at B=32 for ctx=8192 -> collapse is universal.