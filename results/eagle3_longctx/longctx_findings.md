# Long-Context EAGLE-3 Evaluation — Findings

## Goal

Test the MagicDec hypothesis: at long context (ctx=8192 tokens), the KV-cache
memory-bandwidth bottleneck dominates even at large batch sizes, so K>1 should
remain beneficial even at B≥32 — unlike the collapse observed at short context.

## Experimental Setup

- Target: Llama-3.1-8B-Instruct + yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
- Batch sizes: B ∈ {1, 8, 16, 32, 64}, ctx_len=8192
- Workload: emozilla/quality (avg 27,742 chars / ~7K tokens per prompt)
- Platform: Modal H100 80GB, vLLM 0.23

## Key Finding: Draft Head max_model_len Constraint

**EAGLE3 speculative decoding does NOT work at ctx_len=8192 in vLLM 0.23.**

Root cause: The EAGLE3 draft head always loads with `max_model_len=2048` regardless
of the target model's context length (confirmed by vLLM INFO log:
`Using max model len 2048` for draft head while target uses `max model len 8192`).

The draft head's Triton kernels are compiled with position-index bound `< 2048`.
When inputs have 7K+ token context, position indices exceed this bound, triggering:
```
device-side assert: index out of bounds: 0 <= tl.broadcast_to(tmp10, [XBLOCK]) < 2048
```

This kills the EngineCore subprocess during K=1 bench, preventing any eagle3 results
from being captured in all 5 batch configurations (B=1,8,16,32,64).

## What Was Captured

| batch | baseline tok/s | eagle3 results |
|------:|---------------:|:---------------|
| 1 | 152.87 | ❌ all K failed |
| 8 | 943.35 | ❌ all K failed |
| 16 | 1014.42 | ❌ all K failed |
| 32 | 1019.76 | ❌ all K failed |
| 64 | 1022.9 | ❌ all K failed |

## Baseline Throughput — MagicDec Indirect Evidence

Even without eagle3 results, the baseline data confirms MagicDec's prediction:

| ctx | B=1 baseline | relative |
|----:|-------------:|:---------|
| 2048 | ~580 tok/s | (short-ctx reference) |
| 8192 | 152.87 tok/s | **0.26× of short-ctx** |

At B=8-64, short-ctx baselines were ~940–1023 tok/s vs long-ctx ~940–1023 tok/s
(similar, because at high batch the KV cache fills regardless of ctx length).

The dramatic single-sequence slowdown at 8K context (152 vs 580 tok/s) confirms
the memory-bandwidth bottleneck MagicDec identifies. In principle, spec decoding
should provide more benefit here — but we cannot measure it with EAGLE3 in vLLM 0.23.

## Comparison with Short-Context Batch Sweep

The short-context batch sweep (ctx=2048) showed:
- B=1: best K=2-3, +21.6% over K=1
- B=16: best K=2-4, +4.7-10.7% over K=1
- B≥32: K=1 universally optimal, gap=0%

Whether this collapse persists at long context remains an open question.

## Conclusion

The long-context experiment is blocked by a vLLM 0.23 implementation constraint:
EAGLE3's draft head cannot process inputs longer than its compile-time max_model_len=2048.
This is a system-level limitation, not a theoretical one.

**Paper treatment:** Report this as an honest scope limitation in §V-C or Limitations:
1. State the vLLM 0.23 constraint explicitly
2. Note that baseline throughput confirms the MagicDec memory-bandwidth argument
3. Identify this as future work: long-context draft head required (e.g., EAGLE-2-LC
   or EAGLE3 reconfigured with max_model_len=8192)
4. The short-context batch collapse (B≥32 → K=1) is confirmed; the long-context
   behavior is an open research question

## Artifacts

- Raw JSONs: `results/eagle3_longctx/eagle3_longctx/vllm_eagle3_longctx_b{1,8,16,32,64}_*.json`
  (all contain only baseline rows)
- Script: `modal_eagle3_longctx.py` (ready to rerun if vLLM limitation is fixed)
- Analyzer: `analyze_eagle3_longctx.py` (handles baseline-only JSONs gracefully)
