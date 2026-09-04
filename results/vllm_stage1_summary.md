# vLLM Stage 1 — Engine-Grade Speculative Decoding Speedup (real >1×)

**Setup:** Qwen2.5-7B-Instruct on A100-40GB, vLLM 0.23.0 (V1 engine), batch=1,
greedy (temp=0), max_tokens=96, 24 prompts/workload. Net speedup = spec tok/s ÷
no-spec tok/s, both inside the same optimized engine (apples-to-apples).
Method: **ngram** speculative decoding (draft-model path blocked — see note).

## Net speedup vs no-spec (ngram)

| Workload | K=1 | K=2 | K=4 | K=8 | Best fixed |
|---|---|---|---|---|---|
| HumanEval (code) | 1.18× | 1.30× | **1.44×** | 1.41× | K=4 |
| GSM8K (math) | 1.10× | 1.18× | **1.23×** | 1.22× | K=4 |
| MT-Bench (chat) | 0.94× | **1.02×** | 0.99× | 1.02× | K=2 |

## Key finding (motivates the adaptive controller)

**The optimal fixed draft length is workload-dependent, and a single fixed K is
wrong almost everywhere** — exactly the project's premise, now shown on a real
engine:
- K=4 is best for code (**+44%**) but *below parity* for chat (−0.5%): long drafts
  waste compute on unpredictable chat tokens.
- K=2 is best for chat but leaves ~15 pts on the table for code.
- A per-step/per-workload controller that picks K from cheap signals would capture
  the best of each — the adaptive-draft-length thesis.

## Note: draft-model path

The Qwen2.5-7B + Qwen2.5-0.5B draft pair is rejected by vLLM (vocab mismatch:
152064 vs 151936; vLLM requires equal vocab, HF does not). Within Qwen2.5 only
7B/14B/32B/72B share vocab 152064, so a strong-target draft-model run needs an
EAGLE-3 head (verified head exists for 14B: `ruipeterpan/Qwen2.5-14B-Instruct_EAGLE3_UltraChat`)
or a same-vocab pair. ngram needs no draft model and has the same draft-length
knob (`num_speculative_tokens`), so it is a valid vehicle for the thesis.
