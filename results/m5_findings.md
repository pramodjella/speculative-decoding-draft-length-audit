# M5 Findings — Honest Baseline, Lossless Proof, and Engine-Grade Validation

Consolidated write-up of the M5 work, for folding into the manuscript.

## 1. Engine-grade validation on vLLM (NEW — the real >1× result)

The paper's prior physical numbers were <1× because a pure-Python speculative
loop was compared against (and could not match) compiled decode. To obtain a
*fair, production-grade* speedup we re-ran inside vLLM 0.23.0 (V1 engine), where
baseline and speculative decoding share the same CUDA-graphed path.

**Setup:** Qwen2.5-7B-Instruct, A100-40GB, batch=1, greedy (temp=0),
ngram speculative decoding, net speedup = spec tok/s ÷ no-spec tok/s.

| Workload | K=1 | K=2 | K=4 | K=8 | Best fixed |
|---|---|---|---|---|---|
| HumanEval (code) | 1.18× | 1.30× | **1.44×** | 1.41× | K=4 |
| GSM8K (math) | 1.10× | 1.18× | **1.23×** | 1.22× | K=4 |
| MT-Bench (chat) | 0.94× | **1.02×** | 0.99× | 1.02× | K=2 |

**Finding (validates the thesis on a real engine):** the optimal fixed draft
length is workload-dependent and a single K is wrong almost everywhere. K=4 gives
**+44%** on code but falls *below parity* (−0.5%) on chat, where K=2 is best.
This is the exact motivation for a per-step/per-workload adaptive controller,
now demonstrated with production-engine numbers rather than a simulator or a
Python prototype.

## 2. Lossless guarantee — proven exactly in fp32

Speculative decoding with greedy verification is lossless in exact arithmetic.
We verify directly: in fp32, all six controllers reproduce the target's greedy
output **100% token-for-token** across 12 prompts (`m5_equivalence_fp32.json`).
In bf16 we observe occasional divergence; a per-step diagnostic shows every
divergence occurs at an *exact* logit tie (Δ = 0.0000 between the competing
tokens), where argmax tie-breaking differs between the cached single-token path
and the batched verification path — the same nondeterminism affects the baseline
run twice. Thus bf16 divergence is a floating-point artifact, not an algorithmic
change: the controller never alters outputs.

## 3. Why the pure-Python physical speedups were <1× (honest note)

A same-backend honest baseline (`PythonAutoregressiveBaseline`, identical
forward/KV-cache path to the speculative runner — not HF `.generate()`) still
yielded <1× on 7B/1.5B (fixed_4 ≈ 0.75×). Profiling traced this to per-step
Python/CPU overhead: a full-vocab softmax + a CPU↔GPU sync computing draft
entropy on *every* draft token (even for controllers that ignore it), plus
per-token `.item()` syncs and KV-cache cropping. The draft/target FLOP ratio plus
Python launch overhead means the draft forwards cost more than the target time
they save. The *relative* ordering of controllers is preserved, but absolute
net speedup requires an optimized engine — hence the vLLM validation in §1.

## 4. What remains future work

The in-engine *adaptive* controller (per-step K via vLLM's
`custom_class_proposer` extension point, fork-free) is designed but not run; the
adaptive-vs-best-fixed comparison remains in the simulator and the physical
relative-ordering study. vLLM also natively reimplements per-sequence adaptive K
(`DynamicProposer`, PR #26504, acceptance-EWMA) — concurrent work our richer
contextual signals (EntropyLinUCB) extend.
