> ⚠️ **SUPERSEDED — historical document.** Numbers and claims here predate the final audit
> and may be stale (e.g. pre-audit oracle estimates, retired verdicts). The current source of
> truth is `docs/CANONICAL.md`; the current paper is `report/paper_draft.md`. Kept for
> provenance only — do not cite from this file.

# Acceptance-Aware Batch Scheduling for Speculative Decoding
### A 1-page project proposal (top-conference target)

## Problem
In production LLM serving, requests are processed in **continuous batches**. Speculative
decoding (SD) helps at batch=1 but degrades under batching: a single forward verifies the whole
batch, so the **batch's draft length is shared**, and per-token acceptance **varies widely across
concurrent requests** (e.g., code vs open chat). A request that accepts most of a long draft and
one that accepts almost none are forced to use the *same* K. The batch pays the cost of the long
draft but only some requests reap the benefit — wasted verification compute that *reduces*
throughput under load. Our measurements show naive per-request adaptation makes this worse: at
batch 64 it falls to 1.32× vs 1.62× for the best fixed K (batch interference).

## Key insight
Batch interference is usually treated as a *liability* (Nightjar disables speculation under load;
AdaServe trades it against SLOs). We treat it as a **scheduling lever**: if requests were
**grouped by predicted acceptance**, each batch would be homogeneous and could run at *its own*
near-optimal K — high-acceptance batches draft deep, low-acceptance batches draft shallow or skip
speculation — recovering the per-request optimum within the shared-K constraint.

This is only feasible if per-request acceptance is **predictable cheaply and causally**. Our prior
work establishes exactly that on standard separate-draft SD: **draft margin predicts acceptance at
AUC 0.88** (and entropy 0.87). So the scheduler can estimate each request's acceptance online from
one cheap signal and route it to a matching speculation tier.

## Method
1. **Per-request acceptance estimator**: an online estimate `â_r` from a running average of the
   request's recent draft margin / acceptance (cheap, causal, no extra forward).
2. **Acceptance-aware admission/grouping**: maintain a small set of speculation *tiers*
   (e.g., K∈{0,2,4,8}); route each request to the tier maximizing its predicted
   throughput `(accepted(â_r,K)+1)/(K·r+1)`; **batch within a tier** so the shared K is optimal
   for everyone in it.
3. **Load-aware tier gating**: under high load, collapse/disable high-K tiers (verification compute
   is better spent on more requests) — subsumes Nightjar's disable-under-load as a special case.

## Novelty vs prior art
- **Nightjar (2025)**: MAB over speculation length *per batch size* + disable under load — but a
  *single* policy for the whole batch; does **not** group by per-request acceptance.
- **AdaServe (2025)**: SLO-customized SD — orthogonal objective (SLO attainment), not
  throughput-via-homogeneous-batching.
- **SVIP/BanditSpec/SpecKV**: per-step/per-request length at batch=1; ignore batch interference.
- **Ours**: the *batching decision* (which requests share a draft) becomes the control variable,
  enabled by a validated cheap acceptance predictor. To our knowledge unclaimed.

## Experiments
- **H1 (headroom, simulation)**: does acceptance-aware tiered batching beat acceptance-agnostic
  batching + global K in tokens/compute, across realistic acceptance distributions and loads?
  *(This proposal's first gate — see `src/sim_batch_scheduling.py`.)*
- **H2 (real serving)**: implement tier-based routing in vLLM/SGLang continuous batching; measure
  throughput and TTFT/TPOT vs fixed-K SD and Nightjar-style disable-under-load, on a real request
  trace (e.g., AzureLLMInference / ShareGPT) with mixed code/chat/math workloads.
- **H3 (ablation)**: estimator quality (AUC) → realized throughput; tier granularity; load gating.

## Risks
- Real-serving implementation requires continuous-batching internals (engineering; the regime where
  prior vLLM-internals work stalled — budget for SGLang as fallback).
- Grouping adds scheduling latency/fragmentation; gains must exceed it.
- If H1's headroom is small (homogeneous ≈ heterogeneous), drop before H2.

## Why it can clear a top-tier bar
A *new control variable* (batch composition), grounded in a *validated* cheap signal, targeting the
*production* regime (throughput under load) that the latency-only literature underserves — with a
clean simulation→real-serving experimental arc and a falsifiable first gate.
