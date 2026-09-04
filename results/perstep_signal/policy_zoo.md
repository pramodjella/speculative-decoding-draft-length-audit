# Policy-zoo audit — published stopping policies, offline vs wall-clock

> ✅ **FINAL VERDICT (paired, drift-controlled run — `modal_eagle_paired.py`):
> saturation tail-pruning (cumprob, thr=0.05) is a VERIFIED positive; every learned/entropy
> policy loses.** Vs the STRONGEST per-workload fixed depth, per-cycle paired gains:
> GSM8K **+3.58%±0.32** (4/4 cycles), MT-Bench **+4.77%±0.50** (4/4),
> HumanEval +2.90%±1.83 (3/4, borderline). Realized depth 5.62 (adaptive). Fairness checks:
> fixed arms run the same patched codepath; cumprob pays a small extra per-level sync (bias
> against it); outputs lossless-identical; verified tree size unchanged — the gain is pure
> draft-iteration savings.
>
> **Why the contradictions resolved:** raw per-cycle tok/s shows container throughput drifting
> several percent within a run — sequential (unpaired) benching is unreliable at the ±5% effect
> scale. The three earlier answers (+2.4/−1.9/+6.4%) were drift + split artifacts; pairing
> cancels them.
>
> **The surviving policy uses NO learned signal** — it thresholds the cumulative best-path
> probability the drafter already computes (SADDLE/PACER/DDD family; the policy is prior art,
> the audited verdict is ours). Acceptance-prediction policies (hidden probe, entropy) chase
> irreducible variance and lose; saturation detection is deterministic given the tree and wins.
> Scoped amendment integrated into paper_draft/workshop_abstract/DELIVERABLES under delegated
> sign-off; core headline (learned per-step shortening doesn't pay) unchanged.

> **STATUS: provisional positive KILLED by held-out verification (run 1); final iid-split run
> in flight.** Verification (held-out prompts [n:2n], 3× repeated timing) collapsed every
> policy's gain vs the eval-set-tuned fixed depth: cumprob +2.4% → **−1.9%** (+0.1/−2.5/−3.2),
> entropy +1.2% → **−4.7%**, hidden −13.8% → **−17.0%**. The in-distribution +2.4% was
> threshold-selection overfitting — exactly the artifact the literature's evaluation protocol
> permits, demonstrated and caught on ourselves.
>
> **However, run 1's own split was confounded:** contiguous [0:10]/[10:20] is non-iid (mt_bench
> is ordered by category); the best fixed depth flipped d=7(train) → d=5(held-out), a 14%
> fixed-depth gap on mt_bench. Under a train-selected-fixed protocol this ordering artifact
> would flatter the policies. Final run uses an INTERLEAVED (even/odd) split and reports BOTH
> protocols (A: vs eval-selected fixed, strongest baseline; B: vs train-selected fixed,
> deployment protocol). Headline remains frozen; hidden-probe zoo number carries an sklearn-CPU
> per-level overhead not paid by the other policies (canonical hidden number stays the in-loop
> (B) −5–6%).

Four draft-side stopping policies, each mirroring a published method's signal, run through the
SAME verified in-loop EAGLE-3 codepath (`modal_eagle_zoo.py`). For each: the offline-predicted
gain (tree-capture simulator, measured C=0.072, threshold chosen the way the literature does)
vs the wall-clock gain against the best fixed depth (d∈{4..7}, d=7 best on all workloads this
run). Llama-3.1-8B + yuhuili EAGLE3, H100, greedy, n=10/workload.

| Policy (style) | Offline predicted | Wall-clock mean | Per-workload wall-clock |
|---|---:|---:|---|
| cumprob (PACER/SADDLE tail-prune) | **+2.70%** | **+2.40%** | +2.9 / +1.7 / +2.5% |
| entropy (SVIP, loose thr=5.0) | +1.06% | +1.21% | below noise bar — tie |
| margin (AdaEDL-style) | −0.36% | +0.32% | tie |
| hidden probe (SpecDec++-style) | −6.99% | −13.79% | loses everywhere |

Hidden-probe AUC this run: 0.784 (signal present, consistent with prior runs).

## Two findings (both provisional)

**1. Mechanism-matched offline simulation PREDICTS wall-clock.** Sign and ordering correct for
all four policies. This refines the earlier "sign inversion": the inversion arose when a
CHAIN-regime offline analysis (vLLM traces, per-position stop) was deployed into a TREE-regime
engine. When the simulator matches the deployment mechanism (tree capture → tree deployment),
offline metrics track reality. Sharper methodology claim than "cost models mislead."

**2. Tail-pruning ≠ acceptance-prediction.** The one (provisionally) profitable policy does not
predict acceptance — it detects draft-tree saturation (best cumulative path probability < 0.03)
and shaves only the near-worthless tail iterations. Acceptance-prediction stopping (the hidden
probe) still loses badly. Tentative law: *predicting acceptance doesn't pay; detecting tree
saturation pays a little (~+2%).*

## Verification criteria (decides the outcome)

- cumprob held-out mean gain > 2× its standard error, positive on ≥2/3 workloads → RESULT;
  headline gets a scoped amendment (with Yash's sign-off).
- Otherwise → tie/noise; zoo confirms the generalized negative; headline stands verbatim.

Data: `results/eagle_zoo.json`. Verification: `modal_eagle_zoo_verify.py` →
`results/eagle_zoo_verify.json`.

## Released-code audit (Work 3) — blocked on availability

Checked 2026-07-04: **TALON** (arXiv 2601.07353) has no public code (under review at OpenReview,
no repo linked); PACER likewise not found. The feasible audit is therefore the class-level
reimplementation already run: cumprob = the cumulative-confidence class (SADDLE/PACER/DDD),
entropy = the SVIP class, hidden probe = the SpecDec++ trained-head class. Auditing the exact
released implementations through the paired protocol is future work gated on code release.

## Cross-model replication (Work 1) — COMPLETE: tie on the weak head; claim scoped

Paired protocol on DeepSeek-R1-Distill-Llama-8B + yuhuili EAGLE3 head (weak head,
accept-length ~1.5): **vs strongest fixed (d=7): −0.73%±1.19 / −0.86%±2.26 / +1.36%±3.52
(HumanEval/GSM8K/MT-Bench) — a TIE.** Realized depth 6.19 (prunes ~0.8 levels vs Llama's ~1.4).

**Verdict: the survivor's gain is HEAD-DEPENDENT.** Strong instruct head (Llama): +3–5%;
weak head (DeepSeek): ~0. The tail-pruning claim is scoped to strong instruction heads.

**Protocol observation (failure mode 3b):** cycle 4 was contaminated by an INTRA-cycle
throughput step-change — the fixed arms' raw tok/s jumped 14–18% mid-cycle (fixed7 gsm8k
148.5→169.2; mt_bench 131.8→156.2) while configs benched earlier in the rotated order did not.
Pairing cancels smooth drift, not step-changes landing mid-cycle. Cycles 1–3 were directionally
positive (mt_bench +4.2 to +6.4%) but we do NOT claim them; the full-data tie is the result.
Recommendation absorbed into the protocol: ≥6 cycles and/or median-of-cycles statistics.

Qwen3-14B is not loadable in the EAGLE *reference repo* (AngelSlim head is vLLM-format) — but
it DOES work in vLLM, where all offline experiments ran. **Correction to earlier scoping
(2026-07-04): a vLLM-native replication is being attempted** — vLLM's EAGLE-3 drafts a chain
via a Python loop in its V1 proposer; a cumprob early-break can be monkeypatched there under
enforce_eager (same surgery as topK_genrate), with break-then-PAD preserving the engine's
[B,K] contract (all arms pay identical verify cost; differences = pure draft forwards).

> ✅ **FAIR-BASELINE VERDICT (2026-07-07, 3 fresh containers — Yash Q1+Q3 CLOSED):
> the win survives TRUE native baselines, sign-stable, magnitude container-dependent.**
> `modal_vllm_fairbase.py`: three engines in ONE process — native K=2 and K=3 (real
> verification budgets) + patched K=7 cum — paired round-robin ACROSS engines, gated (each
> native provably drafts exactly its K; depths bit-identical across all 3 containers:
> 2.0/3.0/7.0, cum 2.3 — greedy determinism, only the clock varies). vLLM v0.24.0.
>
> **cum0.2 vs strongest native engine, per container (r1/r2/r3):**
> HumanEval +6.93/+6.64/+3.29 → **+5.6%±2.0**; GSM8K +7.08/+8.38/+1.77 → **+5.7%±2.9**;
> MT-Bench +3.50/−0.19/+2.66 → **+2.0%±1.6**. **9/9 cells positive-or-tie (8 positive,
> 1 tie at −0.19±2.04); never a loss.** Defensible claim: **+2 to +8% across containers,
> and one of three workloads (MT-Bench) is a wash** — cross-container spread exceeds within-run SEs, reinforcing the protocol
> finding (single-container numbers deserve limited trust). Mechanism of Q1's answer: at B=1
> the verify pass is flat in K between speculative configs (MEASURED: microbench q=2..8 flat,
> 18.5→18.2ms; see docs/CANONICAL.md), so native verification savings are
> small — the equal-verify design had not been flattering the policy. cum0.05 is consistently
> negative on MT-Bench → thr=0.2 is the robust setting. Data:
> `results/vllm_fairbase_llama8b_{r1,r2,r3}.json`.
>
> **BATCH + TEMPERATURE SCOPING (2026-07-07, `modal_vllm_fairbase2.py` — closes the two
> remaining reviewer gaps, pre-registered expectations confirmed):**
> - **Batch decay (batch-level mean-cum rule; per-request ragged stop impossible in the
>   synchronized loop):** B=1 +2.0/+5.6/+5.7% → B=4 +2.6/+0.4/+0.5% → B=8 −0.9/−3.1/+0.6%.
>   Monotone decay; realized depth drifts up (2.3→2.68) as the mean-rule compromises across
>   requests. **Profit window = latency-sensitive low-batch serving (B≲4)** — consistent with
>   E1 collapse and SPEED-Bench's batch-dependent-K.
> - **Temperature (T=0.8, seeded, rejection-sampling verification): the win SURVIVES —
>   +5.39%±0.99 / +2.05%±0.29 / +1.49%±0.30, all positive-significant**; realized depth 2.24
>   (saturation fires earlier under lower acceptance, as the mechanism predicts).
> Data: `results/vllm_fairbase2_llama8b_{b4,b8,t08}.json`.
> Q2 reconciliation (no contradiction): the −5–6% wall-clock loss was the TRAINED-PROBE policy;
> cumprob won +6.35% while the probe lost −14.7% in the SAME run/prompts/cycles
> (eagle_zoo_verify_iid.json) — the sign split between prediction and saturation-detection IS
> the finding. Q4 acknowledged: cumprob is the PACER/SADDLE class; framing stays "audited
> verdict on a known policy," never "new method". Headline/venue frozen until fair-base lands.

**✅ vLLM-NATIVE RESULT (take 2, 2026-07-04) — survivor REPLICATES in the production engine,
chain regime, Llama-8B.** After the diagnosis below located the true draft loop
(`vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py::_multi_step_decode` — NOT the
parallel dead `vllm.v1.spec_decode.*` path), take 2 passed both fail-fast gates (patched loop
n=70; fixed2 drafts exactly 2) and showed real arm separation (fixed2 163.7 vs fixed7 137.6
tok/s, 19% — and best fixed = k=2–3, matching the E-track tuned K=2). Paired verdict vs the
strongest fixed arm: **cum0.2 wins all 3 workloads — HumanEval +6.43%±0.83, GSM8K +7.46%±1.29,
MT-Bench +2.87%±0.44 (paired)** at adaptive mean draft length 2.3; cum0.05 +0.2–3.2%.
*Design caveat (stated, not hidden):* all arms pay verification for KMAX=7 slots (break-then-pad
keeps the engine contract), so this isolates DRAFT-side savings under equal verify cost; a
production engine rebuilt at K=2 would also trim verify slots. **Qwen3-14B (vLLM chain, same gated harness): cum0.2 TIES-to-slightly-wins** —
+1.26%±0.27 / +0.32%±1.38 / +0.68%±0.50 vs strongest fixed (fixed3/fixed3/fixed2), adaptive
mean draft 3.05; aggressive thr=0.05 HARMFUL (−4.3 to −7.3%, drafts 4.22 — too deep for this
head). Threshold sensitivity is model-dependent; the conservative thr=0.2 never loses.

**FINAL CROSS-ENGINE / CROSS-MODEL PICTURE (all paired, all gated):**
| Setting | tail-prune (conservative thr) vs strongest tuned fixed |
|---|---|
| Llama-8B, vLLM chain | **+2.9 to +7.5%** (paired, all 3 workloads) |
| Llama-8B, EAGLE-repo tree | **+3 to +5%** (2 of 3 strong, 1 borderline) |
| Qwen3-14B, vLLM chain | +0.3 to +1.3% (tie-to-slight-win) |
| DeepSeek-R1 (weak head), tree | ~0 (tie) |

→ Refined survivor claim: **a free, signal-less saturation-pruning policy that ties or beats
the tuned fixed baseline in every setting tested, with gains up to +7% where the head is
strong — while every learned acceptance-prediction policy loses everywhere.**

Data: `results/vllm_tailprune2_{llama8b,qwen14b}.json`; harness `modal_vllm_tailprune2.py`.

**vLLM attempt 1 INVALIDATED by its own validity check (2026-07-04).** First runs on both
models showed NO separation between forced fixed2 and fixed7 arms (ratio 1.001) → a no-op
benchmarked six ways; numbers quarantined. Diagnostic (`modal_vllm_tailprune_diag.py`):
compute_logits hooks fire in-process (n=2884) but **`EagleProposer.propose` is never called
(n=0)** — the dump shows all proposer subclasses inherit ONE `propose` from
`SpecDecodeBaseProposer`, and the engine's drafter instance is not an `EagleProposer`, so the
subclass patch intercepted nothing. Fix: patch the BASE class + fail-fast asserts (n_propose>0;
fixed2 must truncate drafting). Rerun in flight. Protocol lesson #4 for the paper: **always
verify arm separation before reading a benchmark** — it caught this in one glance. B>1 in the reference repo remains infeasible
(single-sequence eagenerate); analytically, verification dominance GROWS with batch (E1:
headroom → 0 at B≥32), so the +3–5% is a B=1, strong-head ceiling.
Data: `results/eagle_paired_deepseek.json`.
