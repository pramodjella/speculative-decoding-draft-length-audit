# How Much of Speculative Decoding's Adaptive Draft-Length Headroom Is Real?
## An Audit Under Tuned Baselines and Paired Wall-Clock Protocols

*Pramod Jella — Vizuara AI Labs, Inference Engineering Research Track*
*Complete paper draft (post-audit; numbers frozen against docs/CANONICAL.md, 2026-07-09).
This document is the source of truth; paper.tex is its LaTeX port.*

---

## Abstract

Adaptive draft-length policies for speculative decoding promise double-digit speedups over any
fixed draft length, citing per-step oracle gaps of +12–23%. **We audit that promise
end-to-end.** A Bayes-ceiling decomposition shows **~80% of the oracle gap is irreducible from
any draft-side signal**; the reachable remainder is real — a hidden-state probe detects it at
AUC 0.79–0.88 — but does not pay: under wall-clock measurement against *tuned native* baselines,
**every learned or entropy-based per-step controller we test loses 2–17%.** Exactly one policy
class survives — signal-free saturation tail-pruning (the PACER/SADDLE family) — and only in a
narrow regime: **+2–5% over the strongest native fixed baseline at batch size 1, with one of
three workloads a wash**; robust to sampling temperature, decaying to zero by batch 8 (driven by the
batch-synchronized stop the engine forces on the adaptive arm, not by verification cost, which
we measure to be flat — the shippable ragged-engine number remains open). Reaching stable verdicts required diagnosing
**six evaluation failure modes** that inflate adaptive-length claims — mechanism-mismatched
offline simulation, in-distribution threshold selection, unpaired sequential GPU benchmarking,
intra-run throughput steps, dead-code instrumentation, and padded-budget baselines — and a
gated, paired benchmarking protocol that we release with all artifacts. The decomposition
explains both verdicts: acceptance *prediction* chases irreducible variance, while saturation
*detection* is deterministic given quantities the drafter already computes.

---

## 1. Introduction

Speculative decoding (SD) accelerates LLM generation by having a cheap draft model propose K
tokens that the target verifies in one parallel pass. The draft length K is the central knob:
too short leaves speedup on the table, too long wastes drafting on rejected tokens. A substantial
literature proposes adapting K per step from decode-time signals. The premise is an oracle
argument: if one knew the acceptance outcome in advance and drafted exactly to it, throughput
would improve by double digits over any fixed K. SpecDecode-Bench [1] recently systematized this
observation on vLLM — establishing that verification dominates SD execution, that acceptance
length varies markedly across positions, requests, and datasets, and that a large gap separates
current methods from the oracle — and issued an explicit call for "a lightweight predictor that
can adaptively select... approaching the oracle bound."

**This paper answers that call, mostly in the negative, and measures exactly where the boundary
lies.** Our contributions:

1. **A Bayes-ceiling decomposition of the per-step oracle** into {already priced by a tuned fixed
   K} + {reachable from a draft-side signal} + {irreducible from any draft-side signal}, showing
   the oracle overstates the draft-side-achievable gain ~5× (~80% is irreducible). Independently
   re-derived; permutation- and measured-cost-controlled.
2. **Signal localization.** The reachable fraction is not where the literature has looked — cheap
   logit features (entropy, margin, SVIP-style) recover +0.3–8% and token position exactly 0% —
   but in the drafter's hidden state, from which a PCA+logistic probe recovers **+16.8 ±2.7% and +17.2 ±2.4% of the
   oracle (+2.0–3.1% net over tuned fixed K, offline)** for the two instruction models.
3. **A wall-clock policy-zoo audit with a drift-controlled protocol.** End-to-end, every
   learned/entropy stopping policy loses to a tuned fixed depth (2–17%), while exactly one
   class survives: signal-free saturation tail-pruning (+2–5%, paired; one of three workloads a wash). Along the way we demonstrate and correct **six evaluation failure modes** that
   inflate adaptive-SD claims (docs/CANONICAL.md) — including unpaired sequential benchmarking
   under container drift, which produced three mutually contradictory answers
   (+2.4/−1.9/+6.4%) from identical configurations until pairing resolved them.

Because exact-verification SD is lossless, every comparison is about speed at unchanged output.

## 2. Related work

**Feature-based drafting.** EAGLE-1/2/3 [2–4] draft from the target's hidden states; EAGLE-2/3
adapt the *tree shape* by draft confidence but commit to a static depth/budget.
**Signal-based length control.** SVIP [5], AdaEDL, DSDE stop drafting on entropy/margin/KL
thresholds. **Learned controllers.** SpecDec++ [6] trains an acceptance head; DISCO a classifier;
BanditSpec [7] and follow-ups treat K as a bandit arm.
**Serving-level adaptation.** Nightjar (load-aware bandit), SADDLE (HPCA'26), TurboSpec gate
speculation by system load — a different, demonstrably profitable lever.
**Benchmarking.** SpecDecode-Bench [1] established verification dominance, the oracle gap, and
acceptance variance on vLLM; we credit those findings and claim only our delta: the
reducible/irreducible split of that gap, the localization of the reducible part to the hidden
state, and its wall-clock non-realizability via early-stopping.
**The honest baseline.** Most adaptive-K papers compare against K=1 or an untuned default.
Against a per-workload tuned fixed K, reported gains of 2–8% compress to within the +2%
per-request oracle we measure (§7).

## 3. Setup

**Models.** Llama-3.1-8B-Instruct + `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`; Qwen3-14B +
`AngelSlim/Qwen3-14B_eagle3`; DeepSeek-R1-Distill-Llama-8B + `yuhuili/EAGLE3-DeepSeek-R1-…`.
**Engine.** vLLM 0.23 (V1), H100, greedy, B=1 unless stated; exact verification (lossless;
fp32 outputs verified token-identical to baseline).
**Workloads.** HumanEval, GSM8K, MT-Bench (30 prompts each); competition math (MATH-500/AIME)
for the reasoning-task control; QuALITY 8K-token passages for long context.
**Baseline.** Best fixed K by grid search over K∈{1..7} per workload — the strongest static
comparison (headline: K=2 → 1.54× measured net speedup on Llama-3.1-8B mixed traffic).
**Capture.** Hooks on the EAGLE3 draft head's `compute_logits` record the complete per-position
draft hidden state (4096/5120-dim, 41K–60K positions per model) with per-step acceptance labels
(`accept[j] = [acc > j]`, a prefix-survival indicator).
**Cost model and its grounding.** Offline policies are scored with speedup = MAT/(1+C·K).
Rather than assume C, we fit it to the measured fixed-K throughput sweep:
net(K) = accept_len(K)/(1+C·K) reproduces measured H100 throughput with RMSE ≈ 0.01 at
C ≈ 0.066–0.072. Headline offline numbers are reported under both C=0.15 (conservative) and
the measured C=0.072; conclusions are unchanged.

## 4. The Bayes-ceiling decomposition

For each model we compute five rungs (8-fold generation-split CV; threshold chosen on train):

- **best fixed K** — tuned static baseline;
- **Bayes(position)** — the best stop-policy given only the position-survival curve (a static
  signal; must degenerate to a fixed K — a built-in correctness check);
- **probe (deployable)** — stop-policy on an out-of-fold PCA-50+logistic probe over the full
  hidden state, threshold chosen on train generations;
- **probe (oracle threshold)** — the *same* probe scores on the *same* held-out rows, but with
  the threshold chosen to maximise performance on those rows. It removes threshold-selection
  error only, not hypothesis-class error, so it upper-bounds *this probe family* rather than
  any reader of the hidden state. Because it differs from the rung above by the threshold
  alone, it bounds it by construction in every fold — itself a positive control;
- **per-step oracle** — K = acc+1 (realization-aware; unreachable by construction).

**Result (recovery = fraction of the fixed→oracle span):**

| Model | Oracle ceiling | Bayes(position) | Probe (deployable) | Probe (oracle thr.) |
|---|---:|---:|---:|---:|
| Llama-3.1-8B (instruct) | +18.1% | +0.0% | +16.8 ±2.7% | +18.2 ±2.6% |
| Qwen3-14B (instruct) | +12.4% | +0.0% | +17.2 ±2.4% | +19.5 ±2.4% |
| DeepSeek-R1 (reasoning) | +4.5% | +0.0% | -0.3 ±0.2% | +0.7 ±0.5% |
| Llama-3.1-8B, math CoT | +23.2% | +0.0% | +9.5 ±3.7% | +12.0 ±3.8% |
| DeepSeek-R1, math CoT | +8.5% | +0.0% | -1.3 ±0.7% | +1.9 ±0.6% |

*All rungs scored on the same test rows within each of 8 generation-split folds, so the
oracle-threshold rung genuinely bounds the deployable probe (nesting control passes in every
fold, all five settings). The fourth rung removes threshold-selection error only, not
hypothesis-class error, so it is a ceiling for this probe family rather than for any reader
of the hidden state.*

Three readings. (i) **Bayes(position) = 0 everywhere**: static signal cannot beat a tuned fixed
K — this validates the pipeline (position degenerates to fixed K exactly) and explains why
position/entropy AUC ≈ 0.5–0.58 bought prior methods nothing. (ii) **probe(oracle threshold) ≈ deploy ≪
oracle**: even with the oracle threshold, the hidden-state ceiling recovers only ~1/5 of the
oracle span — the remaining ~4/5 is verification-outcome variance irreducible from any draft-side
signal. The oracle overstates draft-side-achievable gain ~5×. (iii) The reachable fraction is
**real for the instruction models** (+2.0–3.1% net over tuned K, offline) and absent for the
reasoning head.

**Controls and audit.** (a) *Permutation:* shuffling accept labels across draft blocks (preserving marginals and prefix structure) collapses Llama deployable recovery from +16.8 ±2.6% to −5.6 ±1.4% — a 22.4pp collapse. Both arms are scored with the same paired-within-fold estimator, so they differ only in whether the labels belong to their hidden states: genuine signal, not leakage. (b) *Independent re-derivation:* fresh code (different folds/PCA dim/seed/policy)
reproduces the oracle/fixed/ceiling rungs to <0.1pp and the recovery signs; recovery magnitude is
implementation-sensitive; under the corrected paired computation the two instruct models give **+16.8 ±2.7% and +17.2 ±2.4%**. (c) *Measured-cost grounding:*
re-scoring at the fitted C=0.072 leaves Llama at +17.3% recovery (+2.9% net). (d) *Detection:*
probe AUC 0.79–0.88 vs 0.484 for the earlier 16-dim random projection — the weak probe, not the
hidden state, was the earlier limitation.

## 5. Where the null is confounded: the reasoning head

On identical competition-math inputs, Llama-3.1-8B keeps its signal (+9.5 ±3.7% recovery, +2.5 ±1.0% net)
while DeepSeek-R1 shows none — suggesting a model-class boundary. **However, this is confounded
by draft-head strength**: the DeepSeek head's base acceptance is much lower (accept-length 1.47
vs 2.40 on the same task; position-0 acceptance 0.28 vs 0.55). A weaker head yields a smaller
ceiling and less detectable signal regardless of reasoning. We therefore scope the claim to *this
head* — the DeepSeek-R1-Distill EAGLE head shows no recoverable draft-side signal — and state the
confound as a limitation; separating "reasoning model" from "weaker head" requires a strong
reasoning-model EAGLE head, none of which is currently available.

## 6. Wall-clock: the reachable gain does not survive execution

vLLM 0.23 exposes no per-step draft-length hook, so we test end-to-end in the SafeAILab/EAGLE-3
reference implementation (eager PyTorch — no CUDA-graph penalty; fixed and adaptive share the
same instrumented codepath, equalizing per-step overhead). Llama-3.1-8B, H100, greedy.

**(A) Seed-state controller.** A PCA+ridge probe on the step's seed hidden state sets the draft
depth. *Audit note:* our first capture mispaired seeds with accept-lengths across prompts (one
dangling unverified seed per prompt shifted the global index pairing — caught in code audit,
empirically confirmed by the mismatch equalling the prompt count exactly, fixed, rerun). With
correct pairing the seed carries modest signal (corr 0.26) — and the controller still loses:
**−5.7% / −2.4% / −5.5%** vs the best fixed depth (d=6/7/7) on HumanEval/GSM8K/MT-Bench.

**(B) In-loop within-step early-stop (the faithful test).** A verified copy of the EAGLE-3 draft
routine breaks the depth loop when a per-level hidden-state probe predicts rejection (the shorter
tree is capped safely; the copy reproduces coherent generation). The level probe carries real
signal — **AUC 0.796**, close to the offline 0.84 — yet early-stopping **loses 5–6%**
(−5.1/−5.1/−6.1%) across all stop-thresholds:

| Workload | Fixed (full depth) tok/s | In-loop adaptive tok/s | Gain |
|---|---:|---:|---:|
| HumanEval | 144.5 | 137.2 | −5.1% |
| GSM8K | 128.4 | 121.8 | −5.1% |
| MT-Bench | 133.4 | 125.3 | −6.1% |

**Why real signal still loses.** The draft head is cheap and verification dominates execution
(consistent with SpecDecode-Bench's 42–95% verification share). Drafting deeper is therefore
almost always better: more accepted tokens per expensive verify pass. Early-stopping on
*predicted rejection* saves cheap draft compute but forfeits accepted tokens worth far more.
The chain-regime offline analysis's +2–3% *inverts sign* when deployed into the tree engine —
a mechanism-mismatch failure the literature's evaluation protocol permits.

### 6.1 The policy zoo, and the benchmarking protocol it forced

We extended the audit to four published-style stopping policies through the same verified
codepath: SVIP-style entropy stop, margin stop, SADDLE/PACER-style cumulative-path-probability
stop ("saturation tail-pruning"), and the SpecDec++-style trained hidden-probe stop. Getting a
stable answer required an escalating protocol, each step of which caught a real artifact:

1. *In-distribution selection inflates.* With thresholds selected on the benched prompts (the
   common protocol), tail-pruning showed +2.4%; on held-out prompts this collapsed to −1.9%.
2. *Split ordering confounds.* A contiguous train/held-out split is non-iid on
   category-ordered suites (MT-Bench): the best fixed depth itself flipped (7→5, a 14%
   fixed-depth gap), entangling distribution shift with policy generalization. We use
   interleaved splits.
3. *Unpaired sequential benching is unreliable at the ±5% scale.* Three runs with identical
   thresholds gave +2.4%, −1.9%, and +6.4%: within-container throughput drifts several percent
   between configuration blocks, and consecutive-repeat error bars cannot see it. Our final
   protocol benches all configurations **round-robin with rotated order and per-cycle paired
   differences**, cancelling drift by construction.

**Paired verdict (held-out prompts, vs the strongest per-workload fixed depth):** saturation
tail-pruning (threshold 0.05, selected on train) wins **+3.58%±0.32 on GSM8K (4/4 cycles),
+4.77%±0.50 on MT-Bench (4/4)**, and +2.90%±1.83 on HumanEval (3/4, borderline), at realized
mean depth 5.62 vs fixed 7–8. Outputs are lossless-identical; the verified tree size is
unchanged; the fixed arms run the same instrumented codepath — the gain is pure
draft-iteration savings. Every learned/entropy policy loses or ties.

**The survivor uses no learned signal.** It thresholds the cumulative best-path probability the
drafter already computes — zero model cost, no training, a policy family that exists in prior
art (SADDLE, PACER, DDD). Our contribution is the *audited verdict*: under tuned baselines,
held-out selection, and paired timing — a protocol under which most adaptive-length claims die —
this one class survives, and the decomposition explains why: acceptance *prediction* chases
irreducible realization variance (§4), while saturation *detection* is deterministic given the
tree the drafter has already built. Adaptive *deepening* beyond the default budget remains
untested here (cf. TALON).

**Cross-head replication: the gain is head-dependent.** Repeating the paired protocol on the
weaker DeepSeek-R1-Distill head (accept-length ~1.5 vs Llama's ~2.4) yields a tie:
−0.7%/−0.9%/+1.4% (n.s.) vs the strongest fixed depth, at realized depth 6.19 (pruning only
~0.8 levels vs Llama's ~1.4). Tail-pruning pays when a strong head makes deep fixed budgets
frequently wasteful; on a weak head there is less tail to prune. This run also surfaced a
protocol refinement: one cycle was contaminated by an *intra-cycle* throughput step-change
(the fixed arms' raw tok/s jumped 14–18% mid-cycle), which pairing does not cancel — robust
deployments of the protocol should use ≥6 cycles or median-of-cycles statistics.

**vLLM-native replication (chain regime, production engine).** Since vLLM exposes no per-step
hook, we patched the live draft loop of its V1 engine — located by stack-tracing from a hooked
draft-head call, because vLLM 0.23 ships a *parallel legacy spec-decode package that is dead
code for this configuration*: patching it produces a plausible, silent no-op that only an
arm-separation check catches (a fourth evaluation failure mode; our first attempt was
invalidated exactly this way and quarantined). All arms — forced fixed-k∈{2,3,4,7} and
cumulative-probability pruning — share the patched codepath and pay verification for all 7
slots (break-then-pad preserves the engine contract), isolating draft-side savings under equal
verify cost. Validity: 19% fixed-arm separation, and the strongest fixed arm lands at k=2–3,
matching the independently tuned production K=2. Paired verdict vs the strongest fixed arm:
**Llama-3.1-8B: cum(0.2) +6.4%/+7.5%/+2.9%** (HumanEval/GSM8K/MT-Bench, paired mean±SE) at adaptive mean
draft 2.3; **Qwen3-14B: +1.3%/+0.3%/+0.7%** (tie-to-slight-win) at mean draft 3.05, with the
aggressive threshold (0.05) harmful there (−4 to −7%).

**Fair-baseline verification (native engines, three containers).** Because break-then-pad makes
the fixed arms verify all 7 slots, the win could in principle ride on the baseline's
verification handicap. We therefore loaded *three engines in one process* — native K=2 and K=3
engines, each with its true verification budget, plus the patched K=7 cum engine — and
round-robined *across engines* under the paired protocol, repeated on three fresh containers
(vLLM v0.24). Gates: each native engine provably drafts exactly its K; realized draft lengths
are bit-identical across containers (greedy determinism — only the clock varies). Result:
cum(0.2) vs the strongest *native* engine is **positive-or-tie in 9/9 workload-cells** across
the three containers — HumanEval +5.6%±2.0, GSM8K +5.7%±2.9, MT-Bench +2.0%±1.6 (worst single
cell −0.2±2.0, a tie) — essentially matching the equal-verify numbers, because at B=1 the
verification pass is flat in K between speculative configs — now **measured, not asserted**
(verify microbench, HF-eager isolation, paired protocol): one kernel-regime step from the
non-speculative single-token forward (q=1, 15.8ms) to the multi-token path, then **dead flat
from q=2 to q=8** (18.5→18.2ms at B=1; native-K=2's q=3 vs the padded arm's q=8 differ by
<1%). The native baseline's verification savings are therefore genuinely small. The honest headline is therefore **"+2–5%, and one of three workloads is a wash"** (MT-Bench
+2.0±1.6, worst cell −0.2±2.0); never a significant loss. Cross-container magnitude spread
exceeds within-run error bars, reinforcing that single-container benchmark numbers deserve
limited trust.

**Batch and temperature scoping (same harness).** Extending the fair-baseline design: (i) at
**T=0.8** (seeded sampling, rejection-sampling verification) the win *survives* —
+5.4%/+2.1%/+1.5%, all positive-significant, at realized depth 2.24 (sampling lowers acceptance,
so saturation fires earlier, as the mechanism predicts); (ii) with **batch**, using a
batch-level mean-confidence rule (per-request ragged stopping is impossible in the
batch-synchronized draft loop — itself a finding about adaptive-length methods in serving
engines), the gain **decays monotonically: +2–6% at B=1 → ~+1% at B=4 → 0 to −3% at B=8**,
with realized depth drifting up (2.3→2.68) as the mean rule compromises across requests.
The microbench refines the mechanism: verify latency stays flat even at B=8×q=8 (64 positions ≈
8), so the decay is **not** driven by verification cost growth — it is dominated by the
batch-synchronized (mean) stop wasting per-request adaptivity, forcing rejects on requests whose
chains were still alive.
**Deployability caveat:** the cum arm is forced to pad-verify 7 slots. The microbench shows
padding is ~free at *both* B=1 and B=8 (verify is flat in q even at B=8×q=8), so padding is
not what drives the decay — the batch-synchronized mean stop is. What remains open is
deployability: the config that wins (B=1, padded verify) is not the config one would ship
(ragged), and a ragged engine might decay differently. **The deployable number is therefore open**; what we can claim is that the
observed profit window is latency-sensitive, low-batch serving (B≲4), consistent with our
batch-collapse result and SPEED-Bench's batch-dependent optimal draft lengths. Both outcomes
matched expectations pre-registered before the runs.

**Refined survivor claim (all engines/models, paired and gated):** a free, signal-less
saturation-pruning policy at a conservative threshold **ties or beats the tuned fixed baseline
in every setting tested** — Llama chain +2.9–7.5%, Llama tree +2–5%, Qwen3 chain +0.3–1.3%,
weak-head tree ~0 — while every learned acceptance-prediction policy loses everywhere.

## 7. Supporting results

**Batch collapse.** The adaptive headroom collapses with batch: at B=16 the best-K gap over K=1
is still +5–11%; at B≥32 it is exactly 0% on every workload — the serving regime removes even the
theoretical motivation for per-step control.
**Long context (8K).** Headroom is draft-quality-gated, not KV-bandwidth-gated: Qwen3-14B keeps
1.31–1.39× at ctx=8192 (even at B=32, +3% gap — the only MagicDec-consistent cell), while
Llama-8B and DeepSeek collapse to ~1.0× as their heads' acceptance falls to ~1.1 tokens/step.
**Reconciling prior adaptive-K gains.** AdaEAGLE/SVIP/SpecDec++/DDD report 2–8% gains against
default-K or K=1 baselines; against our tuned fixed K, the per-request oracle is only +2.0% and
online bandits go negative (UCB −4.9%, ε-greedy −6.9%) — their gains largely re-capture the
default→tuned gap rather than genuine per-step signal.

## 8. Discussion

The field's oracle plots are real but misleading as targets: most of the per-step gap is
realization variance no draft-side predictor can reach, and the reachable slice — though
genuinely present in the hidden state — is worth a few percent offline and *negative* end-to-end
via early-stopping. This explains, mechanistically, why the 2025–26 frontier moved to
*load-level* adaptation (speculate-or-not by batch/load), which our batch-collapse data confirms
is the profitable lever, and why signal-level length controllers keep failing against tuned
baselines. For practitioners: ship a tuned, deep fixed draft length; spend adaptivity budget at
the serving level, not per step.

## 9. Limitations (stated, not hidden)

- Offline gains are cost-model figures at B=1 (range +2.0–3.1% net; recovery +16.8–17.2% for instruct models across
  independent re-derivations); end-to-end, early-stopping does not realize them.
- The probe is trained in-distribution (held-out generations, same model/workloads);
  cross-workload/model transfer untested.
- The reasoning-head null is confounded by lower base acceptance (§5); scoped to the head.
- "Irreducible" means irreducible from draft-side signals; privileged post-verification
  quantities are a different question.
- Wall-clock tests early-stopping only; adaptive deepening beyond the default budget is
  untested (cf. TALON). Single-pass timings in §6's first experiments should be read with a
  ~2% noise bar; the §6.1 zoo verdict uses the paired drift-controlled protocol.
- The tail-pruning positive (+2–5%) is verified on one model/GPU pair (Llama-3.1-8B, H100),
  B=1, with HumanEval borderline; cross-model, batched, and vLLM-native replication is open.
- Wall-clock uses the eager reference implementation (vLLM has no per-step hook); relative
  fixed-vs-adaptive comparisons are valid, absolute tok/s is lower than vLLM.

## 10. Conclusion

Against a tuned fixed draft length, ~80% of speculative decoding's celebrated per-step oracle is
irreducible from any draft-side signal; the reducible ~20% lives in the drafter's hidden state,
is worth +2–3% offline for instruction-model heads, and *loses* 2–6% end-to-end when acted on,
because verification dominates. A four-policy wall-clock audit under a paired, drift-controlled
protocol — itself necessitated by six evaluation failure modes we demonstrate — finds exactly
one survivor: signal-free saturation tail-pruning, worth +2–5% at zero model cost. The deployable
answer is a tuned deep draft plus free tail-pruning; the learned per-step signal the field keeps
chasing is real, localized, and — for the shortening lever — not worth acting on.

---

### Reproducibility map

| Result | Script | Artifact |
|---|---|---|
| Hidden-state capture | `modal_eagle3_hidden_full_capture.py` | `results/eagle3_hidden_full/*.parquet` |
| Decomposition | `analyze_bayes_ceiling.py` | `results/perstep_signal/bayes_ceiling.md/.json` |
| Permutation control | `analyze_bayes_ceiling_control.py` | `bayes_ceiling_control.md` |
| Independent audit | `audit_decomposition.py`, `audit_probe_recovery.py` | `pre_write_audit.md` |
| Cost grounding | `measured_cost_ground.py` | `measured_cost_ground_llama8b.json` |
| Reasoning-task control | capture `--wl reasoning` | `bayes_ceiling_reasoning.md` |
| Wall-clock (A)/(B) | `modal_eagle_wallclock.py`, `modal_eagle_inloop.py` | `wallclock_eagle.md`, `eagle_{wallclock,inloop}.json` |

*References: [1] SpecDecode-Bench, arXiv:2601.11580. [2–4] EAGLE-1/2/3. [5] SVIP. [6] SpecDec++.
[7] BanditSpec. (Full citations in LaTeX port.)*
