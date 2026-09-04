> ⚠️ **SUPERSEDED — historical document.** Numbers and claims here predate the final audit
> and may be stale (e.g. pre-audit oracle estimates, retired verdicts). The current source of
> truth is `docs/CANONICAL.md`; the current paper is `report/paper_draft.md`. Kept for
> provenance only — do not cite from this file.

# When Does Adaptive Draft Length Help? A Signal-Limited Analysis of Speculative Decoding on EAGLE-3

*Workshop manuscript draft (6–8 pp). Pramod Kumar Reddy Jella.*

## Abstract
Adaptive draft-length controllers promise to "spend speculation where it pays," and prior
work (SVIP, BanditSpec, AdaEAGLE, SpecKV) reports gains by adjusting how many tokens a drafter
proposes per step. We ask a sharper question: on a strong modern self-speculation system
(EAGLE-3), how much speedup is actually available to a draft-length controller, and what limits
it? Using read-only instrumentation of vLLM that recovers per-step draft entropy, top-1/top-2
margin, target-model verification entropy, and exact accept-run lengths, we evaluate every
standard controller family — fixed-K, entropy-threshold (SVIP), bandits (UCB/ε-greedy),
acceptance-history, acceptance-persistence, and a learned per-step stopper — on
Llama-3.1-8B + EAGLE-3 across 1,295 decode steps, and replicate on Qwen3-14B + EAGLE-3.
We find a **+21.5% per-step oracle ceiling but no cheap controller exceeds ~+3%**, and
per-*request* adaptation is worthless (+2% ceiling; real bandits *lose* 5–9%). Decomposing the
gap, we show the bottleneck is the **signal, not the policy**: on EAGLE-3, draft-side confidence
weakly predicts acceptance (r≈−0.23), and even the strongest signal we find — the target's causal
bonus-token entropy (r≈−0.32) — explains only ~10% of variance. Critically, we show this is a
**property of EAGLE-3, not of speculative decoding in general**: on a standard *separate-draft*
chain, the very same cheap signals predict acceptance *well* (margin AUC 0.88, entropy 0.87),
which both explains why prior controllers (SVIP, +13% on EAGLE-2) succeed there and pinpoints
EAGLE-3's trained, feature-reusing head as what saturates the signal. We conclude that adaptive
draft length is **signal-limited specifically on strong self-speculation systems**, give a
measurable diagnostic for when a controller is worth deploying, and identify the source of the
EAGLE-2-vs-EAGLE-3 discrepancy in the literature.

## 1. Introduction
Speculative decoding (SD) accelerates autoregressive generation by drafting several tokens and
verifying them in one parallel pass. A central knob is the **draft length** K: too few and the
gain is small, too many and rejected drafts waste compute. Because predictability varies within
a generation, a fixed K is "wrong almost everywhere," motivating adaptive controllers that set K
from cheap signals (draft entropy, acceptance history) — SVIP, BanditSpec, AdaEAGLE, SpecKV.

Most of this work predates or sidesteps **EAGLE-3**, today's strongest self-speculation method.
We ask precisely how much headroom an adaptive controller has on EAGLE-3, and — crucially — *why*.
Our contributions:
1. **A per-step headroom methodology**: read-only vLLM instrumentation plus an **oracle-ceiling
   decomposition** separating per-request from per-step headroom, turning "does my controller
   help?" into a measurable quantity computable *before* building a controller.
2. **A negative result with a mechanism**: across Llama-3.1-8B and Qwen3-14B + EAGLE-3, no cheap
   controller beats tuned fixed-K by >3% against +16–21% per-step oracle ceilings. The cause is
   *signal informativeness*, established by exhausting the policy space and measuring
   signal–acceptance correlation directly.
3. **Draft-side vs target-side signal**: the verifier's causal bonus entropy (r≈−0.32) predicts
   acceptance better than the draft entropy/margin (r≈−0.23) that all prior controllers use —
   a redirection of where the signal should come from, yet still too weak to close the gap.
4. **A deployment rule**: cheaply measure the acceptance distribution, its autocorrelation, and
   the signal–acceptance correlation; if the latter is weak (|r|≲0.4), adaptive draft length will
   not pay regardless of the oracle ceiling.

## 2. Related work
**Speculative decoding**: Leviathan et al. (2023); Chen et al. (2023). **Self-speculation /
EAGLE family**: EAGLE-1/2/3 reuse target features for a lightweight trained drafter with dynamic
draft trees. **Adaptive draft length**: SVIP (entropy stop), BanditSpec (UCB over length),
AdaEAGLE (LDLP predictor), SpecKV (MLP predictor), Nightjar/TapOut (bandit planners). All
optimize accepted-tokens-per-step and rely on **draft-side** signals. We do not contradict them;
we show their draft-confidence lever weakens on EAGLE-3 (SVIP-style entropy: +13% on EAGLE-2 →
+2.4% here) and explain why — which none of them measure. We further show a *learned* predictor
(the SpecKV/AdaEAGLE recipe in miniature) reaches only +1% in this regime.

## 3. Method
**Per-step capture.** vLLM exposes no per-step draft signal; its proposer runs in a separate
process through compiled paths. We attach read-only hooks under
`VLLM_ENABLE_V1_MULTIPROCESSING=0` + `enforce_eager=True`: `Eagle3*ForCausalLM.compute_logits`
→ per-position draft entropy + margin; the target model's `compute_logits` → target verification
entropy; `SpecDecodingStats.observe_draft` → per-step accept-run length. Chain drafting is a
prefix, so drafting K′≤K accepts `min(K′, accept_run)`, making offline controller replay **exact**.

**Metrics.** Mean Accepted Tokens/step (MAT, framework-independent) and a cost-model net speedup
`MAT/(1+c·meanK)` with draft/target cost `c=0.15`. Per-request runs use measured wall-clock.
All gains are reported vs the **best tuned fixed K** (the honest baseline).

**Oracle-ceiling decomposition.** We compute two oracles: a *per-request* oracle (best K per
prompt) and a *per-step* oracle (best K per step). Their gap localizes headroom to the within-
stream regime and upper-bounds any realizable controller.

## 4. Experimental setup
Targets/drafts: **Llama-3.1-8B-Instruct + yuhuili/EAGLE3-LLaMA3.1-Instruct-8B**, and
**Qwen3-14B + AngelSlim/Qwen3-14B_eagle3**. Engine vLLM 0.23, single H100, batch 1, greedy.
Workloads: HumanEval (code), GSM8K (math), MT-Bench (chat). Controllers from a shared codebase:
EntropyThreshold (SVIP), EpsilonGreedy, UCB (BanditSpec), AcceptanceHistory, plus a persistence
controller and a learned logistic per-step stopper.

## 5. Results
**Acceptance is short and zero-inflated.** On Llama-3.1-8B, mean accept-length 1.78; **61% of
steps accept zero** draft tokens; lag-1 autocorrelation of the accept-run is 0.247.

**Per-request controllers are futile (Table 1).** The per-request oracle wins only +2%, and every
reactive bandit *loses*.

| Policy | net speedup | vs best fixed |
|---|:---:|:---:|
| Fixed K=2 (best) | 1.538× | — |
| UCB (BanditSpec) | 1.463× | −4.9% |
| ε-greedy | 1.432× | −6.9% |
| AcceptanceHistory | 1.394× | −9.3% |
| Oracle (per-request) | 1.569× | +2.0% |

**Per-step has a large ceiling but cheap controllers can't reach it (Table 2).**

| Policy | speedup~ | vs best fixed |
|---|:---:|:---:|
| Fixed K=2 (best) | 1.211 | — |
| SVIP entropy (τ=6) | 1.240 | +2.4% |
| Margin (τ=0.3) | 1.241 | +2.5% |
| Persistence (best) | 1.245 | +2.9% |
| Learned logistic (held-out) | 1.161 | +1.1% |
| **Oracle (per-step)** | **1.472** | **+21.5%** |

**Signal, not policy (Table 3).** Correlation with the accept-run (more negative = better):

| Signal | r | causal? |
|---|:---:|:---:|
| Draft entropy[0] (SVIP/BanditSpec use) | −0.228 | yes |
| Target entropy[0] (same step) | −0.157 | no |
| **Target bonus entropy, previous step** | **−0.317** | yes |

The causal target signal is the strongest, but a controller gating on it still yields only +2.5%
(|r|=0.32 ⇒ ~10% of variance).

**Generalization (Table 4).** Qwen3-14B replicates the mechanism on a different architecture/size:

| Metric | Llama-3.1-8B | Qwen3-14B |
|---|:---:|:---:|
| Best fixed K | 2 | 1 |
| Per-step oracle ceiling | +21.5% | +16.3% |
| SVIP / Margin / Persistence | +2.4/+2.5/+2.9% | +0.5/+1.7/+2.0% |
| Target-gate controller | +2.5% | −0.7% |

The predictor of controller value is the **signal–acceptance correlation, not the oracle
ceiling**, and it forecasts "ship fixed K" on both pairs.

### 5.5 Scope: the signal is not absent in general — EAGLE-3 saturates it
A natural worry is that acceptance may be *fundamentally* unpredictable. It is not. On a standard
**separate-draft chain** (Llama-3.1-8B target, Llama-3.2-1B draft, 5,084 positions, accept rate
0.79), the same cheap signals predict per-step acceptance *well*:

| Predictor (separate-draft) | AUC |
|---|:---:|
| Draft margin (top1−top2) | **0.878** |
| Target entropy | 0.871 |
| Draft entropy | 0.866 |

Contrast with EAGLE-3, where draft entropy gives r≈−0.23 (AUC ≈ 0.6). The gap is the
**contribution of EAGLE-3's trained, feature-reusing draft head**: by construction it produces
confident drafts whose residual uncertainty no longer tracks acceptance, and it yields short,
zero-inflated accept runs (61% zero). This both (i) explains the literature — SVIP/BanditSpec
report large gains on separate-draft and EAGLE-2 because the signal *is* informative there — and
(ii) scopes our negative result precisely: **adaptive draft length is signal-limited on strong
self-speculation systems, not on speculative decoding in general.** It also implies the right
diagnostic before deploying a controller is to *measure the signal–acceptance AUC on a sample*;
if it is high (≳0.8), a controller will help; if it collapses (≈0.6, EAGLE-3), ship fixed K.

## 6. Discussion
The per-step oracle ceiling (+16–21%) shows genuine within-stream variance: a perfect K-picker
would help substantially. But every cheap signal — entropy, margin, their combination, acceptance
persistence, and a trained predictor — captures ≤~3% of it, because acceptance depends on the
*target's* next-token distribution, unobserved before drafting; the 61% zero-acceptance steps are
not separable from productive ones by any draft-side feature we measured. Even the verifier's own
bonus entropy, the best signal available, is too weak. This yields a practical **deployment rule**:
measure acceptance distribution, autocorrelation, and signal–acceptance correlation cheaply; if
the correlation is weak, adaptive draft length will not pay — ship fixed K.

## 7. Limitations
(i) Speedups use a cost model (c=0.15); MAT is framework-independent but production wall-clock
across batch sizes is future work. (ii) Two model pairs, batch=1, greedy; the serving regime
(batch>1, sampling) is unstudied. (iii) EAGLE-3's production inference is tree-based; our per-step
chain analysis is the natural framing for *length* control but does not model tree drafting. (iv)
We characterize a regime; a different target/draft family (e.g., much larger targets) may have
stronger draft-acceptance correlation, as SVIP's EAGLE-2 result suggests.

## 8. Conclusion
On EAGLE-3 (Llama-3.1-8B and Qwen3-14B), adaptive draft-length control cannot beat tuned fixed K
by more than ~3% despite +16–21% per-step oracle ceilings — a negative result with a precise
cause: the signal, not the policy. Crucially, this is *scoped*: on standard separate-draft SD the
same signals predict acceptance well (AUC ≈ 0.88), so the limitation is EAGLE-3's saturating
trained head, which also resolves the EAGLE-2-vs-EAGLE-3 discrepancy in prior results. We provide
a measurement methodology, a mechanism-backed account, a draft→target signal redirection, a
**signal-AUC deployment diagnostic**, and the scoping result. The open problem we localize:
an acceptance predictor stronger than any draft- or target-side confidence signal, or a different
optimization axis (e.g., certified-quality lossy SD) where the headroom is structural rather than
signal-limited.

*Artifacts:* capture + analysis scripts and per-step traces released; results in
`results/eagle3_controller_findings.md`.
