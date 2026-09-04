> ⚠️ **SUPERSEDED — historical document.** Numbers and claims here predate the final audit
> and may be stale (e.g. pre-audit oracle estimates, retired verdicts). The current source of
> truth is `docs/CANONICAL.md`; the current paper is `report/paper_draft.md`. Kept for
> provenance only — do not cite from this file.

# Workshop submission — locked claim & abstract

*Anchor text for the workshop paper. All numbers are measured; the adaptive-gain figures
are from a per-step cost model (MAT/(1+0.15K)) at B=1 on captured EAGLE-3 traces, H100.*

---

## Title

**How Much of Speculative Decoding's Adaptive Draft-Length Headroom Is Real?
Localizing Per-Step Acceptance Signal to the Drafter's Hidden State**

---

## Abstract

Speculative decoding drafts K tokens and verifies them in parallel; a long line of work
argues that adapting K per step — using entropy, margin, output-variance (SVIP), or learned
controllers — beats a fixed draft length. We show that against a *tuned* fixed K (grid-searched
per workload), this premise mostly evaporates, and we explain precisely why. Capturing per-step
acceptance traces for EAGLE-3 across three target families (Llama-3.1-8B and Qwen3-14B
instruction models, DeepSeek-R1 reasoning), we introduce a **Bayes-ceiling decomposition** that
splits the per-step "oracle" headroom into three parts: the fraction already captured by a tuned
fixed K, the fraction reachable from a draft-time signal, and irreducible aleatoric (verification-
outcome) variance. The widely-cited per-step oracle (+12–18%) is **~80% irreducible** — it
overstates achievable gain roughly five-fold. The reachable ~20% is **invisible to the signals
prior work uses**: cheap logit features recover only +0.3–8%, and a position-only policy recovers
exactly 0%. It is, however, **present in the drafter's hidden state**: a cheap draft-time probe
(PCA + logistic regression, AUC 0.79–0.88 vs 0.484 for a random projection) recovers **~1/5 of the
oracle (+2.3–3.4% net over the tuned baseline, a within-step upper bound)** for the two instruction
models. A label-permutation control (recovery collapses to −4.9%) and an independent re-derivation
confirm the signal is genuine. The one reasoning-distilled head (DeepSeek-R1) shows no recoverable
signal, but its draft head is also substantially weaker (accept-length 1.33 vs 1.92), so we scope
this to the head, not to reasoning models in general. Crucially, an **end-to-end wall-clock audit**
shows the learned gain does **not** survive real execution: even a faithful *in-loop within-step*
probe carrying real signal (AUC 0.80) loses 5–6% vs a tuned fixed depth — verification dominates,
so drafting *shorter* on predicted rejection forfeits accepted tokens worth more than the compute
saved. Extending the audit to a four-policy zoo under a **paired, drift-controlled benchmarking
protocol** (which we show is necessary at the ±5% effect scale), exactly one class survives:
**signal-free saturation tail-pruning** — stop drafting when the cumulative best-path
probability collapses. Replicated across two engines (the EAGLE reference tree drafter and
vLLM's production chain drafter) and three model/head settings, it **ties or beats the strongest
tuned fixed baseline everywhere tested — up to +7.5% (Llama chain), +3–5% (Llama tree),
+0.3–1.3% (Qwen3 chain), ~0 on a weak head — at zero model cost**, while every learned
acceptance-prediction policy loses everywhere. The deployable answer is a tuned deep draft plus
free tail-pruning; the learned-signal premise remains unprofitable, and the decomposition
explains why: acceptance *prediction* chases irreducible variance, saturation *detection* is
deterministic given what the drafter already computed.

---

## Core claim (the one paragraph to defend)

A tuned fixed draft length is a strong baseline that prior adaptive-K methods under-credit by
comparing against default or single-token settings. Against it, the *per-request* adaptive oracle
is only ~+2%, and the much larger *per-step* oracle (+12–18%) is mostly a mirage: our Bayes-ceiling
decomposition shows ~4/5 of it is **irreducible from any draft-side signal** — variance in the
verification outcome that no pre-verification draft-side quantity can predict. The remaining
reachable fraction is not where the literature has looked — not in entropy/margin/SVIP (+0.3–8%)
and not in token position (0%) — but in the drafter's hidden state, from which a cheap draft-time
probe recovers ~1/5 of the oracle (+2.3–3.4% net over the tuned baseline) for the two instruction
targets. The one reasoning-distilled head shows no recoverable signal, but is confounded by a
weaker draft head, so we scope that to the head. And end-to-end, acting on the signal via
early-stopping *loses* wall-clock time (verify dominates). This reframes adaptive draft length
from "tune a controller" to "the reachable signal is in the hidden state, it is small, and it does
not pay off end-to-end via shortening."

---

## Contributions (3)

1. **Bayes-ceiling decomposition** — the first separation of the per-step oracle into
   {priced-by-tuned-K, reachable-from-draft-signal, irreducible}, showing the oracle overstates
   the draft-side-achievable gain ~5× (`analyze_bayes_ceiling.py`; independently re-derived).
2. **Signal localization** — cheap logit features and token position are inert; the hidden state
   carries a permutation-verified +2.3–3.4% net over a tuned fixed K (offline) for the two
   instruction targets (`analyze_bayes_ceiling_control.py`, `audit_probe_recovery.py`).
3. **Wall-clock non-realizability** — even a faithful in-loop within-step probe (AUC 0.80) loses
   5–6% vs a tuned deep draft, because verification dominates; draft-length *shortening* is the
   wrong lever. (The one reasoning head shows no signal but is confounded by lower base acceptance
   — scoped to the head, stated as a limitation.)

Supporting results (same study): tuned fixed K=2 gives 1.54× measured net speedup on
Llama-3.1-8B + EAGLE-3 (B=1, lossless); adaptive headroom collapses to 0 at batch ≥32; and
long-context (8K) headroom is draft-quality-gated, not KV-bandwidth-gated.

---

## Honest scope (state in the paper, do not hide)

- Offline adaptive-gain figures are a per-step cost model at B=1; the range is +2.3–3.4% net
  (recovery +12–19% of the oracle across independent re-derivations). End-to-end, early-stopping
  does not realize it (−5–6% wall-clock).
- The probe is trained in-distribution (held-out generations, same model and workloads);
  cross-workload / cross-model transfer is untested.
- **Draft-head confound (explicit limitation):** the one reasoning head (DeepSeek-R1-Distill) has
  substantially lower base acceptance than the instruction heads (accept-length 1.33 vs 1.92), so
  its null result cannot be attributed to "reasoning" vs "weaker head." Scoped to the head; a
  strong reasoning-model EAGLE head would be needed to separate them (none available).
- We test draft-side signals only ("irreducible from any draft-side signal"). Learned early-stop
  does not pay in wall-clock; the surviving tail-pruning positive (+3–5%, paired protocol) is
  verified on one model/GPU (Llama-3.1-8B, H100) at B=1 with HumanEval borderline —
  cross-model/batch replication open. Drafting *deeper* than the default budget is NOT tested
  end-to-end here (cf. TALON).
