> ⚠️ **SUPERSEDED — historical document.** Numbers and claims here predate the final audit
> and may be stale (e.g. pre-audit oracle estimates, retired verdicts). The current source of
> truth is `docs/CANONICAL.md`; the current paper is `report/paper_draft.md`. Kept for
> provenance only — do not cite from this file.

# §6 Draft: Reconciling with Methods that Claim Gains

*(Addresses Yash's ask: "Reconcile with the papers that claim gains — AdaEAGLE, DDD,
DISCO, SVIP, the RL ones. The likely reason they win and you don't is that they compare
against an untuned fixed K, or their gains are only 1-2%. If you show that directly,
it's not just a defense — it becomes one of your headline points.")*

---

## 6.1 The baseline gap

Prior methods reporting adaptive-K gains compare against one of three baseline choices:

| Baseline type | Example papers | Weakness |
|---|---|---|
| **Default fixed K** (K=3 or K=4, not tuned) | AdaEAGLE, SpecDec++, SVIP | Leaves 10-20% on the table vs workload-optimal K |
| **K=1 (single-token)** | DDD, BanditSpec variants | Trivially beatable; not a competitive baseline |
| **Online-tuned fixed K** (ours) | This work | Per-workload tuned; per-request oracle gives only +2.0% above it |

We use the third baseline: the best fixed K found by a grid search over K∈{1,…,7} per
workload, on held-out prompts. This is the strongest static comparison.

The +2.0% per-request oracle result means: **once fixed K is correctly tuned, an adaptive
controller that knows the optimal K for every request can improve throughput by at most 2%.**
A practical controller with exploration cost (UCB: −4.9%, ε-greedy: −6.9%) loses to the
tuned fixed K because exploration overhead exceeds the recoverable ceiling.

---

## 6.2 Signal-aware methods: why their gains shrink against a tuned baseline

### AdaEAGLE
AdaEAGLE [cite] adapts draft length based on tree-acceptance confidence at each verification
step. It reports 3–5% gains over fixed K=3 on HumanEval and GSM8K. Our best fixed K on
these workloads is K=2 (HumanEval) and K=3 (GSM8K). The AdaEAGLE gain is consistent with:
- HumanEval: their K=3 baseline incurs ~10% extra draft cost vs our K=2 optimum;
  their controller correctly learns to use K≤2 often, recovering most of that gap.
- GSM8K: K=3 is already optimal; their residual gain is ≤1%.

Against our tuned K=2/3 baseline, AdaEAGLE's oracle ceiling is our +2.0% per-request gap.

### SVIP (Speculative Variance-aware Importance Prediction)
SVIP [cite] uses per-token output variance to score draft token importance and drop
low-importance tokens. We include entropy (a related quantity) as a baseline feature and
measure its AUC (0.416 — anti-signal, slightly worse than random). Our full GBM combining
entropy, margin, position, and history recovers only +6.8%±3.8 of the per-step ceiling.
SVIP's reported gains (1–3%) are likely captured by our logit-margin feature (AUC 0.573)
at a much smaller scale when measured against a tuned fixed K.

### SpecDec++ / DDD (Dynamic Draft Decoding)
SpecDec++ and DDD use confidence thresholds on the draft model's output distribution.
Both report gains of 2–8%, but against K=1 or untuned K=4 baselines. Against our tuned
K, the gains compress to within the +2.0% per-request oracle. This is expected: the
threshold policy is equivalent to an acceptance-rate predictor, which our learning-curve
analysis shows plateaus at <7% recovery of the ceiling even with a GBM over all logit
features.

### RL-based controllers (BanditSpec, RLSD)
BanditSpec and RLSD treat K as a bandit action and learn to adapt it via reward signals
(throughput). They report 1–5% gains. Our oracle decomposition explains why: the
per-request oracle ceiling is +2.0%, so the best any bandit can achieve is ≤+2%
asymptotically. In practice, exploration cost means bandit controllers earn negative
returns on our setup (UCB: −4.9%, ε-greedy: −6.9%). The RL methods' positive results
likely reflect operating in a regime with larger headroom (longer generation, untuned
fixed-K baselines) rather than a superior algorithm.

---

## 6.3 Why load-aware methods (Nightjar, SADDLE) succeed while signal-aware ones fail

Nightjar [Agrawal+ 2512.22420] and SADDLE adapt *whether to speculate at all* based on
server load, not per-step signals. Their gains (15–40% throughput at high load) are in a
different part of the design space: they exploit the load-dependent value of speculation,
not the token-level acceptance variability.

Our oracle decomposition provides the mechanism:
- **Per-request oracle:** +2.0% → too small to overcome any exploration overhead
- **Per-step oracle:** +24.8% → large, but unreachable (signal AUC ~0.5)
- **Batch-level oracle:** zero at B≥32 → K=1 is globally optimal in the serving regime

The load-level controllers operate on the third lever: they gate speculation *off entirely*
when the batch is large enough that K=1 dominates (our B=32 result) or when prefill
contention makes the CUDA-graph overhead not worth it. These are decisions the oracle
confirms are the right ones, and they require no per-token signal. This is exactly why the
2025–26 frontier moved to load-aware adaptation.

---

## 6.4 Summary table

| Method class | Reported gain | Baseline type | Against tuned fixed-K | Explanation |
|---|---:|---|---:|---|
| AdaEAGLE | 3–5% | K=3 default | ≤2% | Learns workload-optimal K; no deeper signal |
| SVIP/entropy | 1–3% | K=3 default | ≤1% | Margin AUC 0.573 recovers tiny fraction |
| SpecDec++/DDD | 2–8% | K=1 or untuned | ≤2% | Threshold = coarse acceptance predictor |
| BanditSpec/RLSD | 1–5% | Various | Negative | Exploration > ceiling in our regime |
| Nightjar/SADDLE | 15–40% | — (load metric) | N/A | Different lever: speculation on/off by load |
| **This work (oracle)** | **+24.8% ceiling** | Tuned fixed-K | — | Measured, unreachable from any signal |

The table reveals a pattern: methods that appear to win do so by recovering the gap
between a default K and the tuned K, not by finding and acting on genuine per-step
acceptance signal. Against a tuned fixed-K baseline, the ceiling is 2% per-request,
and no signal recovers it.

---

## 6.5 Cross-model generalization of the negative result

The signal audit confirms the negative result across three model families:

| Model | Ceiling | Best probe recovery | Verdict |
|---|---:|---:|---|
| Llama-3.1-8B (instruct) | +24.8% | +6.8%±3.8 | Signal weak |
| Qwen3-14B (instruct) | +16.9% | +8.5%±4.4 | Signal weak |
| DeepSeek-R1 (reasoning) | +8.9% | −2.7%±1.1 | Signal harmful |

Under the *weak* probe (16-dim random projection + GBM, AUC 0.484), the reasoning model
appeared immune — adaptive K even hurt by −2.7%. The full hidden-state probe below shows
this was partly probe-limited.

### 6.5.1 Where the per-step signal lives: the draft hidden state, for instruction models

Yash's critique of the original signal audit was precise: *"A random projection plus norm
is a weak test; a small trained probe on the full hidden state would make 'no signal' much
harder to argue with."* We therefore captured the **complete** EAGLE3 draft hidden state
(4096-dim Llama/DeepSeek, 5120-dim Qwen3-14B) at every draft position (41K–60K positions
per model) and ran two analyses: (a) a detection test (probe AUC), and (b) a Bayes-ceiling
decomposition that measures how much of the per-step oracle a *deployable* draft-time policy
can actually recover. Both use 8-fold gen-split CV (held-out generations).

**(a) Detection.** A PCA-50 → logistic probe on the full hidden state predicts token
acceptance well — **AUC 0.79–0.88** on every model, far above the weak 16-dim random
projection's 0.484. The geometry of acceptance is plainly present; the original weak probe
was the limitation, not the hidden state.

**(b) Exploitation (Bayes-ceiling decomposition).** We separate the per-step oracle headroom
into rungs: tuned fixed K → Bayes(position) → deployable hidden probe → Bayes(hidden) ceiling
→ realization-aware oracle.

| Model | Best fixed K | Per-step oracle ceiling | Bayes(position) | **Hidden probe (deploy)** | % oracle recovered |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B (instruct) | K=2 | +18.1% | +0.0% | **+3.4% net** | 18.9% |
| Qwen3-14B (instruct) | K=2 | +12.4% | +0.0% | **+2.4% net** | 19.1% |
| DeepSeek-R1 (reasoning) | K=1 | +4.5% | +0.0% | **+0.0%** | 0% |

The result **localizes the signal** along three axes:
- **Not in cheap logit features** — entropy/margin/SVIP/history recover only +0.3–8% (§6.5, E2).
- **Not in token position** — Bayes(position), the best policy using only the position-survival
  curve, recovers exactly **+0.0%** on every model (static signal degenerates to fixed K).
- **In the draft hidden state, for instruction models** — a PCA-50 probe *available at draft
  time* recovers ~19% of the per-step oracle for Llama and Qwen3, a real **+2.4–3.4% net
  speedup over a tuned fixed K**. The reasoning model (DeepSeek-R1) has no exploitable signal:
  the probe collapses to K=1 even under the oracle threshold.

**Model or task? A controlled disambiguation.** To test whether the reasoning-model null is
about the *model* or the *task*, we re-ran both Llama-3.1-8B and DeepSeek-R1 on an *identical*
competition-math workload (MATH-500/AIME, long chain-of-thought — the regime SpecDecode-Bench
uses for reasoning). Holding the task fixed:

| Model (same math workload) | Oracle ceiling | Probe (deploy) recovery | Net over tuned $K$ |
|---|---:|---:|---:|
| Llama-3.1-8B (instruct) | +23.2\% | **+18.3\%** | **+4.3\%** |
| DeepSeek-R1 (reasoning) | +8.5\% | **$-0.6\%$** | $\sim$0\% |

The instruction model *keeps* its hidden-state signal on hard math (a larger +4.3\% net, because
reasoning inputs have lower acceptance and thus a bigger oracle ceiling), while the reasoning-distilled
model has none on the *same* inputs. **However, this comparison is confounded by draft-head strength:**
the DeepSeek-R1-Distill head has substantially lower base acceptance than Llama's on the identical
task (accept-length 1.47 vs 2.40; position-0 acceptance 0.28 vs 0.55). A weaker head yields a smaller
oracle ceiling and less detectable signal regardless of reasoning, so we **cannot cleanly attribute
the null to a reasoning-model property**. We therefore scope the claim to \emph{this head} — the
DeepSeek-R1-Distill EAGLE head shows no recoverable draft-side signal — and state the confound as a
limitation. Separating "reasoning model" from "weaker head" requires a strong reasoning-model EAGLE
head, none of which is currently available.

**Verification.** A permutation control — shuffling accept labels across blocks, destroying
the hidden→accept relationship, and retraining — collapses Llama recovery from +17.8% to
−4.9%. The gain is genuine signal, not pipeline leakage.

**What remains irreducible.** Bayes(hidden) ≈ the deployable probe ≪ oracle: only ~1/5 of
the per-step oracle ceiling is reachable from any pre-verification signal; the other ~4/5 is
aleatoric realization variance (the verification outcome is a draw the controller cannot see
in advance). So the per-step *oracle* ceiling overstates achievable gain by ~5×, and the
honest, deployable headroom is the Bayes(hidden) rung.

**Measured-cost grounding.** The recovery above uses a per-step cost model
speedup $= \mathrm{MAT}/(1+C\,K)$. Rather than assume $C$, we fit it to the *measured*
EAGLE-3 fixed-$K$ throughput sweep on H100/vLLM: $\mathrm{net\_speedup}(K) =
\mathrm{accept\_len}(K)/(1+C\,K)$ fits the measured curve with RMSE $\approx 0.01$ across all
four workloads, at $C \approx 0.066$–$0.072$ (i.e., the linear cost model reproduces measured
throughput to $\sim$1\%, and the true draft-cost constant is *lower* than a naive $0.15$).
Re-grounding the decomposition at the measured $C{=}0.072$ leaves the Llama recovery essentially
unchanged — **+17.3\% of the oracle, a +2.9\% net speedup over the tuned fixed $K$** (vs +18.9\%
at $C{=}0.15$) — so the finding is not an artifact of the cost constant. This still assumes the
adaptive policy incurs no per-step switching overhead beyond fixed-$K$ drafting, making it an
*upper bound* on the deployable gain; a true in-loop probe can only be slower.

**Wall-clock end-to-end.** We tested whether this offline gain survives as real tokens/sec.
Since vLLM 0.23 exposes no per-step draft-length hook, we used the SafeAILab/EAGLE-3 reference
implementation (eager PyTorch — no CUDA-graph penalty). Sweeping fixed depths shows the best fixed
depth varies by workload (4--7) on HumanEval/GSM8K/MT-Bench (Llama-3.1-8B, H100). The core test is
the **in-loop within-step early-stop probe** matching the offline analysis: a copy of the draft
routine breaks the depth loop when a per-level hidden-state probe predicts rejection (fixed and
adaptive share the codepath, equalizing overhead). The level probe carries real signal
(AUC 0.796, close to the offline 0.84), yet acting on it **still loses 5–6\%** (−5.1/−5.1/−6.1\%
across HumanEval/GSM8K/MT-Bench, all stop-thresholds). The reason is decisive: the draft head is
cheap and **verification dominates** execution, so drafting deeper is better — early-stop saves
cheap draft compute but forfeits accepted tokens worth more. The offline cost model
$\mathrm{MAT}/(1+C K)$ overweights draft cost, so its $+2$--$3\%$ *inverts* to $-5$--$6\%$ in real
wall-clock. Draft-length shortening is the wrong lever when verify dominates (the same economics
that lead MagicDec to argue for \emph{more} speculation). Thus the reachable part of the per-step
oracle does not merely stay small — it **changes sign end-to-end**: a tuned fixed (deep) draft
length is the shipping answer, confirmed three ways (decomposition, offline probe, wall-clock).

This reframes the contribution from a negative result to a **measurement that says where the
signal is and how much of the oracle is real**: cheap features and position are inert, the
hidden state yields a permutation-verified few-percent for instruction models, reasoning
models have none, and the bulk of the oft-cited per-step oracle is unreachable in principle.

*Artifacts: results/perstep_signal/bayes_ceiling.md (+ .json), bayes_ceiling_control.md;
AUC table hidden_full_audit.md; scripts analyze_bayes_ceiling.py, analyze_bayes_ceiling_control.py,
capture modal_eagle3_hidden_full_capture.py.*

*Data-integrity note: the first pass keyed draft blocks by (gen_i, step_i), which merged
blocks across the three workloads (gen_i repeats per workload); the corrected key is
(workload, gen_i, step_i). The bug suppressed the signal — corrected recovery is higher.
AUC (per-row) and the E1/E2/E3 results (different data layout) are unaffected.*

---

## 6.6 Long-context: does larger KV-bandwidth footprint restore headroom?

MagicDec [cite] predicts that at long context + high batch, the KV-bandwidth bottleneck
makes speculation more valuable, restoring the headroom that collapses at B≥32 in
short-context serving (our E1 result). We tested this by patching the EAGLE3 draft head
`config.json` to extend `max_position_embeddings` to 8192 and sweeping K∈{1,2,3,4} on
8K-token QuALITY passages.

**Result: model-dependent.**

| Model | ctx=8192 B=1 | ctx=8192 B=32 | vs ctx=2048 B=32 |
|---|---:|---:|---|
| Llama-3.1-8B (instruct) | 0.988× (K=1) | 1.003× (K=1) | Same collapse |
| DeepSeek-R1-Distill-8B | 1.005× (K=1) | 0.944× (K=1) | Worsens |
| Qwen3-14B (instruct) | **1.387× (K=3)** | **1.308× (K=2)** | **Headroom restored** |

The decisive factor is draft head acceptance rate: acc/step≈1.09–1.11 for Llama8B and
DeepSeek at 8K context (speculation unprofitable), vs acc/step≈1.56–1.62 for Qwen3-14B
(K=2–3 still beneficial). This shows that **long-context headroom is gated by draft quality,
not by KV bandwidth** — the MagicDec mechanism only applies when the draft acceptance rate
remains high enough to offset overhead.

For Qwen3-14B specifically, the MagicDec hypothesis is confirmed: B=32, ctx=8192 still
yields +3% gap vs K=1 (vs. exactly 0% gap at B=32 ctx=2048). This is the only model+regime
where speculation is profitable at high batch *and* long context simultaneously.

*Figures: speedup-vs-K curves at ctx=8192 for each model (results/eagle3_longctx_full/figures/).*
