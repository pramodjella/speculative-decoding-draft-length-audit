# CANONICAL NUMBERS — single source of truth

*Every active document (README, DELIVERABLES, paper_draft, paper.tex, workshop_abstract,
ARCHITECTURE) must match this file. Older documents carry a SUPERSEDED banner and may cite
retired numbers — do not propagate them. Last frozen: 2026-07-09 (post fair-baseline +
batch/temperature scoping, post Yash round-3 cleanups).*

## Per-step oracle ceilings (audited pipeline: bayes_ceiling.json, C=0.15, corrected block key)

| Model / workload | Oracle ceiling |
|---|---:|
| Llama-3.1-8B (instruct), mixed | **+18.1%** |
| Qwen3-14B (instruct), mixed | **+12.4%** |
| DeepSeek-R1-Distill (reasoning), mixed | **+4.5%** |
| Llama-3.1-8B, competition math | **+23.2%** |
| DeepSeek-R1-Distill, competition math | **+8.5%** |

Abstract-level statement: **"+12–23% across models"**. ~**80% of the ceiling is irreducible**
from any draft-side signal (Bayes(hidden) ≈ deployable probe ≪ oracle).

**RETIRED:** "+24.9%" and "+21.5%" — earlier estimates from a different capture/protocol
(pre-audit trace pipeline). They appear only in SUPERSEDED-bannered documents.

## Hidden-state probe (offline; a SECTION, not a headline — per mentor review)

Detection AUC 0.79–0.88 (vs 0.484 weak probe). Offline recovery +16.8 ±2.7% / +17.2 ±2.4% of oracle (instruct models; see the corrected paired table below) =
**+2.0–3.1% net upper bound** — which **loses 2–17% end-to-end**. Reasoning-head null is
confounded by head strength (scoped to the head).

**RETIRED:** "+2.3–3.4% net" and "≈+2.2% net" (Llama-reasoning). Both were derived from the
un-paired ladder retired below — +3.41% (llama8b) and +2.36% (qwen14b) read straight off
`bayes_ceiling.json`. Under the corrected paired ladder the net figures are **+3.1 ±0.6
(llama8b), +2.0 ±0.3 (qwen14b), +2.5 ±1.0 (llama8b_reasoning)**, computed per fold as
(deployable − fixed)/fixed and averaged, not as recovery×span. Guarded by `check_stale.sh`.

## The audited survivor: saturation tail-pruning (PACER/SADDLE class, thr=0.2)

Fair baseline = strongest NATIVE fixed-K engine (true verification budget), paired protocol,
3 fresh containers, vLLM v0.24, B=1 greedy:

| Workload | cross-run gain |
|---|---:|
| HumanEval | +5.6% ±2.0 |
| GSM8K | +5.7% ±2.9 |
| MT-Bench | **+2.0% ±1.6 — a WASH** (worst cell −0.2±2.0) |

**Headline sentence (frozen wording):** *"+2–5%, and one of three workloads is a wash; never a
significant loss."* Do NOT write "+4–5% typical".

- **Temperature T=0.8:** survives — +5.4/+2.1/+1.5% (all positive-significant).
- **Batch:** B=4 → +2.6/+0.4/+0.5%; B=8 → −0.9/−3.1/+0.6%. **Mechanism (microbench-corrected):
  the decay is NOT verification-cost growth (verify is flat even at B=8×q=8) — it is the
  batch-synchronized mean-stop wasting per-request adaptivity. CONFOUND CAVEAT (mandatory
  wherever batch decay is claimed):** a ragged engine (per-request stopping) might decay
  differently; the winning config (B=1) is not the shippable ragged config —
  **the deployable number is open.**
- **MEASURED (verify microbench, `modal_verify_microbench.py`):** verify forward latency has
  one kernel-regime step from q=1 (15.8ms, non-speculative) to the multi-token path, then is
  **dead flat q=2→q=8** (18.5→18.2ms at B=1; native-K=2 q=3 vs padded q=8: <1% apart) — and
  flat even at B=8×q=8. Consequences: (a) fair-baseline claim is now measured; (b) the batch
  decay is NOT verify-cost growth — it is the batch-synchronized mean-stop wasting per-request
  adaptivity (mechanism sentence corrected everywhere). Data `results/verify_microbench.json`.

## Per-step oracle decomposition — CORRECTED (paired within fold, 2026-08-20)

Canonical script: `analyze_bayes_ceiling_paired.py` -> `results/perstep_signal/bayes_ceiling_paired.json`.
Every rung scored on the SAME test rows within each of 8 generation-split folds. Positive
controls pass for **all five settings**: nesting (oracle-threshold >= deployable in every
fold) and the position self-test (Bayes(position) == best fixed K).

| setting | deployable recovery | oracle-threshold recovery | ⇒ irreducible | threshold-selection loss |
|---|---:|---:|---:|---:|
| llama8b | +16.8 ±2.7% | +18.2 ±2.6% | **81.8%** | +1.2% of span |
| qwen14b | +17.2 ±2.4% | +19.5 ±2.4% | **80.5%** | +2.2% of span |
| llama8b_reasoning | +9.5 ±3.7% | +12.0 ±3.8% | **88.0%** | +2.6% of span |
| deepseek | -0.3 ±0.2% | +0.7 ±0.5% | **99.3%** | +0.9% of span |
| deepseek_reasoning | -1.3 ±0.7% | +1.9 ±0.6% | **98.1%** | +3.1% of span |

**Headline figures to quote:**
- **Irreducible share: ~80% for instruct models** (80.5-81.8%),
  rising to ~88% for instruct-on-reasoning-workload and **~98-99% for reasoning models**.
- **Deployable recovery: +16.8 to +17.2% of the oracle span for
  instruct models**; +9.5% for Llama on reasoning workload; **negative (-0.3 to -1.3%) for
  DeepSeek** — the probe actively hurts there.
- **Threshold-selection loss: 0.9-3.1% of the span** across all five settings.

**WORDING RULE (mentor round-6).** Do NOT write "threshold tuning is exhausted" as a crisp
finding. The deployable and oracle-threshold intervals OVERLAP in every setting (e.g. Llama
+16.8 ±2.7 vs +18.2 ±2.6), so this is a **direction, not a resolved result**. Approved
phrasing: *"perfect threshold choice would add only ~1-3% of the oracle span, so threshold
selection is unlikely to be the binding constraint."*

**Naming:** the fourth rung is **"probe with oracle threshold"**, NOT "Bayes(hidden) — the
ceiling for anything reading the hidden state". It removes threshold-selection error only,
not hypothesis-class error.

**SUPERSEDED — do not quote:** the un-paired numbers from `analyze_bayes_ceiling.py`
(deployable 18.9% vs "ceiling" 17.8% on Llama; 18.3% vs 15.2% on Llama-reasoning). Those two
rungs were computed on DIFFERENT data (deployable = 8-fold CV mean; ceiling = threshold swept
and scored on the FULL set), so they were not nested and the ceiling was beaten by the probe
it supposedly bounds. Original script retained for the audit trail only.

## Cost model (the speedup formula) — pin this, two values of C are in use

`speedup = (accepted per step + 1) / (1 + C*K)`

Standard speculative-decoding cost model (Leviathan et al. 2023; Chen et al. 2023) with the
numerator taken as the **measured** mean accepted-per-step rather than the papers' modelled
`(1-a^(K+1))/(1-a)`, which assumes independent per-position acceptance. That assumption is
false on our data: lag-1 autocorrelation of per-position d_TV is **rho = 0.307** vs a
position-preserving shuffled null of 0.000 +- 0.0065 (z = 47), the correlation control (BCSD repo).

**Two values of C are legitimately in use — do not "reconcile" them:**

| C | Where | Why |
|---|---|---|
| **0.15** | `analyze_bayes_ceiling.py`, `analyze_perstep_hidden_full.py`, `audit_decomposition.py` | deliberately CONSERVATIVE proxy; the audited ladder is reported at this value |
| **0.066-0.072** | `measured_cost_ground.py` (the fit) | MEASURED: fitted to the EAGLE-3 fixed-K throughput sweep, RMSE ~ 0.01, reproduces H100 throughput to ~1% |

Re-grounding the ladder from the proxy to the measured C moves Llama recovery
**+18.9% -> +17.3%** of the oracle (**+2.9% net over tuned fixed K**). The claim to make is
therefore *robustness*: the conclusion holds at both coefficients. Do NOT write "the ladder is
fitted to real H100 throughput" — the ladder uses the proxy; the fit is a separate grounding
check. (That exact conflation was found in ARCHITECTURE.md Sec 5 on 2026-08-09 and corrected.)

**Assumptions this formula carries:** (i) target verify cost flat in K — MEASURED true at B=1
(q=2-8: 18.5->18.2 ms) and explicitly not assumed at batch; (ii) zero per-step switching
overhead, so every adaptive-policy number from this formula is an UPPER BOUND (this is why the
probe's offline +2-3% became -2 to -17% in wall-clock); (iii) C constant in K (residual ~1%).

## Statistics convention

Paired per-cycle differences vs the strongest baseline arm; report **mean ± SE over cycles**
(4 cycles unless stated). **No σ-multiplier language** ("5–8σ", "9–11σ" are retired phrasing).
Cross-container magnitude spread exceeds within-run SE — magnitudes require multi-container
ranges, not point claims.

## Verdict spine (what the paper claims)

Against tuned native baselines, every learned/entropy per-step controller loses under
wall-clock; exactly one signal-free class (saturation tail-pruning) survives, only at low
batch (B≲4), and the Bayes-ceiling decomposition explains both facts. Six evaluation
pitfalls + the gated paired protocol are the reusable contribution.
