# Deliverables — Adaptive Draft-Length Controllers (Roadmap Completion)

Maps every ROADMAP.pdf milestone/acceptance-check to its artifact. The flagship
regime the roadmap names — **Llama-3.1-8B + EAGLE-3 (`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`)** —
is measured end-to-end on Modal H100 and analysed here; earlier Qwen2.5 simulation/physical/vLLM
tracks remain in place and corroborate it.

## Headline result (Llama-3.1-8B + EAGLE-3, vLLM 0.23, H100, B=1, greedy, lossless)

| question | answer |
|---|---|
| Best fixed K (mixed) | **K=2 → 1.54× net speedup** (HumanEval/GSM8K 1.64–1.65×, MT-Bench 1.36×) |
| RQ1: cheap per-step controller beats best fixed? | **No.** Per-request oracle ceiling only +2.0%; online controllers lose (UCB −4.9%, ε-greedy −6.9%, History −9.3%) to cold-start regret |
| Key nuance | **Per-step** oracle ceiling is **+18.1%** for Llama (audited pipeline; +12–23% across models, see [docs/CANONICAL.md](docs/CANONICAL.md)); cheap signals barely dent it (SVIP entropy +0.3%); a hidden-state probe recovers ~1/5 *offline* (+2.3–3.4% upper bound) but **loses end-to-end** — the only wall-clock survivor is signal-free saturation tail-pruning (E5/E6) |
| RQ2: best signal | top-1 margin (1.273) ≳ entropy/SVIP (1.258) > fixed (1.254); optimum K small (2–3) on every workload |
| RQ3: generalisation | one fixed K=2 within ~1% of per-workload optimum (worst retune gap −2.4%) |

Full prose: [results/insight_report_eagle3_8b.md](results/insight_report_eagle3_8b.md).

## M1 — Literature Review & Scoping
- 12-paper matrix → [literature_matrix.xlsx](literature_matrix.xlsx) (repo root)
- 3 RQs + controller designs + eval plan → manuscript §I
- Fixed-length sweep / best static length → [notebooks/00_fixed_length_sweep.ipynb](notebooks/00_fixed_length_sweep.ipynb); **EAGLE-3 8B sweep** → [results/eagle3_8b/fixedK_by_workload.csv](results/eagle3_8b/fixedK_by_workload.csv)

## M2 — Controller Implementation & Harness
- Controllers (entropy, ε-greedy, UCB, history, LinUCB, oracle) → [src/controllers/](src/controllers/)
- Speculation backbone + per-step hooks → [modal_eagle3_perstep_capture.py](modal_eagle3_perstep_capture.py) (read-only draft+target entropy hooks)
- Mixed-traffic harness / per-step logging → [src/bench/](src/bench/), [src/serve/](src/serve/)
- EDA → [notebooks/01_EDA.ipynb](notebooks/01_EDA.ipynb)
- **Output equivalence (lossless)** → exact-verification EAGLE-3; fp32 100% token-identical, [results/equivalence_postfix_fp32.json](results/equivalence_postfix_fp32.json)

## M3 — Experiments & Ablations
- Controller-vs-fixed (per-request) → [results/eagle3_8b/policies.csv](results/eagle3_8b/policies.csv) (`src/simulate_eagle_controllers.py`)
- Per-step SVIP vs fixed vs oracle → [results/eagle3_8b/perstep_svip.csv](results/eagle3_8b/perstep_svip.csv) (`src/simulate_eagle_perstep.py`)
- Ablations A1 (signal), A2 (controller), A4 (candidate set), A5 (generalisation) → [results/eagle3_8b/ablations.csv](results/eagle3_8b/ablations.csv)
- Error taxonomy (over/under/matched, with examples) → [results/eagle3_8b/error_taxonomy.md](results/eagle3_8b/error_taxonomy.md)
- A3 (batch sweep): shown in simulation (batch interference 1.78×→1.32× at B=64, manuscript §V); per-step EAGLE-3 capture is B=1 (single-stream attribution) — batched EAGLE-3 is future work.
- Generator (one script per table/figure) → [analyze_eagle3_8b.py](analyze_eagle3_8b.py)

## M4 — Analysis, Insights & Manuscript
- Figures (≥200 dpi) → [results/figures_eagle3_8b/](results/figures_eagle3_8b/): speedup-vs-K, accept-vs-K, wasted-tokens, speedup-vs-policy, length-vs-entropy
- Insight report (RQ1–3 with numbers) → [results/insight_report_eagle3_8b.md](results/insight_report_eagle3_8b.md)
- Manuscript → [paper/paper.tex](paper/paper.tex) — **fully ported 2026-07-04** from
  [report/paper_draft.md](report/paper_draft.md) (source of truth; post-audit frozen numbers +
  policy-zoo amendment; 10 sections, 3 tables, repro section). Pending before submission:
  DeepSeek paired-replication row (marked `\pending`), figures, full BibTeX author lists for
  2026 arXiv cites.
- Reproducibility → `modal run modal_eagle3_perstep_capture.py --n 30 --maxtok 128 --maxk 7` then `python analyze_eagle3_8b.py`

## Reproduce the EAGLE-3 8B deliverable
```bash
# 1. capture per-step traces on H100 (writes to Modal volume spec-dec-m5-results)
modal run modal_eagle3_perstep_capture.py --n 30 --maxtok 128 --maxk 7 --tag target_llama8b
modal volume get spec-dec-m5-results eagle3_perstep_target_llama8b.json results/
# 2. fixed-K sweep already at results/eagle3_multik_llama8b.json (modal_vllm_eagle3.py)
# 3. analyse -> CSVs, figures, insight report
python analyze_eagle3_8b.py
```

## Extended Evaluations (2026-06-30, complete)

### E1 — Batch Sweep (Short-context, ctx=2048, B=1→64)

Full sweep confirming that adaptive-K headroom collapses at B≥32.

| batch | best K (humaneval) | speedup | gap vs K=1 |
|------:|-------------------:|--------:|----------:|
| 1 | 2 | 1.642× | +19.9% |
| 8 | 1 | 1.288× | 0% |
| **16** | **2** | **1.376×** | **+10.7%** |
| 32 | 1 | 1.111× | 0% |
| 64 | 1 | 1.077× | 0% |

**Inflection point: B=16→B=32.** At B=16, K>1 still delivers +5–11% over K=1.
At B=32, complete collapse to K=1 on all workloads (humaneval, gsm8k, mt_bench).

Artifacts: [results/eagle3_batch/gap_summary.md](results/eagle3_batch/gap_summary.md),
[results/eagle3_batch/curves.csv](results/eagle3_batch/curves.csv),
[results/eagle3_batch/figures/speedup_vs_K_by_batch.png](results/eagle3_batch/figures/speedup_vs_K_by_batch.png).
Reproduce: `modal run modal_vllm_eagle3.py --batch N` (N=8,16,32,64), then `python analyze_eagle3_batch.py`.

### E2 — Per-Step Signal Audit (Cross-model, 3 architectures)

Confirms that per-step K adaptation is structurally impossible across model families.

| model | oracle ceiling | draft-feature recovery | target-entropy recovery | verdict |
|-------|---------------:|----------------------:|------------------------:|---------|
| Llama-3.1-8B (instruct) | +24.8% | +6.8%±3.8 | +6.8%±5.0 | barely helps |
| Qwen3-14B (instruct) | +16.9% | +8.5%±4.4 | +9.0%±4.4 | barely helps |
| DeepSeek-R1-LLaMA-8B (reasoning) | +8.9% | **−2.7%±1.1** | **−4.9%±2.9** | **predictor hurts** |

**Strongest negative result:** For reasoning models (CoT output), fixed K=1 is already
near-optimal and any predictor introduces harmful noise. Even post-verification target
entropy — which is not available to real controllers — cannot recover the oracle ceiling.

Artifacts: [results/perstep_signal/target_entropy_audit.md](results/perstep_signal/target_entropy_audit.md).
Reproduce: `modal run modal_eagle3_perstep_capture.py --model deepseek_r1_llama8b`, then `python analyze_perstep_target_entropy.py`.

### E3 — Long-Context Evaluation (ctx=8192, vLLM-patched draft head)

Goal: test MagicDec hypothesis that K>1 benefit persists at 8K context even at B≥32.

**Method:** Patched EAGLE3 draft head `config.json` to set `max_position_embeddings=8192`
(removing the 2048-token Triton kernel bound). Long-context prompts from QuALITY dataset.

**Result: Model-dependent. MagicDec partially confirmed for Qwen3-14B only.**

| Model | B=1 speedup | B=32 speedup | Best K (B=1) | Best K (B=32) | MagicDec? |
|---|---:|---:|---:|---:|---|
| Llama-3.1-8B (instruct) | **0.988×** (K=1) | **1.003×** (K=1) | 1 | 1 | No — collapses |
| DeepSeek-R1-Distill-8B (reasoning) | **1.005×** (K=1) | **0.944×** (K=1) | 1 | 1 | No — hurts at B=32 |
| Qwen3-14B (instruct) | **1.387×** (K=3) | **1.308×** (K=2) | 3 | 2 | **Yes — headroom maintained** |

**Key findings:**
1. **Draft head quality is the decisive factor**, not batch size or KV bandwidth.
   Qwen3's EAGLE3 draft head (`AngelSlim/Qwen3-14B_eagle3`) maintains high acceptance
   rate (acc/step≈1.56–1.62) at long context; Llama8B and DeepSeek-R1 heads collapse
   (acc/step≈1.09–1.11), making speculation unprofitable.
2. **Qwen3-14B uniquely preserves headroom at B=32 long-ctx**: this is the only model+regime
   where speculation is beneficial at high batch *and* long context simultaneously.
3. **Short-context batch collapse is complete (B=32 gap=0%) while long-ctx Qwen3-14B B=32
   retains +3% gap** — confirming that context length and draft quality interact independently
   from batch-size collapse.
4. acc/step at 8K context: Qwen3≈1.59, Llama8B≈1.09, DeepSeek≈1.11 (vs ~2.0–2.5 at 2K).

Artifacts: [results/eagle3_longctx_patched/](results/eagle3_longctx_patched/) (6 JSONs),
[results/eagle3_longctx_full/gap_summary.md](results/eagle3_longctx_full/gap_summary.md),
figures at [results/eagle3_longctx_full/figures/](results/eagle3_longctx_full/figures/).

Reproduce:
```bash
PYTHONIOENCODING=utf-8 modal run modal_vllm_eagle3_longctx_patched.py --model llama8b --batch 1 --tag lcp_b1
# ... (see script for all 6 combinations)
python analyze_eagle3_longctx_full.py
```

### E4 — Full Hidden-State Probe (strengthened per-step signal test)

Goal: address Yash's critique: "A random projection plus norm is a weak test; a small
trained probe on the full hidden state would make 'no signal' much harder to argue with."

**Method:** Captured complete EAGLE3 draft hidden state (4096-dim Llama/DeepSeek,
5120-dim Qwen3-14B) at every draft position (41K–60K positions/model). Trained
logistic regression + 2-layer MLP (256→64) + PCA-50 → logistic regression in 8-fold
gen-split CV. Threshold chosen on TRAIN; cost model MAT/(1+0.15·K).

**Result: the draft hidden state carries genuinely exploitable per-step signal — for
instruction models. This is a positive result, and it localizes the signal precisely.**

> **Data-integrity note.** The first pass of this audit grouped draft blocks by
> `(gen_i, step_i)`, but `gen_i` repeats across the 3 workloads, so blocks from
> humaneval/gsm8k/mt_bench were merged (~2.4 per key). The correct key is
> `(workload, gen_i, step_i)` (every block then has exactly 7 positions, constant `acc`).
> The bug *suppressed* the signal; all recovery numbers below are the corrected values.
> AUC is per-row and was unaffected. E1/E2/E3 use a different (nested, safe) data layout
> and are not affected.

Two measurements, both on held-out generations (8-fold gen-split CV):

**(a) Detection (AUC, full hidden state vs the old 16-dim random projection):**

| Model | RP-16 AUC (old weak probe) | Full-probe AUC |
|---|---:|---:|
| Llama-3.1-8B (instruct) | 0.484 | **0.842** |
| Qwen3-14B (instruct) | 0.484 | **0.876** |
| DeepSeek-R1 (reasoning) | 0.484 | **0.870** |

**(b) Exploitation — Bayes-ceiling decomposition (`analyze_bayes_ceiling.py`):**

| Model | Best fixed K | Per-step oracle ceiling | Bayes(position) | **Probe (deploy)** net gain | % of oracle recovered |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B (instruct) | K=2 | +18.1% | +0.0% | **+3.4%** | 18.9% |
| Qwen3-14B (instruct) | K=2 | +12.4% | +0.0% | **+2.4%** | 19.1% |
| DeepSeek-R1 (reasoning) | K=1 | +4.5% | +0.0% | **+0.0%** | 0% |

**Key findings:**
1. **The per-step signal lives in the draft hidden state — not in cheap features, not in
   position.** Cheap logit features fail (E2: +0.3–8%); static position signal degenerates
   to fixed K (Bayes(position) = +0.0% everywhere); but a PCA-50 logistic probe on the full
   hidden state, **available at draft time**, recovers ~19% of the per-step oracle for
   instruction models — a real **+2.4–3.4% net speedup over a *tuned* fixed K** (B=1, cost
   model MAT/(1+0.15K)).
2. **Model-class split.** Instruction models (Llama, Qwen3) have exploitable signal;
   the reasoning model (DeepSeek-R1) has none — the probe collapses to K=1 even with the
   oracle threshold. This matches E2's reasoning-model negative.
3. **Permutation-verified.** Shuffling accept labels across blocks collapses Llama recovery
   from +17.8% to −4.9% (`analyze_bayes_ceiling_control.py`) — the gain is genuine signal,
   not a leak.
4. **The rest of the oracle is irreducible.** Bayes(hidden) ≈ deploy ≪ oracle: ~4/5 of the
   per-step oracle ceiling is aleatoric realization variance that no pre-verification signal
   can reach. The oracle ceiling overstates achievable gain; the recoverable fraction is ~19%.
5. **Measured-cost grounded (not just a proxy).** Fitting the cost model to the measured
   EAGLE-3 fixed-K throughput sweep gives C≈0.066–0.072 (RMSE≈0.01; the model reproduces
   measured H100 throughput to ~1%). Re-grounding at the measured C=0.072 leaves Llama recovery
   at +17.3% of the oracle (**+2.9% net over tuned fixed K**) — stable vs the +18.9% proxy.
   Still an upper bound (assumes no per-step switching overhead); wall-clock in-loop is the
   remaining step. Script: `measured_cost_ground.py`.
6. **Reasoning head: no recoverable signal — but CONFOUNDED by head strength (narrowed).**
   Re-running Llama-3.1-8B and DeepSeek-R1 on identical competition-math (MATH-500/AIME): Llama
   keeps signal (+18.3% recovery, +4.3% net), DeepSeek has none (~0%). **Caveat (audit, Yash):**
   the DeepSeek-R1-Distill head is substantially weaker (accept-length 1.47 vs Llama 2.40 on the
   same task; pos-0 acceptance 0.276 vs 0.547), so the null cannot be cleanly attributed to
   "reasoning model" vs "weaker head." Claim scoped to *this head*, stated as a limitation; a
   strong reasoning-model EAGLE head would be needed to separate them (none available).
   Artifacts: `results/perstep_signal/bayes_ceiling_reasoning.md`, `pre_write_audit.md`.

**Net (offline):** for instruction models a draft-time *within-step* hidden-state probe
recovers ~1/5 of the per-step oracle (+2.4–3.4% net over tuned fixed K, cost-model + measured-cost
grounded); for reasoning models fixed K=1 is optimal. **But see E5:** a *deployable* per-step
controller does not realize this end-to-end, so tuned fixed length remains the shipping answer.

Artifacts: [results/perstep_signal/bayes_ceiling.md](results/perstep_signal/bayes_ceiling.md)
(+ .json), [results/perstep_signal/bayes_ceiling_control.md](results/perstep_signal/bayes_ceiling_control.md),
AUC table [results/perstep_signal/hidden_full_audit.md](results/perstep_signal/hidden_full_audit.md)
(recovery column superseded by bayes_ceiling).
Reproduce: `python analyze_bayes_ceiling.py` then `python analyze_bayes_ceiling_control.py`.

### E5 — Wall-clock end-to-end (deployable adaptive depth vs tuned fixed)

Goal: does the offline within-step gain survive as a *real* tokens/sec speedup? vLLM 0.23 has
no per-step K hook, so we test in the **SafeAILab/EAGLE-3 reference repo** (plain PyTorch, eager
— no CUDA-graph penalty). A PCA-50+ridge probe sets draft depth per step from the seed hidden
state; fixed and adaptive run the same codepath (equal overhead). Llama-3.1-8B, H100, greedy.

Two controllers tested; **both lose to a tuned fixed depth — including the faithful in-loop probe.**

**(A) Seed-based reactive** (`modal_eagle_wallclock.py`): probe sets depth from the seed hidden
state. Code audit (pre_write_audit.md §5) found the first run's capture mispaired seeds↔labels
across prompts (its "no signal, corr 0.05" was an artifact); fixed and rerun. **Corrected: the
seed carries modest signal (corr 0.26) — and the controller STILL loses** −5.7% / −2.4% / −5.5%
vs best fixed depth (d=6/7/7). Best fixed depth varies by workload — workload-level headroom is
real and captured by a per-workload tuned fixed.

**(B) In-loop within-step early-stop** (`modal_eagle_inloop.py`) — the FAITHFUL test matching the
offline analysis (copy of topK_genrate breaks the depth loop when a per-level hidden-state probe
predicts rejection; tt_eff caps the shorter tree; copy verified to generate coherently):

| Workload | Fixed (full depth) tok/s | In-loop adaptive tok/s | Gain | mean depth |
|---|---:|---:|---:|---:|
| HumanEval | 144.5 | 137.2 | **−5.1%** | 6.6 |
| GSM8K | 128.4 | 121.8 | **−5.1%** | 4.5 |
| MT-Bench | 133.4 | 125.3 | **−6.1%** | 5.7 |

Within-step level probe **AUC = 0.796** (real signal, ≈ offline 0.84) — yet acting on it LOSES
5–6% across all stop-thresholds. **Key insight:** the draft head is cheap and *verification
dominates*, so drafting deeper is better; early-stop saves cheap draft compute but forfeits
accepted tokens worth more. The offline cost model overweights draft cost, so its +2–3% inverts
to −5–6% in real wall-clock. Draft-length *shortening* is the wrong lever when verify dominates.

**Conclusion (measured 3 ways):** adaptive draft-length shortening does not pay off —
(1) ~80% of the oracle is irreducible; (2) offline the reachable ~20% is real but small; (3) end-to-end
even a real-signal within-step probe (AUC 0.80) LOSES 5–6%. Ship a tuned fixed (deep) draft length.
This is a complete **measurement** result, not a method claim.

Artifacts: [results/perstep_signal/wallclock_eagle.md](results/perstep_signal/wallclock_eagle.md),
`results/{eagle_wallclock,eagle_inloop}.json` (Modal volume). Scripts: `modal_eagle_wallclock.py`,
`modal_eagle_inloop.py`.

### E6 — Policy-Zoo Audit + Paired Benchmarking (main-track extension)

Goal: does ANY published-style draft-stopping policy beat a tuned fixed depth under honest
evaluation? Four policies (SVIP-entropy, margin, SADDLE/PACER-style cumulative-probability,
SpecDec++-style trained hidden probe) through the same verified in-loop codepath
(`modal_eagle_zoo.py`), then an escalating verification ladder that itself became a finding.

**Three evaluation failure modes demonstrated and caught (each inflates adaptive-SD claims):**
1. **Mechanism mismatch** — chain-regime offline analysis deployed into a tree engine
   (the earlier +2–3%→−5–6% inversion).
2. **In-distribution threshold selection** — zoo run: cumprob +2.4% on the prompts its
   threshold was tuned on; −1.9% on held-out (verify run 1).
3. **Unpaired sequential benching** — three runs gave +2.4/−1.9/+6.4% with identical
   thresholds; raw per-cycle data shows several-% container drift within a run. Paired
   round-robin (order rotated per cycle) is required at the ±5% effect scale
   (`modal_eagle_paired.py`).

**Final paired, drift-controlled verdict (held-out prompts, vs STRONGEST per-workload fixed):**

| Policy | GSM8K | MT-Bench | HumanEval | Verdict |
|---|---:|---:|---:|---|
| cumprob (saturation tail-prune, thr=0.05) | **+3.58%±0.32** | **+4.77%±0.50** | +2.90%±1.83 | **survives** (realized depth 5.62) |
| entropy (SVIP-style) | — | — | — | loses/tie (unpaired runs; never survived held-out) |
| hidden probe (SpecDec++-style) | — | — | — | loses everywhere (−12 to −17%, incl. probe overhead) |

**The survivor uses NO learned signal** — it thresholds the cumulative best-path probability the
drafter already computes (zero model cost; policy class = SADDLE/PACER/DDD prior art — the
audited verdict under tuned-baseline + paired protocol is ours). Acceptance-prediction policies
chase irreducible variance and lose; saturation detection is deterministic given the tree.

**Scoped amendment to the headline:** learned per-step shortening still doesn't pay (core
unchanged); the deployable answer in the tree regime is a tuned deep draft **plus free
saturation tail-pruning (+3–5%, strong instruction heads at B=1)**.

**Cross-head replication (done):** on the weaker DeepSeek-R1-Distill head the paired protocol
gives a TIE (−0.7/−0.9/+1.4%, n.s.; realized depth 6.19, pruning only ~0.8 levels) → the gain is
**head-dependent**. The run also surfaced protocol failure-mode 3b: an intra-cycle throughput
step-change (fixed arms jumped 14–18% mid-cycle) that pairing cannot cancel → use ≥6 cycles /
median-of-cycles. Data: `results/eagle_paired_deepseek.json`.

**vLLM-NATIVE replication (done — production engine, chain regime):** patched the LIVE V1 draft
loop (`worker/gpu/spec_decode/autoregressive/speculator.py::_multi_step_decode`, located by
stack-trace; the parallel `vllm.v1.spec_decode.*` package is DEAD code for eagle3 — patching it
was a silent no-op caught only by the arm-separation gate = failure-mode #4; attempt 1
quarantined). Equal-verify design (all arms verify KMAX=7 slots; break-then-pad). Gates: 19%
fixed-arm separation; best fixed k=2–3 matches E-track tuned K=2. **Llama-8B: cum0.2 beats
strongest fixed +6.4/+7.5/+2.9% (paired mean±SE over cycles), mean draft 2.3. Qwen3-14B: +1.3/+0.3/+0.7%
(tie-to-slight-win), mean draft 3.05; thr=0.05 harmful there.** Production K=2 engine baseline
(verify-slot savings included) = future work. Data: `results/vllm_tailprune2_*.json`;
harness `modal_vllm_tailprune2.py`; diagnostics `modal_vllm_tailprune_diag*.py` + pkg dumps.

**Final survivor claim (all engines/models, paired+gated):** signal-free saturation pruning at a
conservative threshold **ties or beats tuned fixed everywhere tested** (Llama chain +2.9–7.5%,
Llama tree +3–5%, Qwen3 chain +0.3–1.3%, weak-head tree ~0); every learned policy loses.

Artifacts: [results/perstep_signal/policy_zoo.md](results/perstep_signal/policy_zoo.md),
`results/{eagle_zoo,eagle_zoo_verify,eagle_zoo_verify_iid,eagle_paired}.json`.
Scripts: `modal_eagle_zoo.py`, `modal_eagle_zoo_verify.py`, `modal_eagle_paired.py`.

## Honest scope note
On a strong, cheap EAGLE-3 head the deployable answer is a tuned fixed K — adaptive
control's realisable gain is ~2% at request level. The scientifically interesting,
reproducible finding is the **gap between the large per-step oracle headroom (+25%) and
the tiny gain from draft entropy (+0.3%)**, which redirects future work from bandit
tuning toward stronger per-step acceptance signals.

The extended evaluations (E1–E4) generalize this finding:
- E1: at B≥32, adaptive K headroom collapses to exactly 0 — the serving regime removes
  even the theoretical motivation for per-step controllers
- E2: across instruction AND reasoning model families, no signal (including oracle
  post-verification signals) can recover the ceiling; reasoning models are actively hurt
- E3: long-context headroom is **model-dependent** (Qwen3-14B maintains 1.3–1.4×, Llama8B/DeepSeek
  collapse); draft head quality, not batch size, is the decisive factor
- E4: full hidden-state probe (4096/5120-dim) localizes the per-step signal — it is NOT in
  cheap features (E2) or token position (Bayes(position)=0%), but a draft-time PCA-50 probe
  on the hidden state recovers ~19% of the per-step oracle for **instruction** models
  (+2.4–3.4% net over a *tuned* fixed K; Llama, Qwen3) and 0% for the **reasoning** model
  (DeepSeek). Permutation-verified (shuffled labels → recovery collapses to ~0). The
  remaining ~4/5 of the oracle is irreducible realization variance. This is a positive,
  model-class-dependent result — it revises the earlier (block-key-bugged) "doesn't convert"
  claim upward to a real few-percent for instruct models.
