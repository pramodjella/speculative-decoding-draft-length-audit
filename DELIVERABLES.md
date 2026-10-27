# Deliverables: experiments, scripts and data

This file maps each experiment behind the paper (E1 to E7) to the scripts that ran it and the
files that hold its results. The first table lists the numbers the paper uses. The sections
after it give the setup and result of each experiment. Numbers that were later corrected are
kept at the end, fenced and marked as superseded, so the history of each figure can be
followed; none of them should be quoted.

Hardware for every GPU run: one NVIDIA H100 80GB through Modal. Greedy decoding, batch size 1
unless stated. The vLLM tail-pruning runs use eager mode (`enforce_eager=True`, no CUDA graphs). The `modal_*.py` scripts need an H100 and a Modal account; the `analyze_*.py`
scripts run on a CPU from the released files.

## Current numbers

| Finding | Number | Experiment | Data |
|---|---|---|---|
| Per-step oracle gain over the best fixed K | +18.1% for Llama-3.1-8B; +4.5 to +23.2% over the five settings | E4 | `results/perstep_signal/bayes_ceiling_paired.json` |
| Share of that gain a draft-side signal can reach | about one fifth (deployable probe +16.8% of the span for Llama-3.1-8B, +17.2% for Qwen3-14B); the rest is irreducible | E4 | same |
| Net offline gain of the hidden-state probe over a tuned fixed K | +3.1% (Llama-3.1-8B), +2.0% (Qwen3-14B), +2.5% (Llama-3.1-8B on math); none for DeepSeek-R1-Distill | E4 | same |
| Hidden-state probe AUC | 0.84 to 0.88 | E4 | `results/perstep_signal/hidden_full_audit.json` |
| In-loop probe, wall clock | −5.1 to −6.1% against the fixed depth | E5 | `DELIVERABLES.md` (E5 table) |
| Learned-probe policy, wall clock | −12 to −17%, probe overhead included | E6 | `results/eagle_zoo_verify.json` |
| Saturation tail-pruning, eager tree engine, held-out prompts | +2.9 to +4.8% over the strongest fixed depth (Llama-3.1-8B) | E6 | `results/eagle_paired.json` |
| Saturation tail-pruning cum(0.2), vLLM, three containers | HumanEval +5.6 ± 1.2%, GSM8K +5.7 ± 2.0%, MT-Bench +2.0 ± 1.1% | E6 | `results/vllm_fairbase_llama8b_r{1,2,3}.json` |
| cum(0.2) on Spec-Bench, 480 prompts, 512 tokens, three fresh containers | +6.8 ± 0.2% (mean ± SE) with each category against its own best fixed K; +4.8 to +8.7% by category | E7 | `results/specbench/` |
| Larger batches | Spec-Bench, each category at its own best fixed K: +1.8% [−2.5, +5.0] at B=4 and +2.4% [−2.0, +5.7] at B=8, not distinguishable from zero; fair-baseline workloads: about +1% at B=4, −3 to +1% at B=8 | E6, E7 | `results/vllm_fairbase2_llama8b_b{4,8}.json`, `results/specbench/specbench_spec_tb1.json` |
| Outputs against non-speculative greedy decoding | 78 to 84% identical token for token, the same rate as stock speculative decoding; every mismatch at a top-1/top-2 gap of at most 0.25 nats | E7 | `results/specbench/specbench_reference_full.json` |
| Verification cost against K at B=1 | 15.8 ms for one token, then 18.5 to 18.2 ms for 2 to 8 query positions | E6 | `results/verify_microbench.json` |
| Per-request controllers, replayed on captured traces | all lose to the tuned fixed K; the per-request oracle is +2.0% | E6 | `results/eagle3_controller_findings.md` |

## E1. Batch sweep (context 2048, B = 1 to 64)

How the best fixed K and its gap over K=1 change with batch size, on HumanEval, GSM8K and
MT-Bench (vLLM 0.23, Llama-3.1-8B with its EAGLE-3 head).

| batch | best K (HumanEval) | speedup | gap over K=1 |
|------:|-------------------:|--------:|-------------:|
| 1 | 2 | 1.642× | +19.9% |
| 8 | 1 | 1.288× | 0% |
| 16 | 2 | 1.376× | +10.7% |
| 32 | 1 | 1.111× | 0% |
| 64 | 1 | 1.077× | 0% |

At B=16 a K above 1 still gives +5 to 11% over K=1. From B=32 the best K is 1 on all three
workloads, so there is nothing left for an adaptive controller to recover.

Files: `results/eagle3_batch/` (`gap_summary.md`, `curves.csv`, `gap_vs_batch.csv`, figure).
Run: `modal run modal_vllm_eagle3.py --batch N` for N = 8, 16, 32, 64, then
`python analyze_eagle3_batch.py`.

## E2. Cheap per-step signals (three model families)

Whether logit-level features available at draft time (entropy, top-1 margin and similar) can
recover the per-step oracle. They recover little; on the reasoning model a predictor does worse
than fixed K=1, and even the target's own entropy after verification, which no real controller
can see, does not reach the oracle. These runs used an earlier pipeline whose oracle figures
were later corrected, so their table sits under Superseded numbers below. The paper's figures
for this question come from the decomposition in E4.

Files: `results/perstep_signal/target_entropy_audit.md`, `results/perstep_signal/audit.md`.
Run: `modal run modal_eagle3_perstep_capture.py --model <model>`, then
`python analyze_perstep_target_entropy.py`.

## E3. Long context (8192 tokens)

Whether the benefit of K above 1 survives long context at high batch. The EAGLE-3 head's
`max_position_embeddings` was raised to 8192 to lift a 2048-token kernel bound; prompts come
from QuALITY.

| Model | B=1 speedup | B=32 speedup | Best K (B=1) | Best K (B=32) |
|---|---:|---:|---:|---:|
| Llama-3.1-8B (instruct) | 0.988× (K=1) | 1.003× (K=1) | 1 | 1 |
| DeepSeek-R1-Distill-8B (reasoning) | 1.005× (K=1) | 0.944× (K=1) | 1 | 1 |
| Qwen3-14B (instruct) | 1.387× (K=3) | 1.308× (K=2) | 3 | 2 |

The result depends on the draft head. Qwen3-14B's head keeps a high acceptance rate at 8K
context (about 1.6 accepted tokens per step) and keeps its headroom at B=32. The Llama and
DeepSeek heads fall to about 1.1 accepted tokens per step, against 2.0 to 2.5 at 2K, and
speculation stops paying.

Files: `results/eagle3_longctx_patched/` (six JSONs), `results/eagle3_longctx_full/`
(`gap_summary.md`, figures).
Run: `modal run modal_vllm_eagle3_longctx_patched.py --model llama8b --batch 1 --tag lcp_b1`
(the script lists all six combinations), then `python analyze_eagle3_longctx_full.py`.

## E4. Hidden-state probe and the decomposition (the paper's decomposition table)

The complete EAGLE-3 draft hidden state (4096 dimensions for Llama and DeepSeek, 5120 for
Qwen3-14B) was captured at every draft position, 41K to 60K positions per model. Probes
(logistic regression, a 256-64 MLP, and PCA-50 followed by logistic regression) were trained
in 8-fold cross-validation split by generation; thresholds are chosen on the training folds.

Detection, on held-out generations:

| Model | 16-dim random projection AUC | full hidden state AUC |
|---|---:|---:|
| Llama-3.1-8B (instruct) | 0.484 | 0.842 |
| Qwen3-14B (instruct) | 0.484 | 0.876 |
| DeepSeek-R1-Distill (reasoning) | 0.484 | 0.870 |

The decomposition scores every rung of the ladder (oracle, best fixed, position, deployable
probe, probe with the oracle threshold) on the same test rows in each fold, so the ceiling
bounds the deployable probe and each difference is a paired per-fold statistic:

| Setting | oracle gain over best fixed K | deployable probe (% of span) | probe, oracle threshold (% of span) | net gain |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | +18.1% | 16.8 ± 2.7 | 18.2 ± 2.6 | +3.1 ± 0.6% |
| Qwen3-14B | +12.4% | 17.2 ± 2.4 | 19.5 ± 2.4 | +2.0 ± 0.3% |
| DeepSeek-R1-Distill | +4.5% | −0.3 ± 0.2 | 0.7 ± 0.5 | 0.0% |
| Llama-3.1-8B, math | +23.2% | 9.5 ± 3.7 | 12.0 ± 3.8 | +2.5 ± 1.0% |
| DeepSeek-R1-Distill, math | +8.5% | −1.3 ± 0.7 | 1.9 ± 0.6 | −0.1 ± 0.1% |

About four fifths of the oracle gain is variance in the verification outcome that the draft
side does not determine. Token position recovers nothing. On the DeepSeek-R1-Distill head the
probe finds nothing either; that head is also much weaker (accepted length 1.47 against 2.40
for Llama on the same math task), so the null cannot be attributed to reasoning models in
general.

Controls (the paper's appendix): shuffling the acceptance labels across blocks drops the
recovery to zero (`analyze_bayes_ceiling_control_paired.py`); the top-line numbers were
re-derived by an independent script (`audit_decomposition.py`, `audit_probe_recovery.py`); the
cost model was fitted to the measured fixed-K throughput (C = 0.066 to 0.072, reproducing
measured H100 throughput to about 1%, `measured_cost_ground.py`).

Span convention, for anyone reconciling the decomposition table by hand: recovery is computed
per fold and averaged, so each fold's recovery is a fraction of its own oracle span. The
`threshold_selection_loss_pct_of_span` field in `bayes_ceiling_paired.json` divides by the
whole-capture span instead and reads 0.87 to 3.11. Both are correct; they differ only in the
denominator, and the two conventions shift the recovery columns by at most 1.6 points.

A data-layout bug in the first pass of this audit grouped draft blocks by `(gen_i, step_i)`,
which merged blocks across workloads. The correct key is `(workload, gen_i, step_i)`. The bug
suppressed the signal; every number above uses the corrected key. AUC is per row and was not
affected, and E1 to E3 use a different layout.

Files: `results/perstep_signal/bayes_ceiling_paired.json` and `.md` (the paper's source),
`bayes_ceiling_control_paired.json`, `hidden_full_audit.json` and `.md`,
`bayes_ceiling_reasoning.md`, `pre_write_audit.md`.
Run: `modal run modal_eagle3_hidden_full_capture.py`, then
`python analyze_bayes_ceiling_paired.py`. The captures (about 1.5 GB of parquet) are not in the
repository; the outputs above are, for every fold.

## E5. Wall clock: acting on the probe

Whether the offline gain survives as tokens per second. vLLM 0.23 has no per-step draft-length
hook, so this runs in the SafeAILab/EAGLE-3 reference implementation (PyTorch, eager). Fixed
and adaptive arms run the same code path. Llama-3.1-8B.

(A) Seed-based controller (`modal_eagle_wallclock.py`): a probe sets the depth from the seed
hidden state. The seed carries some signal (correlation 0.26), and the controller still loses
5.7%, 2.4% and 5.5% to the best fixed depth (6, 7 and 7). A first version of this run paired
seeds with the wrong labels across prompts; `pre_write_audit.md` section 5 records the fix.

(B) In-loop early stop (`modal_eagle_inloop.py`): a copy of the tree search stops expanding
when a per-level hidden-state probe predicts rejection.

| Workload | fixed full depth (tok/s) | in-loop adaptive (tok/s) | gain | mean depth |
|---|---:|---:|---:|---:|
| HumanEval | 144.5 | 137.2 | −5.1% | 6.6 |
| GSM8K | 128.4 | 121.8 | −5.1% | 4.5 |
| MT-Bench | 133.4 | 125.3 | −6.1% | 5.7 |

The per-level probe has AUC 0.796 in the loop, close to the offline 0.84, and acting on it
loses 5 to 6% at every threshold tried. The draft head is cheap and verification dominates
the step, so stopping early saves little draft time and gives up accepted tokens. The offline
cost model gives draft cost too much weight, which is why its +2 to 3% turns negative here.

Files: `results/perstep_signal/wallclock_eagle.md`, `results/eagle_wallclock.json`. The
in-loop harness wrote to a remote volume, so the table above is the record of its cells.

## E6. Published-style policies, the paired protocol, and the vLLM replication

Four draft-stopping policies of the kinds in the literature (an entropy rule in the style of
SVIP, a top-1 margin rule, a cumulative-probability rule in the style of DDD and Wang et al. (2026), and
a trained hidden-state probe in the style of SpecDec++) run through the same in-loop code path
(`modal_eagle_zoo.py`), then a sequence of checks that became part of the result.

Evaluation failure modes this run exposed (each one inflates adaptive-length results):
1. Mechanism mismatch: an analysis made in the chain regime, deployed into a tree engine.
2. Where the fixed baseline is chosen: in the interleaved-split run the cumulative rule read
   +9.8% against the fixed depth chosen on the training prompts and +6.4% against the best fixed
   depth on the benched prompts (`results/eagle_zoo_verify_iid.json`, protocols B and A). A
   contiguous train/bench split confounds the comparison further: MT-Bench is ordered by
   category, its best fixed depth flips from 7 to 5 across the split, and the cumulative rule read
   −1.9% there (`results/eagle_zoo_verify.json`, a different container). The first pass, with
   thresholds chosen on the benched prompts (`results/eagle_zoo.json`), read +2.4%.
3. Unpaired sequential timing: one fixed arm read 147.0, 129.2 and 135.2 tok/s across three
   repeats in one run (`results/eagle_zoo_verify.json`). Per-cycle data show drift of several percent within a run, so arms are now timed
   round-robin with the order rotated each cycle (`modal_eagle_paired.py`).

Paired verdict on held-out prompts against the strongest fixed depth per workload
(Llama-3.1-8B, eager tree):

| Policy | GSM8K | MT-Bench | HumanEval |
|---|---:|---:|---:|
| cumulative probability, threshold 0.05 (realized depth 5.62) | +3.58% ± 0.32 | +4.77% ± 0.50 | +2.90% ± 1.83 |
| entropy (SVIP style) | – | – | – |
| hidden-state probe (SpecDec++ style) | −12 to −17%, probe overhead included | | |

The entropy rule never survived the held-out runs. The rule that survives uses no learned
signal: it thresholds the cumulative best-path probability the drafter already computes.

On the weaker DeepSeek-R1-Distill head the same protocol gives a tie (−0.7%, −0.9%, +1.4%, not
significant; realized depth 6.19), so the gain depends on the head. That run also showed a
fourth failure mode: a throughput step of 14 to 18% partway through a cycle, which pairing
cannot cancel; the protocol now uses at least six cycles and reports the median over cycles
(`results/eagle_paired_deepseek.json`).

vLLM replication (the fair-baseline table). The live V1 draft loop
(`worker/gpu/spec_decode/autoregressive/speculator.py::_multi_step_decode`) was patched; the
parallel `vllm.v1.spec_decode` package turned out to be unused for EAGLE-3, and patching it was
a silent no-op that only the arm-separation gate caught. cum(0.2) runs in a K=7 engine against
native fixed-K engines with their own verification budgets, paired round-robin, on three
fresh containers:

| Workload | cont. 1 | cont. 2 | cont. 3 | mean ± SE |
|---|---:|---:|---:|---:|
| HumanEval | +6.9 | +6.6 | +3.3 | +5.6 ± 1.2 |
| GSM8K | +7.1 | +8.4 | +1.8 | +5.7 ± 2.0 |
| MT-Bench | +3.5 | −0.2 | +2.7 | +2.0 ± 1.1 |

At B=4 the gain falls to about +1%, and at B=8 to between −3 and +1%; we attribute the decay to
the batch-level stopping rule, which we did not test directly (`modal_vllm_fairbase2.py`).
Verification time at B=1, 4 and 8 is flat in K from
2 to 8 query positions (`modal_verify_microbench.py`). An earlier vLLM run with padded fixed
arms (`modal_vllm_tailprune2.py`) gave +6.4%, +7.5% and +2.9% for Llama-3.1-8B and +1.3%,
+0.3% and +0.7% for Qwen3-14B.

Files: `results/perstep_signal/policy_zoo.md`, `results/eagle_zoo.json`,
`results/eagle_zoo_verify.json`, `results/eagle_zoo_verify_iid.json`, `results/eagle_paired.json`,
`results/vllm_fairbase_llama8b_r{1,2,3}.json`, `results/vllm_fairbase2_llama8b_{b4,b8,t08}.json`,
`results/vllm_tailprune2_*.json`, `results/verify_microbench.json`.
Scripts: `modal_eagle_zoo.py`, `modal_eagle_zoo_verify.py`, `modal_eagle_paired.py`,
`modal_vllm_fairbase.py`, `modal_vllm_fairbase2.py`, `modal_vllm_tailprune2.py`,
`modal_verify_microbench.py`; per-request controllers: `modal_eagle3_traces.py` (capture) and
`src/simulate_eagle_controllers.py` (replay).

## E7. Spec-Bench rerun of the fair-baseline result (vLLM 0.24.0, B = 1, 4, 8)

The fair-baseline comparison repeated on a public benchmark with longer outputs, a batch
sweep, and a token-by-token check against non-speculative greedy decoding.

- Workload: all 480 Spec-Bench prompts in six categories (MT-Bench first turns, translation,
  summarization, QA, math reasoning, RAG), chat template, greedy, 512 new tokens.
- Arms: native fixed-K vLLM engines and the patched K=7 engine running cum(0.2), in one
  process. Each prompt is timed under every arm with the arm order rotated; gains are paired
  per prompt with bootstrap 95% intervals.
- Best fixed depth on this workload: K=5 for MT-Bench, K=4 for summarization and QA, K=4 or 5
  for math and RAG (it differs between containers), and K=2 for translation. Three fresh
  containers, each category against its own best fixed K:

  | category | cont. 1 | cont. 2 | cont. 3 | mean ± SE |
  |---|---:|---:|---:|---:|
  | MT-Bench | +6.4 | +7.0 | +5.8 | +6.4 ± 0.4 |
  | Translation | +4.9 | +4.7 | +5.0 | +4.8 ± 0.1 |
  | Summarization | +7.4 | +7.4 | +7.2 | +7.3 ± 0.1 |
  | QA | +7.6 | +7.1 | +7.4 | +7.3 ± 0.2 |
  | Math | +7.3 | +6.6 | +5.1 | +6.3 ± 0.7 |
  | RAG | +8.0 | +10.5 | +7.5 | +8.7 ± 0.9 |
  | All, pooled | +7.0 [+6.2, +7.7] | +7.2 [+6.1, +7.9] | +6.4 [+5.7, +7.0] | +6.8 ± 0.2 |

  All 18 per-container category intervals are above zero (the lowest lower bound is +3.2%).
  Realized depth 3.98. The pooled row keeps each category at its own best depth; comparing
  every category with the single best depth over all prompts (K=4) reads higher, +7.7% on
  container 1, because translation's best depth is K=2.
- Against native K=2 and 3, the fair-baseline configuration, on two fresh containers: +9.5% and
  +11.7%. That baseline is under-tuned for this workload, so these overstate the gain.
- Batch (40 prompts per category), each category against its own best fixed K: +1.8%
  [−2.5, +5.0] at B=4 and +2.4% [−2.0, +5.7] at B=8, not distinguishable from zero. Per category
  the values swing widely on 5 to 10 chunks each, so `summary.md` and the paper's batch table
  give each one with its interval.
- Padded fixed depths K=2 to 5 on 8 prompts per category put the best depth at 4 or 5 in five
  of six categories; cum(0.2) beats the best padded depth by +6.4% [+3.8, +8.5].
- Token identity: 78 to 84% of outputs match the non-speculative greedy reference token for
  token, for stock speculative decoding and for cum(0.2) alike. Every mismatch is at a position
  where the target's top-1 minus top-2 log-probability gap is at most 0.25 nats.

Files: `results/specbench/`: `specbench_spec_t1.json`, `specbench_spec_t2.json`,
`specbench_spec_t3.json` (three containers; native K=4, 5 and padded K=2),
`specbench_spec_c1.json`, `specbench_spec_c2.json` (native K=2, 3),
`specbench_spec_tb1.json` and `specbench_spec_b1.json` (batch), `specbench_reference_full.json`
(non-speculative reference with top-2 log-probabilities), `summary.md`, `summary.json`.
Scripts: `modal_vllm_specbench_rerun.py` (runs), `analyze_specbench_rerun.py` (summary, tables
and figure; runs on a CPU).

## Earlier work in this repository

Before the EAGLE-3 measurements, the project built controllers and a simulator around
Qwen2.5 draft-target pairs. That code and its results remain:
- Controllers (entropy, epsilon-greedy, UCB, history, LinUCB, oracle): `src/controllers/`;
  harness and logging: `src/bench/`, `src/serve/`; unit tests: `tests/`.
- Fixed-K sweep and exploration: `notebooks/`; EAGLE-3 fixed-K sweep by workload:
  `results/eagle3_8b/fixedK_by_workload.csv`.
- Per-request controllers and per-step SVIP simulation on the EAGLE-3 traces:
  `results/eagle3_8b/policies.csv`, `perstep_svip.csv`, `ablations.csv`, `error_taxonomy.md`
  (`src/simulate_eagle_controllers.py`, `src/simulate_eagle_perstep.py`,
  `analyze_eagle3_8b.py`).
- An early output-equivalence spot check on the Qwen2.5 pure-Python harness,
  `results/equivalence_postfix_fp32.json`, covers 4 prompts and is not clean (`"all_pass":
  false`; 89.6% token match for the epsilon-greedy run). The token-identity check that the
  paper uses is the Spec-Bench one in E7.

The working notes under `results/` (`*_findings.md`, `insight_report*.md` and similar) were
written while the project ran. Some describe these earlier phases and use older table
numbers; where they differ from this file, this file is current.

## Superseded numbers

<!-- superseded-block: start (E2's first pipeline and the un-paired ladder, kept for the audit trail) -->
E2's cheap-signal audit, from the first pipeline (oracle figures superseded by E4):

| model | oracle gain | draft-feature recovery | target-entropy recovery |
|-------|---------------:|----------------------:|------------------------:|
| Llama-3.1-8B (instruct) | +24.8% | +6.8% ± 3.8 | +6.8% ± 5.0 |
| Qwen3-14B (instruct) | +16.9% | +8.5% ± 4.4 | +9.0% ± 4.4 |
| DeepSeek-R1-Distill-Llama-8B (reasoning) | +8.9% | −2.7% ± 1.1 | −4.9% ± 2.9 |

The first version of the decomposition (`analyze_bayes_ceiling.py` → `bayes_ceiling.json`)
scored the rungs on different rows, so the ceiling did not bound the deployable probe. Its
values, superseded by the paired table in E4:

| Model | best fixed K | per-step oracle | Bayes(position) | probe net gain | % of oracle |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | K=2 | +18.1% | +0.0% | +3.4% | 18.9% |
| Qwen3-14B | K=2 | +12.4% | +0.0% | +2.4% | 19.1% |
| DeepSeek-R1 | K=1 | +4.5% | +0.0% | +0.0% | 0% |

The same pass gave +18.3% recovery and +4.3% net for Llama-3.1-8B on competition math, and a
measured-cost re-grounding of +17.3% recovery (+2.9% net). The paired values that replace all
of these are +3.1%, +2.0% and +2.5% net.
<!-- superseded-block: end -->
