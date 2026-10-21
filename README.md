# Following the Speedup: An Audit of Adaptive Draft Length in Speculative Decoding

Code, raw measurements and analysis scripts for the submission. For double-blind review,
names in comments and notes read `[name withheld]`.

## What is measured

Speculative decoding drafts K tokens with a small model and verifies them with the target
model in one pass. Adaptive draft length changes K at every step. The paper compares it
against a tuned fixed K, on one H100 at batch size 1 unless stated.

| Question | Result | Where |
|---|---|---|
| How much of the oracle gain can a draft-side signal reach? | About 20%. The other ~80% is variance that the draft-side state does not determine. | Table 2, Figure 2 |
| Where does the reachable part live? | In the drafter's hidden state: probe AUC 0.84 to 0.88. Logit features and token position recover almost none of it. | Section 6, `results/perstep_signal/` |
| Does acting on that signal pay in a running engine? | No: 2 to 17% slower than a tuned fixed depth. | Tables 3 and 4 |
| Which policy survives? | Signal-free saturation tail-pruning, cum(0.2). In vLLM: +2 to +6% over the fastest native fixed-K engine on three workloads (one within noise), and +6.8 ± 0.2% (mean ± SE over three fresh containers) on all 480 Spec-Bench prompts at 512 output tokens, with each category against its own best fixed depth (K = 4 or 5; K = 2 for translation). Every category gains on every container. The threshold 0.2 was fixed on the original workloads before the Spec-Bench runs. | Tables 5 and 6 |
| What happens at larger batch? | On Spec-Bench, with each category at its own best fixed depth: +1.8% at batch 4 and +2.4% at batch 8, neither distinguishable from zero. | Table 11, Figure 6 |
| How faithful are the outputs? | No less faithful than stock vLLM speculative decoding: 78 to 84% of Spec-Bench outputs match non-speculative greedy decoding token for token, the same rate as stock speculative decoding, and every mismatch is at a near-tie between the target's top two tokens (log-probability gap at most 0.25). | Table 10 |

## Quick start: recompute on a CPU

No GPU is needed for this part. Tested with Python 3.11 on a CPU-only machine.

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib

python analyze_specbench_rerun.py   # Spec-Bench: rebuilds results/specbench/summary.md exactly,
                                    # and writes Tables 6, 9, 10, 11 and Figure 6 under paper/
python analyze_make_figures.py      # Figures 1 to 5, written to paper/figures/
python analyze_eagle3_batch.py      # batch sweep: results/eagle3_batch/*.csv
python analyze_eagle3_longctx.py    # long-context headroom
python measured_cost_ground.py      # measured verification-cost constant
```

`pytest -q tests --ignore=tests/test_linucb_simulation.py` runs the 18 unit tests of the
controller library in `src/` (that file is a script: `python tests/test_linucb_simulation.py`).

## Where each table and figure comes from

Table and figure numbers follow the submitted PDF. "GPU harness" scripts produced the data and
need an H100 and a [Modal](https://modal.com) account. Everything else reads the released files.

| Paper | Data in this repository | GPU harness | CPU analysis |
|---|---|---|---|
| Figure 1, the audit in one picture | `results/perstep_signal/bayes_ceiling_paired.json`, `results/eagle_wallclock.json` | – | `analyze_make_figures.py` |
| Table 2, decomposition; Figure 2 | `results/perstep_signal/bayes_ceiling_paired.json` (every fold), `bayes_ceiling_paired.md` | `modal_eagle3_hidden_full_capture.py` | `analyze_bayes_ceiling_paired.py` (needs the captures, see below) |
| Appendix B, controls | `results/perstep_signal/bayes_ceiling_control_paired.json`, `hidden_full_audit.json` | same | `audit_decomposition.py`, `audit_probe_recovery.py` (need the captures) |
| Table 3, in-loop probe | cells in `DELIVERABLES.md`, section E5 | `modal_eagle_inloop.py` | – |
| Table 4, policy zoo | `results/eagle_zoo_verify.json` (held-out), `results/eagle_zoo_verify_iid.json` | `modal_eagle_zoo_verify.py` | – |
| Table 5, fair baseline | `results/vllm_fairbase_llama8b_r{1,2,3}.json` (three fresh containers) | `modal_vllm_fairbase.py` | – |
| Table 6, Spec-Bench at batch 1 | `results/specbench/specbench_spec_t1.json`, `_t2.json`, `_t3.json` (three fresh containers; native K = 4, 5 and padded K = 2) | `modal_vllm_specbench_rerun.py` | `analyze_specbench_rerun.py` |
| Table 9, Spec-Bench against K = 2, 3 | `results/specbench/specbench_spec_c1.json`, `specbench_spec_c2.json` (native K = 2, 3 on two containers) | `modal_vllm_specbench_rerun.py` | `analyze_specbench_rerun.py` |
| Figure 3, why pairing is needed | `results/vllm_tailprune2_llama8b.json` | – | `analyze_make_figures.py` |
| Table 7, paired verdict | `results/eagle_paired.json`, `results/eagle_paired_deepseek.json` | `modal_eagle_paired.py` | – |
| Figure 4, batch decay; temperature | `results/vllm_fairbase2_llama8b_b4.json`, `_b8.json`, `_t08.json` | `modal_vllm_fairbase2.py` | `analyze_make_figures.py` |
| Figure 5, the survivor across settings | paired gains from the runs above, entered in the script | – | `analyze_make_figures.py` |
| Table 8, online controllers | `results/eagle3_controller_findings.md` | `modal_controller_benchmark.py` | – |
| Table 10, token identity | `results/specbench/specbench_reference_full.json` (greedy reference with top-2 log-probabilities) | `modal_vllm_specbench_rerun.py` | `analyze_specbench_rerun.py` |
| Table 11, Figure 6, Spec-Bench batch | `results/specbench/specbench_spec_tb1.json`, `specbench_spec_b1.json` | `modal_vllm_specbench_rerun.py` | `analyze_specbench_rerun.py` |
| Verification cost against K | `results/verify_microbench.json` | `modal_verify_microbench.py` | – |
| Batch collapse, long context | `results/eagle3_batch/`, `results/eagle3_longctx*/` | `modal_vllm_eagle3.py`, `modal_*longctx*.py` | `analyze_eagle3_batch.py`, `analyze_eagle3_longctx.py` |

`DELIVERABLES.md` maps each experiment (E1 to E7) to the script and file behind it. Its first
table lists every number the paper uses; the Spec-Bench numbers come from
`results/specbench/summary.md`. Values that were later corrected are kept at the end of
`DELIVERABLES.md` under a clear banner, so the history of each number can be followed.

## How the timing was done

- All arms of a comparison run in one process on the same GPU, round-robin with the order
  rotated each cycle. Gains are paired differences (per cycle, or per prompt for Spec-Bench)
  with bootstrap 95% intervals. Unpaired, back-to-back timing gave contradictory answers from
  identical settings, which is why the protocol is paired.
- The baseline for each workload is its fastest fixed depth.
- Spec-Bench rerun: all 480 prompts (MT-Bench first turns plus five other categories), chat
  template, greedy, 512 new tokens, vLLM 0.24.0 pinned, one H100. The run log at the end of
  `results/specbench/summary.md` lists the time and approximate cost of each run.
- Model pairs, versions and settings are in the paper's Setup section and at the top of each
  `modal_*.py` script.

## Repository layout

```
analyze_*.py, audit_*.py   CPU analyses over the released files
modal_*.py                 GPU harnesses (H100 through Modal)
results/                   raw per-run JSON, CSV and summaries behind every table and figure
src/                       controller library, simulators and the speculative-decoding runner
tests/                     unit tests for src/
docs/ARCHITECTURE.md       how the code fits together
configs/, data/            experiment configs and dataset cards
notebooks/                 early exploration
scripts/check_stale.sh     checks shipped documents for retired numbers
```

## Working notes

The `*_findings.md`, `*_summary.md` and `insight_report*.md` files under `results/` are notes
written while the project ran. Some describe earlier phases (for example, simulator-only
results) and use the table numbers of earlier drafts. Where they differ from the paper, the
paper, this README and `DELIVERABLES.md` are current.

## Known gaps

- The per-position hidden-state captures behind Table 2 (`results/eagle3_hidden_full/*.parquet`,
  about 1.5 GB) are too large for this repository. The decomposition's outputs are here in full,
  for every fold, so its arithmetic can be checked; recomputing it from scratch means
  re-running the capture on an H100.
- Table 3's harness wrote to a remote volume, so its raw JSON is not here. Its cells are
  recorded in `DELIVERABLES.md` (E5).
- The `modal_*.py` scripts need an H100 and a Modal account. They are included so the
  measurements can be read and re-run; all CPU analyses work from the released files.

## License

MIT, see `LICENSE`.
