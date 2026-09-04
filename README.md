# Following the Speedup: An Audit of Adaptive Draft Length in Speculative Decoding

Artifact repository for the paper. Pramod Kumar Reddy Jella, Yash Dixit, Naman Dwivedi,
Raj Dandekar, Rajat Dandekar, Sreedath Panat.

📄 [paper/paper.pdf](paper/paper.pdf) · 📊 [DELIVERABLES.md](DELIVERABLES.md) (result → script →
artifact map) · 📌 [docs/CANONICAL.md](docs/CANONICAL.md) (single source of truth for every
number)

---

## The result

Adaptive draft length promises a double-digit speedup. The paper follows that number
through three gates and finds most of it was never reachable.

- **~80% of the oracle gap is irreducible** from any draft-side signal. The oracle overstates
  the draft-side-achievable gain by roughly 5×.
- **The reachable remainder is real but does not pay.** It lives in the drafter's hidden state
  (probe AUC 0.79–0.88), is worth +2.0–3.1% offline — and **loses 2–17% in wall-clock** when
  acted on, because verification dominates execution.
- **Exactly one policy class survives:** signal-free saturation tail-pruning (the PACER/SADDLE
  family), **+2–5% over the strongest *native* tuned baseline at batch 1, with one of three
  workloads a wash**. Robust to sampling temperature; decays to zero by batch 8.
- **Six evaluation failure modes** inflate adaptive-length claims. We demonstrate each on our
  own numbers and release the paired, drift-controlled protocol that survives them.

The mechanism ties the two verdicts together: acceptance **prediction** chases irreducible
realization variance, while saturation **detection** is deterministic given quantities the
drafter has already computed.

### Scope, stated plainly

The winning configuration (B=1) is not the one you would ship. Per-request ragged stopping is
impossible in a batch-synchronized draft loop, and the batch-level mean stop is what erodes
the gain — not verification cost, which we measure to be flat in *K*. **The deployable
ragged-engine number is open.** We report MT-Bench at +2.0 ±1.6 as a wash rather than as a win.

---

## Reproducing the paper

Every number printed in the paper's tables is released as raw JSON under
[`results/`](results/), so any figure can be checked directly against its artifact without a
GPU. What each column below means:

- **Verify** — runs on a fresh clone, CPU only. Recomputes the paper's numbers from released
  JSON.
- **Regenerate** — re-runs the measurement itself. Needs an H100, driven through
  [Modal](https://modal.com).

```bash
pip install -r requirements.txt
```

| Paper element | Verify (fresh clone, CPU) | Regenerate (H100) |
|---|---|---|
| Table I — decomposition | `results/perstep_signal/bayes_ceiling_paired.json` † | `modal run modal_eagle3_hidden_full_capture.py` then `python analyze_bayes_ceiling_paired.py` |
| Table II — in-loop probe | `results/eagle_wallclock.json` | `modal run modal_eagle_inloop.py` |
| Table III — paired verdict | `results/eagle_paired.json` | `modal run modal_eagle_paired.py` |
| Table IV — policy zoo | `results/eagle_zoo_verify{,_iid}.json` | `modal run modal_eagle_zoo_verify.py` |
| Table V — fair baseline | `results/vllm_fairbase_llama8b_r{1,2,3}.json` | `modal run modal_vllm_fairbase.py` |
| Batch decay (Fig. 5) | `results/vllm_fairbase2_llama8b_b{4,8}.json` | `modal run modal_vllm_fairbase.py` |
| Table VII — online controllers | `results/eagle3_controller_findings.md` | — |
| Cost-model grounding | `python measured_cost_ground.py` | — |
| Batch collapse, long context | `python analyze_eagle3_batch.py`, `python analyze_eagle3_longctx.py` | — |
| All five figures | `python analyze_make_figures.py` | — |

† **The one reproducibility gap worth flagging.** `analyze_bayes_ceiling_paired.py` reads the
per-position hidden-state captures (`results/eagle3_hidden_full/hidden_full_*.parquet`,
~1.5 GB across five settings). Those are too large for git and are **not** in this repository,
so the decomposition cannot be re-derived from a clone — only regenerated from the capture
step. Its *outputs* are released in full, including the per-fold values behind every error
bar, so the arithmetic is checkable. Ask if you want the parquets; we can host them.

Then rebuild the PDF:

```bash
python analyze_make_figures.py && bash scripts/check_stale.sh && cd paper && pdflatex paper.tex
```

`scripts/check_stale.sh` is a release gate: it fails if any retired number reappears in a
shipped document, and runs the LaTeX structural checks. Run it before sending anything.

---

## Layout

```
paper/            paper.tex, the five figures it uses, built PDF/DOCX, arXiv tarball
results/          raw per-run JSON behind every table and figure
analyze_*.py      CPU analyses over the released JSON
modal_*.py        H100 capture / wall-clock harnesses (Modal)
src/              simulator-era controller library (bandits, LinUCB, entropy policies)
docs/             CANONICAL.md (numbers), ARCHITECTURE.md (how it fits together),
                  RESEARCH_GUIDE.md (orientation), ARXIV_SUBMISSION.md (posting procedure)
scripts/          release gates + the Word export
```

Start at [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) if you want the guided tour, or
[DELIVERABLES.md](DELIVERABLES.md) if you want a specific number's provenance.

## Conventions this repo enforces

- **docs/CANONICAL.md is the single source of truth.** Every other document defers to it, and
  retired figures are kept there under an explicit banner so the audit trail survives.
- **Paired, drift-controlled timing.** Arms are benched round-robin with rotated order and
  per-cycle paired differences; report mean ± SE over cycles. Unpaired sequential benchmarking
  gave us three contradictory answers from identical configurations.
- **Baselines are tuned, and "strongest" means the one that minimises our own reported gain.**

## A note on scope

Budget-certified speculative decoding (BCSD), the follow-on line, lives in a separate
repository. Nothing in this paper depends on it.

## Citation

```bibtex
@article{jella2026adaptivedraft,
  title  = {Following the Speedup: An Audit of Adaptive Draft Length in
            Speculative Decoding},
  author = {Jella, Pramod Kumar Reddy and Dixit, Yash and Dwivedi, Naman and
            Dandekar, Raj and Dandekar, Rajat and Panat, Sreedath},
  year   = {2026}
}
```

_Vizuara AI Labs — for research and educational purposes._
