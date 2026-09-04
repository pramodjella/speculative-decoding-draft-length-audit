# Referee report — `paper/paper.tex` (6 pp, IEEEtran), pre-arXiv

> **STATUS: CLOSED — every item below is resolved as of commit `ae8d955` + follow-up.**
> The paper is now 7 pp / 4 figures / 6 tables, compiles with 0 errors, 0 overfull boxes and
> 0 unresolved references, and all 41 numbers printed in its tables reconcile against the
> released artifacts. The tarball extracts and builds standalone in a clean directory.
> The one blocker that remains is not editorial: **Yash's go-ahead and Vizuara's sign-off on
> the author sequence.** See `docs/ARXIV_SUBMISSION.md`.
>
> One finding surfaced during the fixes that is not in the report below: the net range
> **"+2.3–3.4%" was derived from the retired un-paired ladder** (Llama +3.41, Qwen +2.36 read
> off `bayes_ceiling.json`, the artifact CANONICAL marks "do not quote"). Corrected to
> **+2.0–3.1%** paper-wide, with per-fold SEs now in Table I, and guarded by `check_stale.sh`.

Reviewed against the compiled `paper/paper.pdf` (2026-08-25 build), the LaTeX log, and the
released artifacts in `results/`. Reviewer stance: a hostile-but-fair MLSys/ICLR referee who
downloads the repo and checks numbers.

---

## Verdict

**Accept the science; do not post this build.** The core result is genuinely valuable and
well-defended: the reducible/irreducible split of the per-step oracle is a real contribution,
the permutation control and independent re-derivation are the right controls, and the
paired drift-controlled protocol is the kind of methodological artifact the field actually
needs. The negative-result framing is honest without being self-flagellating.

But the paper is **under-evidenced relative to its own claims** and has **presentation defects
that render on the page**. Three of the abstract's headline numbers cannot be checked from the
body, one table is physically clipped off the page edge, and the single most valuable dataset
in the repo (the policy zoo) never appears. arXiv postings are permanent; every item under
BLOCKERS below is visible to a reader in the first five minutes.

Estimated work: **1–2 days.** Most of it is moving data you already have into tables.

---

## BLOCKERS — fix before posting

### B1. Table I is clipped off the page edge

The log says `Overfull \hbox (77.41257pt too wide) in paragraph at lines 91--103`. On an IEEE
252 pt column that is a ~30% overrun, and it is not cosmetic: **page 2 shows the fifth column
truncated mid-token.** The header reads `(oracl` and the cells read `+18.2 ±`, `+19.5 ±`,
`+0.7 ±`, `+12.0 ±`, `+1.9 ±` — every error bar in the `Bayes (oracle thr.)` column is gone.

This is the paper's headline table. A reader cannot read one of its five columns.

Fix: drop the `Bayes (position)` column (it is `+0.0%` in all five rows — state it once in the
caption as the correctness check it is), shorten the headers, and add
`\setlength{\tabcolsep}{4pt}`. That alone clears the overrun with room to spare.

### B2. Table I mixes two different units in adjacent `%` columns

Column 2 (`Oracle ceiling`) is an **absolute** cost-model speedup gain. Columns 3–5 are
**fractions of the fixed→oracle span**. Both print as `%` with no marking.

The consequence is that the table looks internally contradictory: Qwen3-14B reads
`Oracle ceiling +12.4%` but `Bayes (oracle thr.) +19.5%` — a rung *above* the ceiling it is
supposed to sit under. Same for DeepSeek math CoT and Llama math CoT. A referee who does not
read the caption closely concludes the decomposition is broken.

Fix: rename to `Oracle span (abs. %)` and `Recovery (% of span)`, or add a units row. Consider
adding the derived net column (`+2.3–3.4%`) since that is the number the abstract quotes and
it currently appears nowhere in the table.

### B3. Four `\pending{surname}` placeholders in the author block

They render in orange on page 1 exactly as intended:
*Naman [surname], Dr. Raj [surname], Rajat [surname], Sreedath [surname].*

arXiv author names cannot be corrected without a new version. Resolve before upload, and strip
the `\pending` macro and the internal status comment at the top of `paper.tex` (it currently
says "TAB-corruption repaired", which does not belong in a public source package).

### B4. The headline loss range "2–17%" is unsupported in the body and contradicted by the conclusion

Three incompatible numbers for the same result:

| Location | Claim |
|---|---|
| Abstract | learned/entropy controllers "lose **2–17%**" |
| Contribution 3 | "loses to a tuned fixed depth (**2–17%**)" |
| Conclusion | "*loses* **2–6%** end-to-end when acted on" |
| Body evidence shown | −2.4 to −6.1% (§VI-A/B); −4 to −7% (Qwen) |

Nothing in the body reaches −17%. A referee reads this as either sloppiness or inflation of a
negative result.

The number *is* real — I recomputed it from `results/eagle_zoo_verify.json`: the trained
hidden-probe stop loses a mean of **−17.0%** on held-out prompts, and the entropy stop's best
cell is **−2.0%**. So "2–17%" is exactly `[min entropy loss, mean probe loss]`. The problem is
purely that **the evidence is not in the paper** (see B5). Fix B5 and this claim becomes
defensible; then correct the Conclusion to match.

### B5. Contribution 3's central evidence — the policy zoo — has no table

The paper claims a "wall-clock **policy-zoo** audit" over four policies × three workloads and
prints only the survivor (Table III). The four losing arms appear nowhere. The load-bearing
claim of the paper — *every learned/entropy policy loses* — is asserted, not shown.

The data exists and is clean. From `results/eagle_zoo_verify.json` (held-out) and
`results/eagle_zoo_verify_iid.json` (in-distribution), mean ± SE over 3 repeats, vs the
**strongest** fixed depth in each run:

**Held-out selection (the paper's protocol), best fixed = K=5 on all three:**

| Policy | HumanEval | GSM8K | MT-Bench | mean | realized depth |
|---|---|---|---|---|---|
| SVIP-style entropy stop | −2.0 ± 0.5 | −4.3 ± 0.4 | −7.7 ± 0.6 | **−4.7** | 5.92 |
| Saturation tail-pruning (SADDLE/PACER) | +0.1 ± 0.7 | −2.5 ± 0.2 | −3.2 ± 0.2 | **−1.9** | 5.49 |
| Trained hidden-probe stop (SpecDec++-style) | −14.0 ± 0.6 | −14.7 ± 0.3 | −22.3 ± 0.3 | **−17.0** | 6.95 |

**In-distribution selection (the common protocol), best fixed = K=8/6/5:**

| Policy | HumanEval | GSM8K | MT-Bench | mean |
|---|---|---|---|---|
| SVIP-style entropy stop | +0.9 ± 0.5 | +2.6 ± 0.7 | +1.1 ± 1.3 | **+1.5** |
| Saturation tail-pruning | +4.5 ± 0.8 | +7.3 ± 1.2 | +7.3 ± 0.8 | **+6.4** |
| Trained hidden-probe stop | −15.1 ± 1.4 | −12.8 ± 0.8 | −16.2 ± 0.8 | **−14.7** |

Two things this table buys you that the current draft does not have:

1. It substantiates B4 and the "every policy loses" claim with actual cells.
2. **It is the cleanest demonstration of failure mode #2 in the paper.** The same policy on the
   same hardware goes `+6.4% → −1.9%` when threshold selection moves off the benched prompts,
   *and the best fixed depth itself moves K=8/6/5 → K=5/5/5*. Right now failure mode #2 is
   supported by a single sentence ("+2.4% → −1.9%"); this makes it a figure-grade result.

⚠️ **One caveat you must handle explicitly if you add this.** In this unpaired run the survivor
(cumprob) reads **−1.9% mean held-out**, while Table III reports **+2.90 to +4.77%** on the same
three workloads. Different protocol (3 sequential repeats vs 4-cycle rotated round-robin) and a
different container — this *is* your failure mode #3 firing. Add one sentence saying the zoo
table is the pre-pairing evidence and Table III supersedes it for the survivor, or a referee
will read the two tables as a contradiction and stop there. Handled well, it strengthens the
protocol argument; left implicit, it sinks it.

### B6. "Six evaluation failure modes" is a headline contribution that is never enumerated

Abstract and Contribution 3 both promise six. The body gives a numbered list of **three**
(§VI-A), mentions "a fourth evaluation failure mode" parenthetically inside a 372-word
paragraph, and scatters the other two (mechanism-mismatch in §VI, padded-budget in §VI-C).
A reader cannot assemble six.

This is likely the most reusable thing in the paper — it deserves to be the most findable.
Make it a compact table: *mode / how it manifests / what it did to our number / the fix*.
You already have a measured before/after for four of the six:

| # | Failure mode | Measured effect on our own number |
|---|---|---|
| 1 | Mechanism-mismatched offline simulation | chain-regime +2–3% → **sign inversion** in tree engine |
| 2 | In-distribution threshold selection | +6.4% → **−1.9%** (zoo, held-out) |
| 3 | Unpaired sequential GPU benchmarking | **+2.4 / −1.9 / +6.4%** from identical configs |
| 4 | Intra-run throughput step-changes | fixed-arm tok/s jumped **14–18%** mid-cycle |
| 5 | Dead-code instrumentation | silent no-op; first vLLM attempt **invalidated** |
| 6 | Padded-budget baselines | motivated the 3-engine native rerun (verify flat, <1%) |

### B7. No repository URL anywhere in the paper

The abstract says "a gated, paired benchmarking protocol that we release with all artifacts";
Contribution 3 says "(see docs/CANONICAL.md)"; §Reproducibility cites "DELIVERABLES.md
(E1–E6)" and eleven bare script filenames. **The paper never gives a URL.** Every one of those
pointers is unresolvable for an arXiv reader.

`git remote` says `https://github.com/pramodjella/adaptive-draft-length-audit`. Add it as a
footnote on page 1 and in §Reproducibility, and delete the inline `docs/CANONICAL.md` from the
contribution list — an internal repo path in a contribution bullet reads as an unfinished draft.

Also confirm the repo is **public and complete** before posting. A reproducibility section that
points at a 404 is worse than no reproducibility section.

---

## MAJOR — referees will hit these

### M1. Figure 3's embedded title is truncated mid-word

`fig_paired_protocol.png` renders its title as *"Paired round-robin benchmarking: arms
separate; drift is visible and cance"* — the word is cut off inside the PNG, so no LaTeX change
fixes it. The `cum0.05` legend label also sits on top of the red data line at cycle 2.
Regenerate with a shorter title and the legend moved outside the axes.

### M2. All three figures are unreferenced in the text

`fig:decomp`, `fig:survivor`, `fig:protocol` all have `\label`s and **zero `\ref`s**. Three
floats with no callout. (`sec:setup` is also unreferenced.) Add "(Fig. 1)" etc. at the point of
claim.

### M3. Figure 1's rung names don't match Table I's

Fig. 1's bars read `best fixed K / Bayes(position) / probe (deploy) / **Bayes(hidden)** /
per-step oracle`, while the rung list and Table I call that fourth rung **`probe (oracle
threshold)`**. Same quantity, two names, one page apart. Fig. 1 also says "~18% of span" and
"~82%" where the text says "~1/5" and "~80%" — harmonize to one convention.

### M4. The 372-word paragraph

§VI-C's "vLLM-native replication" paragraph is 372 words and contains **four distinct
experiments**: the V1 patch, the three-container fair-baseline rerun, the verify microbench,
and the honest-headline restatement. Page 4's right column has no white space from top to
bottom. This is where a tired referee gives up.

Split into three labelled paragraphs or subsections, and put the 9/9 fair-baseline cells in a
table — right now the paper's single most defensible positive result (three fresh containers,
gated, native baselines) is three numbers buried mid-paragraph.

### M5. "Paired verdict (Table III)." is a sentence fragment

Line 176 is `\textbf{Paired verdict (Table~\ref{tab:paired}).}` followed immediately by two
`figure` environments, with the continuation starting a new paragraph after them. On page 3 it
renders as a bold fragment stranded at the column bottom with no verb. Move the figures or
merge the fragment into the following sentence.

### M6. Inconsistent sectioning in §VI

The reader sees hand-numbered run-ins **(A) Seed-state controller** and **(B) In-loop
within-step early-stop**, then a real `\subsection` that renders as **A. The Policy Zoo**.
Two different "A"s in one section. Promote all three to `\subsection`.

### M7. Six pages of IEEEtran is the wrong container for this content

arXiv has no page limit, and the paper is visibly fighting the format. Five substantial
experiments — temperature robustness, batch decay, cross-head replication, the Qwen chain
replication, the verify microbench — are each compressed to one or two sentences with no table,
and the batch-decay result (a headline scoping claim, `+2–6% → ~+1% → 0 to −3%`) appears in
prose only.

Recommend expanding to 10–12 pages: add the zoo table (B5), the failure-mode table (B6), the
9/9 fair-baseline table (M4), and a batch-decay figure. The science supports it; the current
compression is actively hiding evidence you already paid for.

### M8. Notation debt

`MAT` in `speedup = MAT/(1+CK)` is never expanded. `C` is fitted before being defined as a
draft:target per-token cost ratio. `acc` appears in `K = acc+1` before definition. There is no
notation block. For a paper whose central object is a cost model, define it once formally.

### M9. The headline number is scoped four different ways

Abstract `+2–5%`; §VI-C `Llama chain +2.9–7.5%`; fair-baseline `+5.6/+5.7/+2.0`; refined
survivor claim `Llama chain +2.9–7.5%, Llama tree +2–5%, Qwen3 +0.3–1.3%, weak head ~0`. All
are defensible individually, but the reader must reconstruct which one is *the* claim. State
one headline with its regime, once, and make every other figure an explicitly scoped refinement
of it.

### M10. "SPEED-Bench" is cited by name with no reference

§VI-C, batch paragraph: "consistent with … SPEED-Bench's batch-dependent optimal draft lengths."
There is no `\cite`, no bibliography entry, and every other mention in the paper says
*SpecDecode-Bench*. Either it is a distinct paper missing from the bibliography or a stale
mis-name — either way it is a dangling attribution to a named benchmark.

### M11. "Pre-registered" is used in its strong sense without an artifact

"Both outcomes matched expectations pre-registered before the runs." In this repo,
"pre-registration" means *τ selected on the train split* (`modal_bcsd_*.py`) — a different and
weaker thing than a pre-registered directional hypothesis, and those scripts belong to the BCSD
line, not this paper. Either point to a timestamped artifact or soften to "matched the
mechanism's prediction stated in advance." Referees in this area treat the word literally.

---

## MINOR

- **Bibliography.** Four entries carry **no authors at all**: `dsde`, `saddle`, `pacer`,
  `turbospec`. `hou2025banditspec`, `adaedl`, and `turbospec` lack venue or arXiv ID.
  `paper/downloads/` already holds PDFs whose filenames encode the first author
  (`adasd_lu.pdf`, `nightjar_li.pdf`, `banditspec_hou.pdf`) — recoverable in minutes. arXiv
  moderation and referees both read reference lists as a proxy for care.
- **Verify every 2026 arXiv ID resolves** before posting: `2601.11580`, `2601.07353`,
  `2602.01274`, `2512.22420`. A wrong ID in a permanent posting is hard to live down.
- **Setup section justification.** The `\texttt{}` model paths produce
  `Underfull \hbox (badness 10000)` ×4 at lines 68–74; page 2 shows word gaps wide enough to
  drive through. Wrap in `\sloppy` or break the model list into a small table.
- **§V (reasoning-head confound)** is honest, but giving a self-identified confound its own
  top-level section over-weights it. Fold into §IV as a paragraph; keep the limitation bullet.
- **Title** is 20 words across two subtitle clauses. The question alone carries it.
- Consider promoting the batch-decay result to a figure — it is the sharpest practical
  guidance in the paper ("the profit window is `B ≲ 4`") and currently has no visual.

---

## What is working, and should not be touched

- The five-rung ladder with **two built-in positive controls** (position degenerating to fixed
  `K` exactly; oracle-threshold bounding deploy in every fold) is unusually well-designed. Say
  so more loudly — most referees have never seen a decomposition that self-validates.
- The **permutation control** (+16.8 → −5.6, 22.4 pp collapse, same paired-within-fold
  estimator on both arms) is the right control, correctly executed and correctly reported.
- The **three-container fair-baseline rerun** with bit-identical realized draft lengths as a
  determinism gate is the strongest single piece of evidence in the paper. Give it a table.
- Reporting **`+2.0 ± 1.6` as "a wash"** rather than as a win is the credibility anchor of the
  whole document. Keep it exactly as it is.
- The **mechanism explanation** — prediction chases irreducible variance, detection is
  deterministic given what the drafter already computed — is what makes this an audit rather
  than a list of negative results. It is currently stated three times (abstract, §VI-C,
  conclusion); that repetition is earned.

---

## Suggested order of work

1. B3 (surnames) + B7 (repo URL) — minutes, unblocks everything else.
2. B1 + B2 (Table I) — the clipped column is the worst single defect on the page.
3. B5 + B4 (zoo table, then reconcile 2–17% vs 2–6%) — biggest scientific gain per hour.
4. B6 (failure-mode table) — makes the most reusable contribution findable.
5. M1–M6 (figure regen, refs, fragment, sectioning) — an afternoon.
6. M7 (expand past six pages) if you want this to read as the definitive audit rather than a
   compressed workshop paper. Recommended.
7. Minors + `scripts/check_stale.sh` + three-pass rebuild + re-verify the tarball.
