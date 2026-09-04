# arXiv submission — procedure for this paper

Everything needed is prepared. `paper/arxiv_submission.tar.gz` has been verified to compile
standalone from a clean directory (9 pages, 5 figures, pdflatex ×3, no external .bib,
zero errors, zero unresolved references, zero overfull boxes).

---

## Before you upload — one thing only you can settle

**Yash's go-ahead**, and Vizuara's confirmation of the authorship sequence. He asked to be told
before it goes up.

Author names are final in the PDF — *Pramod Kumar Reddy Jella, Yash Dixit, Naman Dwivedi,
Raj Dandekar, Rajat Dandekar, Sreedath Panat*. Titles are omitted, as ML venues do not use
them in the byline, and the affiliation line was removed at the authors' request. Check the
sequence is the one Vizuara wants **before** submitting: arXiv postings are permanent and
author names cannot be edited without submitting a new version.

**The artifact repo must be public when this posts.** The paper prints
`github.com/pramodjella/speculative-decoding-draft-length-audit` on page 1 and the
Reproducibility section names eleven scripts. It has flipped to private at least once; verify
with an anonymous fetch, not an authenticated one, since `gh` will succeed either way.

To recompile after editing:

```bash
cd paper && pdflatex -interaction=nonstopmode paper.tex && pdflatex paper.tex && pdflatex paper.tex
```

---

## Step 1 — account and endorsement

- Register at <https://arxiv.org/user/register> using an **institutional email if possible**
  (a vizuara.com address will smooth endorsement; gmail alone often triggers the endorsement
  requirement).
- First-time submitters to **cs.LG / cs.CL** usually need an **endorsement**: arXiv shows an
  endorsement code, and an existing arXiv author in that category vouches for you. Ask Yash or
  Dr. Raj — anyone who has posted to cs.LG recently can endorse. Do this *first*; it can take
  a day or two and blocks everything else.

## Step 2 — start the submission

<https://arxiv.org/submit> → **Start New Submission**.

- **License:** CC BY 4.0 is the usual choice for preprints you may later publish; arXiv's
  default (non-exclusive license to distribute) is the most conservative. TMLR accepts
  either, so pick CC BY 4.0 unless Vizuara has a policy.
- **Archive/subject:** primary **cs.LG** (Machine Learning). Cross-list **cs.CL** (Computation
  and Language). Optionally cross-list **cs.PF** (Performance) — the paper is a systems
  measurement study, and cs.PF readers are the audience for the benchmarking-protocol content.

## Step 3 — upload

Upload **`paper/arxiv_submission.tar.gz`**. It contains exactly:

```
paper.tex
figures/fig_waterfall.png
figures/fig_decomposition.png
figures/fig_survivor.png
figures/fig_paired_protocol.png
figures/fig_batch_decay.png
```

Notes on why it is packaged this way:

- **Source, not PDF.** arXiv strongly prefers LaTeX source and will build the PDF itself.
  PDF-only submissions are accepted but flagged and look worse.
- **No `.bib` / `.bbl` needed.** The paper uses 20 inline `\bibitem` entries, so arXiv's
  compiler resolves references without BibTeX. (If you ever switch to `\bibliography{...}`,
  you **must** include the generated `.bbl`, because arXiv does not run BibTeX.)
- **No `.aux`, `.log`, `.out`, `.pdf`** — arXiv regenerates these, and stale ones cause
  build errors. They are gitignored for the same reason.

## Step 4 — metadata

- **Title:** Following the Speedup: An Audit of Adaptive Draft Length in Speculative
  Decoding
- **Authors:** in the Vizuara sequence, matching the compiled PDF exactly. arXiv wants
  `Surname, Firstname` per line.
- **Abstract:** paste from `paper/paper.tex` (the `\begin{abstract}` block), as **plain text**
  — strip the `\textbf{}`, `\emph{}` and `$...$` markup. arXiv renders a little TeX in
  abstracts but plain prose is safer.
- **Comments field:** worth stating the artifact position, e.g.
  *"9 pages, 5 figures. All numbers reproduce from released raw JSON; benchmarking protocol
  and analysis scripts released with the paper."*

## Step 5 — check the build, then submit

- arXiv compiles and shows you a **preview PDF**. **Read it before submitting** — check that
  Table I's rightmost column is not clipped (it was, until this round), that all five figures
  appear, and that Table VI (the six failure modes) spans the full page width.
- Submit. You get an identifier like `arXiv:2608.XXXXX`.
- Announcement is next business day at 20:00 ET; there is a hold window in which you can
  replace the source.

## Step 6 — afterwards

- Put the arXiv ID in `README.md` and `docs/CANONICAL.md`.
- For **TMLR**: submit through OpenReview. TMLR permits and expects preprints, so the arXiv
  posting does not conflict — but check any workshop CFP's archival policy *before*
  submitting there, since a non-archival workshop is fine while an archival one could
  complicate the TMLR submission.

---

## Regenerating the package

If `paper.tex` or the figures change, run all four steps — skipping any one of them has
shipped a stale artifact at least once:

```bash
python analyze_make_figures.py       # figures are generated, never hand-edited
bash scripts/check_stale.sh          # must PASS: guards retired claims + LaTeX health
cd paper && pdflatex -interaction=nonstopmode paper.tex && pdflatex paper.tex && pdflatex paper.tex
cd .. && bash scripts/build_docx.sh  # the Word copy goes stale silently otherwise
```

then rebuild the tarball with only the source and the five referenced figures — never the
whole `paper/` directory, which contains build artifacts, the Word export and the literature
spreadsheet:

```bash
cd paper && tar -czf arxiv_submission.tar.gz paper.tex figures/fig_waterfall.png figures/fig_decomposition.png figures/fig_survivor.png figures/fig_paired_protocol.png figures/fig_batch_decay.png
```

Then extract it into an empty directory and compile there. The tarball building in *your*
`paper/` directory proves nothing — it can silently pick up files the tarball does not carry.

---

## Venue timelines (verified 2026-09-04)

Confirmed from primary sources, not listing sites. Each row says where it came from so it can
be re-checked.

### arXiv — no deadline, ready to post

| item | value | source |
|---|---|---|
| Primary category | **cs.LG** (its description names *methodology* explicitly) | arxiv.org/category_taxonomy |
| Cross-list | **cs.PF** (*performance measurement and evaluation* — fits the protocol work), then **cs.CL** | same |
| Time to live | submit before 14:00 US Eastern on a weekday → public next day 20:00 ET | info.arxiv.org/help/availability.html |

Endorsement is the only thing that can delay posting, and it cannot be checked from outside
the account. Log in, start a cs.LG submission, and arXiv states whether one is needed.
Endorsers need papers **in cs.LG**, submitted **between 3 months and 5 years ago**. A
qualifying institutional address can trigger automatic endorsement, so a vizuara.com address
is worth trying first (info.arxiv.org/help/endorsement.html).

### TMLR — rolling, our archival home

Open now; the only pause on their calendar was 2 Dec 2025 – 5 Jan 2026 and has passed. Expect
a similar December pause. Target is a decision ~9 weeks after submission; 2025 medians were 91
days (short) and 104 (long). At 9 pages we are under their 12-page cutoff, so reviews are due
in 2 weeks rather than 4. arXiv preprints are explicitly permitted provided the TMLR version
does not link to a named copy.

### NeurIPS 2026 — ODI, and the deadline moved

**ENLSP does not exist in 2026.** It ran 2021–2024 and is not among the 102 accepted workshops
(blog.neurips.cc/2026/08/10). The efficiency slot went to new workshops, most already closed:
AXIOM 29 Aug, LIGHT 30 Aug.

**ODI — On-Device Intelligence: Foundation Models under Real-World Constraints** (Sydney) is
the one that fits and is still open. Site odi2026.github.io, contact odi.neurips2026@gmail.com.

| item | value |
|---|---|
| Paper deadline | **5 September 2026, 23:59 AoE** — *extended* from 29 August |
| | equivalently 6 Sep 11:59 UTC, or 6 Sep 17:29 IST |
| Author notification | **29 September 2026, 23:59 AoE** |
| Archival? | **Non-archival.** "Both tracks are non-archival and may be submitted elsewhere." |
| Review | Double-blind, short papers, best-paper award with a 15-minute oral |
| Workshop day | 11–12 December 2026, Sydney |

Two sources agree on the deadline: the workshop's own `content/dates.md` and the OpenReview
invitation `NeurIPS.cc/2026/Workshop/ODI/-/Submission` (`duedate` = 2026-09-06 11:59 UTC).
AoE is UTC−12, so the two are the same instant.

**Non-archival matters:** it means submitting to ODI does not block the TMLR submission, which
was the open risk when the status was unknown.

**Useful method.** Workshop sites are often JS-rendered and NeurIPS's own listing returns 403
to automated requests. Two routes that work:

```bash
# every 2026 workshop venue id
curl -s "https://api2.openreview.net/groups?prefix=NeurIPS.cc/2026/Workshop&limit=200"
# the authoritative deadline for one of them (duedate is epoch ms); note api2, not api
curl -s "https://api2.openreview.net/invitations?id=NeurIPS.cc/2026/Workshop/ODI/-/Submission"
```

The group record also carries `website` and `contact`. For a GitHub Pages workshop site whose
rendered page will not load, read the repo source instead (`content/dates.md` here).
