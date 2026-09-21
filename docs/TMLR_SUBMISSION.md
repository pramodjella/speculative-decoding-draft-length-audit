# TMLR submission — procedure for this paper

The TMLR source lives in `paper/tmlr/` and is **generated**, not hand-edited. Edit
`paper/paper.tex` (the single source for both containers), then:

```bash
python3 scripts/build_tmlr.py                      # regenerate paper/tmlr/paper_tmlr.tex
cd paper/tmlr && pdflatex paper_tmlr && bibtex paper_tmlr && pdflatex paper_tmlr && pdflatex paper_tmlr
```

Verified: 16 pages, pdflatex ×3 + bibtex, **0 errors, 0 overfull boxes, 0 unresolved
references or citations**, all fonts embedded and subsetted.

---

## Why TMLR

arXiv declined the preprint and placed a standing restriction on the submitting account:
submissions now need a journal reference or DOI. Peer review therefore has to come first, and
`report/venue_authorship_memo.md` already listed TMLR as option **A2**. It fits unusually
well: rolling submission with no deadline, and a stated bar of *technical correctness over
subjective significance* — the right bar for a negative-result measurement paper. Acceptance
gives the journal reference arXiv is asking for (ISSN 2835-8856), after which the paper can be
posted.

---

## Before you submit — four things only you can settle

**1. Double-blind is enforced.** TMLR states that non-anonymous submissions are *rejected
without review*. The generated PDF is clean: zero occurrences of the author names, the
affiliation, or the GitHub account in the rendered text, and no author or title in the PDF
metadata. `scripts/build_tmlr.py` re-checks this on every run. **Do not** hand-edit
`paper_tmlr.tex` to add the repository link back.

**2. Decide how reviewers see the artifact.** Right now the paper says the repository is
"withheld here to preserve anonymity and linked in the camera-ready". That is honest and
costs nothing, but it does deny reviewers the artifact, and this paper's whole argument is
that its numbers reconcile against released data. The standard fix is an anonymised mirror at
<https://anonymous.4open.science>. If you create one, uncomment the second `\artifacturl`
definition in the preamble and point it there.

**3. Confirm the author list.** `paper/tmlr/paper_tmlr.tex` carries all six authors in the
`\author{}` block, hidden by the anonymous mode. `report/venue_authorship_memo.md` proposed
two (first author + senior author) with the others in acknowledgements, and that question was
never closed in writing. Settle it before the camera-ready, not after.

**4. Sync the public artifact repository.** It is still on the 2026-09-04 state: retired
numbers in `DELIVERABLES.md` and a pre-fix `paper.tex`. Run
`scripts/sync_public_artifact.sh <path>`. This matters less for a blind submission than it did
for arXiv, but the repository is public *now*, so it is contradicting the paper in the open.

---

## Submitting on OpenReview

1. TMLR submissions go through OpenReview. Create the submission and upload
   **`paper/tmlr/paper_tmlr.pdf`** — the PDF only; the source is not required.
2. Enter the title and abstract as plain text. Strip nothing: the abstract has no markup.
3. TMLR asks the submitting author to certify a few things on the form, including that the
   submission is anonymised and is not concurrently under review elsewhere. Both hold.
4. Expect an Action Editor assignment, then reviewers. The review period is short by journal
   standards (roughly two to three months to a first decision), and the whole exchange is
   public on OpenReview.

## Afterwards

- On acceptance, switch `\usepackage{tmlr}` to `\usepackage[accepted]{tmlr}`, fill in
  `\month`, `\year` and `\openreview`, point `\artifacturl` at the real repository, and
  rebuild. That reveals the authors and adds the journal header.
- Then post to arXiv **with the TMLR journal reference attached**, which is what the
  moderators asked for. Keep `docs/ARXIV_SUBMISSION.md` for that step.
- Put the OpenReview link and the journal reference in `README.md` and `docs/CANONICAL.md`.

## What the converter changes, and why

| From (IEEEtran) | To (TMLR) |
|---|---|
| `\documentclass[conference]{IEEEtran}`, two columns | `\documentclass[10pt]{article}` + `tmlr.sty`, one column, 6.5in |
| `\IEEEauthorblockN/A` | `\author{\name ... \addr ...}`, hidden by anonymous mode |
| GitHub URL, twice | `\artifacturl`, anonymity-preserving for review |
| `\cite` (numeric) | `\citep` (natbib author-year; `tmlr.sty` requires natbib) |
| inline `thebibliography` | `references.bib` + `tmlr.bst` |
| `table*` spanning two columns | `table`, prose columns wrapped with `p{}` |
| `\includegraphics[width=\columnwidth]` | `width=0.8\textwidth` |
| `\IEEEkeywords` | dropped — no such block in TMLR |
| comments naming people | stripped |
