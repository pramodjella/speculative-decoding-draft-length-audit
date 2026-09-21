#!/usr/bin/env python3
"""Generate the TMLR submission source from paper/paper.tex.

The IEEEtran conference paper and the TMLR journal submission are the same work in two
containers, so the body prose is single-sourced: this script converts, it does not fork.
Edit paper/paper.tex, re-run this, rebuild. What it changes:

  * preamble          IEEEtran two-column -> article 10pt + tmlr.sty (single column, 6.5in)
  * authorship        \\IEEEauthorblock* -> TMLR \\author{\\name ... \\addr ...}. Kept in the
                      source but HIDDEN by \\usepackage{tmlr}: TMLR is double blind and says
                      non-anonymous submissions are rejected without review. Switching to
                      \\usepackage[accepted]{tmlr} reveals them for the camera-ready.
  * artifact link     the GitHub URL de-anonymises the authors in one click, so it is replaced
                      by \\artifactwhere, which points reviewers at the anonymised supplementary
                      material and, one commented line later, at the real repository.
  * citations         \\cite -> \\citep (tmlr.sty requires natbib; every call site here is
                      "SystemName \\cite{}", which is the parenthetical form)
  * bibliography      inline thebibliography -> \\bibliography{references} + tmlr.bst
  * floats            table* -> table (no two-column spanning), figures to 0.8\\textwidth
  * keywords          \\IEEEkeywords dropped (no such block in TMLR)
  * comments          LaTeX comments naming people are stripped; source uploads leak them

Usage: python3 scripts/build_tmlr.py
"""
import os
import re
import sys

SRC = os.path.join("paper", "paper.tex")
DST = os.path.join("paper", "tmlr", "paper_tmlr.tex")

PREAMBLE = r"""\documentclass[10pt]{article}

% TMLR double-blind submission. The [accepted] option reveals the authors and adds the
% journal header for the camera-ready; [preprint] de-anonymises without the TMLR branding.
\usepackage{tmlr}
%\usepackage[accepted]{tmlr}
%\usepackage[preprint]{tmlr}

\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage{url}

% Filled in for the camera-ready.
\def\month{MM}
\def\year{YYYY}
\def\openreview{\url{https://openreview.net/forum?id=XXXX}}

% The artifact pointer names the authors' GitHub account, which would defeat double-blind
% review, so for review it points at the anonymised supplementary material instead. It is a
% full predicate so the same macro reads correctly in both sentences that use it. For the
% camera-ready, comment the first line and uncomment the second.
\newcommand{\artifactwhere}{as anonymised supplementary material accompanying this submission}
%\newcommand{\artifactwhere}{at \url{https://github.com/pramodjella/speculative-decoding-draft-length-audit}}

% Unnumbered footnote for the artifact pointer on page 1.
\newcommand\blfootnote[1]{%
  \begingroup
  \renewcommand\thefootnote{}\footnote{#1}%
  \addtocounter{footnote}{-1}%
  \endgroup
}

\title{Following the Speedup: An Audit of Adaptive Draft Length in Speculative Decoding}

% Hidden while \usepackage{tmlr} is anonymous; revealed by the [accepted] option.
\author{\name Pramod Kumar Reddy Jella \email pramodjella1993@gmail.com \\
      \addr Vizuara AI Labs
      \AND
      \name Yash Dixit \\
      \addr Vizuara AI Labs
      \AND
      \name Naman Dwivedi \\
      \addr Vizuara AI Labs
      \AND
      \name Raj Dandekar \\
      \addr Vizuara AI Labs
      \AND
      \name Rajat Dandekar \\
      \addr Vizuara AI Labs
      \AND
      \name Sreedath Panat \\
      \addr Vizuara AI Labs}

\begin{document}
% Long model paths and URLs overflow under default tolerances; prefer loose interword
% spacing over text in the margin.
\sloppy

\maketitle
\blfootnote{Code, raw per-run JSON, and the benchmarking protocol are released
\artifactwhere.}
"""


def convert(text):
    steps = []

    def note(msg):
        steps.append(msg)

    # --- preamble: everything up to and including the page-one artifact footnote ---
    anchor = "\n\\begin{abstract}"
    idx = text.find(anchor)
    if idx < 0:
        sys.exit("could not find \\begin{abstract}")
    body = text[idx:]
    note("preamble replaced (article 10pt + tmlr.sty, TMLR \\author, anonymised artifact link)")

    # --- keywords: no such block in TMLR ---
    body, n = re.subn(r"\n\\begin\{IEEEkeywords\}.*?\\end\{IEEEkeywords\}\n", "\n", body,
                      flags=re.S)
    note(f"IEEEkeywords block removed ({n})")

    # --- citations: tmlr.sty requires natbib ---
    body, n = re.subn(r"\\cite\{", r"\\citep{", body)
    note(f"\\cite -> \\citep ({n})")

    # --- floats: single column, so nothing spans ---
    body, n = re.subn(r"\\(begin|end)\{table\*\}", r"\\\1{table}", body)
    note(f"table* -> table ({n // 2})")
    body, n = re.subn(r"\\includegraphics\[width=\\columnwidth\]",
                      r"\\includegraphics[width=0.8\\textwidth]", body)
    note(f"figure width \\columnwidth -> 0.8\\textwidth ({n})")

    # --- the failure-mode table was a 7in table*; at 6.5in its prose columns must wrap ---
    body, n = re.subn(
        r"\\begin\{tabular\}\{rlll\}",
        r"\\begin{tabular}{r>{\\raggedright\\arraybackslash}p{0.20\\textwidth}"
        r">{\\raggedright\\arraybackslash}p{0.29\\textwidth}"
        r">{\\raggedright\\arraybackslash}p{0.34\\textwidth}}",
        body)
    note(f"failure-mode table given wrapping columns ({n})")

    # --- artifact URL: both call sites go through the anonymisable macro ---
    # "at" and the URL are on separate source lines, so match across the break.
    body, n = re.subn(
        r"at\s*\n?\s*\\url\{https://github\.com/pramodjella/speculative-decoding-draft-length-audit\}",
        r"\\artifactwhere", body)
    note(f"'at <URL>' -> \\artifactwhere ({n})")

    # --- bibliography: hand list -> bibtex ---
    body, n = re.subn(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
                      "\\\\bibliographystyle{tmlr}\n\\\\bibliography{references}", body,
                      flags=re.S)
    note(f"thebibliography -> \\bibliography{{references}} ({n})")

    # --- comments naming people leak if the source is uploaded ---
    out = []
    dropped = 0
    for line in (PREAMBLE + body).split("\n"):
        if re.match(r"^\s*%", line) and re.search(
                r"Naman|Yash|Pramod|Dandekar|Panat|Dixit|Dwivedi", line):
            dropped += 1
            continue
        out.append(line)
    note(f"comments naming people stripped ({dropped})")
    return "\n".join(out), steps


def main():
    if not os.path.exists(SRC):
        sys.exit(f"missing {SRC}")
    text = open(SRC, encoding="utf-8").read()
    converted, steps = convert(text)
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    open(DST, "w", encoding="utf-8").write(converted)
    for s in steps:
        print("  " + s)
    print(f"\n  wrote {DST}")

    leaks = [(i, l) for i, l in enumerate(converted.split("\n"), 1)
             if re.search(r"github\.com/pramodjella|Vizuara", l) and not l.lstrip().startswith("%")]
    real = [(i, l) for i, l in leaks if "\\addr" not in l and "\\name" not in l]
    print(f"  de-anonymisation check: {len(real)} uncommented mention(s) outside \\author")
    for i, l in real:
        print(f"    line {i}: {l.strip()[:90]}")


if __name__ == "__main__":
    main()
