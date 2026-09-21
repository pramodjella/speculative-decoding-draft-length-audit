#!/usr/bin/env python3
"""Build the anonymised artifact bundle that goes to OpenReview as supplementary material.

TMLR is double blind, so the public repository cannot be linked from the submission. But this
paper's whole argument is that its numbers reconcile against released measurements, and a
reviewer who cannot check them has to take that on trust. Supplementary material closes the
gap without an anonymising proxy: the same tree, with the identifying parts removed.

Source is the PUBLIC repo (already free of the follow-on line's code), minus git history,
minus the built paper, minus anything naming an author, an affiliation or the account. Every
file is scanned after copying, and the build fails rather than shipping a leak.

Usage: build_tmlr_supplement.py <public-repo-path> [out.zip]
"""
import os
import re
import shutil
import sys
import tempfile
import zipfile

# Whole paths that identify the authors or only make sense with the paper attached.
DROP_PATHS = {
    "README.md",                    # names all six authors and carries the BibTeX block
    "docs/ARXIV_SUBMISSION.md",     # names the authors and the account
    "docs/TMLR_SUBMISSION.md",      # names the account
    "paper",                        # the de-anonymised PDF and its source
    "report",                       # internal review correspondence
    "literature_matrix.xlsx",
    "scripts/build_docx.sh",        # its pandoc metadata block lists every author by name
}
IDENTIFIERS = ["Vizuara", "pramodjella", "Pramod", "Jella", "Dixit", "Dwivedi",
               "Dandekar", "Panat", "vizuara.com"]
SCAN_EXT = (".md", ".py", ".sh", ".txt", ".json", ".csv", ".tex", ".cu", ".ipynb")

README = """# Artifact for "Following the Speedup: An Audit of Adaptive Draft Length in Speculative Decoding"

Anonymised supplementary material for TMLR review. Author names, the affiliation, the
repository URL and the built paper have been removed; nothing else has been changed.

## Reconciling the paper's tables

Every number in Tables 1-7 comes from the JSON under `results/`. The quickest checks:

| Paper | Artifact |
|---|---|
| Table 2, decomposition | `results/perstep_signal/bayes_ceiling_paired.json` -- per-fold `rec_deployable_pct`, `rec_ceiling_pct`; both self-tests (`position_selftest_gap == 0`, ceiling >= deployable) hold in 8/8 folds |
| Table 4, policy zoo | `results/eagle_zoo_verify.json` (held-out), `results/eagle_zoo_verify_iid.json` (in-distribution) |
| Table 5, paired verdict | `results/eagle_paired.json` -- `summary.<workload>.vs_best_fixed` |
| Table 6, fair baseline | `results/vllm_fairbase_llama8b_r{1,2,3}.json` -- `cum_arms.cum0.2.vs_strongest_native` |
| Table 7, controllers | `results/eagle3_controller_findings.md` |
| Verify microbench | `results/verify_microbench.json` |
| Permutation control | `results/perstep_signal/bayes_ceiling_control_paired.json` |

`DELIVERABLES.md` maps each experiment (E1-E6) to the script and artifact that produced it.
Its section (b) keeps a superseded un-paired ladder behind an explicit banner, for the audit
trail; the paper quotes the paired values, and the banner says so.

Table 3 (the in-loop probe) is the one table whose raw JSON is not here: that harness writes
to a remote volume. Its cells are recorded in `DELIVERABLES.md` under E5.

## Running things

`requirements.txt` covers the analysis scripts. The capture and wall-clock harnesses
(`modal_*.py`) need an H100 and a Modal account, so they are included to be read rather than
re-run. The analysis scripts (`analyze_*.py`) run on the released JSON on any machine.
"""


def redact_lines(path, fn):
    """Rewrite a file line by line. Used where an identity sits in the contents, not the path:
    a developer's absolute path in a script header, an author byline in a findings note."""
    if not os.path.exists(path):
        return
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    open(path, "w", encoding="utf-8").write("\n".join(fn(l) for l in lines))


def main():
    if not 2 <= len(sys.argv) <= 3:
        sys.exit(f"usage: {sys.argv[0]} <public-repo-path> [out.zip]")
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) == 3 else "paper/tmlr/supplementary_anonymous.zip"

    tmp = tempfile.mkdtemp()
    root = os.path.join(tmp, "artifact")
    shutil.copytree(src, root, ignore=shutil.ignore_patterns(".git", "*.pyc", "__pycache__"))

    for rel in DROP_PATHS:
        p = os.path.join(root, rel)
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)
    open(os.path.join(root, "README.md"), "w", encoding="utf-8").write(README)

    # Two files carry an identity inside their contents rather than in their path.
    redact_lines(os.path.join(root, "analyze_bayes_ceiling_control.py"),
                 lambda l: re.sub(r"r'F:[^']*'", "r'<path to the capture parquet>'", l))
    redact_lines(os.path.join(root, "results", "insight_report.md"),
                 lambda l: "**Author:** anonymised for review" if l.startswith("**Author:**") else l)

    # Scan what survived. Anything still naming an author fails the build.
    leaks = []
    for dirpath, dirs, files in os.walk(root):
        for f in files:
            if not f.endswith(SCAN_EXT):
                continue
            p = os.path.join(dirpath, f)
            try:
                body = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for i, line in enumerate(body.split("\n"), 1):
                for ident in IDENTIFIERS:
                    if ident in line:
                        leaks.append((os.path.relpath(p, root), i, ident, line.strip()[:80]))
    if leaks:
        print(f"REFUSING: {len(leaks)} identifying mention(s) survive:")
        for rel, i, ident, line in leaks[:40]:
            print(f"  {rel}:{i}  [{ident}]  {line}")
        sys.exit(1)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirs, files in os.walk(root):
            for f in sorted(files):
                p = os.path.join(dirpath, f)
                z.write(p, os.path.relpath(p, tmp))
                n += 1
    shutil.rmtree(tmp)
    print(f"  {n} files, {os.path.getsize(out) / 1e6:.1f} MB -> {out}")
    print("  anonymisation: clean")


if __name__ == "__main__":
    main()
