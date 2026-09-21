#!/usr/bin/env python3
"""Hold the unpublished BCSD line out of the public artifact repo.

The public repo carries no BCSD *code* -- that split was done correctly. docs/RESEARCH_GUIDE.md
is the problem: it is an internal orientation document that describes the follow-on line end to
end. Not only in its own sections (the Lane B file map, the function-by-function walkthrough of
the in-engine harness, and the math from the eta-rule and ledger through the sequence
certificate and the Hellinger identity) but woven through Parts 1-3 as well, in the
architecture diagrams and the decision log.

Cutting it section by section was tried and rejected: it still left the mechanism visible in
the Part 1 and Part 3 diagrams, a dangling "see Part 5" cross-reference, and a decision-table
row weighing the line against a competitor's paper. A half-redacted document is worse than
either extreme.

So the guide is withheld whole. Nothing in the paper points at it -- paper.tex cites
DELIVERABLES.md and scripts/check_stale.sh -- and docs/ARCHITECTURE.md already serves as the
public "how the code fits together" document. The README link is retargeted to match.

Idempotent. Usage: redact_public_docs.py <public-repo-path>
"""
import os
import subprocess
import sys

GUIDE = "docs/RESEARCH_GUIDE.md"


def drop_guide(pub):
    path = os.path.join(pub, GUIDE)
    if not os.path.exists(path):
        return "already withheld"
    tracked = subprocess.run(["git", "-C", pub, "ls-files", "--error-unmatch", GUIDE],
                             capture_output=True).returncode == 0
    if tracked:
        subprocess.run(["git", "-C", pub, "rm", "-q", GUIDE], check=True)
    else:
        os.remove(path)
    return "withheld (internal orientation doc; describes the unpublished line throughout)"


def fix_architecture(pub):
    path = os.path.join(pub, "docs", "ARCHITECTURE.md")
    if not os.path.exists(path):
        return "absent"
    text = open(path, encoding="utf-8").read()
    old = ("> **Looking for the deep version?** [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md) covers the\n"
           "> full evolution of the research (every fork and why), file-by-file and\n"
           "> function-by-function flows including the BCSD in-engine harness, and the math\n"
           "> explained with child-level examples (η-rule, ledger, chain gating, the exact\n"
           "> Hellinger identity). This page remains the quick Lane-A (audit) intro.\n\n")
    if old not in text:
        return "already current"
    open(path, "w", encoding="utf-8").write(text.replace(old, ""))
    return "removed the pointer to the withheld guide"


def fix_readme(pub):
    path = os.path.join(pub, "README.md")
    if not os.path.exists(path):
        return "absent"
    text = open(path, encoding="utf-8").read()
    old = "RESEARCH_GUIDE.md (orientation), ARXIV_SUBMISSION.md (posting procedure)"
    new = "ARXIV_SUBMISSION.md (posting procedure)"
    if old not in text:
        return "already current"
    open(path, "w", encoding="utf-8").write(text.replace(old, new))
    return "dropped the link to the withheld guide"


def scan(pub):
    hits = []
    for root, dirs, files in os.walk(pub):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            if not f.endswith((".md", ".py", ".sh", ".tex")):
                continue
            if f == os.path.basename(__file__):   # this scanner names what it looks for
                continue
            p = os.path.join(root, f)
            try:
                body = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for i, line in enumerate(body.split("\n"), 1):
                if "BCSD" in line:
                    hits.append((os.path.relpath(p, pub), i, line.strip()[:88]))
    return hits


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <public-repo-path>")
    pub = sys.argv[1]
    print(f"  {GUIDE}: {drop_guide(pub)}")
    print(f"  docs/ARCHITECTURE.md: {fix_architecture(pub)}")
    print(f"  README.md: {fix_readme(pub)}")
    hits = scan(pub)
    print(f"\n  remaining mentions by name ({len(hits)}) -- review, none disclose the method:")
    for rel, i, line in hits:
        print(f"    {rel}:{i}  {line}")


if __name__ == "__main__":
    main()
