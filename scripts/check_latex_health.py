"""Structural health checks for paper/paper.tex.

Every bug guarded here actually shipped past a string-only stale-check and was caught only
by a mentor reading the rendered output, or by an attempted compile:

  1. double-escaped percent (\\\\% instead of \\%)  -- data cells rendered as literal backslashes
  2. lone CR bytes                                 -- LaTeX reads them as line ends, which
                                                      turned a \\caption into a runaway argument
  3. stray C0 control bytes (BEL/BS/VT/FF/ESC)     -- a Python "\\a"/"\\b"/"\\f" escape ate the
                                                      backslash and the macro's first letter
  4. PDF older than source                         -- the committed artifact silently going stale

Exit 0 = healthy, 1 = problems (printed).
Usage: python scripts/check_latex_health.py [path/to/paper.tex]
"""
import os
import sys

BS = chr(92)
CONTROL = {7: "BEL (likely ate a backslash-a, e.g. approx)",
           8: "BS  (likely ate a backslash-b)",
           11: "VT  (likely ate a backslash-v)",
           12: "FF  (likely ate a backslash-f)",
           27: "ESC (likely ate a backslash-e)"}


def check(tex_path):
    problems = []
    if not os.path.exists(tex_path):
        return [f"missing file: {tex_path}"]
    raw = open(tex_path, "rb").read()
    text = raw.decode("utf-8", errors="replace")

    n_dbl = text.count(BS + BS + "%")
    if n_dbl:
        problems.append(f"double-escaped percent ({BS}{BS}%) x{n_dbl} "
                        f"-- renders as a literal backslash in data cells")

    lone = [i for i in range(len(raw) - 1)
            if raw[i:i + 1] == b"\r" and raw[i + 1:i + 2] != b"\n"]
    if lone:
        problems.append(f"lone CR bytes x{len(lone)} -- LaTeX treats these as line ends "
                        f"and they can break a caption into a runaway argument")

    for code, why in CONTROL.items():
        n = raw.count(bytes([code]))
        if n:
            problems.append(f"stray control byte U+{code:04X} x{n} -- {why}")

    pdf = os.path.splitext(tex_path)[0] + ".pdf"
    if os.path.exists(pdf) and os.path.getmtime(tex_path) > os.path.getmtime(pdf):
        problems.append("paper.pdf is older than paper.tex -- recompile before shipping")
    elif not os.path.exists(pdf):
        problems.append("paper.pdf does not exist -- compile before shipping")

    return problems


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("paper", "paper.tex")
    probs = check(path)
    for p in probs:
        print(f"LATEX: {p}")
    sys.exit(1 if probs else 0)
