"""Compile paper/paper.tex to PDF on Modal (no local LaTeX toolchain).

Yash's audit item: the committed PDF is six weeks stale and predates every correction, and
the results table did not compile (double-escaped percents, a CR-shattered \\rightarrow).
Both source bugs are fixed; this actually rebuilds the artifact and reports any remaining
LaTeX errors rather than assuming success.

CPU only, runs in well under a minute of compute.

Run: modal run modal_compile_paper.py
Out: paper/paper.pdf (downloaded), plus the compile log on stdout.
"""
import os
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("texlive-latex-base", "texlive-latex-recommended", "texlive-latex-extra",
                 "texlive-fonts-recommended", "texlive-publishers", "latexmk")
)
app = modal.App("compile-paper")

HERE = os.path.dirname(os.path.abspath(__file__))
image = image.add_local_dir(os.path.join(HERE, "paper"), remote_path="/work/paper")


@app.function(image=image, timeout=1800)
def compile_tex():
    import subprocess, pathlib
    os.chdir("/work/paper")
    log = ""
    for i in range(3):                       # 3 passes: refs, citations, final
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            "paper.tex"], capture_output=True, text=True)
        log = r.stdout + r.stderr
        if r.returncode != 0:
            tail = [l for l in log.splitlines()
                    if l.startswith("!") or "Error" in l or "Undefined" in l][:25]
            return {"ok": False, "pass": i + 1, "errors": tail,
                    "log_tail": log.splitlines()[-40:]}
    pdf = pathlib.Path("paper.pdf")
    if not pdf.exists():
        return {"ok": False, "errors": ["pdflatex returned 0 but produced no PDF"],
                "log_tail": log.splitlines()[-40:]}
    warns = [l for l in log.splitlines()
             if "Warning" in l and "Font" not in l][:20]
    pages = log.count("[") if "Output written" not in log else None
    m = [l for l in log.splitlines() if "Output written" in l]
    return {"ok": True, "bytes": pdf.stat().st_size, "output_line": m[0] if m else "",
            "warnings": warns, "pdf": pdf.read_bytes()}


@app.local_entrypoint()
def main():
    res = compile_tex.remote()
    if not res["ok"]:
        print("COMPILE FAILED at pass", res.get("pass"))
        for e in res["errors"]:
            print("  ", e)
        print("--- log tail ---")
        for l in res["log_tail"]:
            print("  ", l)
        raise SystemExit(1)
    out = os.path.join(HERE, "paper", "paper.pdf")
    with open(out, "wb") as f:
        f.write(res["pdf"])
    print("COMPILE OK")
    print(" ", res["output_line"])
    print(f"  wrote {out} ({res['bytes']} bytes)")
    if res["warnings"]:
        print("  warnings:")
        for w in res["warnings"]:
            print("   ", w)
