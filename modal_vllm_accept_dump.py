"""Dump vLLM 0.23 acceptance/rejection-sampling source (CPU-only) for the eta-rule harness.

Finds where spec-decode acceptance is decided: dumps rejection-sampler files plus every file
under v1/ whose text mentions acceptance, with full source for the top candidates.
"""
import os, modal

image = (modal.Image.debian_slim(python_version="3.12").pip_install("vllm==0.23.0"))
app = modal.App("vllm-accept-dump")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)


@app.function(image=image, timeout=900, volumes={"/root/out": vol})
def run():
    import vllm, pathlib
    root = pathlib.Path(vllm.__file__).parent / "v1"
    hits = []
    for p in sorted(root.rglob("*.py")):
        try:
            t = p.read_text()
        except Exception:
            continue
        score = t.count("accept") + t.count("reject")
        if score >= 3:
            hits.append((score, p, t))
    hits.sort(key=lambda x: -x[0])
    out = ["=== files mentioning accept/reject (score, path) ===\n"]
    for sc, p, _ in hits[:25]:
        out.append(f"{sc:4d}  {p.relative_to(root.parent)}\n")
    out.append("\n\n")
    for sc, p, t in hits[:6]:
        out.append(f"\n\n########## FILE ({sc}): {p.relative_to(root.parent)} "
                   f"({len(t.splitlines())} lines) ##########\n{t}")
    blob = "".join(out)
    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/vllm_accept_src.txt", "w") as f:
        f.write(blob)
    vol.commit()
    return {"top_files": [str(p.relative_to(root.parent)) for _, p, _ in hits[:8]],
            "chars": len(blob)}


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(run.remote(), indent=2))
