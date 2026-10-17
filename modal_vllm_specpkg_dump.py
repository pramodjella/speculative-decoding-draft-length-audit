"""Dump the entire vllm/v1/worker/gpu/spec_decode/ package source to the volume (CPU-only)."""
import os, modal

image = (modal.Image.debian_slim(python_version="3.12").pip_install("vllm"))
app = modal.App("vllm-specpkg-dump")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)


@app.function(image=image, timeout=600, volumes={"/root/out": vol})
def run():
    import vllm, pathlib
    root = pathlib.Path(vllm.__file__).parent / "v1" / "worker" / "gpu" / "spec_decode"
    out = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root)
        txt = p.read_text()
        out.append(f"\n\n########## FILE: {rel} ({len(txt.splitlines())} lines) ##########\n{txt}")
    blob = "".join(out)
    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/vllm_gpu_specdecode_pkg.txt", "w") as f:
        f.write(blob)
    vol.commit()
    return {"files": [str(p.relative_to(root)) for p in sorted(root.rglob('*.py'))],
            "chars": len(blob)}


@app.local_entrypoint()
def main():
    print(run.remote())
