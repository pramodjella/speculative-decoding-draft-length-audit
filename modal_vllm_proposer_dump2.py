"""Dump vLLM V1 EAGLE proposer source to the volume as a file (no GPU needed)."""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
app = modal.App("vllm-proposer-dump2")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)


@app.function(image=image, timeout=900, volumes={"/root/out": vol})
def run():
    import inspect, importlib, pkgutil, os
    import vllm
    lines = [f"vllm version: {vllm.__version__}\n"]
    import vllm.v1.spec_decode as SD
    mods = [m.name for m in pkgutil.iter_modules(SD.__path__)]
    lines.append(f"spec_decode submodules: {mods}\n")
    for modname in mods:
        try:
            mod = importlib.import_module(f"vllm.v1.spec_decode.{modname}")
        except Exception as e:
            lines.append(f"[{modname}] import failed: {e}\n")
            continue
        for cn in dir(mod):
            obj = getattr(mod, cn)
            if isinstance(obj, type) and obj.__module__ == mod.__name__:
                methods = [m for m in dir(obj)
                           if not m.startswith("_") and callable(getattr(obj, m, None))]
                lines.append(f"\n### class {modname}.{cn}  methods={methods}\n")
                for meth in ("propose", "propose_tree", "prepare_inputs", "load_model"):
                    fn = getattr(obj, meth, None)
                    if fn is None:
                        continue
                    try:
                        src = inspect.getsource(fn)
                        lines.append(f"\n----- SOURCE {cn}.{meth} -----\n{src}\n")
                    except Exception as e:
                        lines.append(f"[{cn}.{meth}] getsource failed: {e}\n")
    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/vllm_proposer_src.txt", "w") as f:
        f.write("".join(lines))
    vol.commit()
    total = sum(len(l) for l in lines)
    return {"chars": total, "modules": mods}


@app.local_entrypoint()
def main():
    print(run.remote())
