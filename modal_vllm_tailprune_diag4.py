"""Diag 4 (decisive): stack-trace the draft loop from inside the compute_logits hook.

gc found no 'propose'-typed objects, yet draft-head compute_logits fires in-process. A stack
trace captured mid-drafting names the exact module.function that runs the draft loop. Also:
broad gc scan (any type defining 'propose'), and dump of the runner's drafting entrypoint.

Run: PYTHONIOENCODING=utf-8 modal run modal_vllm_tailprune_diag4.py
Out: /root/out/vllm_draft_stack.txt
"""
import os, modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0",
          "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
          "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
)
app = modal.App("vllm-tailprune-diag4")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

TARGET = "meta-llama/Llama-3.1-8B-Instruct"
EAGLE  = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


@app.function(image=image, gpu="H100", timeout=2400, volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run():
    import gc, inspect, traceback, importlib, pkgutil, torch

    OUT = []

    # hook draft-head compute_logits to capture stacks
    import vllm.model_executor.models as MM
    ST = {"n": 0, "stacks": []}
    def wrap_logits(fn):
        def patched(self, *a, **k):
            ST["n"] += 1
            if ST["n"] in (30, 60):        # mid-drafting, past init/profile calls
                ST["stacks"].append("".join(traceback.format_stack()[-18:]))
            return fn(self, *a, **k)
        return patched
    for _, name, _ in pkgutil.iter_modules(MM.__path__):
        if "eagle" not in name.lower():
            continue
        try:
            mod = importlib.import_module(f"vllm.model_executor.models.{name}")
        except Exception:
            continue
        for cn in dir(mod):
            obj = getattr(mod, cn)
            if isinstance(obj, type) and "eagle3" in cn.lower() and hasattr(obj, "compute_logits"):
                obj.compute_logits = wrap_logits(obj.compute_logits)

    from vllm import LLM, SamplingParams
    llm = LLM(model=TARGET,
              speculative_config={"method": "eagle3", "model": EAGLE,
                                  "num_speculative_tokens": 7},
              gpu_memory_utilization=0.7, max_model_len=2048,
              enforce_eager=True, disable_log_stats=True)
    llm.generate(["What is 17*24? Show your steps."],
                 SamplingParams(temperature=0.0, max_tokens=32), use_tqdm=False)

    OUT.append(f"compute_logits calls: {ST['n']}\n")
    for i, s in enumerate(ST["stacks"]):
        OUT.append(f"\n===== STACK #{i} (draft-head compute_logits) =====\n{s}\n")
        print(f"===== STACK #{i} =====\n{s}", flush=True)

    # broad gc scan: ANY type defining propose
    types_found = {}
    for o in gc.get_objects():
        try:
            t = type(o)
            if "propose" in getattr(t, "__dict__", {}) or any(
                    "propose" in getattr(b, "__dict__", {}) for b in t.__mro__[1:2]):
                types_found[f"{t.__module__}.{t.__name__}"] = t
        except Exception:
            continue
    OUT.append(f"\ntypes with propose in dict/mro1: {sorted(types_found)}\n")
    print("propose-bearing live types:", sorted(types_found), flush=True)

    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/vllm_draft_stack.txt", "w") as f:
        f.write("".join(OUT))
    vol.commit()
    return {"n_logits": ST["n"], "n_stacks": len(ST["stacks"]),
            "propose_types": sorted(types_found)}


@app.local_entrypoint()
def main():
    print(run.remote())
