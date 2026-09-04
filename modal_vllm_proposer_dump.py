"""Diagnostic: dump vLLM 0.23 V1 EAGLE proposer source + verify Qwen3 loads eagerly.

Phase 1 of the vLLM-native tail-pruning replication. Before patching the draft loop, we need
the ACTUAL propose() source (version-specific), the class layout, and confirmation that
Qwen3-14B + AngelSlim eagle3 head runs under enforce_eager with our usual hooks.

Run: PYTHONIOENCODING=utf-8 modal run modal_vllm_proposer_dump.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0",
          "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
          "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
)
app = modal.App("vllm-proposer-dump")

TARGET = "Qwen/Qwen3-14B"
EAGLE  = "AngelSlim/Qwen3-14B_eagle3"


@app.function(image=image, gpu="H100", timeout=2400,
              secrets=[modal.Secret.from_name("huggingface")])
def run():
    import inspect, importlib, pkgutil, torch
    import vllm
    print("vllm", vllm.__version__, flush=True)

    # ── 1. locate the V1 spec-decode proposer modules and dump propose() ── #
    import vllm.v1.spec_decode as SD
    print("spec_decode submodules:", [m.name for m in pkgutil.iter_modules(SD.__path__)],
          flush=True)
    for modname in [m.name for m in pkgutil.iter_modules(SD.__path__)]:
        try:
            mod = importlib.import_module(f"vllm.v1.spec_decode.{modname}")
        except Exception as e:
            print(f"[{modname}] import failed: {e}", flush=True)
            continue
        for cn in dir(mod):
            obj = getattr(mod, cn)
            if isinstance(obj, type) and hasattr(obj, "propose"):
                print(f"\n===== {modname}.{cn}.propose SOURCE =====", flush=True)
                try:
                    src = inspect.getsource(obj.propose)
                    print(src, flush=True)
                except Exception as e:
                    print(f"getsource failed: {e}", flush=True)
                # also dump __init__ signature + attributes of interest
                try:
                    print(f"--- {cn}.__init__{inspect.signature(obj.__init__)}", flush=True)
                except Exception:
                    pass

    # ── 2. confirm Qwen3 + AngelSlim head loads and generates eagerly ────── #
    from vllm import LLM, SamplingParams
    llm = LLM(model=TARGET,
              speculative_config={"method": "eagle3", "model": EAGLE,
                                  "num_speculative_tokens": 7},
              gpu_memory_utilization=0.8, max_model_len=2048,
              enforce_eager=True, disable_log_stats=False)
    out = llm.generate(["What is 17*24? Show your steps."],
                       SamplingParams(temperature=0.0, max_tokens=48), use_tqdm=False)
    txt = out[0].outputs[0].text
    print("\nQWEN3 EAGER GEN OK:", repr(txt[:160]), flush=True)
    return {"ok": True}


@app.local_entrypoint()
def main():
    print(run.remote())
