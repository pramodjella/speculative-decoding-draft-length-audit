"""Diag 5: dump the LIVE speculator (vllm/v1/worker/gpu/ path) — class source + nature of
its propose attribute (class method vs instance-attr compiled wrapper).

Stack (diag4): gpu/model_runner.py:1440 -> self.speculator.propose(...). Everything under
vllm.v1.spec_decode.* is a parallel dead path for this config.

Run: PYTHONIOENCODING=utf-8 modal run modal_vllm_tailprune_diag5.py
Out: /root/out/vllm_speculator_src.txt
"""
import os, modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0",
          "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
          "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
)
app = modal.App("vllm-tailprune-diag5")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

TARGET = "meta-llama/Llama-3.1-8B-Instruct"
EAGLE  = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


@app.function(image=image, gpu="H100", timeout=2400, volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run():
    import inspect
    from vllm import LLM, SamplingParams

    llm = LLM(model=TARGET,
              speculative_config={"method": "eagle3", "model": EAGLE,
                                  "num_speculative_tokens": 7},
              gpu_memory_utilization=0.7, max_model_len=2048,
              enforce_eager=True, disable_log_stats=True)
    llm.generate(["hi"], SamplingParams(temperature=0.0, max_tokens=8), use_tqdm=False)

    # navigate to the live runner with fallbacks
    eng = llm.llm_engine
    core = getattr(eng, "engine_core", eng)
    core = getattr(core, "engine_core", core)
    ex = getattr(core, "model_executor", None)
    w = getattr(ex, "driver_worker", None)
    runner = getattr(w, "model_runner", w)
    print("runner type:", type(runner).__module__, type(runner).__name__, flush=True)

    spec = getattr(runner, "speculator", None)
    print("speculator:", type(spec).__module__, type(spec).__name__, flush=True)

    OUT = [f"runner: {type(runner).__module__}.{type(runner).__name__}\n",
           f"speculator: {type(spec).__module__}.{type(spec).__name__}\n",
           f"speculator mro: {[c.__name__ for c in type(spec).__mro__]}\n"]

    p = spec.propose
    OUT.append(f"\npropose object: {type(p)}  repr={repr(p)[:200]}\n")
    OUT.append(f"propose in type dict: {'propose' in type(spec).__dict__}\n")
    OUT.append(f"propose in instance dict: {'propose' in getattr(spec, '__dict__', {})}\n")
    for attr in ("__wrapped__", "__func__", "fn", "_torchdynamo_orig_callable"):
        if hasattr(p, attr):
            OUT.append(f"propose.{attr}: {repr(getattr(p, attr))[:200]}\n")

    inst_keys = sorted(getattr(spec, "__dict__", {}).keys())
    OUT.append(f"\nspeculator instance attrs: {inst_keys}\n")

    try:
        src = inspect.getsource(type(spec))
        OUT.append(f"\n===== FULL CLASS SOURCE {type(spec).__name__} =====\n{src}\n")
    except Exception as e:
        OUT.append(f"class getsource failed: {e}\n")
        try:
            fn = getattr(type(spec), "propose", None) or (
                p.__func__ if hasattr(p, "__func__") else p)
            OUT.append(f"\n===== propose SOURCE =====\n{inspect.getsource(fn)}\n")
        except Exception as e2:
            OUT.append(f"propose getsource failed: {e2}\n")

    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/vllm_speculator_src.txt", "w") as f:
        f.write("".join(OUT))
    vol.commit()
    print("dumped", sum(len(x) for x in OUT), "chars", flush=True)
    return {"speculator": f"{type(spec).__module__}.{type(spec).__name__}"}


@app.local_entrypoint()
def main():
    print(run.remote())
