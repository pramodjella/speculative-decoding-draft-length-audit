"""Diagnostic: WHY did the patched propose have no effect? (arms didn't separate)

Checks, in one cheap run (~10 min):
  1. Does patched propose EXECUTE at all? (counter + pid print from inside it)
  2. Do the logits hooks execute? (counter + pid)
  3. Same process as main? (os.getpid comparison -> engine-core subprocess hypothesis)
  4. Ground truth independent of process boundaries: SpecDecoding acceptance metrics
     (disable_log_stats=False) under forced fixed2 vs fixed7 — if the arm works,
     acceptance-per-step MUST drop to <=2 under fixed2.

Run: PYTHONIOENCODING=utf-8 modal run modal_vllm_tailprune_diag.py
"""
import os, time, json
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0",
          "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
          "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
)
app = modal.App("vllm-tailprune-diag")

TARGET = "meta-llama/Llama-3.1-8B-Instruct"
EAGLE  = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


@app.function(image=image, gpu="H100", timeout=2400,
              secrets=[modal.Secret.from_name("huggingface")])
def run():
    import torch, importlib, pkgutil
    print(f"MAIN pid={os.getpid()}", flush=True)

    EL = {"arm": ("fixed", 7), "n_propose": 0, "n_logits": 0}

    import vllm.model_executor.models as MM
    def wrap_logits(fn):
        def patched(self, *a, **k):
            EL["n_logits"] += 1
            if EL["n_logits"] in (1, 2):
                print(f"[hook] compute_logits pid={os.getpid()} call#{EL['n_logits']}",
                      flush=True)
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

    import vllm.v1.spec_decode.eagle as E
    orig_propose = E.EagleProposer.propose
    def propose_probe(self, *a, **k):
        EL["n_propose"] += 1
        if EL["n_propose"] in (1, 2):
            print(f"[probe] propose pid={os.getpid()} call#{EL['n_propose']} "
                  f"arm={EL['arm']} num_spec={self.num_speculative_tokens}", flush=True)
        out = orig_propose(self, *a, **k)
        # arm: fixed-k truncation by PADDING the returned tensor after column k-?
        kind, val = EL["arm"]
        if kind == "fixed" and val < out.shape[1]:
            out[:, val:] = out[:, val - 1:val]   # repeat column -> forced rejection
        return out
    E.EagleProposer.propose = propose_probe
    print("propose PROBE installed", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=TARGET,
              speculative_config={"method": "eagle3", "model": EAGLE,
                                  "num_speculative_tokens": 7},
              gpu_memory_utilization=0.7, max_model_len=2048,
              enforce_eager=True, disable_log_stats=False)
    sp = SamplingParams(temperature=0.0, max_tokens=96)

    from datasets import load_dataset
    prompts = list(load_dataset("openai/gsm8k", "main", split="test")["question"])[1:9:2]

    def counters():
        # read cumulative spec-decode counters from engine metrics if reachable
        return {"n_propose": EL["n_propose"], "n_logits": EL["n_logits"]}

    def bench(arm):
        EL["arm"] = arm
        torch.cuda.synchronize(); t0 = time.time(); ntok = 0
        for p in prompts:
            out = llm.generate([p], sp, use_tqdm=False)
            ntok += len(out[0].outputs[0].token_ids)
        torch.cuda.synchronize()
        return round(ntok / (time.time() - t0), 2)

    print("warmup...", flush=True)
    llm.generate(["hi"], SamplingParams(temperature=0.0, max_tokens=8), use_tqdm=False)
    print(f"after warmup: {counters()}", flush=True)

    t7 = bench(("fixed", 7))
    c7 = counters()
    print(f"fixed7: {t7} tok/s  counters={c7}", flush=True)
    t2 = bench(("fixed", 2))
    c2 = counters()
    print(f"fixed2: {t2} tok/s  counters={c2}", flush=True)
    print(f"SEPARATION: fixed2/fixed7 = {t2/t7:.3f} (expect ~0.7-0.9 if arm effective)",
          flush=True)
    return {"fixed7_tps": t7, "fixed2_tps": t2, "counters": c2}


@app.local_entrypoint()
def main():
    print(run.remote())
