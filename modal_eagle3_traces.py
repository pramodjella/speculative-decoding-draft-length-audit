"""EAGLE-3 MULTI-K per-prompt traces for Llama-3.1-8B -> offline controller sim.

vLLM 0.23.0 has NO in-engine hook to vary EAGLE-3's K per step, so we capture
EAGLE-3 acceptance at EACH fixed K per prompt and run controllers OFFLINE to ask:
does a per-request adaptive-K controller beat fixed K=3 on EAGLE-3?

Captures, for each (prompt, K in KS):
  - accept_len = accepted_draft_tokens/num_spec_steps + 1   (deterministic, greedy)
  - latency    = wall-clock for that prompt (single-shot, noisy -> use for cost ref)
Plus a no-spec baseline latency per prompt. Counters read by DIFFING vLLM's
cumulative spec-decode counters (real 0.23 names, no `_total` suffix).

Output: /root/out/eagle3_multik_llama8b.json
Run: PYTHONIOENCODING=utf-8 modal run modal_eagle3_traces.py
"""
import os, json, time
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_ATTENTION_BACKEND": "FLASH_ATTN"})
)
app = modal.App("eagle3-traces")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

TARGET = "meta-llama/Llama-3.1-8B-Instruct"
EAGLE = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
KS = [int(x) for x in os.environ.get("TK_KS", "1,2,3,5,7").split(",")]
N = int(os.environ.get("TK_N", "30"))           # prompts per workload
MAXTOK = int(os.environ.get("TK_MAXTOK", "128"))
ACC = "vllm:spec_decode_num_accepted_tokens"     # confirmed 0.23 names (no _total)
STEPS = "vllm:spec_decode_num_drafts"


@app.function(image=image, gpu="H100", timeout=5400,
              volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run():
    import torch
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    import vllm; print("vllm", vllm.__version__, flush=True)

    def take(x, n): return list(x)[:n]
    workloads = {
        "humaneval": take(load_dataset("openai/openai_humaneval", split="test")["prompt"], N),
        "gsm8k": take(load_dataset("openai/gsm8k", "main", split="test")["question"], N),
        "mt_bench": take(load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
                         .map(lambda x: {"p": x["prompt"][0]})["p"], N),
    }
    flat = [(w, i, p) for w, ps in workloads.items() for i, p in enumerate(ps)]
    sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK)

    def counter(llm, name):
        try:
            ms = llm.get_metrics()
        except Exception:
            ms = llm.llm_engine.get_metrics()
        for m in ms:
            if m.name == name:
                v = getattr(m, "value", None)
                if v is None: v = getattr(m, "count", None)
                return float(v) if v is not None else None
        return None

    def timed(llm, p):
        torch.cuda.synchronize(); t0 = time.time()
        out = llm.generate([p], sp, use_tqdm=False)
        torch.cuda.synchronize()
        return time.time() - t0, len(out[0].outputs[0].token_ids)

    rows = []  # one row per (prompt, K) + baseline rows (K=0)

    # ---- baseline (no spec) ----
    print("=== baseline ===", flush=True)
    base = LLM(model=TARGET, gpu_memory_utilization=0.6, max_model_len=2048,
               enforce_eager=False)
    base.generate(flat[0][2:3][0:1] if False else [flat[0][2]], sp, use_tqdm=False)  # warmup
    for w, i, p in flat:
        dt, gen = timed(base, p)
        rows.append({"workload": w, "idx": i, "K": 0, "latency": round(dt, 4),
                     "gen": gen, "accept_len": None})
    del base; torch.cuda.empty_cache()

    # ---- EAGLE-3 at each fixed K ----
    for K in KS:
        print(f"=== eagle3 K={K} ===", flush=True)
        try:
            llm = LLM(model=TARGET,
                      speculative_config={"method": "eagle3", "model": EAGLE,
                                          "num_speculative_tokens": K},
                      gpu_memory_utilization=0.6, max_model_len=2048,
                      enforce_eager=False, disable_log_stats=False)
        except Exception as e:
            print(f"  [init FAILED K={K}] {repr(e)[:200]}", flush=True)
            continue
        llm.generate([flat[0][2]], sp, use_tqdm=False)  # warmup
        prev_a = counter(llm, ACC) or 0.0
        prev_s = counter(llm, STEPS) or 0.0
        for w, i, p in flat:
            dt, gen = timed(llm, p)
            a = counter(llm, ACC); s = counter(llm, STEPS)
            da = (a - prev_a) if a is not None else None
            ds = (s - prev_s) if s is not None else None
            prev_a = a if a is not None else prev_a
            prev_s = s if s is not None else prev_s
            al = (da / ds + 1.0) if (da is not None and ds) else None
            rows.append({"workload": w, "idx": i, "K": K, "latency": round(dt, 4),
                         "gen": gen, "accept_len": (round(al, 4) if al else None)})
        del llm; torch.cuda.empty_cache()

    os.makedirs("/root/out", exist_ok=True)
    payload = {"target": TARGET, "eagle_head": EAGLE, "ks": KS, "n_per_wl": N, "rows": rows}
    with open("/root/out/eagle3_multik_llama8b.json", "w") as f:
        json.dump(payload, f, indent=2)
    vol.commit()
    print(f"\nSAVED {len(rows)} rows ({len(KS)} K x {len(flat)} prompts + baseline)", flush=True)
    return {"n_rows": len(rows), "ks": KS, "n_prompts": len(flat)}


@app.local_entrypoint()
def main():
    res = run.remote()
    print(f"\n===== captured {res['n_rows']} rows: K={res['ks']} x {res['n_prompts']} prompts =====")
    print("  pull eagle3_multik_llama8b.json from the spec-dec-m5-results volume for offline sim.")
