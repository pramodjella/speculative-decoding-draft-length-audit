"""Validate + benchmark the custom_class adaptive-length ngram proposer in vLLM.

Llama-3.1-8B target (ungated mirror), A100, batch=1, greedy. Compares:
  baseline (no spec) | native ngram fixed K | custom adaptive proposer (history/ucb)
Net speedup = spec tok/s / no-spec tok/s, all inside vLLM's optimized engine.
"""
import os, time, json
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets")
    .env({
        "VLLM_USE_V1": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "PYTHONPATH": "/root/project/src",   # so vLLM's worker can import the proposer
    })
    .add_local_dir("src", "/root/project/src")
)
app = modal.App("vllm-adaptive-proposer")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

TARGET = os.environ.get("VA_TARGET", "NousResearch/Meta-Llama-3.1-8B-Instruct")
N = int(os.environ.get("VA_N", "16"))
MAXTOK = int(os.environ.get("VA_MAXTOK", "96"))
PROPOSER = "vllm_adaptive_proposer.AdaptiveNgramProposer"


@app.function(image=image, gpu="A100", timeout=3600, volumes={"/root/out": vol})
def run():
    import torch
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    import vllm; print("vllm", vllm.__version__, "torch", torch.__version__, flush=True)

    def take(x, n): return list(x)[:n]
    workloads = {
        "humaneval": take(load_dataset("openai/openai_humaneval", split="test")["prompt"], N),
        "gsm8k": take(load_dataset("openai/gsm8k", "main", split="test")["question"], N),
    }
    sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK)

    def bench(llm, prompts):
        llm.generate(prompts[:1], sp, use_tqdm=False)   # warmup
        torch.cuda.synchronize(); t0 = time.time(); ntok = 0
        for p in prompts:
            ntok += len(llm.generate([p], sp, use_tqdm=False)[0].outputs[0].token_ids)
        torch.cuda.synchronize()
        return ntok / (time.time() - t0)

    results = []
    def add(name, w, tps, base):
        r = {"method": name, "workload": w, "tok_per_s": round(tps, 2),
             "net_speedup": round(tps / base, 3) if base else None}
        results.append(r); print(r, flush=True)

    # ---- baseline (no spec) ----
    print("\n=== baseline ===", flush=True)
    base = LLM(model=TARGET, gpu_memory_utilization=0.85, max_model_len=2048)
    base_tps = {}
    for w, prompts in workloads.items():
        base_tps[w] = bench(base, prompts); add("baseline", w, base_tps[w], base_tps[w])
    del base; torch.cuda.empty_cache()

    def run_cfg(label, spec_cfg, env=None):
        if env:
            os.environ.update(env)
        print(f"\n=== {label}  cfg={spec_cfg}  env={env} ===", flush=True)
        try:
            llm = LLM(model=TARGET, speculative_config=spec_cfg,
                      gpu_memory_utilization=0.85, max_model_len=2048)
        except Exception as e:
            print(f"  [INIT FAILED] {label}: {repr(e)[:400]}", flush=True)
            results.append({"method": label, "error": repr(e)[:400]}); return
        for w, prompts in workloads.items():
            try:
                add(label, w, bench(llm, prompts), base_tps[w])
            except Exception as e:
                print(f"  [BENCH FAILED] {label} {w}: {repr(e)[:200]}", flush=True)
        del llm; torch.cuda.empty_cache()

    # ---- native ngram fixed K (reference) ----
    for k in (2, 4):
        run_cfg(f"ngram_fixed{k}",
                {"method": "ngram", "num_speculative_tokens": k,
                 "prompt_lookup_max": 3, "prompt_lookup_min": 1})

    # ---- custom adaptive proposer ----
    for ctrl in ("fixed", "history", "ucb"):
        run_cfg(f"adaptive_{ctrl}",
                {"model": PROPOSER, "num_speculative_tokens": 4},
                env={"VS_CTRL": ctrl, "VS_NGRAM_N": "2"})

    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/vllm_adaptive_results.json", "w") as f:
        json.dump({"target": TARGET, "results": results}, f, indent=2)
    vol.commit()
    return results


@app.local_entrypoint()
def main():
    res = run.remote()
    print("\n===== vLLM adaptive proposer =====")
    for r in res:
        if "error" in r:
            print(f"  {r['method']:16s} INIT FAILED: {r['error'][:120]}")
        else:
            print(f"  {r['method']:16s} {r['workload']:10s} {r['tok_per_s']:>7} tok/s  {r.get('net_speedup')}x")
