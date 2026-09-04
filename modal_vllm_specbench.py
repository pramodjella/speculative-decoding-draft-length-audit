"""Stage 1: real engine-grade speculative-decoding speedup curve on vLLM (A100).

Measures TRUE net speedup vs no-spec, all inside the same optimized vLLM engine
(baseline and spec share the CUDA-graphed path), so any >1x is a genuine win —
the result the pure-Python loop could not produce.

Sweeps num_speculative_tokens K and tries multiple spec methods, recording which
initialize on this vLLM version so the run yields a real number regardless of
version quirks:
  1. ngram        (no draft model; always supported in V1) -> guaranteed datapoint
  2. draft model  (Qwen2.5-0.5B drafting for 7B)            -> the project's setup
  3. eagle/eagle3 (if a head checkpoint id is provided)

Batch=1 (the regime where per-step adaptive draft length is well-defined).
This Stage-1 result is itself publishable; Stage 2 adds the adaptive controller.
"""
import os, json, time
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets")
    .env({
        "VLLM_USE_V1": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        # debian_slim has no nvcc; FlashInfer JIT-compiles its sampler kernel and
        # crashes. We decode greedily (temp=0) so we don't need FlashInfer at all.
        # Force the prebuilt FlashAttention path (ships compiled, no nvcc needed).
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    })
)
app = modal.App("vllm-specbench-stage1")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

TARGET = os.environ.get("VS_TARGET", "Qwen/Qwen2.5-7B-Instruct")
DRAFT = os.environ.get("VS_DRAFT", "Qwen/Qwen2.5-0.5B-Instruct")
EAGLE = os.environ.get("VS_EAGLE", "")  # optional EAGLE head id
# NOTE: module-level constants re-evaluate INSIDE the Modal container from its
# (empty) env, so local `export VS_*` does NOT propagate — edit these defaults
# to change the run. Trimmed for budget: each (method,K) needs a fresh 7B load.
N = int(os.environ.get("VS_NPROMPTS", "24"))
MAXTOK = int(os.environ.get("VS_MAXTOK", "96"))
KS = [int(x) for x in os.environ.get("VS_KS", "1,2,4,8").split(",")]


@app.function(image=image, gpu="A100", timeout=7200,
              volumes={"/root/out": vol})
def run():
    import torch
    from vllm import LLM, SamplingParams
    from datasets import load_dataset

    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    import vllm; print("vllm", vllm.__version__)

    # ---- workloads (real) ----
    def take(x, n): return list(x)[:n]
    he = take(load_dataset("openai/openai_humaneval", split="test")["prompt"], N)
    gsm = take(load_dataset("openai/gsm8k", "main", split="test")["question"], N)
    try:
        mt = take(load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
                  .map(lambda x: {"p": x["prompt"][0]})["p"], N)
    except Exception:
        mt = take(["Tell me about " + t for t in ["the ocean", "history", "music",
                   "space", "cooking"]] * 20, N)
    workloads = {"humaneval": he, "gsm8k": gsm, "mt_bench": mt}
    all_prompts = {w: p for w, p in workloads.items()}

    sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK)

    def bench(llm, prompts):
        # batch=1 latency, summed over prompts; warmup first prompt
        llm.generate(prompts[:1], sp, use_tqdm=False)
        torch.cuda.synchronize()
        t0 = time.time(); ntok = 0
        for p in prompts:
            out = llm.generate([p], sp, use_tqdm=False)
            ntok += len(out[0].outputs[0].token_ids)
        torch.cuda.synchronize()
        dt = time.time() - t0
        return ntok, dt, ntok / dt

    results = []

    def add(method, k, w, ntok, dt, tps, base_tps):
        spd = tps / base_tps if base_tps else 0
        row = {"method": method, "K": k, "workload": w,
               "tokens": ntok, "secs": round(dt, 3),
               "tok_per_s": round(tps, 2), "net_speedup": round(spd, 3)}
        results.append(row); print(row)

    # ---- baseline (no spec) ----
    print("\n=== baseline (no speculative decoding) ===")
    base = LLM(model=TARGET, gpu_memory_utilization=0.85, max_model_len=2048,
               enforce_eager=False)
    base_tps = {}
    for w, prompts in all_prompts.items():
        ntok, dt, tps = bench(base, prompts)
        base_tps[w] = tps
        add("baseline", 0, w, ntok, dt, tps, tps)
    del base
    torch.cuda.empty_cache()

    # ---- spec configs to try ----
    def spec_cfgs(k):
        cfgs = []
        cfgs.append(("ngram", {"method": "ngram", "num_speculative_tokens": k,
                               "prompt_lookup_max": 3, "prompt_lookup_min": 1}))
        cfgs.append(("draft", {"model": DRAFT, "num_speculative_tokens": k}))
        if EAGLE:
            cfgs.append(("eagle", {"method": "eagle", "model": EAGLE,
                                   "num_speculative_tokens": k}))
        return cfgs

    working = {}  # method -> bool
    for k in KS:
        for method, cfg in spec_cfgs(k):
            if working.get(method) is False:
                continue  # method already proven broken on this version
            print(f"\n=== method={method} K={k} ===")
            try:
                llm = LLM(model=TARGET, speculative_config=cfg,
                          gpu_memory_utilization=0.85, max_model_len=2048,
                          enforce_eager=False)
            except Exception as e:
                print(f"  [init FAILED] method={method}: {repr(e)[:300]}")
                working[method] = False
                continue
            working[method] = True
            for w, prompts in all_prompts.items():
                try:
                    ntok, dt, tps = bench(llm, prompts)
                    add(method, k, w, ntok, dt, tps, base_tps[w])
                except Exception as e:
                    print(f"  [bench FAILED] {method} K={k} {w}: {repr(e)[:200]}")
            del llm
            torch.cuda.empty_cache()

    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/vllm_stage1_results.json", "w") as f:
        json.dump({"target": TARGET, "draft": DRAFT, "results": results,
                   "methods_working": working}, f, indent=2)
    vol.commit()
    return {"results": results, "methods_working": working}


@app.local_entrypoint()
def main():
    res = run.remote()
    print("\n===== vLLM Stage 1 speedup summary =====")
    print("methods working:", res["methods_working"])
    # best speedup per (method, workload)
    best = {}
    for r in res["results"]:
        if r["method"] == "baseline":
            continue
        key = (r["method"], r["workload"])
        if key not in best or r["net_speedup"] > best[key]["net_speedup"]:
            best[key] = r
    for (m, w), r in sorted(best.items()):
        print(f"  {m:8s} {w:10s} best K={r['K']}  net_speedup={r['net_speedup']}x  ({r['tok_per_s']} tok/s)")
