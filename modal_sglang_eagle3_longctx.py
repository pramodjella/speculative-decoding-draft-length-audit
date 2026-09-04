"""Long-context EAGLE-3 sweep — SGLang (reference EAGLE3 implementation).

SGLang is the EAGLE3 paper's own inference stack and handles max_model_len correctly
(inherits from the target model config, not the draft head). It avoids the vLLM 0.23
Triton kernel bound issue entirely.

Models covered (all three used in the paper):
  llama8b   : Llama-3.1-8B-Instruct  + yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
  qwen14b   : Qwen3-14B              + AngelSlim/Qwen3-14B_eagle3
  deepseek  : DeepSeek-R1-Distill-Llama-8B + yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B

Approach: SGLang offline Engine API.  Each K value reinitialises the engine (SGLang
sets speculative_num_draft_tokens at engine init time). Engine.generate() returns
output token counts directly for throughput measurement.

Run commands:
  modal run modal_sglang_eagle3_longctx.py --model llama8b  --batch 1  --tag sgl_b1
  modal run modal_sglang_eagle3_longctx.py --model llama8b  --batch 16 --tag sgl_b16
  modal run modal_sglang_eagle3_longctx.py --model llama8b  --batch 32 --tag sgl_b32
  modal run modal_sglang_eagle3_longctx.py --model qwen14b  --batch 1  --tag sgl_b1  --gpu-mem 0.88
  modal run modal_sglang_eagle3_longctx.py --model qwen14b  --batch 16 --tag sgl_b16 --gpu-mem 0.90
  modal run modal_sglang_eagle3_longctx.py --model deepseek --batch 1  --tag sgl_b1
  modal run modal_sglang_eagle3_longctx.py --model deepseek --batch 32 --tag sgl_b32

Outputs: eagle3_longctx_sglang/{model}_b{batch}_{tag}.json
Download: modal volume get spec-dec-m5-results eagle3_longctx_sglang/ results/eagle3_longctx_sglang/
"""
import os, json, time
import modal

# SGLang + FlashInfer for H100 (SM90, CUDA 12.4).
# Must use cuda:devel base because deep_gemm (a sglang dep) needs nvcc at runtime.
# debian_slim only has CUDA runtime; devel includes nvcc.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        "torch>=2.4",
        "sglang[all]>=0.4.0",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.4/",
    )
    .pip_install("datasets", "huggingface_hub")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "TOKENIZERS_PARALLELISM": "false",
    })
)
app = modal.App("sglang-eagle3-longctx")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

# ── model registry ─────────────────────────────────────────────────────────── #
MODEL_REGISTRY = {
    "llama8b": (
        "meta-llama/Llama-3.1-8B-Instruct",
        "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    ),
    "qwen14b": (
        "Qwen/Qwen3-14B",
        "AngelSlim/Qwen3-14B_eagle3",
    ),
    "deepseek": (
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
    ),
}

CTX_LEN = 8192
KS      = [1, 2, 3, 4]
N       = 8
MAXTOK  = 256


@app.function(image=image, gpu="H100", timeout=14400,
              volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(target, eagle, ks, n, maxtok, gpu_mem, tag, batch, ctx_len, model_key):
    import torch
    import sglang as sgl
    from datasets import load_dataset

    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    print("sglang", sgl.__version__)
    print(f">>> [{tag}] model={model_key} target={target}")
    print(f"    eagle={eagle}  K={ks}  batch={batch}  ctx_len={ctx_len}")

    # ── workloads ────────────────────────────────────────────────────────── #
    def load_quality(n):
        try:
            ds = load_dataset("emozilla/quality", split="validation")
            prompts = []
            for ex in ds:
                article  = ex.get("article", "")
                question = ex.get("question", "What is the main point of this passage?")
                max_chars = (ctx_len - maxtok - 400) * 4
                article = article[:max_chars]
                prompts.append(
                    f"Read the following passage carefully:\n\n{article}\n\n"
                    f"Question: {question}\nAnswer:"
                )
                if len(prompts) >= n:
                    break
            return prompts[:n]
        except Exception as e:
            print(f"  [quality load failed: {e}]")
            return None

    def synthetic_long(n):
        para = (
            "The Inflation Reduction Act of 2022 is a landmark piece of legislation "
            "that reformed the US tax code, healthcare subsidies, and energy policy. "
            "It allocates substantial funding for clean energy tax credits and deficit "
            "reduction measures. Key provisions include drug price negotiations for "
            "Medicare, expanded ACA subsidies, and major investments in domestic "
            "semiconductor production. The act represents the largest climate "
            "investment in US history. "
        )
        target_chars = (ctx_len - maxtok - 200) * 4
        long_text = (para * ((target_chars // len(para)) + 1))[:target_chars]
        return [f"Summarize the following document:\n\n{long_text}\n\nSummary:"] * n

    qual = load_quality(n)
    workloads = {}
    if qual and len(qual) >= 2:
        workloads["quality"] = qual
        print(f"  quality: {len(qual)} prompts, avg {sum(len(p) for p in qual)//len(qual)} chars")
    else:
        workloads["synthetic_long"] = synthetic_long(n)
        print(f"  synthetic: {n} prompts (fallback)")

    sp = {"temperature": 0.0, "max_new_tokens": maxtok}
    results = []

    def add(method, k, w, ntok, dt, base_tps, aps=None):
        tps = ntok / dt if dt > 0 else 0
        row = {
            "method": method, "K": k, "workload": w,
            "model": model_key, "batch": batch, "ctx_len": ctx_len,
            "tokens": ntok, "secs": round(dt, 3),
            "tok_per_s": round(tps, 2),
            "net_speedup": round(tps / base_tps, 3) if base_tps else 0,
            "accepted_tokens_per_step": aps,
        }
        results.append(row); print(row)

    def make_engine(spec_k=None):
        """Create SGLang engine, with or without EAGLE3 speculation."""
        kwargs = dict(
            model_path=target,
            context_length=ctx_len,
            max_running_requests=max(batch, 1),
            mem_fraction_static=gpu_mem,
            dtype="bfloat16",
            log_level="warning",
        )
        if spec_k is not None:
            kwargs.update(
                speculative_algorithm="EAGLE3",
                speculative_draft_model_path=eagle,
                speculative_num_draft_tokens=spec_k,
            )
        return sgl.Engine(**kwargs)

    def bench_engine(engine, prompts):
        """Warmup + timed batch generation. Returns (ntok, dt, tok/s)."""
        # Warmup with one prompt
        engine.generate(prompts[:1], sampling_params=sp)
        torch.cuda.synchronize()

        t0 = time.time(); ntok = 0
        for i in range(0, len(prompts), batch):
            grp  = prompts[i:i+batch]
            outs = engine.generate(grp, sampling_params=sp)
            # SGLang returns list of dicts with 'token_ids' or 'text'
            for o in outs:
                tids = o.get("token_ids") or o.get("output_ids") or []
                # fallback: estimate from text length
                if not tids and "text" in o:
                    tids = o["text"].split()
                ntok += len(tids)
        torch.cuda.synchronize()
        dt = time.time() - t0
        return ntok, dt, ntok / dt

    # ── baseline (no spec) ──────────────────────────────────────────────── #
    print("\n=== baseline ===")
    base_tps = {}
    try:
        engine = make_engine(spec_k=None)
        for w, prompts in workloads.items():
            ntok, dt, tps = bench_engine(engine, prompts)
            base_tps[w] = tps
            add("baseline", 0, w, ntok, dt, tps)
        engine.shutdown()
        del engine
    except Exception as e:
        print(f"  [baseline FAILED]: {repr(e)[:400]}")
        # Can't compute speedup without baseline — abort
        raise

    # ── EAGLE-3 fixed-K sweep ─────────────────────────────────────────────── #
    for k in ks:
        print(f"\n=== sglang eagle3 K={k} ===")
        try:
            engine = make_engine(spec_k=k)
        except Exception as e:
            print(f"  [engine init FAILED] K={k}: {repr(e)[:400]}")
            continue
        for w, prompts in workloads.items():
            try:
                ntok, dt, tps = bench_engine(engine, prompts)
                add("eagle3_sglang", k, w, ntok, dt, base_tps.get(w, 0))
            except Exception as e:
                print(f"  [bench FAILED] K={k} {w}: {repr(e)[:200]}")
        try:
            engine.shutdown()
        except Exception:
            pass
        del engine

    # ── save ─────────────────────────────────────────────────────────────── #
    out_dir  = "/root/out/eagle3_longctx_sglang"
    out_path = f"{out_dir}/{model_key}_b{batch}_{tag}.json"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"target": target, "eagle_head": eagle, "model": model_key,
                   "batch": batch, "ctx_len": ctx_len, "results": results}, f, indent=2)
    vol.commit()
    print(f"\n>>> wrote {out_path}")
    return {"results": results, "out_path": out_path}


@app.local_entrypoint()
def main(model: str = "llama8b", ks: str = "1,2,3,4",
         n: int = N, maxtok: int = MAXTOK, gpu_mem: float = 0.80,
         tag: str = "run", batch: int = 1, ctx_len: int = CTX_LEN):
    if model not in MODEL_REGISTRY:
        print(f"Unknown model '{model}'. Choose from: {list(MODEL_REGISTRY)}")
        return
    target, eagle = MODEL_REGISTRY[model]
    ks_list = [int(x) for x in ks.split(",")]
    print(f"Running: model={model}  batch={batch}  ctx_len={ctx_len}  K={ks_list}")
    res = run.remote(target, eagle, ks_list, n, maxtok, gpu_mem,
                     tag, batch, ctx_len, model)
    print(f"\n===== SGLang long-ctx EAGLE3 [{model}] B={batch} ctx={ctx_len} =====")
    best = {}
    for r in res["results"]:
        if r["method"] == "baseline":
            continue
        w = r["workload"]
        if w not in best or r["net_speedup"] > best[w]["net_speedup"]:
            best[w] = r
    if best:
        for w, r in sorted(best.items()):
            print(f"  {w:16s} best K={r['K']}  speedup={r['net_speedup']}x"
                  f"  ({r['tok_per_s']:.1f} tok/s)")
    else:
        print("  No eagle3 results — check logs for errors.")
    print(f"\n  saved: {res.get('out_path', '?')}")
