"""Long-context EAGLE-3 sweep — vLLM with draft-head config patch.

PROBLEM: vLLM 0.23 reads max_position_embeddings from the EAGLE3 head's config.json
and compiles Triton kernels with that bound. All known EAGLE3 heads ship with
max_position_embeddings=2048, causing device-side assertion failures at ctx>2048.

FIX: download the draft head explicitly, patch max_position_embeddings in its
config.json to match ctx_len, then pass the local path to speculative_config.
Triton JIT-compiles fresh on each Modal container, so the patched value takes effect.

Models covered (all three used in the paper):
  llama8b   : Llama-3.1-8B-Instruct  + yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
  qwen14b   : Qwen3-14B              + AngelSlim/Qwen3-14B_eagle3
  deepseek  : DeepSeek-R1-Distill-Llama-8B + yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B

Run commands:
  # One model at a time (run all three in parallel for speed):
  modal run modal_vllm_eagle3_longctx_patched.py --model llama8b  --batch 1  --tag lc_patched_b1
  modal run modal_vllm_eagle3_longctx_patched.py --model llama8b  --batch 16 --tag lc_patched_b16
  modal run modal_vllm_eagle3_longctx_patched.py --model llama8b  --batch 32 --tag lc_patched_b32
  modal run modal_vllm_eagle3_longctx_patched.py --model qwen14b  --batch 1  --tag lc_patched_b1  --gpu-mem 0.88
  modal run modal_vllm_eagle3_longctx_patched.py --model qwen14b  --batch 16 --tag lc_patched_b16 --gpu-mem 0.90
  modal run modal_vllm_eagle3_longctx_patched.py --model deepseek --batch 1  --tag lc_patched_b1
  modal run modal_vllm_eagle3_longctx_patched.py --model deepseek --batch 32 --tag lc_patched_b32

Outputs: eagle3_longctx_patched/{model}_{batch}_{tag}.json
Download: modal volume get spec-dec-m5-results eagle3_longctx_patched/ results/eagle3_longctx_patched/
"""
import os, json, time, shutil
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets", "huggingface_hub")
    .env({
        "VLLM_USE_V1": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    })
)
app = modal.App("vllm-eagle3-longctx-patched")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

# ── model registry ─────────────────────────────────────────────────────────── #
# (target, eagle_head, short_name)
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
N       = 8       # prompts per workload
MAXTOK  = 256     # longer outputs for realistic long-ctx measurement


@app.function(image=image, gpu="H100", timeout=14400,
              volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(target, eagle, ks, n, maxtok, gpu_mem, tag, batch, ctx_len, model_key):
    import torch
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    import vllm; print("vllm", vllm.__version__)
    print(f">>> [{tag}] model={model_key} target={target}")
    print(f"    eagle={eagle}  K={ks}  batch={batch}  ctx_len={ctx_len}")

    # ── patch draft head config ──────────────────────────────────────────── #
    print("\n--- patching draft head config ---")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        draft_cache = snapshot_download(eagle, token=hf_token)
    except Exception as e:
        print(f"  [FATAL] could not download {eagle}: {e}")
        raise

    draft_local = "/tmp/eagle3_head_patched"
    if os.path.exists(draft_local):
        shutil.rmtree(draft_local)
    shutil.copytree(draft_cache, draft_local)

    cfg_path = os.path.join(draft_local, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    orig = cfg.get("max_position_embeddings", "not set")
    print(f"  original max_position_embeddings = {orig}")
    cfg["max_position_embeddings"] = ctx_len
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  patched  max_position_embeddings = {ctx_len}")

    # ── workloads ───────────────────────────────────────────────────────── #
    def load_quality(n):
        try:
            ds = load_dataset("emozilla/quality", split="validation")
            prompts = []
            for ex in ds:
                article = ex.get("article", "")
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

    sp = SamplingParams(temperature=0.0, max_tokens=maxtok)

    def accepted_per_step(llm):
        try:
            metrics = llm.llm_engine.get_metrics()
            d = {m.name: getattr(m, "value", getattr(m, "sum", None)) for m in metrics}
            acc   = d.get("vllm:spec_decode_num_accepted_tokens")
            steps = d.get("vllm:spec_decode_num_drafts")
            if acc is not None and steps:
                return acc / steps + 1.0
        except Exception as e:
            print(f"  [metrics: {repr(e)[:120]}]")
        return float("nan")

    def bench(llm, prompts, read_accept=False):
        llm.generate(prompts[:max(1, min(batch, len(prompts)))], sp, use_tqdm=False)
        torch.cuda.synchronize()
        t0 = time.time(); ntok = 0
        for i in range(0, len(prompts), batch):
            outs = llm.generate(prompts[i:i+batch], sp, use_tqdm=False)
            for o in outs:
                ntok += len(o.outputs[0].token_ids)
        torch.cuda.synchronize()
        dt  = time.time() - t0
        aps = accepted_per_step(llm) if read_accept else float("nan")
        return ntok, dt, ntok / dt, aps

    results = []

    def add(method, k, w, ntok, dt, tps, base_tps, aps):
        row = {
            "method": method, "K": k, "workload": w,
            "model": model_key, "batch": batch, "ctx_len": ctx_len,
            "tokens": ntok, "secs": round(dt, 3),
            "tok_per_s": round(tps, 2),
            "net_speedup": round(tps / base_tps, 3) if base_tps else 0,
            "accepted_tokens_per_step": (round(aps, 4) if aps == aps else None),
        }
        results.append(row); print(row)

    # ── baseline ─────────────────────────────────────────────────────────── #
    print("\n=== baseline ===")
    llm_kwargs = dict(
        gpu_memory_utilization=gpu_mem,
        max_model_len=ctx_len,
        enforce_eager=False,
        disable_log_stats=False,
    )
    if batch > 1:
        llm_kwargs["max_num_seqs"] = batch
    base = LLM(model=target, **llm_kwargs)
    base_tps = {}
    for w, prompts in workloads.items():
        ntok, dt, tps, _ = bench(base, prompts)
        base_tps[w] = tps
        add("baseline", 0, w, ntok, dt, tps, tps, float("nan"))
    del base; torch.cuda.empty_cache()

    # ── EAGLE-3 fixed-K sweep (patched draft head) ────────────────────────── #
    for k in ks:
        cfg_spec = {"method": "eagle3", "model": draft_local,
                    "num_speculative_tokens": k}
        print(f"\n=== eagle3_patched K={k} ===")
        try:
            llm = LLM(model=target, speculative_config=cfg_spec, **llm_kwargs)
        except Exception as e:
            print(f"  [init FAILED] K={k}: {repr(e)[:400]}")
            continue
        for w, prompts in workloads.items():
            try:
                ntok, dt, tps, aps = bench(llm, prompts, read_accept=True)
                add("eagle3_patched", k, w, ntok, dt, tps, base_tps[w], aps)
            except Exception as e:
                print(f"  [bench FAILED] K={k} {w}: {repr(e)[:200]}")
        del llm; torch.cuda.empty_cache()

    # ── save ─────────────────────────────────────────────────────────────── #
    out_dir  = "/root/out/eagle3_longctx_patched"
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
    print(f"\n===== vLLM-patched long-ctx [{model}] B={batch} ctx={ctx_len} =====")
    best = {}
    for r in res["results"]:
        if r["method"] == "baseline":
            continue
        w = r["workload"]
        if w not in best or r["net_speedup"] > best[w]["net_speedup"]:
            best[w] = r
    if best:
        for w, r in sorted(best.items()):
            aps = r["accepted_tokens_per_step"]
            print(f"  {w:16s} best K={r['K']}  speedup={r['net_speedup']}x"
                  f"  acc/step={aps}  ({r['tok_per_s']:.1f} tok/s)")
    else:
        print("  No eagle3 results — patch may not have worked; check logs above.")
    print(f"\n  saved: {res.get('out_path', '?')}")
