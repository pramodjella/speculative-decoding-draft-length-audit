"""ACTIVE — EAGLE-3 fixed-K headroom sweep on H100. K in {3,4,5} (vLLM-blog range).

Run command (single line):
    PYTHONIOENCODING=utf-8 modal run modal_vllm_eagle3.py
Batch sweep (B=16 added to show the inflection point between B=8 and B=32):
    modal run modal_vllm_eagle3.py --batch 8  --tag b8   --gpu-mem 0.85
    modal run modal_vllm_eagle3.py --batch 16 --tag b16  --gpu-mem 0.87  --ks 1,2,3,4
    modal run modal_vllm_eagle3.py --batch 32 --tag b32  --gpu-mem 0.9
    modal run modal_vllm_eagle3.py --batch 64 --tag b64  --gpu-mem 0.9
  Outputs land in volume as eagle3_batch/vllm_eagle3_b{N}_{tag}.json — separate
  from the B=1 deliverable trace (eagle3_multik_llama8b.json), so analyze_eagle3_8b.py
  keeps reading what it always read.

EAGLE-3 headroom experiment (the "cheaper EAGLE-style draft" fork from Yash's
feedback). This is the higher-headroom regime that the local 8 GB box cannot host
(EAGLE-3 needs a 7B+ target + vLLM/Linux). It is the cloud counterpart to the
local higher-gap sweep in run_full_pipeline.py.

Why EAGLE-3 here:
  - The draft is a tiny trained head on the target's own features, so draft cost r
    is very low (r ~ 0.1-0.3) -> exactly the sweet spot the r x maxK analysis
    (analyze_r_sweep.py) flagged as having the most adaptive-K headroom.
  - It still exposes per-step draft logits, so the contextual controllers
    (entropy / margin / LinUCB) remain usable in the eventual Stage-2 in-engine run.
  - Published EAGLE-3 speedups are in the 2-3x range, the regime where a
    draft-length controller has real room to win -- unlike the tight 1.5-3B pairs.

What it measures (batch=1, greedy), for a fixed-K sweep:
  1. net wall-clock speedup vs no-spec (same vLLM CUDA-graphed engine) -- secondary.
  2. accepted tokens per target step (HEADLINE) -- derived from vLLM's spec-decode
     acceptance counters; deterministic, the quantity the controller optimizes.

Mirrors modal_vllm_specbench.py (Stage 1). Stage 2 (in-engine adaptive K via
vllm.v1.spec_decode.custom_class_proposer) remains future work.
"""
import os, json, time
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets")
    .env({
        "VLLM_USE_V1": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        # debian_slim has no nvcc -> FlashInfer sampler JIT crashes. We decode
        # greedily (temp=0) so force the prebuilt FlashAttention path instead.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    })
)
app = modal.App("vllm-eagle3-headroom")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

# NOTE: module-level constants re-evaluate INSIDE the Modal container from its
# (empty) env, so local `export` does NOT propagate -- edit these defaults to
# change the run (same gotcha as modal_vllm_specbench.py).
TARGET = os.environ.get("VE_TARGET", "meta-llama/Llama-3.1-8B-Instruct")
# Verified EAGLE-3 head for Llama-3.1-8B (yuhuili = original EAGLE authors).
# Alt verified head: "ruipeterpan/Qwen2.5-14B-Instruct_EAGLE3_UltraChat" (target
# Qwen2.5-14B) if a Qwen target is preferred.
EAGLE = os.environ.get("VE_EAGLE", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
N = int(os.environ.get("VE_NPROMPTS", "4"))      # small: fast headroom probe
MAXTOK = int(os.environ.get("VE_MAXTOK", "64"))  # shorter gens -> faster bench
# K in {3,4,5}: the num_speculative_tokens range used in vLLM EAGLE-3 examples
# and EAGLE papers. EAGLE may cap K by head depth; unsupported K fail-init + skip.
KS = [int(x) for x in os.environ.get("VE_KS", "3,4,5").split(",")]


@app.function(image=image, gpu="H100", timeout=7200,
              volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])  # Llama is gated
def run(target, eagle, ks, n, maxtok, gpu_mem, tag, batch):
    import torch
    from vllm import LLM, SamplingParams
    from datasets import load_dataset

    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    import vllm; print("vllm", vllm.__version__)
    print(f">>> [{tag}] target={target} eagle={eagle} K={ks} batch={batch} gpu_mem={gpu_mem}")
    if batch > n:
        print(f"  [warn] batch={batch} > n={n} per workload — last group will be"
              f" undersized; pass --n {batch} (or more) for a full sweep")

    # ---- workloads (real) ----
    def take(x, m): return list(x)[:m]
    he = take(load_dataset("openai/openai_humaneval", split="test")["prompt"], n)
    gsm = take(load_dataset("openai/gsm8k", "main", split="test")["question"], n)
    try:
        mt = take(load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
                  .map(lambda x: {"p": x["prompt"][0]})["p"], n)
    except Exception:
        mt = take(["Tell me about " + t for t in ["the ocean", "history", "music",
                   "space", "cooking"]] * 20, n)
    workloads = {"humaneval": he, "gsm8k": gsm, "mt_bench": mt}

    sp = SamplingParams(temperature=0.0, max_tokens=maxtok)

    def accepted_per_step_from_metrics(llm):
        """Headline metric from vLLM spec-decode counters.

        accepted_tokens_per_step = accepted_draft_tokens / num_spec_steps + 1
        (each step emits the accepted draft run plus the bonus token). The exact
        metrics API varies by vLLM version, so this is defensive; returns NaN if
        the counters can't be read (then fall back to wall-clock only).
        """
        try:
            metrics = llm.llm_engine.get_metrics()  # V1
            d = {}
            for m in metrics:
                d[m.name] = getattr(m, "value", getattr(m, "sum", None))
            # vLLM 0.23 counter names have NO `_total` suffix (confirmed via recon)
            acc = d.get("vllm:spec_decode_num_accepted_tokens")
            steps = d.get("vllm:spec_decode_num_drafts")
            if acc is not None and steps:
                return acc / steps + 1.0
        except Exception as e:
            print(f"  [metrics read failed: {repr(e)[:160]}]")
        return float("nan")

    def bench(llm, prompts, read_accept=False):
        # warmup: same shape as the real call so the scheduler sees the same batch size
        llm.generate(prompts[:max(1, min(batch, len(prompts)))], sp, use_tqdm=False)
        torch.cuda.synchronize()
        t0 = time.time(); ntok = 0
        # Submit prompts in groups of `batch` so vLLM's scheduler forms a real batched
        # decode (verification cost ~ B*K — the headroom Yash is pointing at).
        for i in range(0, len(prompts), batch):
            grp = prompts[i : i + batch]
            outs = llm.generate(grp, sp, use_tqdm=False)
            for o in outs:
                ntok += len(o.outputs[0].token_ids)
        torch.cuda.synchronize()
        dt = time.time() - t0
        aps = accepted_per_step_from_metrics(llm) if read_accept else float("nan")
        return ntok, dt, ntok / dt, aps

    results = []

    def add(method, k, w, ntok, dt, tps, base_tps, aps):
        spd = tps / base_tps if base_tps else 0
        row = {"method": method, "K": k, "workload": w, "batch": batch,
               "tokens": ntok, "secs": round(dt, 3), "tok_per_s": round(tps, 2),
               "net_speedup": round(spd, 3),
               "accepted_tokens_per_step": (round(aps, 4) if aps == aps else None)}
        results.append(row); print(row)

    # ---- baseline (no spec) ----
    print("\n=== baseline (no speculative decoding) ===")
    # Only pin max_num_seqs in the batch-sweep path so B=1 behaves identically to
    # the original deliverable run (vLLM default ~256).
    llm_kwargs = dict(gpu_memory_utilization=gpu_mem, max_model_len=2048,
                      enforce_eager=False, disable_log_stats=False)
    if batch > 1:
        llm_kwargs["max_num_seqs"] = batch
    base = LLM(model=target, **llm_kwargs)
    base_tps = {}
    for w, prompts in workloads.items():
        ntok, dt, tps, _ = bench(base, prompts)
        base_tps[w] = tps
        add("baseline", 0, w, ntok, dt, tps, tps, float("nan"))
    del base
    torch.cuda.empty_cache()

    # ---- EAGLE-3 fixed-K sweep ----
    for k in ks:
        cfg = {"method": "eagle3", "model": eagle, "num_speculative_tokens": k}
        print(f"\n=== method=eagle3 K={k} ===")
        try:
            llm = LLM(model=target, speculative_config=cfg, **llm_kwargs)
        except Exception as e:
            print(f"  [init FAILED] eagle3 K={k}: {repr(e)[:300]}")
            continue
        for w, prompts in workloads.items():
            try:
                ntok, dt, tps, aps = bench(llm, prompts, read_accept=True)
                add("eagle3", k, w, ntok, dt, tps, base_tps[w], aps)
            except Exception as e:
                print(f"  [bench FAILED] eagle3 K={k} {w}: {repr(e)[:200]}")
        del llm
        torch.cuda.empty_cache()

    # Namespace under eagle3_batch/ so the batch sweep can't overwrite the B=1
    # deliverable trace (vllm_eagle3_*.json at the root of the volume) that
    # analyze_eagle3_8b.py already reads.
    if batch > 1:
        out_dir = "/root/out/eagle3_batch"
        out_path = f"{out_dir}/vllm_eagle3_b{batch}_{tag}.json"
    else:
        out_dir = "/root/out"
        out_path = f"{out_dir}/vllm_eagle3_{tag}.json"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"target": target, "eagle_head": eagle, "batch": batch,
                   "results": results}, f, indent=2)
    vol.commit()
    return {"results": results, "out_path": out_path}


@app.local_entrypoint()
def main(target: str = TARGET, eagle: str = EAGLE, ks: str = "3,4,5",
         n: int = 4, maxtok: int = 64, gpu_mem: float = 0.6, tag: str = "run",
         batch: int = 1):
    ks_list = [int(x) for x in ks.split(",")]
    res = run.remote(target, eagle, ks_list, n, maxtok, gpu_mem, tag, batch)
    print(f"\n===== EAGLE-3 headroom summary [{tag}] batch={batch} =====")
    print(f"  saved {res.get('out_path', '?')}")
    best = {}
    for r in res["results"]:
        if r["method"] == "baseline":
            continue
        w = r["workload"]
        if w not in best or r["net_speedup"] > best[w]["net_speedup"]:
            best[w] = r
    for w, r in sorted(best.items()):
        aps = r["accepted_tokens_per_step"]
        print(f"  {w:10s} best K={r['K']}  speedup={r['net_speedup']}x  "
              f"acc/step={aps}  ({r['tok_per_s']} tok/s)")
