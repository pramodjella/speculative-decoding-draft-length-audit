"""Long-context batch sweep for EAGLE-3 / Llama-3.1-8B.

Yash's "most valuable experiment left": does the K->1 batch collapse
(observed at short context, max_model_len=2048) persist at long context?

MagicDec (2408.11049) explains that at short context + large batch the bottleneck
is compute (not KV bandwidth), making spec decoding degrade with larger K. But at
LONG context, the KV cache is huge -> the bottleneck reverts to memory bandwidth
even at large batch -> spec decoding should stay beneficial with K > 1.

This script sweeps (context_len=8192, B in {1,8,16,32,64}) to characterize whether:
  (a) K->1 collapse holds at 8K context too  (negative: same result, stronger claim)
  (b) K > 1 stays optimal at 8K context      (positive: headroom survives, good news)

Workloads:
  - GovReport (LongBench): ~8K-token policy documents, summarization output ~500 tok
  - SCROLLS/QuALITY: 4K-10K token passages with comprehension questions

Run commands:
  # B=1 (reference — identical setup to short-ctx to isolate context-length effect)
  modal run modal_eagle3_longctx.py --batch 1  --tag longctx_b1

  # Batch sweep (B=16 fills the gap between B=8 and B=32 to show the inflection point)
  modal run modal_eagle3_longctx.py --batch 8  --tag longctx_b8  --gpu-mem 0.85
  modal run modal_eagle3_longctx.py --batch 16 --tag longctx_b16 --gpu-mem 0.87
  modal run modal_eagle3_longctx.py --batch 32 --tag longctx_b32 --gpu-mem 0.90
  modal run modal_eagle3_longctx.py --batch 64 --tag longctx_b64 --gpu-mem 0.92

Outputs land in volume: eagle3_longctx/vllm_eagle3_longctx_b{N}_{tag}.json
(separate namespace from eagle3_batch/ short-ctx results)
"""
import os, json, time
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets")
    .env({
        "VLLM_USE_V1": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    })
)
app = modal.App("vllm-eagle3-longctx")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

TARGET  = os.environ.get("VE_TARGET", "meta-llama/Llama-3.1-8B-Instruct")
EAGLE   = os.environ.get("VE_EAGLE",  "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
# K sweep: 1..5 to find best K per batch (no skipping)
KS      = [int(x) for x in os.environ.get("VE_KS", "1,2,3,4,5").split(",")]
N       = int(os.environ.get("VE_NPROMPTS", "8"))   # per workload
MAXTOK  = int(os.environ.get("VE_MAXTOK", "256"))   # longer outputs for long-ctx
CTX_LEN = int(os.environ.get("VE_CTX", "8192"))     # the key variable vs short-ctx


@app.function(image=image, gpu="H100", timeout=10800,
              volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(target, eagle, ks, n, maxtok, gpu_mem, tag, batch, ctx_len):
    import torch
    from vllm import LLM, SamplingParams
    from datasets import load_dataset

    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    import vllm; print("vllm", vllm.__version__)
    print(f">>> [{tag}] target={target} eagle={eagle} K={ks} "
          f"batch={batch} ctx_len={ctx_len} gpu_mem={gpu_mem}")

    # ── long-context workloads ────────────────────────────────────────────── #

    def take(lst, m): return list(lst)[:m]

    def load_govreport(n):
        """GovReport: long US government policy documents -> summarize.
        Average input ~9K tokens. We take the first n docs from the train set.
        Prompt format mirrors LongBench convention.
        """
        try:
            ds = load_dataset("tau/scrolls", "gov_report", split="validation")
            prompts = []
            for ex in ds:
                doc   = ex.get("input", "") or ex.get("text", "")
                # trim to fit ctx_len-ish (rough char estimate: 4 chars/token)
                max_chars = (ctx_len - maxtok - 200) * 4
                doc = doc[:max_chars]
                prompts.append(
                    f"Summarize the following government report in 3-5 sentences:\n\n"
                    f"{doc}\n\nSummary:"
                )
                if len(prompts) >= n:
                    break
            return prompts[:n]
        except Exception as e:
            print(f"  [govreport load failed: {e}] — falling back to synthetic long prompts")
            return None

    def load_quality(n):
        """QuALITY: 4-10K-token passages + multiple-choice comprehension questions."""
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

    def synthetic_long(n, ctx_len):
        """Fallback: repeat a paragraph until prompt is ~ctx_len tokens long."""
        paragraph = (
            "The Inflation Reduction Act of 2022 represents a landmark piece of "
            "legislation that significantly reformed the United States tax code, "
            "healthcare subsidies, and energy policy. It allocated substantial "
            "funding for clean energy tax credits, drug price negotiations, and "
            "deficit reduction measures over the following decade. "
        )
        target_chars = (ctx_len - maxtok - 100) * 4
        long_text = (paragraph * ((target_chars // len(paragraph)) + 1))[:target_chars]
        return [
            f"Summarize the key provisions of the following policy document:\n\n"
            f"{long_text}\n\nSummary:"
        ] * n

    # try real datasets; fall back to synthetic
    gov = load_govreport(n)
    qual = load_quality(n)
    syn = synthetic_long(n, ctx_len)

    workloads = {}
    if gov is not None and len(gov) >= 2:
        workloads["govreport"] = gov
        print(f"  govreport: {len(gov)} prompts, "
              f"avg len {sum(len(p) for p in gov)//len(gov)} chars")
    if qual is not None and len(qual) >= 2:
        workloads["quality"] = qual
        print(f"  quality:   {len(qual)} prompts, "
              f"avg len {sum(len(p) for p in qual)//len(qual)} chars")
    if not workloads:
        workloads["synthetic_long"] = syn
        print(f"  synthetic: {len(syn)} prompts (fallback)")

    if not workloads:
        raise RuntimeError("No workloads could be loaded")

    sp = SamplingParams(temperature=0.0, max_tokens=maxtok)

    def accepted_per_step(llm):
        try:
            metrics = llm.llm_engine.get_metrics()
            d = {}
            for m in metrics:
                d[m.name] = getattr(m, "value", getattr(m, "sum", None))
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
            grp  = prompts[i: i + batch]
            outs = llm.generate(grp, sp, use_tqdm=False)
            for o in outs:
                ntok += len(o.outputs[0].token_ids)
        torch.cuda.synchronize()
        dt  = time.time() - t0
        aps = accepted_per_step(llm) if read_accept else float("nan")
        return ntok, dt, ntok / dt, aps

    results = []

    def add(method, k, w, ntok, dt, tps, base_tps, aps):
        spd = tps / base_tps if base_tps else 0
        row = {
            "method": method, "K": k, "workload": w, "batch": batch,
            "ctx_len": ctx_len, "tokens": ntok, "secs": round(dt, 3),
            "tok_per_s": round(tps, 2), "net_speedup": round(spd, 3),
            "accepted_tokens_per_step": (round(aps, 4) if aps == aps else None),
        }
        results.append(row); print(row)

    # ── baseline (no spec) ──────────────────────────────────────────────────
    print("\n=== baseline (no speculative decoding) ===")
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

    # ── EAGLE-3 fixed-K sweep ───────────────────────────────────────────────
    for k in ks:
        cfg = {"method": "eagle3", "model": eagle, "num_speculative_tokens": k}
        print(f"\n=== eagle3 K={k} ===")
        try:
            llm = LLM(model=target, speculative_config=cfg, **llm_kwargs)
        except Exception as e:
            print(f"  [init FAILED] K={k}: {repr(e)[:300]}")
            continue
        for w, prompts in workloads.items():
            try:
                ntok, dt, tps, aps = bench(llm, prompts, read_accept=True)
                add("eagle3", k, w, ntok, dt, tps, base_tps[w], aps)
            except Exception as e:
                print(f"  [bench FAILED] K={k} {w}: {repr(e)[:200]}")
        del llm; torch.cuda.empty_cache()

    # ── save ────────────────────────────────────────────────────────────────
    out_dir  = "/root/out/eagle3_longctx"
    out_path = f"{out_dir}/vllm_eagle3_longctx_b{batch}_{tag}.json"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"target": target, "eagle_head": eagle, "batch": batch,
                   "ctx_len": ctx_len, "results": results}, f, indent=2)
    vol.commit()
    print(f"\n>>> wrote {out_path}")
    return {"results": results, "out_path": out_path}


@app.local_entrypoint()
def main(target: str = TARGET, eagle: str = EAGLE, ks: str = "1,2,3,4,5",
         n: int = N, maxtok: int = MAXTOK, gpu_mem: float = 0.7,
         tag: str = "run", batch: int = 1, ctx_len: int = CTX_LEN):
    ks_list = [int(x) for x in ks.split(",")]
    res = run.remote(target, eagle, ks_list, n, maxtok, gpu_mem, tag, batch, ctx_len)
    print(f"\n===== Long-ctx EAGLE-3 [{tag}] B={batch} ctx={ctx_len} =====")
    best = {}
    for r in res["results"]:
        if r["method"] == "baseline":
            continue
        w = r["workload"]
        if w not in best or r["net_speedup"] > best[w]["net_speedup"]:
            best[w] = r
    for w, r in sorted(best.items()):
        aps = r["accepted_tokens_per_step"]
        print(f"  {w:14s} best K={r['K']}  speedup={r['net_speedup']}x  "
              f"acc/step={aps}  ({r['tok_per_s']} tok/s)")
    print(f"\n  saved: {res.get('out_path', '?')}")
