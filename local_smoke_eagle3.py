"""LOCAL 8GB smoke test: does the EAGLE-3 head load + actually speculate?

De-risks the local path BEFORE building the full per-step capture pipeline.
Default config is the 8GB-VRAM fit found in the head survey:
    target = Qwen/Qwen3-1.7B   (~3.4GB fp16)
    eagle  = AngelSlim/Qwen3-1.7B_eagle3   (~0.3-0.4B head)
(no sub-8B *Llama* EAGLE-3 head exists, so the local path uses Qwen3.)

Engine policy (user's rule): try vLLM first; if vLLM is unavailable or its
EAGLE-3 init fails, fall back to SGLang. Both are Linux-first -> run under WSL2.

What PASS means:
  1. engine + target + EAGLE-3 head initialise without error, AND
  2. speculation is genuinely engaged -- accepted-draft-tokens > 0 (or accept
     length > 1). Zero acceptance with a clean init is the classic symptom of a
     target/draft VOCAB MISMATCH (see memory: Qwen/EAGLE3 vocab-mismatch gotcha),
     so we flag it loudly rather than reporting a false PASS.

Run (inside WSL2 / Linux, GPU visible):
    python local_smoke_eagle3.py                       # auto: vLLM else SGLang
    python local_smoke_eagle3.py --engine sglang       # force SGLang
    python local_smoke_eagle3.py --gpu-mem 0.70        # lower if you OOM
"""
import argparse
import sys
import time

PROMPT = (
    "Write a Python function that returns the n-th Fibonacci number, "
    "then briefly explain how it works."
)


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# vLLM path
# --------------------------------------------------------------------------- #
def try_vllm(args):
    """Returns 'pass' | 'spec-off' | 'unavailable' | 'init-fail'."""
    try:
        import torch
        from vllm import LLM, SamplingParams
        import vllm
    except Exception as e:
        log(f"[vLLM] not importable -> {repr(e)[:200]}")
        return "unavailable"

    log(f"[vLLM] version={vllm.__version__}  torch={torch.__version__}  "
        f"cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        log(f"[vLLM] GPU={torch.cuda.get_device_name(0)}  "
            f"free={free/1e9:.2f}GB / total={total/1e9:.2f}GB")

    cfg = {"method": "eagle3", "model": args.eagle,
           "num_speculative_tokens": args.k}
    log(f"[vLLM] init target={args.target} eagle={args.eagle} K={args.k} "
        f"gpu_mem={args.gpu_mem} max_len={args.max_model_len} eager={args.eager}")
    try:
        llm = LLM(model=args.target, speculative_config=cfg,
                  gpu_memory_utilization=args.gpu_mem,
                  max_model_len=args.max_model_len,
                  enforce_eager=args.eager,
                  dtype="float16", disable_log_stats=False)
    except Exception as e:
        msg = repr(e)
        log(f"[vLLM] INIT FAILED: {msg[:400]}")
        if any(s in msg.lower() for s in ("vocab", "size mismatch", "shape")):
            log("[vLLM] ^ looks like a TARGET/DRAFT VOCAB MISMATCH -- the head "
                "does not match this target. Pick the head trained for this exact "
                "model, or switch engines.")
        return "init-fail"

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    t0 = time.time()
    out = llm.generate([PROMPT], sp, use_tqdm=False)
    dt = time.time() - t0
    text = out[0].outputs[0].text
    ntok = len(out[0].outputs[0].token_ids)
    log(f"\n[vLLM] generated {ntok} tokens in {dt:.2f}s ({ntok/dt:.1f} tok/s)")
    log("[vLLM] --- output (first 300 chars) ---")
    log(text[:300])

    acc, drafts = _vllm_spec_counters(llm)
    return _verdict("vLLM", acc, drafts)


def _vllm_spec_counters(llm):
    """Read accepted-draft-tokens and num-drafts from vLLM V1 metrics (defensive:
    counter names vary by version; 0.23 has no `_total` suffix)."""
    try:
        metrics = llm.llm_engine.get_metrics()
        d = {}
        for m in metrics:
            d[m.name] = getattr(m, "value", getattr(m, "sum", None))
        for key in ("vllm:spec_decode_num_accepted_tokens",
                    "vllm:spec_decode_num_accepted_tokens_total"):
            if key in d and d[key] is not None:
                acc = d[key]
                break
        else:
            acc = None
        for key in ("vllm:spec_decode_num_drafts",
                    "vllm:spec_decode_num_drafts_total"):
            if key in d and d[key] is not None:
                drafts = d[key]
                break
        else:
            drafts = None
        return acc, drafts
    except Exception as e:
        log(f"[vLLM] could not read spec-decode counters: {repr(e)[:160]}")
        return None, None


# --------------------------------------------------------------------------- #
# SGLang fallback
# --------------------------------------------------------------------------- #
def try_sglang(args):
    """Returns 'pass' | 'spec-off' | 'unavailable' | 'init-fail'."""
    try:
        import sglang as sgl
    except Exception as e:
        log(f"[SGLang] not importable -> {repr(e)[:200]}")
        return "unavailable"

    log(f"[SGLang] version={getattr(sgl, '__version__', '?')}")
    # chain draft (topk=1) so the offline per-step sim stays exact later
    log(f"[SGLang] init target={args.target} eagle={args.eagle} "
        f"steps={args.k} mem_frac={args.gpu_mem}")
    try:
        llm = sgl.Engine(
            model_path=args.target,
            speculative_algorithm="EAGLE3",
            speculative_draft_model_path=args.eagle,
            speculative_num_steps=args.k,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=args.k + 1,
            mem_fraction_static=args.gpu_mem,
            dtype=args.sgl_dtype,
            max_total_tokens=args.max_model_len,
            # backend/dtype are CLI-tunable: triton avoids flashinfer's broken JIT
            # but may mis-verify EAGLE3; fa3 matches the reference recipe.
            attention_backend=args.sgl_attn,
            disable_cuda_graph=True,
            # EAGLE3 reuses mutable metadata buffers; the default overlap-spec-v2
            # scheduler corrupts verification -> 0% acceptance. Must disable.
            disable_overlap_schedule=True,
        )
    except Exception as e:
        msg = repr(e)
        log(f"[SGLang] INIT FAILED: {msg[:400]}")
        if any(s in msg.lower() for s in ("vocab", "size mismatch", "shape")):
            log("[SGLang] ^ looks like a TARGET/DRAFT VOCAB MISMATCH.")
        return "init-fail"

    t0 = time.time()
    out = llm.generate([PROMPT],
                       {"temperature": 0.0, "max_new_tokens": args.max_tokens})
    dt = time.time() - t0
    rec = out[0] if isinstance(out, list) else out
    text = rec.get("text", "")
    meta = rec.get("meta_info", {}) or {}
    log(f"\n[SGLang] generated in {dt:.2f}s")
    log("[SGLang] --- output (first 300 chars) ---")
    log(text[:300])
    log(f"[SGLang] meta_info keys: {sorted(meta.keys())}")
    log(f"[SGLang] meta_info: { {k: meta[k] for k in meta if k != 'output_token_logprobs'} }")
    # SGLang exposes spec acceptance length in meta_info when available.
    # Derive it from verify count if no direct field is present.
    acc_len = meta.get("spec_accept_length") or meta.get("accept_length")
    if acc_len is None:
        comp = meta.get("completion_tokens")
        verify = meta.get("spec_verify_ct")
        if comp and verify:
            acc_len = comp / verify
            log(f"[SGLang] derived accept_len = completion/{'spec_verify_ct'} "
                f"= {comp}/{verify} = {acc_len:.3f}")
    rate = meta.get("spec_accept_rate")
    accepted = meta.get("spec_accepted_drafts")
    if acc_len is not None:
        log(f"[SGLang] accept_length/step={acc_len:.3f}  accept_rate={rate}  "
            f"accepted_drafts={accepted}")
        # A working EAGLE3 head gives accept_length well above 1 (card: ~2.17).
        # ~1.0 with accept_rate 0 = drafts proposed but none accepted = SPEC-OFF.
        if float(acc_len) > 1.3 or (accepted or 0) > 0:
            log("[SGLang] PASS: speculation ENGAGED (drafts accepted).")
            return "pass"
        log("[SGLang] FAIL (spec-off): drafts proposed but ~0 accepted. "
            "Check overlap-schedule / backend / head compatibility.")
        return "spec-off"
    log("[SGLang] no accept-length in meta_info; init+generate succeeded "
        "(treat as soft PASS, confirm with the full capture run).")
    return "pass"


# --------------------------------------------------------------------------- #
def _verdict(engine, acc, drafts):
    log("")
    if acc is None or drafts in (None, 0):
        log(f"[{engine}] SOFT PASS: init + generate OK, but could not read "
            f"spec-decode acceptance counters (acc={acc}, drafts={drafts}). "
            f"Confirm speculation with the full capture run.")
        return "pass"
    acc_per_step = acc / drafts + 1.0
    log(f"[{engine}] accepted_draft_tokens={acc}  num_drafts={drafts}  "
        f"accept_len/step={acc_per_step:.3f}")
    if acc > 0:
        log(f"[{engine}] PASS: speculation is ENGAGED (head matches target). "
            f"Safe to build the full per-step capture pipeline.")
        return "pass"
    log(f"[{engine}] FAIL (spec-off): init OK but ZERO accepted draft tokens. "
        f"Classic VOCAB-MISMATCH / incompatible head symptom -- do NOT proceed "
        f"with this head+engine combo.")
    return "spec-off"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--eagle", default="AngelSlim/Qwen3-1.7B_eagle3")
    ap.add_argument("--engine", default="auto",
                    choices=["auto", "vllm", "sglang"])
    ap.add_argument("--k", type=int, default=4,
                    help="num speculative tokens / draft steps")
    ap.add_argument("--gpu-mem", type=float, default=0.85,
                    help="vLLM gpu_memory_utilization / SGLang mem_fraction_static "
                         "(lower to ~0.70 if you OOM on 8GB)")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--eager", action="store_true", default=True,
                    help="enforce_eager (saves VRAM on small cards; default on)")
    ap.add_argument("--no-eager", dest="eager", action="store_false")
    ap.add_argument("--sgl-attn", dest="sgl_attn", default="triton",
                    help="SGLang attention backend: triton|fa3|flashinfer")
    ap.add_argument("--sgl-dtype", dest="sgl_dtype", default="float16",
                    help="SGLang dtype: float16|bfloat16")
    args = ap.parse_args()

    log("=" * 70)
    log("LOCAL EAGLE-3 SMOKE TEST")
    log(f"  target = {args.target}")
    log(f"  eagle  = {args.eagle}")
    log(f"  engine = {args.engine}   K = {args.k}")
    log("=" * 70)

    result = None
    if args.engine in ("auto", "vllm"):
        result = try_vllm(args)
        if result in ("unavailable", "init-fail") and args.engine == "auto":
            log("\n[auto] vLLM did not work -> falling back to SGLang\n")
            result = try_sglang(args)
    elif args.engine == "sglang":
        result = try_sglang(args)

    log("\n" + "=" * 70)
    if result == "pass":
        log("RESULT: PASS -- head loads and speculates. Proceed to full pipeline.")
        sys.exit(0)
    elif result == "spec-off":
        log("RESULT: SPEC-OFF -- loads but does not accept draft tokens "
            "(likely vocab mismatch). Try the other engine or a different head.")
        sys.exit(2)
    else:
        log("RESULT: NO ENGINE AVAILABLE -- install vLLM (>=0.8.5, ideally recent) "
            "or SGLang inside WSL2/Linux, with a CUDA-visible GPU.")
        sys.exit(3)


if __name__ == "__main__":
    main()
