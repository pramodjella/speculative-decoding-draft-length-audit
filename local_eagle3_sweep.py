"""LOCAL SGLang EAGLE-3 fixed-K sweep for Qwen3-1.7B (8GB-VRAM roadmap track).

Captures, per (workload, prompt, K): wall-clock latency, generated tokens, and
EAGLE-3 mean accept length (completion_tokens / spec_verify_ct). K=0 is the
no-speculation baseline.

Emits ONE K per process invocation to a shard file (so each K gets a clean GPU /
engine teardown -- multi-engine in one process is flaky). run_sweep.sh loops the
Ks and merges the shards into results/eagle3_multik_qwen17b.json, which has the
SAME schema as eagle3_multik_llama8b.json and therefore feeds the existing
src/simulate_eagle_controllers.py unchanged.

Engine: SGLang (vLLM 0.23 fails on WSL2 -- UVA-not-available). Chain draft
(eagle_topk=1, num_draft_tokens=K+1) so the prefix property holds, matching the
offline per-step sim assumptions.

Usage (one K):
    python local_eagle3_sweep.py --k 3 --n 20 --out /root/shards/k3.json
"""
import argparse, json, os, time


def load_prompts(n):
    from datasets import load_dataset
    def take(x, m):
        return list(x)[:m]
    wl = {}
    wl["humaneval"] = take(load_dataset("openai/openai_humaneval", split="test")["prompt"], n)
    wl["gsm8k"] = take(load_dataset("openai/gsm8k", "main", split="test")["question"], n)
    try:
        wl["mt_bench"] = take(load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
                              .map(lambda x: {"p": x["prompt"][0]})["p"], n)
    except Exception:
        wl["mt_bench"] = take(["Tell me about " + t for t in
                               ["the ocean", "history", "music", "space", "cooking"]] * 50, n)
    return wl


def build_engine(args):
    import sglang as sgl
    # triton attention backend avoids flashinfer's nvcc/cccl JIT (broken on the
    # mixed-minor pip CUDA-13 stack); cuda graph disabled for the same reason.
    common = dict(model_path=args.target, dtype="float16",
                  mem_fraction_static=args.mem_frac, disable_cuda_graph=True,
                  attention_backend="triton", disable_overlap_schedule=True,
                  max_total_tokens=args.max_total, skip_tokenizer_init=False)
    if args.k == 0:
        return sgl.Engine(**common)                       # no-spec baseline
    return sgl.Engine(speculative_algorithm="EAGLE3",
                      speculative_draft_model_path=args.eagle,
                      speculative_num_steps=args.k,
                      speculative_eagle_topk=1,
                      speculative_num_draft_tokens=args.k + 1,
                      **common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--eagle", default="AngelSlim/Qwen3-1.7B_eagle3")
    ap.add_argument("--maxtok", type=int, default=128)
    ap.add_argument("--mem-frac", dest="mem_frac", type=float, default=0.80)
    ap.add_argument("--max-total", dest="max_total", type=int, default=4096)
    args = ap.parse_args()

    wl = load_prompts(args.n)
    llm = build_engine(args)
    sp = {"temperature": 0.0, "max_new_tokens": args.maxtok}

    rows = []
    # warmup (not recorded) so the first timed gen isn't penalised by lazy init
    try:
        llm.generate(next(iter(wl.values()))[0], sp)
    except Exception as e:
        print("warmup failed:", repr(e)[:200], flush=True)

    for w, prompts in wl.items():
        for idx, p in enumerate(prompts):
            t0 = time.time()
            out = llm.generate(p, sp)
            dt = time.time() - t0
            rec = out[0] if isinstance(out, list) else out
            meta = rec.get("meta_info", {}) or {}
            gen = meta.get("completion_tokens")
            verify = meta.get("spec_verify_ct")
            if args.k == 0:
                accept_len = None
            else:
                accept_len = (round(gen / verify, 4) if verify else None)
            rows.append({"workload": w, "idx": idx, "K": args.k,
                         "latency": round(dt, 4), "gen": gen, "accept_len": accept_len})
            print(f"{w} idx={idx} K={args.k} lat={dt:.3f} gen={gen} "
                  f"verify={verify} accept_len={accept_len}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f)
    print(f"WROTE {len(rows)} rows -> {args.out}", flush=True)
    try:
        llm.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
