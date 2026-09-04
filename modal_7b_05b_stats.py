"""Focused stats for Yash on the ACTUAL pair: Qwen2.5-7B target / Qwen2.5-0.5B draft.

Produces exactly what was asked, batch=1, bf16, SDPA, KV-cache, warmup excluded:
  - acceptance rate per draft length K (measured, from our harness)
  - net speedup vs the STANDARD baseline target.generate()  (what papers report)
  - HF built-in assisted-generation speedup (the recommended gut-check)
  - for transparency, speedup vs a lean pure-Python autoregressive baseline too
across humaneval / gsm8k / mt_bench.
"""
import os, sys, time, json
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers==4.44.2", "accelerate", "datasets", "numpy")
    .add_local_dir("src", "/root/project/src")
)
app = modal.App("qwen7b-05b-stats")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

TARGET = "Qwen/Qwen2.5-7B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
N = int(os.environ.get("ST_N", "24"))
MNT = int(os.environ.get("ST_MNT", "96"))
KS = [1, 2, 4, 8]


@app.function(image=image, gpu="A100", timeout=3600, volumes={"/root/out": vol})
def run():
    os.chdir("/root/project")
    sys.path.insert(0, "/root/project/src")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from serve.physical_runner import PhysicalSpeculativeRunner, PythonAutoregressiveBaseline

    def load(mid):
        tok = AutoTokenizer.from_pretrained(mid)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        m = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa").eval()
        return m, tok

    tgt, tok = load(TARGET)
    drf, _ = load(DRAFT)
    dev = next(tgt.parameters()).device
    print(f"target={TARGET} draft={DRAFT}  GPU mem={torch.cuda.memory_allocated()/1e9:.1f}GB")

    def take(x, n): return list(x)[:n]
    workloads = {
        "humaneval": take(load_dataset("openai/openai_humaneval", split="test")["prompt"], N),
        "gsm8k": take(load_dataset("openai/gsm8k", "main", split="test")["question"], N),
        "mt_bench": take(load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
                         .map(lambda x: {"p": x["prompt"][0]})["p"], N),
    }

    lean = PythonAutoregressiveBaseline(tgt, tok)

    @torch.no_grad()
    def gen_plain(p):
        ids = tok.encode(p, return_tensors="pt").to(dev)
        out = tgt.generate(ids, max_new_tokens=MNT, do_sample=False, pad_token_id=tok.eos_token_id)
        return out.shape[1] - ids.shape[1]

    @torch.no_grad()
    def gen_assisted(p, k):
        ids = tok.encode(p, return_tensors="pt").to(dev)
        tgt.generation_config.num_assistant_tokens = k
        tgt.generation_config.num_assistant_tokens_schedule = "constant"
        out = tgt.generate(ids, max_new_tokens=MNT, do_sample=False,
                           assistant_model=drf, pad_token_id=tok.eos_token_id)
        return out.shape[1] - ids.shape[1]

    def timed(fn, prompts):
        fn(prompts[0]); torch.cuda.synchronize()
        t0 = time.time(); ntok = 0
        for p in prompts:
            ntok += fn(p)
        torch.cuda.synchronize()
        return ntok / (time.time() - t0)

    rows = []
    for w, prompts in workloads.items():
        print(f"\n===== {w} (n={len(prompts)}) =====")
        tps_plain = timed(gen_plain, prompts)
        tps_lean = timed(lambda p: lean.generate(p, max_new_tokens=MNT)["generated_tokens"], prompts)
        print(f"plain generate(): {tps_plain:.1f} tok/s | lean python AR: {tps_lean:.1f} tok/s")

        # our harness at each fixed K: tok/s, speedup vs generate(), acceptance rate
        for k in KS:
            runner = PhysicalSpeculativeRunner(tgt, drf, tok, controller=k)
            # warmup + acceptance accumulation
            accs, steps_tot, acc_tot = [], 0, 0
            runner.generate(prompts[0], max_new_tokens=MNT)
            torch.cuda.synchronize(); t0 = time.time(); ntok = 0
            for p in prompts:
                r = runner.generate(p, max_new_tokens=MNT)
                ntok += r["generated_tokens"]; steps_tot += r["steps"]
                acc_tot += r["avg_accepted"] * max(1, r["steps"])
            torch.cuda.synchronize()
            tps = ntok / (time.time() - t0)
            acc_rate = (acc_tot / max(1, steps_tot)) / k  # mean accepted draft toks / K
            row = {"workload": w, "method": f"ours_fixed{k}", "K": k,
                   "tok_per_s": round(tps, 1),
                   "speedup_vs_generate": round(tps / tps_plain, 3),
                   "speedup_vs_lean": round(tps / tps_lean, 3),
                   "acceptance_rate": round(acc_rate, 3)}
            rows.append(row); print(row)

        # HF built-in assisted generation gut-check at K=2,4 (may be unsupported:
        # Qwen2.5-7B and 0.5B have different vocab sizes, so HF rejects the pair).
        for k in [2, 4]:
            try:
                tps = timed(lambda p: gen_assisted(p, k), prompts)
                row = {"workload": w, "method": f"hf_assisted_K{k}", "K": k,
                       "tok_per_s": round(tps, 1),
                       "speedup_vs_generate": round(tps / tps_plain, 3),
                       "speedup_vs_lean": round(tps / tps_lean, 3),
                       "acceptance_rate": None}
            except Exception as e:
                row = {"workload": w, "method": f"hf_assisted_K{k}", "K": k,
                       "tok_per_s": None, "speedup_vs_generate": None,
                       "speedup_vs_lean": None, "acceptance_rate": None,
                       "error": f"unsupported: {repr(e)[:120]}"}
            rows.append(row); print(row)

        rows.append({"workload": w, "method": "plain_generate", "K": 0, "tok_per_s": round(tps_plain, 1),
                     "speedup_vs_generate": 1.0, "speedup_vs_lean": round(tps_plain / tps_lean, 3),
                     "acceptance_rate": None})

    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/qwen7b_05b_stats.json", "w") as f:
        json.dump({"target": TARGET, "draft": DRAFT, "batch_size": 1,
                   "max_new_tokens": MNT, "n_prompts": N, "rows": rows}, f, indent=2)
    vol.commit()
    return rows


@app.local_entrypoint()
def main():
    rows = run.remote()
    print("\n===== Qwen2.5-7B / 0.5B stats (batch=1) =====")
    print(f"{'workload':10} {'method':16} {'tok/s':>7} {'vs gen':>7} {'vs lean':>8} {'accept':>7}")
    for r in rows:
        acc = f"{r['acceptance_rate']:.0%}" if r['acceptance_rate'] is not None else "-"
        print(f"{r['workload']:10} {r['method']:16} {r['tok_per_s']:>7} "
              f"{r['speedup_vs_generate']:>6}x {r['speedup_vs_lean']:>7}x {acc:>7}")
