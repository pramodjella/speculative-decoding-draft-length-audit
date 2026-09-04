"""Stats on a MATCHED-vocab pair: Llama-3.1-8B target / Llama-3.2-1B draft.

Unlike Qwen (7B vocab 152064 vs 0.5B/1.5B 151936), Llama 3.1-8B and 3.2-1B share
the identical tokenizer and vocab_size=128256, so HF built-in assisted generation
(Yash's gut-check) and vLLM draft-model spec both accept the pair. This run does
the transformers side: standard generate() baseline, HF assisted-gen, and our
custom harness, reporting acceptance rate + speedup vs generate(), batch=1.

Uses ungated mirrors so it runs without a Meta license/HF token.
"""
import os, sys, time, json
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    # transformers >= 4.45 is required to parse the Llama-3.2 tokenizer.json;
    # 4.44.2 raises "data did not match any variant ... ModelWrapper".
    .pip_install("torch", "transformers==4.46.3", "accelerate", "datasets", "numpy")
    .env({"PYTHONUNBUFFERED": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir("src", "/root/project/src")
)
app = modal.App("llama-8b-1b-stats")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

# unsloth hosts ungated, faithful copies of both, so target+draft share an
# identical tokenizer (required for HF assisted generation).
TARGET_CANDIDATES = [
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
DRAFT_CANDIDATES = [
    "unsloth/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]
N = int(os.environ.get("ST_N", "24"))
MNT = int(os.environ.get("ST_MNT", "96"))
KS = [1, 2, 4, 8]


@app.function(image=image, gpu="A100", timeout=3600, volumes={"/root/out": vol})
def run():
    import traceback
    try:
        return _run()
    except Exception:
        print("FATAL EXCEPTION:\n" + traceback.format_exc(), flush=True)
        raise


def _run():
    os.chdir("/root/project")
    sys.path.insert(0, "/root/project/src")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from serve.physical_runner import PhysicalSpeculativeRunner, PythonAutoregressiveBaseline
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)

    def try_load(cands):
        for mid in cands:
            try:
                tok = AutoTokenizer.from_pretrained(mid)
                if tok.pad_token_id is None:
                    tok.pad_token_id = tok.eos_token_id
                m = AutoModelForCausalLM.from_pretrained(
                    mid, torch_dtype=torch.bfloat16, device_map="cuda",
                    attn_implementation="sdpa").eval()
                print(f"  loaded {mid}")
                return m, tok, mid
            except Exception as e:
                print(f"  skip {mid}: {repr(e)[:120]}")
        raise RuntimeError(f"none of {cands} loaded")

    print("Loading target..."); tgt, tok, tgt_id = try_load(TARGET_CANDIDATES)
    print("Loading draft...");  drf, dtok, drf_id = try_load(DRAFT_CANDIDATES)
    dev = next(tgt.parameters()).device
    tv, dv = tgt.config.vocab_size, drf.config.vocab_size
    print(f"target={tgt_id} (vocab {tv})  draft={drf_id} (vocab {dv})  match={tv==dv}")
    print(f"GPU mem={torch.cuda.memory_allocated()/1e9:.1f}GB")

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

        for k in KS:
            runner = PhysicalSpeculativeRunner(tgt, drf, tok, controller=k)
            steps_tot, acc_tot = 0, 0
            runner.generate(prompts[0], max_new_tokens=MNT)
            torch.cuda.synchronize(); t0 = time.time(); ntok = 0
            for p in prompts:
                r = runner.generate(p, max_new_tokens=MNT)
                ntok += r["generated_tokens"]; steps_tot += r["steps"]
                acc_tot += r["avg_accepted"] * max(1, r["steps"])
            torch.cuda.synchronize()
            tps = ntok / (time.time() - t0)
            acc_rate = (acc_tot / max(1, steps_tot)) / k
            row = {"workload": w, "method": f"ours_fixed{k}", "K": k, "tok_per_s": round(tps, 1),
                   "speedup_vs_generate": round(tps / tps_plain, 3),
                   "acceptance_rate": round(acc_rate, 3)}
            rows.append(row); print(row)

        for k in [2, 4]:
            try:
                tps = timed(lambda p: gen_assisted(p, k), prompts)
                row = {"workload": w, "method": f"hf_assisted_K{k}", "K": k, "tok_per_s": round(tps, 1),
                       "speedup_vs_generate": round(tps / tps_plain, 3), "acceptance_rate": None}
            except Exception as e:
                row = {"workload": w, "method": f"hf_assisted_K{k}", "K": k, "tok_per_s": None,
                       "speedup_vs_generate": None, "acceptance_rate": None,
                       "error": f"{repr(e)[:140]}"}
            rows.append(row); print(row)

        rows.append({"workload": w, "method": "plain_generate", "K": 0, "tok_per_s": round(tps_plain, 1),
                     "speedup_vs_generate": 1.0, "acceptance_rate": None})

    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/llama_8b_1b_stats.json", "w") as f:
        json.dump({"target": tgt_id, "draft": drf_id, "vocab_match": tv == dv,
                   "batch_size": 1, "max_new_tokens": MNT, "n_prompts": N, "rows": rows}, f, indent=2)
    vol.commit()
    return {"target": tgt_id, "draft": drf_id, "vocab_match": tv == dv, "rows": rows}


@app.local_entrypoint()
def main():
    res = run.remote()
    print(f"\n===== Llama-3.1-8B / 3.2-1B stats (batch=1)  vocab_match={res['vocab_match']} =====")
    print(f"target={res['target']}  draft={res['draft']}")
    print(f"{'workload':10} {'method':16} {'tok/s':>7} {'vs gen':>7} {'accept':>7}")
    for r in res["rows"]:
        acc = f"{r['acceptance_rate']:.0%}" if r['acceptance_rate'] is not None else "-"
        sg = f"{r['speedup_vs_generate']}x" if r['speedup_vs_generate'] is not None else "FAIL"
        tps = r['tok_per_s'] if r['tok_per_s'] is not None else "-"
        print(f"{r['workload']:10} {r['method']:16} {str(tps):>7} {sg:>7} {acc:>7}")
