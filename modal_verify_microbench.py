"""VERIFY-LATENCY-vs-K microbench (Yash round-3, item 1).

The fair-baseline argument rests on one asserted sentence: "at B=1 the verification pass is
memory-bound and nearly flat in K." This measures it directly: the TARGET model's forward
latency over q = K+1 query positions against a 512-token KV prefix, for q in {1,2,3,4,5,8}
and B in {1,4,8} (the B>1 rows document where flatness breaks, supporting the batch-decay story).

Protocol: our own medicine — all (B,q) cells benchmarked round-robin per cycle with rotated
order, 5 cycles, CUDA-event timing, median-of-30 per cell per cycle, reported mean±SE over
cycles. Pure HF transformers eager (engine-independent isolation of the verify forward).

Run: PYTHONIOENCODING=utf-8 modal run modal_verify_microbench.py
Out: /root/out/verify_microbench.json
"""
import os, json
import modal

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "transformers", "accelerate", "numpy"))
app = modal.App("verify-microbench")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

TARGET = "meta-llama/Llama-3.1-8B-Instruct"
PREFIX_LEN = 512
QS = [1, 2, 3, 4, 5, 8]
BS = [1, 4, 8]
CYCLES = 5
REPS = 30
WARM = 8


@app.function(image=image, gpu="H100", timeout=3600, volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run():
    import torch, numpy as np, copy
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(TARGET)
    model = AutoModelForCausalLM.from_pretrained(
        TARGET, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    dev = "cuda"
    print("model loaded", flush=True)

    text = ("The quick brown fox jumps over the lazy dog. " * 200)
    base_ids = tok(text, return_tensors="pt").input_ids[0][:PREFIX_LEN]

    # build a prefix KV cache per batch size
    caches = {}
    for B in BS:
        ids = base_ids.unsqueeze(0).repeat(B, 1).to(dev)
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
        caches[B] = out.past_key_values
        print(f"prefix cache built B={B}", flush=True)

    def time_cell(B, q):
        """Median latency (ms) of a forward over q positions with the prefix cache."""
        ids = torch.randint(100, 20000, (B, q), device=dev)
        pos = torch.arange(PREFIX_LEN, PREFIX_LEN + q, device=dev).unsqueeze(0).repeat(B, 1)
        cache = caches[B]
        can_crop = hasattr(cache, "crop")
        times = []
        with torch.no_grad():
            for r in range(WARM + REPS):
                if can_crop:
                    cache.crop(PREFIX_LEN)
                    c = cache
                else:
                    c = copy.deepcopy(cache)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(input_ids=ids, position_ids=pos, past_key_values=c, use_cache=True)
                end.record()
                torch.cuda.synchronize()
                if r >= WARM:
                    times.append(start.elapsed_time(end))
        if can_crop:
            cache.crop(PREFIX_LEN)
        return float(np.median(times))

    cells = [(B, q) for B in BS for q in QS]
    results = {f"B{B}_q{q}": [] for B, q in cells}
    for c in range(CYCLES):
        order = cells[c % len(cells):] + cells[:c % len(cells)]
        for B, q in order:
            ms = time_cell(B, q)
            results[f"B{B}_q{q}"].append(round(ms, 4))
        print(f"cycle {c+1}/{CYCLES} done", flush=True)

    # summary: mean±SE per cell; ratios vs q=1 within each B
    import numpy as np
    summary = {}
    print("\n=== verify forward latency (ms), mean±SE over cycles ===", flush=True)
    for B in BS:
        row = {}
        base = np.mean(results[f"B{B}_q1"])
        for q in QS:
            v = results[f"B{B}_q{q}"]
            m, se = float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))
            row[f"q{q}"] = {"ms_mean": round(m, 4), "ms_se": round(se, 4),
                            "ratio_vs_q1": round(m / base, 4)}
        summary[f"B{B}"] = row
        rs = "  ".join(f"q={q}:{row[f'q{q}']['ms_mean']:.2f}ms({row[f'q{q}']['ratio_vs_q1']:.3f}x)"
                       for q in QS)
        print(f"B={B}: {rs}", flush=True)
    for B in BS:
        r8 = summary[f"B{B}"]["q8"]["ratio_vs_q1"]
        print(f">>> B={B}: verifying 8 positions costs {r8:.3f}x of verifying 1 "
              f"({'~FLAT' if r8 < 1.15 else 'NOT flat'})", flush=True)

    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/verify_microbench.json", "w") as f:
        json.dump({"target": TARGET, "prefix_len": PREFIX_LEN, "cycles": CYCLES,
                   "reps": REPS, "raw_ms": results, "summary": summary}, f, indent=2)
    vol.commit()
    return {B: summary[f"B{B}"]["q8"]["ratio_vs_q1"] for B in BS}


@app.local_entrypoint()
def main():
    print(json.dumps(run.remote(), indent=2))
