"""Work IN the EAGLE-3 repo: real WALL-CLOCK sweep over tree size (total_token) per
workload. EAGLE-3 analog of "best fixed K". If one size wins everywhere -> adaptive
tree size is pointless; if it varies by workload -> there's headroom to adapt.
Validated recipe: transformers 4.53.1 / torch 2.6.0.
"""
import os, time, json
import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("git")
         .pip_install("torch==2.6.0", "transformers==4.53.1", "accelerate==0.26.0",
                      "sentencepiece", "huggingface_hub", "fschat", "datasets")
         .run_commands("git clone --depth 1 https://github.com/SafeAILab/EAGLE.git /root/EAGLE")
         .env({"PYTHONPATH": "/root/EAGLE"}))
app = modal.App("eagle-ttsweep")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

BASE = "meta-llama/Llama-3.1-8B-Instruct"
EA = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
N = int(os.environ.get("TT_N", "10"))
MAXTOK = int(os.environ.get("TT_MAXTOK", "128"))
TTS = [int(x) for x in os.environ.get("TT_VALS", "-1,32,48,60,80").split(",")]


@app.function(image=image, gpu="H100", timeout=5400, volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run():
    import torch
    from eagle.model.ea_model import EaModel
    from datasets import load_dataset

    def take(x, n): return list(x)[:n]
    workloads = {
        "humaneval": take(load_dataset("openai/openai_humaneval", split="test")["prompt"], N),
        "gsm8k": take(load_dataset("openai/gsm8k", "main", split="test")["question"], N),
        "mt_bench": take(load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
                         .map(lambda x: {"p": x["prompt"][0]})["p"], N),
    }

    results = []
    for tt in TTS:
        model = EaModel.from_pretrained(use_eagle3=True, base_model_path=BASE, ea_model_path=EA,
                                        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                        device_map="auto", total_token=tt)
        model.eval()
        tok = model.get_tokenizer()
        # warmup
        wi = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                     add_generation_prompt=True, return_tensors="pt")
        wi = (wi["input_ids"] if hasattr(wi, "keys") else wi).to(model.base_model.device)
        with torch.no_grad():
            model.eagenerate(wi, temperature=0.0, max_new_tokens=8)
        for w, prompts in workloads.items():
            torch.cuda.synchronize(); t0 = time.time(); ntok = 0
            for p in prompts:
                enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                              add_generation_prompt=True, return_tensors="pt")
                ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(model.base_model.device)
                with torch.no_grad():
                    out = model.eagenerate(ids, temperature=0.0, max_new_tokens=MAXTOK)
                ntok += (out[0].shape[0] if out[0].dim() == 1 else out[0].shape[1]) - ids.shape[1]
            torch.cuda.synchronize(); dt = time.time() - t0
            tps = ntok / dt
            results.append({"total_token": tt, "workload": w, "tok_per_s": round(tps, 2), "tokens": ntok})
            print(f"  tt={tt:>3} {w:10s} {tps:7.1f} tok/s", flush=True)
        del model; torch.cuda.empty_cache()

    # best total_token per workload (real wall-clock)
    print("\n=== best total_token per workload (real wall-clock) ===", flush=True)
    best = {}
    for r in results:
        w = r["workload"]
        if w not in best or r["tok_per_s"] > best[w]["tok_per_s"]:
            best[w] = r
    for w, r in sorted(best.items()):
        print(f"  {w:10s} best total_token={r['total_token']}  ({r['tok_per_s']} tok/s)", flush=True)
    bts = set(r["total_token"] for r in best.values())
    print(f"\n  -> best total_token {'VARIES by workload' if len(bts) > 1 else 'is the SAME everywhere'} "
          f"({sorted(bts)}) => adaptation {'has headroom' if len(bts) > 1 else 'is pointless'}", flush=True)

    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/eagle_ttsweep.json", "w") as f:
        json.dump({"target": BASE, "results": results}, f)
    vol.commit()
    return {"best": {w: r["total_token"] for w, r in best.items()}}


@app.local_entrypoint()
def main():
    print(run.remote())
