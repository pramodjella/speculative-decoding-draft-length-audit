"""Step 3b: correct per-step capture on the EAGLE reference loop + VALIDATE the
pairing vLLM failed. Single hook on evaluate_posterior(logits, candidates) ->
along the accepted path: draft token, target dist (entropy + argmax), acceptance.

Validation: on accepted positions (j < accept_length) the draft token MUST equal
the target argmax (greedy SD) -> match should be ~100% (vLLM capture gave 1%).
Also dumps target entropy per position. Working recipe: transformers 4.53.1.
"""
import os, json
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.6.0", "transformers==4.53.1", "accelerate==0.26.0",
                 "sentencepiece", "huggingface_hub", "fschat")
    .run_commands("git clone --depth 1 https://github.com/SafeAILab/EAGLE.git /root/EAGLE")
    .env({"PYTHONPATH": "/root/EAGLE"})
)
app = modal.App("eagle-capture")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

BASE = "meta-llama/Llama-3.1-8B-Instruct"
EA = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
N = int(os.environ.get("EC_N", "8"))
MAXTOK = int(os.environ.get("EC_MAXTOK", "96"))


@app.function(image=image, gpu="H100", timeout=3600,
              volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run():
    import torch
    import eagle.model.ea_model as EM
    from eagle.model.ea_model import EaModel

    REC = {"steps": []}

    orig_ep = EM.evaluate_posterior
    def ep_hook(logits, candidates, logits_processor, *a, **k):
        out = orig_ep(logits, candidates, logits_processor, *a, **k)
        try:
            bc, acc_len = int(out[0]), int(out[1])
            L = candidates.shape[1]
            positions = []
            # EAGLE verify: logits[bc, j] predicts candidates[bc, j+1].
            # accepted drafts = candidates[bc, 1..acc_len], verified by logits[bc, 0..acc_len-1].
            for j in range(min(acc_len + 1, L - 1)):
                tj = int(candidates[bc, j + 1])
                pj = logits[bc, j].float().softmax(-1)
                targ = int(pj.argmax())
                t2 = torch.topk(pj, 2).values
                ent = float(-(pj * pj.clamp_min(1e-9).log2()).sum())
                positions.append({"t": tj, "targ": targ, "match": int(tj == targ),
                                  "tent": round(ent, 4), "tmargin": round(float(t2[0] - t2[1]), 4),
                                  "accepted": int(j < acc_len)})
            REC["steps"].append({"accept_length": acc_len, "positions": positions})
        except Exception as e:
            print("ep_hook err", repr(e)[:160], flush=True)
        return out
    EM.evaluate_posterior = ep_hook

    model = EaModel.from_pretrained(use_eagle3=True, base_model_path=BASE, ea_model_path=EA,
                                    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                    device_map="auto", total_token=-1)
    model.eval()
    tok = model.get_tokenizer()

    prompts = ([
        "Write a python function to compute the nth Fibonacci number.",
        "What is 248 * 17? Show your steps.",
        "Summarize the causes of World War I in three sentences.",
        "Explain how a hash map works.",
        "Translate 'good morning, how are you?' into French.",
        "Write a haiku about autumn.",
        "What is the derivative of x^3 + 2x?",
        "List three uses of graphene.",
    ] * 3)[:max(N, 8)]

    gens = []
    for p in prompts:
        REC["steps"] = []
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt").to(model.base_model.device)
        with torch.no_grad():
            model.eagenerate(ids, temperature=0.0, max_new_tokens=MAXTOK)
        gens.append({"steps": list(REC["steps"])})

    # ---- VALIDATION ----
    tot = match = 0
    accs = []
    for g in gens:
        for s in g["steps"]:
            accs.append(s["accept_length"])
            for pos in s["positions"]:
                if pos["accepted"]:
                    tot += 1; match += pos["match"]
    import statistics as st
    rate = 100 * match / max(1, tot)
    print(f"\n===== VALIDATION =====", flush=True)
    print(f"  steps={sum(len(g['steps']) for g in gens)}  accepted positions checked={tot}", flush=True)
    print(f"  draft==target on accepted positions: {match}/{tot} = {rate:.1f}%  (vLLM gave 1%; expect ~100%)", flush=True)
    print(f"  mean accept_length+1 (MAT) = {1 + st.mean(accs):.3f}", flush=True)

    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/eagle_ref_capture_llama8b.json", "w") as f:
        json.dump({"target": BASE, "eagle": EA, "gens": gens}, f)
    vol.commit()
    return {"steps": sum(len(g['steps']) for g in gens), "match_rate": round(rate, 1),
            "mat": round(1 + st.mean(accs), 3)}


@app.local_entrypoint()
def main():
    r = run.remote()
    print(f"\n===== EAGLE-ref capture: steps={r['steps']} MAT={r['mat']} "
          f"accepted-match={r['match_rate']}% =====")
    print("  PAIRING VALID (correct substrate!) -> add draft-q hook for SVIP/BCSD"
          if r["match_rate"] > 95 else
          f"  !! match {r['match_rate']}% -- alignment still off, inspect")
