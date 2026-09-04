"""OPTION 1 gate: can a learned PROBE on the target hidden state predict per-step
draft acceptance far better than cheap scalar signals (entropy/margin)? If yes,
a controller on it could capture much of the +21.5% per-step oracle ceiling that
entropy-based controllers can't (signal-limited result). Clean separate-draft chain.

Per generated position i: target last-hidden H_i [4096]; draft/target dists q_i,p_i;
label y_i = (argmax q_i == argmax p_i) (would-accept under greedy SD); scalars
tent_i (target entropy), dent_i (draft entropy), margin_i. Train a logistic probe
on H -> y (train/test split by sequence) and compare AUC vs the scalar signals,
including the CAUSAL probe on H_{i-1} (available before drafting position i).
"""
import os, json
import modal

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.5.1", "transformers==4.47.1", "accelerate==1.2.1",
                      "datasets", "sentencepiece"))
app = modal.App("priv-probe")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

TARGET = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT = os.environ.get("PP_DRAFT", "meta-llama/Llama-3.2-1B")  # weaker -> acceptance variance to predict
N = int(os.environ.get("PP_N", "40"))
MAXTOK = int(os.environ.get("PP_MAXTOK", "128"))


@app.function(image=image, gpu="H100", timeout=3600, volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(TARGET)
    tgt = AutoModelForCausalLM.from_pretrained(TARGET, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    drf = AutoModelForCausalLM.from_pretrained(DRAFT, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    dev = tgt.device

    def take(x, n): return list(x)[:n]
    prompts = (take(load_dataset("openai/openai_humaneval", split="test")["prompt"], N // 2)
               + take(load_dataset("openai/gsm8k", "main", split="test")["question"], N - N // 2))

    H, Y, TENT, DENT, MARG, SEQ = [], [], [], [], [], []
    for si, p in enumerate(prompts):
        enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt")
        ids = (enc["input_ids"] if (isinstance(enc, dict) or hasattr(enc, "keys")) else enc).to(dev)
        with torch.no_grad():
            gen = tgt.generate(ids, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        S = gen
        with torch.no_grad():
            to = tgt(S, output_hidden_states=True)
            Pl = to.logits[0].float()
            Hs = to.hidden_states[-1][0].float()        # [T, 4096] target last hidden
            Ql = drf(S).logits[0].float()
        lo, hi = ids.shape[1] - 1, S.shape[1] - 1       # generated region
        for i in range(lo, hi):
            pi = Pl[i].softmax(-1); qi = Ql[i].softmax(-1)
            y = int(int(qi.argmax()) == int(pi.argmax()))
            t2 = torch.topk(qi, 2).values
            H.append(Hs[i]); Y.append(y); SEQ.append(si)
            TENT.append(float(-(pi * pi.clamp_min(1e-9).log2()).sum()))
            DENT.append(float(-(qi * qi.clamp_min(1e-9).log2()).sum()))
            MARG.append(float(t2[0] - t2[1]))

    import statistics as st
    Ht = torch.stack(H).to(dev)                          # [n, 4096]
    Yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    n = len(Y)
    print(f"positions={n}  accept_rate={Yt.mean():.3f}", flush=True)

    def auc(scores, labels):
        s = torch.as_tensor(scores, dtype=torch.float32, device=dev)
        y = torch.as_tensor(labels, dtype=torch.float32, device=dev)
        pos, neg = s[y == 1], s[y == 0]
        if len(pos) == 0 or len(neg) == 0: return float("nan")
        # P(score_pos > score_neg) via rank-sum
        alls = torch.cat([pos, neg]); order = alls.argsort()
        ranks = torch.empty_like(order, dtype=torch.float32); ranks[order] = torch.arange(len(alls), device=dev, dtype=torch.float32)
        r_pos = ranks[:len(pos)].sum()
        return float((r_pos - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))

    # scalar-signal AUCs (negate so higher=more likely accept where appropriate)
    print("scalar signal AUC (predicting acceptance):", flush=True)
    print(f"  target entropy : {auc([-x for x in TENT], Y):.3f}", flush=True)
    print(f"  draft entropy  : {auc([-x for x in DENT], Y):.3f}", flush=True)
    print(f"  draft margin   : {auc(MARG, Y):.3f}", flush=True)

    # train/test split by sequence
    seqs = torch.tensor(SEQ, device=dev)
    uniq = sorted(set(SEQ)); cut = uniq[int(0.7 * len(uniq))]
    tr = seqs < cut; te = ~tr
    mu, sd = Ht[tr].mean(0), Ht[tr].std(0) + 1e-6
    Xtr = (Ht[tr] - mu) / sd; Xte = (Ht[te] - mu) / sd
    ytr, yte = Yt[tr], Yt[te]
    w = torch.zeros(Ht.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.01, weight_decay=1e-3)
    for ep in range(300):
        opt.zero_grad()
        logit = Xtr @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = (Xte @ w + b).sigmoid()
    probe_auc = auc(pred.tolist(), yte.tolist())
    # scalar AUC on the SAME test split (fair comparison)
    te_idx = te.nonzero().flatten().tolist()
    tent_te = [-TENT[i] for i in te_idx]; yte_l = [Y[i] for i in te_idx]
    print(f"\n=== HELD-OUT TEST ({int(te.sum())} pos) ===", flush=True)
    print(f"  HIDDEN-STATE PROBE AUC : {probe_auc:.3f}", flush=True)
    print(f"  target entropy AUC     : {auc(tent_te, yte_l):.3f}", flush=True)
    gap = probe_auc - auc(tent_te, yte_l)
    print(f"\nGATE: probe AUC - entropy AUC = {gap:+.3f}", flush=True)
    print(f"  -> {'GO: hidden state holds acceptance signal entropy misses (option 1 viable)' if probe_auc > 0.70 and gap > 0.07 else 'WEAK: hidden state not much better than entropy'}", flush=True)

    with open("/root/out/priv_probe.json", "w") as f:
        json.dump({"target": TARGET, "draft": DRAFT, "n": n, "accept_rate": float(Yt.mean()),
                   "probe_auc": round(probe_auc, 3),
                   "entropy_auc": round(auc(tent_te, yte_l), 3), "gap": round(gap, 3)}, f)
    vol.commit()
    return {"probe_auc": round(probe_auc, 3), "entropy_auc": round(auc(tent_te, yte_l), 3),
            "accept_rate": round(float(Yt.mean()), 3), "n": n}


@app.local_entrypoint()
def main():
    r = run.remote()
    print(f"\n===== OPTION-1 probe gate: n={r['n']} accept={r['accept_rate']} "
          f"PROBE_AUC={r['probe_auc']} vs entropy_AUC={r['entropy_auc']} =====")
    print("  GO -> hidden-state probe beats entropy; build the probe controller, chase the ceiling"
          if r["probe_auc"] > 0.70 and (r["probe_auc"] - r["entropy_auc"]) > 0.07
          else "  WEAK -> even the hidden state doesn't predict acceptance well; ceiling may be irreducible")
