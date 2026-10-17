"""Step 3c: add the DRAFT-q hook to the validated EAGLE capture.

Forward-hook on ea_layer.lm_head collects full draft logits per node; topK_genrate
return gives retrieve_indices (node->path map); evaluate_posterior gives the
accepted path (best_candidate, accept_length, candidates) + target logits.

Diagnostic-first: dumps draft-logit shape (reduced vocab?) and VALIDATES the draft
node->path mapping (draft argmax must == drafted token on accepted positions, like
the target check). If valid, saves per-accepted-position draft entropy/margin +
target entropy/dist for SVIP/BCSD. Recipe: transformers 4.53.1 / torch 2.6.0.
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
app = modal.App("eagle-capture2")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

BASE = "meta-llama/Llama-3.1-8B-Instruct"
EA = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
N = int(os.environ.get("EC_N", "6"))
MAXTOK = int(os.environ.get("EC_MAXTOK", "96"))


@app.function(image=image, gpu="H100", timeout=3600,
              volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run():
    import torch
    import eagle.model.ea_model as EM
    from eagle.model.ea_model import EaModel

    LOG = {"dbuf": [], "ri": None, "records": [], "dshape": None,
           "dmatch": 0, "dtot": 0}

    model = EaModel.from_pretrained(use_eagle3=True, base_model_path=BASE, ea_model_path=EA,
                                    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                    device_map="auto", total_token=-1)
    model.eval()
    tok = model.get_tokenizer()
    ea = model.ea_layer

    # EAGLE-3 reduced(32k)->full vocab map: full_id = reduced_id + d2t[reduced_id]
    d2t = None
    for attr in ("d2t",):
        if hasattr(ea, attr):
            try: d2t = getattr(ea, attr).detach().to("cpu")
            except Exception: pass
    if d2t is None:
        print("d2t not found; ea attrs:", [a for a in dir(ea) if not a.startswith('_')][:50], flush=True)
    else:
        print(f"captured d2t shape {tuple(d2t.shape)}", flush=True)
    LOG["d2t"] = d2t

    # 1) draft lm_head forward hook -> collect full draft logits per node
    def lm_hook(mod, inp, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        LOG["dbuf"].append(o.detach().float().view(-1, o.shape[-1]))
        if LOG["dshape"] is None:
            LOG["dshape"] = tuple(o.shape)
    if hasattr(ea, "lm_head"):
        ea.lm_head.register_forward_hook(lm_hook)
        print("hooked ea_layer.lm_head", flush=True)
    else:
        print("!! ea_layer has no lm_head; attrs:", [a for a in dir(ea) if 'head' in a.lower()], flush=True)

    # 2) wrap topK_genrate -> reset draft buffer, capture retrieve_indices + draft_tokens
    TK = type(ea)
    orig_tk = TK.topK_genrate
    def tk_wrap(self, *a, **k):
        LOG["dbuf"] = []
        out = orig_tk(self, *a, **k)
        try:
            LOG["ri"] = out[1].detach()           # retrieve_indices [paths, depth]
            LOG["dtokens"] = out[0].detach().view(-1)  # draft_tokens [total]
            LOG["dnl"] = torch.cat(LOG["dbuf"], 0) if LOG["dbuf"] else None  # [nodes, dvocab]
        except Exception as e:
            print("tk_wrap err", repr(e)[:140], flush=True)
        return out
    TK.topK_genrate = tk_wrap

    # 3) wrap evaluate_posterior -> build per-accepted-position record
    orig_ep = EM.evaluate_posterior
    def ep_wrap(logits, candidates, lp, *a, **k):
        out = orig_ep(logits, candidates, lp, *a, **k)
        try:
            bc, al = int(out[0]), int(out[1])
            dnl, ri = LOG.get("dnl"), LOG.get("ri")
            rec = {"accept_length": al, "positions": []}
            for j in range(al):
                tj = int(candidates[bc, j + 1])
                pj = logits[bc, j].float().softmax(-1)
                pent = float(-(pj * pj.clamp_min(1e-9).log2()).sum())
                pos = {"t": tj, "tent": round(pent, 4),
                       "tmatch": int(int(pj.argmax()) == tj)}
                # node a draft token sits at = ri[bc, .]; a node's logits predict its
                # CHILD, so token at path pos j+1 was drafted by PARENT node ri[bc, j].
                # Test both hypotheses: parent (j) vs self (j+1).
                d2t = LOG.get("d2t")
                def full_argmax(node):
                    red = int(dnl[node].argmax())
                    return red + int(d2t[red]) if d2t is not None else red
                if dnl is not None and ri is not None:
                    for hyp, hidx in (("parent", j), ("self", j + 1)):
                        if hidx < ri.shape[1]:
                            node = int(ri[bc, hidx])
                            if 0 <= node < dnl.shape[0]:
                                m = int(full_argmax(node) == tj)
                                LOG.setdefault("hyp", {}).setdefault(hyp, [0, 0])
                                LOG["hyp"][hyp][0] += m; LOG["hyp"][hyp][1] += 1
                    # store entropy/margin from the PARENT node (the drafting dist)
                    pnode = int(ri[bc, j])
                    if 0 <= pnode < dnl.shape[0]:
                        qj = dnl[pnode].softmax(-1)
                        dent = float(-(qj * qj.clamp_min(1e-9).log2()).sum())
                        d2 = torch.topk(qj, 2).values
                        pos.update({"dent": round(dent, 4),
                                    "dmargin": round(float(d2[0] - d2[1]), 4),
                                    "dmatch": int(full_argmax(pnode) == tj)})
                        LOG["dtot"] += 1; LOG["dmatch"] += int(full_argmax(pnode) == tj)
                rec["positions"].append(pos)
            LOG["records"].append(rec)
        except Exception as e:
            print("ep_wrap err", repr(e)[:160], flush=True)
        return out
    EM.evaluate_posterior = ep_wrap

    prompts = ([
        "Write a python function to compute the nth Fibonacci number.",
        "What is 248 * 17? Show your steps.",
        "Summarize the causes of World War I in three sentences.",
        "Explain how a hash map works.",
        "Translate 'good morning' into French and German.",
        "Write a haiku about autumn.",
    ] * 2)[:max(N, 6)]

    gens = []
    for p in prompts:
        LOG["records"] = []
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt").to(model.base_model.device)
        with torch.no_grad():
            model.eagenerate(ids, temperature=0.0, max_new_tokens=MAXTOK)
        gens.append({"steps": list(LOG["records"])})

    print(f"\n===== DRAFT-q DIAGNOSTIC =====", flush=True)
    print(f"  draft lm_head out shape sample: {LOG['dshape']}  (vocab dim = {LOG['dshape'][-1] if LOG['dshape'] else '?'})", flush=True)
    print(f"  DRAFT (parent-node) validation: draft_argmax==token "
          f"{LOG['dmatch']}/{LOG['dtot']} = {100*LOG['dmatch']/max(1,LOG['dtot']):.1f}%", flush=True)
    for hyp, (m, t) in LOG.get("hyp", {}).items():
        print(f"    hypothesis '{hyp}': {m}/{t} = {100*m/max(1,t):.1f}%", flush=True)
    # also show a sample record
    for g in gens[:1]:
        for s in g["steps"][:2]:
            print("  step:", s["accept_length"], s["positions"][:2], flush=True)

    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/eagle_ref_draftq_llama8b.json", "w") as f:
        json.dump({"target": BASE, "eagle": EA, "draft_vocab": (LOG['dshape'][-1] if LOG['dshape'] else None),
                   "draft_map_match_pct": round(100*LOG['dmatch']/max(1,LOG['dtot']), 1), "gens": gens}, f)
    vol.commit()
    return {"draft_vocab": (LOG['dshape'][-1] if LOG['dshape'] else None),
            "draft_match_pct": round(100*LOG['dmatch']/max(1,LOG['dtot']), 1),
            "steps": sum(len(g['steps']) for g in gens)}


@app.local_entrypoint()
def main():
    r = run.remote()
    print(f"\n===== draft-q capture: draft_vocab={r['draft_vocab']} "
          f"draft_map_match={r['draft_match_pct']}% steps={r['steps']} =====")
    print("  DRAFT MAPPING VALID -> draft entropy/margin usable for SVIP; rerun analyses"
          if r["draft_match_pct"] > 90 else
          "  !! draft mapping off -- inspect node ordering / reduced-vocab")
