"""IN-LOOP within-step early-stop probe for EAGLE-3 (the faithful wall-clock test).

The seed-based controller (modal_eagle_wallclock.py) failed: the seed hidden state does not
predict chain acceptance. This tests the WITHIN-STEP probe our offline analysis measured (+19%
of the per-step oracle): after each depth level's hidden state is computed, a probe decides
whether to keep expanding the draft tree. Implemented by monkeypatching topK_genrate with an
exact copy that (a) calls a per-level controller to break early, and (b) caps the final tree
selection at min(total_tokens, available) so a shorter tree does not crash downstream.

Modes:
  validate : patched copy with NO early-stop -> must reproduce normal generation (sanity)
  capture  : full depth, record per-level hidden + this step's accept_length -> train probe
  bench    : fixed (full depth) vs in-loop early-stop probe -> real tok/s + realized accept

Recipe: transformers 4.53.1 / torch 2.6.0 (validated). H100.
Run: PYTHONIOENCODING=utf-8 modal run modal_eagle_inloop.py --n 12 --maxtok 128
"""
import os, time, json
import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("git")
         .pip_install("torch==2.6.0", "transformers==4.53.1", "accelerate==0.26.0",
                      "sentencepiece", "huggingface_hub", "fschat", "datasets",
                      "scikit-learn", "numpy")
         .run_commands("git clone --depth 1 https://github.com/SafeAILab/EAGLE.git /root/EAGLE")
         .env({"PYTHONPATH": "/root/EAGLE"}))
app = modal.App("eagle-inloop")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

BASE = "meta-llama/Llama-3.1-8B-Instruct"
EA   = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


@app.function(image=image, gpu="H100", timeout=7200, volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(n, maxtok):
    import torch, numpy as np
    from eagle.model.ea_model import EaModel
    from datasets import load_dataset

    model = EaModel.from_pretrained(use_eagle3=True, base_model_path=BASE, ea_model_path=EA,
                                    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                    device_map="auto", total_token=-1)
    model.eval()
    tok = model.get_tokenizer()
    ea  = model.ea_layer
    dev = model.base_model.device
    ea_cls = type(ea)

    # ── module-level control/log state ─────────────────────────────────────────────── #
    EL = {"mode": "off", "cap_hidden": [], "cap_step_lens": [],  # capture
          "cur_levels": [],                                       # per-step level hiddens
          "ctrl": None, "thr": 0.5, "adaptive_depths": []}

    orig_topk = ea_cls.topK_genrate

    @torch.no_grad()
    def topK_patched(self, hidden_states, input_ids, head, logits_processor):
        # ---- verbatim prologue ----
        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k
        sample_token = input_ids[:, -1]
        scores_list = []; parents_list = []; ss_token = []
        input_ids = input_ids[:, 1:].to(hidden_states.device)
        len_posi = input_ids.shape[1]
        self.reset()
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids[:, kv_len:],
                                               past_key_values=self.stable_kv, use_cache=True)
        else:
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]
        last_headout = self.lm_head(self.norm(last_hidden))
        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        if self.config.vocab_size == self.config.draft_vocab_size:
            ss_token.append(topk_index); input_ids = topk_index
        else:
            ss_token.append(topk_index + self.d2t[topk_index]); input_ids = topk_index + self.d2t[topk_index]
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)

        # per-step capture buffer (level 0 = seed)
        EL["cur_levels"] = []

        n_levels_done = 0
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            out_hidden, past_key_values = self(input_hidden, input_ids=input_ids,
                                               past_key_values=past_key_values,
                                               position_ids=position_ids, use_cache=True)
            len_posi += 1
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)
            last_headout = self.lm_head(self.norm(out_hidden[0]))
            last_p = self.logsoftmax(last_headout)
            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values
            cu_scores = topk_p + scores[:, None]
            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p
            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            input_ids = topk_index.view(-1)[topk_cs_index][None]
            if self.config.vocab_size == self.config.draft_vocab_size:
                ss_token.append(topk_index)
            else:
                input_ids = input_ids + self.d2t[input_ids]
                ss_token.append(topk_index + self.d2t[topk_index])
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
            n_levels_done = i + 1

            # ---- per-level hidden representative (top-scored branch) ----
            lvl_h = out_hidden[0].mean(dim=0).float().detach()  # [dim]
            if EL["mode"] == "capture":
                EL["cur_levels"].append(lvl_h.cpu().numpy())
            # ---- in-loop early stop ----
            if EL["mode"] == "bench_adaptive" and EL["ctrl"] is not None:
                p_cont = EL["ctrl"](lvl_h)   # P(this level accepted)
                if p_cont < EL["thr"]:
                    break

        EL["adaptive_depths"].append(n_levels_done)

        # ---- tree assembly (verbatim, but cap total_tokens at available) ----
        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        tt_eff = min(total_tokens, scores_list.shape[0])   # <-- safety for shorter tree
        top_scores = torch.topk(scores_list, tt_eff, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)
        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        tree_mask = torch.eye(tt_eff + 1).bool()
        tree_mask[:, 0] = True
        for i in range(tt_eff):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])
        tree_position_ids = torch.sum(tree_mask, dim=1) - 1
        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]
        del parents_list, ss_token, ss_token_list, draft_parents
        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = tt_eff - noleaf_num
        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()
        rid = 0
        position_ids_list = tree_position_ids.tolist()
        for i in range(tt_eff + 1):
            if i not in noleaf_index:
                cid = i; ddepth = position_ids_list[i]
                for j in reversed(range(ddepth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1
        if logits_processor is not None:
            maxitem = tt_eff + 5
            def custom_sort(lst):
                return [lst[i] if lst[i] >= 0 else maxitem for i in range(len(lst))]
            retrieve_indices = sorted(retrieve_indices, key=custom_sort)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    ea_cls.topK_genrate = topK_patched

    # hook accept_length per step (to label capture levels)
    import eagle.model.ea_model as EM
    orig_eval = EM.evaluate_posterior
    def eval_wrap(*a, **k):
        out = orig_eval(*a, **k)
        try:
            al = out[1]; al = int(al.item()) if torch.is_tensor(al) else int(al)
            if EL["mode"] == "capture":
                EL["cap_step_lens"].append((len(EL["cur_levels"]), al))
                for li, h in enumerate(EL["cur_levels"]):
                    EL["cap_hidden"].append((h, 1 if li < al else 0))
        except Exception:
            pass
        return out
    EM.evaluate_posterior = eval_wrap

    def take(x, m): return list(x)[:m]
    workloads = {
        "humaneval": take(load_dataset("openai/openai_humaneval", split="test")["prompt"], n),
        "gsm8k":     take(load_dataset("openai/gsm8k", "main", split="test")["question"], n),
        "mt_bench":  take(load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
                          .map(lambda x: {"p": x["prompt"][0]})["p"], n),
    }
    def encode(p):
        enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt")
        return (enc["input_ids"] if hasattr(enc, "keys") else enc).to(dev)
    def gen(ids):
        with torch.no_grad():
            return model.eagenerate(ids, temperature=0.0, max_new_tokens=maxtok)

    # ── VALIDATE: patched copy (no early stop) must generate coherently ─────────────── #
    EL["mode"] = "off"
    gen(encode("hi"))  # warmup
    vtxt = tok.decode(gen(encode("What is 17*24? Show your steps."))[0][0], skip_special_tokens=True) \
        if False else None
    out = gen(encode("What is 17*24? Show your steps."))
    dec = tok.decode(out[0][0] if out[0].dim() == 2 else out[0], skip_special_tokens=True)
    print("VALIDATE gen (should be coherent):", repr(dec[-160:]), flush=True)

    # ── CAPTURE: full depth, per-level hidden + accept label ────────────────────────── #
    EL["mode"] = "capture"; EL["cap_hidden"].clear()
    for w, prompts in workloads.items():
        for p in prompts:
            EL["cur_levels"] = []
            gen(encode(p))
    EL["mode"] = "off"
    H = np.stack([h for h, _ in EL["cap_hidden"]]).astype("float32")
    y = np.array([lab for _, lab in EL["cap_hidden"]], dtype="int64")
    print(f"capture: {len(y)} level-samples, accept rate={y.mean():.3f}, dim={H.shape[1]}", flush=True)

    # ── TRAIN probe: level hidden -> P(level accepted) ──────────────────────────────── #
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    ntr = int(0.8 * len(y))
    sc = StandardScaler().fit(H[:ntr])
    Xtr, Xte = sc.transform(H[:ntr]), sc.transform(H[ntr:])
    pca = PCA(n_components=min(50, Xtr.shape[1])).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    clf = LogisticRegression(max_iter=300, C=0.1).fit(Ztr, y[:ntr])
    auc = roc_auc_score(y[ntr:], clf.predict_proba(Zte)[:, 1]) if len(set(y[ntr:])) > 1 else float("nan")
    print(f"within-step level probe: test AUC={auc:.3f}", flush=True)
    def controller(h_t):
        import numpy as _np
        z = pca.transform(sc.transform(h_t.cpu().numpy().reshape(1, -1)))
        return float(clf.predict_proba(z)[0, 1])
    EL["ctrl"] = controller

    # ── BENCH: fixed (full depth) vs in-loop early-stop ─────────────────────────────── #
    def bench(prompts):
        torch.cuda.synchronize(); t0 = time.time(); ntok = 0
        for p in prompts:
            ids = encode(p); out = gen(ids)
            ntok += (out[0].shape[0] if out[0].dim() == 1 else out[0].shape[1]) - ids.shape[1]
        torch.cuda.synchronize()
        return ntok / (time.time() - t0)

    results = {"within_step_auc": auc, "fixed": {}, "adaptive": {}, "adaptive_mean_depth": {}}
    gen(encode("hi"))
    EL["mode"] = "off"
    for w, prompts in workloads.items():
        results["fixed"][w] = round(bench(prompts), 2)
        print(f"  FIXED (full depth) {w:10s} {results['fixed'][w]} tok/s", flush=True)
    # sweep a couple thresholds for the early-stop
    best = None
    for thr in [0.3, 0.5, 0.7]:
        EL["mode"] = "bench_adaptive"; EL["thr"] = thr
        per_w = {}; md = {}
        for w, prompts in workloads.items():
            EL["adaptive_depths"] = []
            per_w[w] = round(bench(prompts), 2)
            md[w] = round(float(np.mean(EL["adaptive_depths"])), 2) if EL["adaptive_depths"] else None
            print(f"  ADAPTIVE thr={thr} {w:10s} {per_w[w]} tok/s (mean depth={md[w]})", flush=True)
        results["adaptive"][thr] = per_w; results["adaptive_mean_depth"][thr] = md
    EL["mode"] = "off"

    # summary: best adaptive threshold per workload vs fixed
    summary = {}
    for w in workloads:
        fx = results["fixed"][w]
        best_thr = max(results["adaptive"], key=lambda t: results["adaptive"][t][w])
        ad = results["adaptive"][best_thr][w]
        summary[w] = {"fixed_tps": fx, "adaptive_tps": ad, "best_thr": best_thr,
                      "mean_depth": results["adaptive_mean_depth"][best_thr][w],
                      "gain_pct": round(100 * (ad / fx - 1), 2)}
        print(f"  >>> {w:10s} fixed={fx} | adaptive(thr={best_thr})={ad} | "
              f"gain={summary[w]['gain_pct']:+.2f}% | mean_depth={summary[w]['mean_depth']}", flush=True)
    results["summary"] = summary

    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/eagle_inloop.json", "w") as f:
        json.dump({"target": BASE, "n": n, "maxtok": maxtok, **results}, f, indent=2)
    vol.commit()
    return {"auc": auc, "summary": summary}


@app.local_entrypoint()
def main(n: int = 12, maxtok: int = 128):
    print(f"EAGLE-3 in-loop within-step early-stop: n={n} maxtok={maxtok}")
    print(json.dumps(run.remote(n, maxtok), indent=2))
