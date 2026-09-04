"""POLICY-ZOO AUDIT: published draft-stopping policies vs tuned fixed depth, offline vs wall-clock.

Main-track experiment. Runs FOUR draft-side stopping policies through the SAME verified in-loop
EAGLE-3 codepath (patched topK_genrate from modal_eagle_inloop.py, validated to generate
coherently), measuring for each:
  (1) OFFLINE-predicted gain vs best fixed depth (cost model, measured C) — what the literature's
      evaluation methodology would report;
  (2) WALL-CLOCK gain vs best fixed depth — reality.

Policies (each stops the depth loop when its rule fires, mirroring the cited method's signal):
  entropy  : stop when draft next-token entropy > tau            (SVIP-style)
  margin   : stop when top1-top2 prob margin < tau               (AdaEDL/margin-style)
  cumprob  : stop when cumulative best-path prob < tau           (PACER/SADDLE-style)
  hidden   : stop when PCA-50+LR probe P(accept) < tau           (SpecDec++-style trained head)

If offline says + and wall-clock says − across the zoo, the sign-inversion is a property of the
evaluation methodology, not of one controller. Llama-3.1-8B + yuhuili EAGLE3, H100, greedy.

Run: PYTHONIOENCODING=utf-8 modal run modal_eagle_zoo.py --n 10 --maxtok 128
Out: /root/out/eagle_zoo.json
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
app = modal.App("eagle-zoo")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

BASE = "meta-llama/Llama-3.1-8B-Instruct"
EA   = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
C_MEAS = 0.072   # measured cost constant (measured_cost_ground.py)

GRIDS = {
    "entropy": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
    "margin":  [0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
    "cumprob": [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4],
    "hidden":  [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
}


@app.function(image=image, gpu="H100", timeout=10800, volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(n, maxtok, fixed_depths):
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
    MAXD = int(getattr(ea, "depth", 7))
    print(f"MAXD={MAXD}", flush=True)

    EL = {"mode": "off", "policy": None, "thr": None, "ctrl": None,
          "cur_levels": [], "cap_steps": []}

    orig_topk = ea_cls.topK_genrate

    @torch.no_grad()
    def topK_patched(self, hidden_states, input_ids, head, logits_processor):
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

        EL["cur_levels"] = []

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

            # per-level signals from the best branch (row 0 = best previous-level branch)
            row = last_p[0]                                   # log-probs [vocab]
            p_row = torch.exp(row)
            ent0 = float(-(p_row * row).sum())                # nats
            t2 = torch.topk(p_row, 2).values
            margin0 = float(t2[0] - t2[1])
            cum0 = float(torch.exp(scores[0]))                # best cumulative path prob
            h0 = out_hidden[0].mean(dim=0).float().detach()   # mean level hidden [dim]

            if EL["mode"] == "capture":
                EL["cur_levels"].append(
                    {"ent": ent0, "margin": margin0, "cum": cum0, "h": h0.cpu().numpy()})
            elif EL["mode"] == "bench" and EL["policy"] is not None:
                pol, thr = EL["policy"], EL["thr"]
                stop = False
                if pol == "entropy":  stop = ent0 > thr
                elif pol == "margin": stop = margin0 < thr
                elif pol == "cumprob": stop = cum0 < thr
                elif pol == "hidden": stop = EL["ctrl"](h0) < thr
                if stop:
                    break

        # tree assembly with tt_eff safety cap (verified copy)
        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        tt_eff = min(total_tokens, scores_list.shape[0])
        top_scores = torch.topk(scores_list, tt_eff, dim=-1)
        top_scores_index = torch.sort(top_scores.indices).values
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
        del parents_list, scores_list, ss_token, ss_token_list, draft_parents
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
                cid = i; dd = position_ids_list[i]
                for j in reversed(range(dd + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1
        if logits_processor is not None:
            maxitem = tt_eff + 5
            def cs(lst): return [x if x >= 0 else maxitem for x in lst]
            retrieve_indices = sorted(retrieve_indices, key=cs)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    ea_cls.topK_genrate = topK_patched

    import eagle.model.ea_model as EM
    orig_eval = EM.evaluate_posterior
    def eval_wrap(*a, **k):
        out = orig_eval(*a, **k)
        try:
            al = out[1]; al = int(al.item()) if torch.is_tensor(al) else int(al)
            if EL["mode"] == "capture" and EL["cur_levels"]:
                EL["cap_steps"].append((list(EL["cur_levels"]), al))
                EL["cur_levels"] = []
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

    # ── CAPTURE (full depth) ─────────────────────────────────────────────── #
    EL["mode"] = "off"; gen(encode("hi"))
    EL["mode"] = "capture"; EL["cap_steps"] = []
    for w, prompts in workloads.items():
        for p in prompts:
            EL["cur_levels"] = []
            gen(encode(p))
    EL["mode"] = "off"
    steps = EL["cap_steps"]
    print(f"capture: {len(steps)} steps", flush=True)

    # ── train hidden probe (SpecDec++-style head) ────────────────────────── #
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    H, y = [], []
    for levels, al in steps:
        for li, lv in enumerate(levels):
            H.append(lv["h"]); y.append(1 if li < al else 0)
    H = np.stack(H).astype("float32"); y = np.array(y)
    ntr = int(0.8 * len(y))
    sc = StandardScaler().fit(H[:ntr]); pca = PCA(n_components=50).fit(sc.transform(H[:ntr]))
    clf = LogisticRegression(max_iter=300, C=0.1).fit(pca.transform(sc.transform(H[:ntr])), y[:ntr])
    auc = roc_auc_score(y[ntr:], clf.predict_proba(pca.transform(sc.transform(H[ntr:])))[:, 1])
    print(f"hidden probe AUC={auc:.3f}", flush=True)
    def hidden_score(h_t):
        z = pca.transform(sc.transform(h_t.cpu().numpy().reshape(1, -1)))
        return float(clf.predict_proba(z)[0, 1])
    EL["ctrl"] = hidden_score
    hid_scores_cached = clf.predict_proba(pca.transform(sc.transform(H)))[:, 1]

    # ── OFFLINE simulation: each policy + fixed depths, cost model C_MEAS ──── #
    def stop_len(levels, pol, thr, hs_iter):
        for li, lv in enumerate(levels):
            fire = False
            if pol == "entropy":  fire = lv["ent"] > thr
            elif pol == "margin": fire = lv["margin"] < thr
            elif pol == "cumprob": fire = lv["cum"] < thr
            elif pol == "hidden": fire = next(hs_iter) < thr
            if fire:
                return li + 1
        return len(levels)

    def offline_speedup_policy(pol, thr):
        tot_acc, tot_k, ns = 0, 0, 0
        hi = iter(hid_scores_cached)
        for levels, al in steps:
            k = stop_len(levels, pol, thr, hi)
            tot_acc += min(al, k); tot_k += k; ns += 1
        return (tot_acc / ns + 1) / (1 + C_MEAS * (tot_k / ns))

    def offline_speedup_fixed(d):
        tot_acc, ns = 0, 0
        for levels, al in steps:
            k = min(d, len(levels))
            tot_acc += min(al, k); ns += 1
        return (tot_acc / ns + 1) / (1 + C_MEAS * d)

    off_fixed = {d: offline_speedup_fixed(d) for d in fixed_depths}
    best_off_fixed = max(off_fixed.values())
    offline = {}
    for pol, grid in GRIDS.items():
        best_thr, best_sp = None, -1
        for thr in grid:
            sp = offline_speedup_policy(pol, thr)
            if sp > best_sp:
                best_sp, best_thr = sp, thr
        offline[pol] = {"thr": best_thr, "speedup": round(best_sp, 4),
                        "gain_vs_fixed_pct": round(100 * (best_sp / best_off_fixed - 1), 2)}
        print(f"OFFLINE {pol:8s} thr={best_thr}  predicted gain vs best fixed: "
              f"{offline[pol]['gain_vs_fixed_pct']:+.2f}%", flush=True)

    # ── WALL-CLOCK bench: fixed sweep + each policy at its offline-best thr ── #
    def bench(prompts):
        torch.cuda.synchronize(); t0 = time.time(); ntok = 0
        for p in prompts:
            ids = encode(p); out = gen(ids)
            ntok += (out[0].shape[0] if out[0].dim() == 1 else out[0].shape[1]) - ids.shape[1]
        torch.cuda.synchronize()
        return ntok / (time.time() - t0)

    results = {"auc": round(float(auc), 3), "offline": offline,
               "offline_fixed": {d: round(s, 4) for d, s in off_fixed.items()},
               "fixed": {}, "policies": {}}
    EL["mode"] = "off"; gen(encode("hi"))
    for d in fixed_depths:
        ea.depth = d
        per_w = {w: round(bench(pr), 2) for w, pr in workloads.items()}
        results["fixed"][d] = per_w
        print(f"  FIXED d={d}: {per_w}", flush=True)
    ea.depth = MAXD

    for pol in GRIDS:
        EL["mode"] = "bench"; EL["policy"] = pol; EL["thr"] = offline[pol]["thr"]
        per_w = {w: round(bench(pr), 2) for w, pr in workloads.items()}
        results["policies"][pol] = per_w
        print(f"  POLICY {pol} thr={offline[pol]['thr']}: {per_w}", flush=True)
    EL["mode"] = "off"; EL["policy"] = None

    # ── summary: offline-predicted vs wall-clock gain per policy ───────────── #
    summary = {}
    for pol in GRIDS:
        wc_gains = []
        for w in workloads:
            best_fx = max(results["fixed"][d][w] for d in fixed_depths)
            wc_gains.append(100 * (results["policies"][pol][w] / best_fx - 1))
        summary[pol] = {
            "offline_predicted_gain_pct": offline[pol]["gain_vs_fixed_pct"],
            "wallclock_gain_pct_by_workload": {w: round(g, 2) for w, g in
                                               zip(workloads, wc_gains)},
            "wallclock_mean_gain_pct": round(float(np.mean(wc_gains)), 2),
        }
        print(f">>> {pol:8s} offline {summary[pol]['offline_predicted_gain_pct']:+.2f}%  "
              f"wallclock mean {summary[pol]['wallclock_mean_gain_pct']:+.2f}%", flush=True)
    results["summary"] = summary

    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/eagle_zoo.json", "w") as f:
        json.dump({"target": BASE, "n": n, "maxtok": maxtok, "C": C_MEAS,
                   "fixed_depths": fixed_depths, **results}, f, indent=2)
    vol.commit()
    return summary


@app.local_entrypoint()
def main(n: int = 10, maxtok: int = 128, depths: str = "4,5,6,7"):
    ds = [int(x) for x in depths.split(",")]
    print(f"policy zoo: n={n} maxtok={maxtok} fixed={ds}")
    print(json.dumps(run.remote(n, maxtok, ds), indent=2))
