"""VERIFICATION of the provisional policy-zoo positive (cumprob +2.4%), per audit-first rule.

The zoo run (modal_eagle_zoo.py, results/eagle_zoo.json) found cumulative-path-probability
tail pruning beating the best fixed depth on all 3 workloads (+1.7/+2.9/+2.5%). PROVISIONAL:
single-pass timing, thresholds selected on the benched prompts. This script runs the two
checks that decide whether it becomes a result:

  1. HELD-OUT prompts: capture + threshold selection + probe training on prompts [0:n],
     wall-clock bench on UNSEEN prompts [n:2n].
  2. REPEATED timing: R bench passes per configuration -> mean +/- std.

Also logs each policy's realized mean draft depth (proves it adapts rather than sitting at a
constant), and includes d=8 in the fixed sweep. Policies: cumprob + entropy (provisional
positives) + hidden (negative control). Llama-3.1-8B + yuhuili EAGLE3, H100, greedy.

Run: PYTHONIOENCODING=utf-8 modal run modal_eagle_zoo_verify.py --n 10 --maxtok 128 --repeats 3
Out: /root/out/eagle_zoo_verify.json
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
app = modal.App("eagle-zoo-verify")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

BASE = "meta-llama/Llama-3.1-8B-Instruct"
EA   = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
C_MEAS = 0.072

GRIDS = {
    "entropy": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
    "cumprob": [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4],
    "hidden":  [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
}


@app.function(image=image, gpu="H100", timeout=10800, volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(n, maxtok, repeats, fixed_depths):
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
          "cur_levels": [], "cap_steps": [], "depths": []}

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
        n_done = 0

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
            n_done = i + 1

            row = last_p[0]
            p_row = torch.exp(row)
            ent0 = float(-(p_row * row).sum())
            t2 = torch.topk(p_row, 2).values
            margin0 = float(t2[0] - t2[1])
            cum0 = float(torch.exp(scores[0]))
            h0 = out_hidden[0].mean(dim=0).float().detach()

            if EL["mode"] == "capture":
                EL["cur_levels"].append(
                    {"ent": ent0, "margin": margin0, "cum": cum0, "h": h0.cpu().numpy()})
            elif EL["mode"] == "bench" and EL["policy"] is not None:
                pol, thr = EL["policy"], EL["thr"]
                stop = False
                if pol == "entropy":  stop = ent0 > thr
                elif pol == "cumprob": stop = cum0 < thr
                elif pol == "hidden": stop = EL["ctrl"](h0) < thr
                if stop:
                    break

        if EL["mode"] == "bench" and EL["policy"] is not None:
            EL["depths"].append(n_done)

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

    # INTERLEAVED (even/odd) split — iid across dataset ordering. The first verify run used
    # contiguous [0:n]/[n:2n], which is non-iid (mt_bench is ordered by category): best fixed
    # depth flipped 7->5 across the split, entangling distribution shift with generalization.
    def take_even(x, m): return list(x)[0:2 * m:2]
    def take_odd(x, m):  return list(x)[1:2 * m:2]
    he = load_dataset("openai/openai_humaneval", split="test")["prompt"]
    gk = load_dataset("openai/gsm8k", "main", split="test")["question"]
    mb = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train").map(
        lambda x: {"p": x["prompt"][0]})["p"]
    train_wl = {"humaneval": take_even(he, n), "gsm8k": take_even(gk, n),
                "mt_bench": take_even(mb, n)}
    bench_wl = {"humaneval": take_odd(he, n), "gsm8k": take_odd(gk, n),
                "mt_bench": take_odd(mb, n)}
    def encode(p):
        enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt")
        return (enc["input_ids"] if hasattr(enc, "keys") else enc).to(dev)
    def gen(ids):
        with torch.no_grad():
            return model.eagenerate(ids, temperature=0.0, max_new_tokens=maxtok)

    # ── capture on TRAIN prompts only ────────────────────────────────────── #
    EL["mode"] = "off"; gen(encode("hi"))
    EL["mode"] = "capture"; EL["cap_steps"] = []
    for w, prompts in train_wl.items():
        for p in prompts:
            EL["cur_levels"] = []
            gen(encode(p))
    EL["mode"] = "off"
    steps = EL["cap_steps"]
    print(f"capture (train prompts): {len(steps)} steps", flush=True)

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    H, y = [], []
    for levels, al in steps:
        for li, lv in enumerate(levels):
            H.append(lv["h"]); y.append(1 if li < al else 0)
    H = np.stack(H).astype("float32"); y = np.array(y)
    sc = StandardScaler().fit(H); pca = PCA(n_components=50).fit(sc.transform(H))
    clf = LogisticRegression(max_iter=300, C=0.1).fit(pca.transform(sc.transform(H)), y)
    def hidden_score(h_t):
        z = pca.transform(sc.transform(h_t.cpu().numpy().reshape(1, -1)))
        return float(clf.predict_proba(z)[0, 1])
    EL["ctrl"] = hidden_score
    hid_scores_cached = clf.predict_proba(pca.transform(sc.transform(H)))[:, 1]

    def stop_len(levels, pol, thr, hs_iter):
        for li, lv in enumerate(levels):
            fire = False
            if pol == "entropy":  fire = lv["ent"] > thr
            elif pol == "cumprob": fire = lv["cum"] < thr
            elif pol == "hidden": fire = next(hs_iter) < thr
            if fire:
                return li + 1
        return len(levels)

    def offline_policy(pol, thr):
        tot_acc, tot_k, ns = 0, 0, 0
        hi = iter(hid_scores_cached)
        for levels, al in steps:
            k = stop_len(levels, pol, thr, hi)
            tot_acc += min(al, k); tot_k += k; ns += 1
        return (tot_acc / ns + 1) / (1 + C_MEAS * (tot_k / ns))

    def offline_fixed(d):
        tot_acc, ns = 0, 0
        for levels, al in steps:
            tot_acc += min(al, min(d, len(levels))); ns += 1
        return (tot_acc / ns + 1) / (1 + C_MEAS * d)

    off_fixed = {d: offline_fixed(d) for d in fixed_depths}
    best_off_fixed = max(off_fixed.values())
    bd_train = max(off_fixed, key=off_fixed.get)   # train-selected fixed depth (protocol B)
    print(f"train-selected fixed depth (offline sim): d={bd_train}", flush=True)
    offline = {}
    for pol, grid in GRIDS.items():
        best_thr, best_sp = None, -1
        for thr in grid:
            sp = offline_policy(pol, thr)
            if sp > best_sp:
                best_sp, best_thr = sp, thr
        offline[pol] = {"thr": best_thr,
                        "gain_pct": round(100 * (best_sp / best_off_fixed - 1), 2)}
        print(f"OFFLINE(train) {pol:8s} thr={best_thr}  gain {offline[pol]['gain_pct']:+.2f}%",
              flush=True)

    # ── wall-clock on HELD-OUT prompts, repeated ─────────────────────────── #
    def bench_once(prompts):
        torch.cuda.synchronize(); t0 = time.time(); ntok = 0
        for p in prompts:
            ids = encode(p); out = gen(ids)
            ntok += (out[0].shape[0] if out[0].dim() == 1 else out[0].shape[1]) - ids.shape[1]
        torch.cuda.synchronize()
        return ntok / (time.time() - t0)

    results = {"offline": offline, "fixed": {}, "policies": {}, "policy_mean_depth": {}}
    EL["mode"] = "off"; gen(encode("hi"))
    for d in fixed_depths:
        ea.depth = d
        per_w = {}
        for w, prompts in bench_wl.items():
            runs = [round(bench_once(prompts), 2) for _ in range(repeats)]
            per_w[w] = runs
            print(f"  FIXED d={d} {w:10s} {runs}", flush=True)
        results["fixed"][d] = per_w
    ea.depth = MAXD

    for pol in GRIDS:
        EL["mode"] = "bench"; EL["policy"] = pol; EL["thr"] = offline[pol]["thr"]
        per_w = {}; EL["depths"] = []
        for w, prompts in bench_wl.items():
            runs = [round(bench_once(prompts), 2) for _ in range(repeats)]
            per_w[w] = runs
            print(f"  POLICY {pol} {w:10s} {runs}", flush=True)
        results["policies"][pol] = per_w
        md = round(float(np.mean(EL["depths"])), 2) if EL["depths"] else None
        sd = round(float(np.std(EL["depths"])), 2) if EL["depths"] else None
        results["policy_mean_depth"][pol] = {"mean": md, "std": sd}
        print(f"  {pol} realized depth: {md}±{sd}", flush=True)
    EL["mode"] = "off"; EL["policy"] = None

    # ── summary: held-out mean±std gains vs best fixed (matched by mean) ──── #
    summary = {}
    for pol in GRIDS:
        gains_a, gains_b = [], []
        for w in bench_wl:
            fx_means = {d: float(np.mean(results["fixed"][d][w])) for d in fixed_depths}
            best_d_eval = max(fx_means, key=fx_means.get)
            po_runs = results["policies"][pol][w]
            # Protocol A: vs best fixed selected on the EVAL set (strongest baseline)
            fx_runs = results["fixed"][best_d_eval][w]
            g = 100 * (np.mean(po_runs) / np.mean(fx_runs) - 1)
            se = 100 * np.sqrt(np.var(po_runs) / len(po_runs) +
                               np.var(fx_runs) / len(fx_runs)) / np.mean(fx_runs)
            gains_a.append((w, round(float(g), 2), round(float(se), 2), best_d_eval))
            # Protocol B: vs fixed selected on TRAIN (deployment protocol, same for policy)
            fx_runs_b = results["fixed"][bd_train][w]
            gb = 100 * (np.mean(po_runs) / np.mean(fx_runs_b) - 1)
            gains_b.append((w, round(float(gb), 2), bd_train))
        mean_a = round(float(np.mean([g for _, g, _, _ in gains_a])), 2)
        mean_b = round(float(np.mean([g for _, g, _ in gains_b])), 2)
        summary[pol] = {"offline_gain_pct": offline[pol]["gain_pct"],
                        "protocolA_vs_eval_fixed": gains_a, "protocolA_mean_pct": mean_a,
                        "protocolB_vs_train_fixed": gains_b, "protocolB_mean_pct": mean_b,
                        "realized_depth": results["policy_mean_depth"][pol]}
        print(f">>> {pol:8s} offline {offline[pol]['gain_pct']:+.2f}%  "
              f"A(vs eval-fixed) {mean_a:+.2f}%  B(vs train-fixed d={bd_train}) {mean_b:+.2f}%",
              flush=True)
    results["summary"] = summary
    results["bd_train"] = bd_train

    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/eagle_zoo_verify.json", "w") as f:
        json.dump({"target": BASE, "n": n, "maxtok": maxtok, "repeats": repeats,
                   "fixed_depths": fixed_depths, **results}, f, indent=2)
    vol.commit()
    return summary


@app.local_entrypoint()
def main(n: int = 10, maxtok: int = 128, repeats: int = 3, depths: str = "5,6,7,8"):
    ds = [int(x) for x in depths.split(",")]
    print(f"zoo VERIFY: n={n} (train [0:n], bench [n:2n]) maxtok={maxtok} repeats={repeats} fixed={ds}")
    print(json.dumps(run.remote(n, maxtok, repeats, ds), indent=2))
