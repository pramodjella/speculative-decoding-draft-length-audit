"""PAIRED alternating benchmark: cumprob(thr=0.05) vs fixed depths, drift-controlled.

Three prior runs gave contradictory cumprob-vs-fixed answers (+2.4% / -1.9% / +6.4%) with
identical thresholds -> suspect: fixed configs always benched BEFORE policies, so slow container
drift (thermals/clock/host contention) biases the comparison; consecutive-repeat SEs miss it.

Design: round-robin the 5 configs [fixed d=5,6,7,8, cumprob@0.05] within each cycle, rotating
the order every cycle; C cycles give C time-matched paired samples per config. Paired per-cycle
relative differences cancel drift. Prompts = interleaved ODD set (same as iid verify bench).

Run: PYTHONIOENCODING=utf-8 modal run modal_eagle_paired.py --n 10 --maxtok 128 --cycles 4
Out: /root/out/eagle_paired.json
"""
import os, time, json
import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("git")
         .pip_install("torch==2.6.0", "transformers==4.53.1", "accelerate==0.26.0",
                      "sentencepiece", "huggingface_hub", "fschat", "datasets", "numpy")
         .run_commands("git clone --depth 1 https://github.com/SafeAILab/EAGLE.git /root/EAGLE")
         .env({"PYTHONPATH": "/root/EAGLE"}))
app = modal.App("eagle-paired")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

MODELS = {
    "llama8b":  ("meta-llama/Llama-3.1-8B-Instruct",
                 "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"),
    "deepseek": ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                 "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"),
}
BASE = "meta-llama/Llama-3.1-8B-Instruct"
EA   = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
CUM_THR = 0.05


@app.function(image=image, gpu="H100", timeout=10800, volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(n, maxtok, cycles, model_key="llama8b"):
    import torch, numpy as np
    from eagle.model.ea_model import EaModel
    from datasets import load_dataset

    BASE, EA = MODELS[model_key]
    print(f"model={model_key}  target={BASE}  head={EA}", flush=True)
    model = EaModel.from_pretrained(use_eagle3=True, base_model_path=BASE, ea_model_path=EA,
                                    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                    device_map="auto", total_token=-1)
    model.eval()
    tok = model.get_tokenizer()
    ea  = model.ea_layer
    dev = model.base_model.device
    ea_cls = type(ea)
    MAXD = int(getattr(ea, "depth", 7))

    EL = {"cum_on": False, "thr": CUM_THR, "depths": []}

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

            if EL["cum_on"] and float(torch.exp(scores[0])) < EL["thr"]:
                break

        if EL["cum_on"]:
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

    def take_odd(x, m): return list(x)[1:2 * m:2]
    he = load_dataset("openai/openai_humaneval", split="test")["prompt"]
    gk = load_dataset("openai/gsm8k", "main", split="test")["question"]
    mb = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train").map(
        lambda x: {"p": x["prompt"][0]})["p"]
    workloads = {"humaneval": take_odd(he, n), "gsm8k": take_odd(gk, n),
                 "mt_bench": take_odd(mb, n)}

    def encode(p):
        enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt")
        return (enc["input_ids"] if hasattr(enc, "keys") else enc).to(dev)

    def bench_once(prompts):
        torch.cuda.synchronize(); t0 = time.time(); ntok = 0
        for p in prompts:
            ids = encode(p)
            with torch.no_grad():
                out = model.eagenerate(ids, temperature=0.0, max_new_tokens=maxtok)
            ntok += (out[0].shape[0] if out[0].dim() == 1 else out[0].shape[1]) - ids.shape[1]
        torch.cuda.synchronize()
        return ntok / (time.time() - t0)

    def set_config(cfg):
        kind, val = cfg
        if kind == "fixed":
            EL["cum_on"] = False; ea.depth = int(val)
        else:
            EL["cum_on"] = True; EL["thr"] = float(val); ea.depth = MAXD

    configs = [("fixed", 5), ("fixed", 6), ("fixed", 7), ("fixed", 8), ("cum", CUM_THR)]
    names = [f"{k}{v}" for k, v in configs]

    # warmup
    EL["cum_on"] = False; ea.depth = MAXD
    with torch.no_grad():
        model.eagenerate(encode("hi"), temperature=0.0, max_new_tokens=8)

    # round-robin cycles, rotating order each cycle to cancel intra-cycle drift
    tps = {w: {nm: [] for nm in names} for w in workloads}
    for c in range(cycles):
        order = configs[c % len(configs):] + configs[:c % len(configs)]
        for cfg in order:
            nm = f"{cfg[0]}{cfg[1]}"
            set_config(cfg)
            for w, prompts in workloads.items():
                t = bench_once(prompts)
                tps[w][nm].append(round(t, 2))
            print(f"cycle {c+1}/{cycles} {nm:10s} " +
                  " ".join(f"{w}:{tps[w][nm][-1]}" for w in workloads), flush=True)

    # paired analysis: per cycle, cumprob vs each fixed on the same cycle index
    summary = {}
    cum_nm = f"cum{CUM_THR}"
    for w in workloads:
        per_fixed = {}
        for kind, val in configs:
            if kind != "fixed":
                continue
            nm = f"fixed{val}"
            diffs = [100 * (a / b - 1) for a, b in zip(tps[w][cum_nm], tps[w][nm])]
            per_fixed[nm] = {"mean_gain_pct": round(float(np.mean(diffs)), 2),
                             "se": round(float(np.std(diffs) / np.sqrt(len(diffs))), 2),
                             "diffs": [round(d, 2) for d in diffs]}
        best_nm = max(per_fixed, key=lambda k: -per_fixed[k]["mean_gain_pct"])
        summary[w] = {"vs_each_fixed": per_fixed,
                      "vs_best_fixed": {"config": best_nm, **per_fixed[best_nm]}}
        print(f">>> {w:10s} cumprob vs best fixed ({best_nm}): "
              f"{per_fixed[best_nm]['mean_gain_pct']:+.2f}% ±{per_fixed[best_nm]['se']:.2f} "
              f"(paired, {cycles} cycles)", flush=True)

    md = round(float(np.mean(EL["depths"])), 2) if EL["depths"] else None
    print(f"cumprob realized depth: {md}", flush=True)

    os.makedirs("/root/out", exist_ok=True)
    suffix = "" if model_key == "llama8b" else f"_{model_key}"
    with open(f"/root/out/eagle_paired{suffix}.json", "w") as f:
        json.dump({"target": BASE, "model": model_key, "n": n, "maxtok": maxtok,
                   "cycles": cycles, "thr": CUM_THR, "tps": tps, "summary": summary,
                   "cum_depth_mean": md}, f, indent=2)
    vol.commit()
    return summary


@app.local_entrypoint()
def main(n: int = 10, maxtok: int = 128, cycles: int = 4, model: str = "llama8b"):
    print(f"paired bench: model={model} n={n} maxtok={maxtok} cycles={cycles} thr={CUM_THR}")
    print(json.dumps(run.remote(n, maxtok, cycles, model), indent=2))
