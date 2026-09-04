"""WALL-CLOCK test of adaptive draft length in the EAGLE-3 reference repo (plain PyTorch,
eager — no CUDA-graph penalty, unlike vLLM which has no per-step hook at all).

Mechanism: EAGLE's ea_layer.topK_genrate drafts to `self.depth` (read once at the top of the
call). We set self.depth per generation-step from a cheap probe on the step's SEED hidden
state, so the draft tree is built consistently for the chosen depth (no fragile mid-loop break).

Phases:
  1) CAPTURE  : run at full depth, record (seed hidden state, this step's accept_length).
  2) TRAIN    : PCA-50 + ridge predictor  seed_hidden -> accept_length.
  3) BENCHMARK: measure real tok/s for
        - fixed depth d (sweep) through the SAME wrapper codepath (probe runs, output ignored)
        - adaptive depth = clip(round(pred_accept_len)+1, 1, MAXD)
     Fair comparison: both paths run the probe forward, so per-step control overhead is equal;
     the only difference is how many draft iterations run. Reported: adaptive tok/s vs best fixed.

Validated recipe: transformers 4.53.1 / torch 2.6.0 (same as modal_eagle_ttsweep.py).

Run (validation): PYTHONIOENCODING=utf-8 modal run modal_eagle_wallclock.py --n 6 --maxtok 96
Run (full):       PYTHONIOENCODING=utf-8 modal run modal_eagle_wallclock.py --n 30 --maxtok 160
Output: /root/out/eagle_wallclock.json  (download: modal volume get spec-dec-m5-results eagle_wallclock.json results/)
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
app = modal.App("eagle-wallclock")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

BASE = "meta-llama/Llama-3.1-8B-Instruct"
EA   = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


@app.function(image=image, gpu="H100", timeout=7200, volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(n, maxtok, depths):
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
    MAXD = int(getattr(ea, "depth", 5))
    print(f"ea_layer default depth (MAXD) = {MAXD}", flush=True)

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

    # ── hooks: seed hidden per step (topK_genrate) + accept_length per step ──────────── #
    import eagle.model.ea_model as EM
    LOG = {"seed": [], "acc": [], "ctrl": None, "probe_on": False, "fixed_depth": None,
           "adaptive_depths": []}

    ea_cls = type(ea)
    orig_topk = ea_cls.topK_genrate
    def topk_wrap(self, hidden_states, input_ids, head, logits_processor=None, *a, **k):
        # seed hidden = last row of hidden_states (the token we draft from)
        h = hidden_states
        if torch.is_tensor(h):
            seed = h.detach().reshape(-1, h.shape[-1])[-1].float().cpu().numpy()
        else:
            seed = None
        # decide depth for THIS step
        if LOG["probe_on"] and LOG["ctrl"] is not None and seed is not None:
            self.depth = int(LOG["ctrl"](seed))
            LOG["adaptive_depths"].append(self.depth)
        elif LOG["fixed_depth"] is not None:
            self.depth = int(LOG["fixed_depth"])
        # capture mode records the seed (full-depth run)
        if seed is not None and LOG.get("capture"):
            LOG["seed"].append(seed)
        return orig_topk(self, hidden_states, input_ids, head, logits_processor, *a, **k)
    ea_cls.topK_genrate = topk_wrap

    orig_eval = EM.evaluate_posterior
    def eval_wrap(*a, **k):
        out = orig_eval(*a, **k)
        # out = (best_candidate, accept_length, ...) ; accept_length is a tensor/int
        try:
            al = out[1]
            al = int(al.item()) if torch.is_tensor(al) else int(al)
            if LOG.get("capture"):
                LOG["acc"].append(al)
        except Exception:
            pass
        return out
    EM.evaluate_posterior = eval_wrap

    def warmup():
        with torch.no_grad():
            model.eagenerate(encode("hi"), temperature=0.0, max_new_tokens=8)

    # ── PHASE 1: capture (full depth) ───────────────────────────────────────────────── #
    # BUGFIX (audit 2026-07-03): pair seeds<->accept_lengths PER PROMPT, not by global flat
    # index. Each prompt leaves one dangling seed (the final drafted tree is never verified),
    # so global index-pairing drifted by one per preceding prompt, mispairing ~all data after
    # prompt 1 and invalidating the original "seed has no signal" readout.
    LOG["fixed_depth"] = MAXD; LOG["probe_on"] = False; LOG["capture"] = True
    warmup()
    cap_prompts = [p for w in workloads for p in workloads[w]]
    seed_list, acc_list = [], []
    dangling = 0
    for p in cap_prompts:
        LOG["seed"].clear(); LOG["acc"].clear()
        with torch.no_grad():
            model.eagenerate(encode(p), temperature=0.0, max_new_tokens=maxtok)
        mp = min(len(LOG["seed"]), len(LOG["acc"]))   # within-prompt truncation only
        dangling += len(LOG["seed"]) - mp
        seed_list.extend(LOG["seed"][:mp]); acc_list.extend(LOG["acc"][:mp])
    LOG["capture"] = False
    m = len(acc_list)
    print(f"capture: {m} PER-PROMPT paired samples ({dangling} dangling seeds dropped)", flush=True)
    seeds = np.stack(seed_list).astype("float32")
    accs  = np.array(acc_list, dtype="float32")
    print(f"accept_length dist: mean={accs.mean():.2f} min={accs.min()} max={accs.max()}", flush=True)

    # ── PHASE 2: train PCA-50 + ridge  seed -> accept_length ────────────────────────── #
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    # split by generation order (first 80% train)
    ntr = int(0.8 * m)
    sc = StandardScaler().fit(seeds[:ntr])
    Xtr = sc.transform(seeds[:ntr]); Xte = sc.transform(seeds[ntr:])
    pca = PCA(n_components=min(50, Xtr.shape[1])).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    ridge = Ridge(alpha=10.0).fit(Ztr, accs[:ntr])
    pred_te = ridge.predict(Zte)
    from numpy import corrcoef
    r = float(corrcoef(pred_te, accs[ntr:])[0, 1]) if m - ntr > 2 else float("nan")
    print(f"seed->accept_len probe: test corr={r:.3f}", flush=True)

    def controller(seed_vec):
        z = pca.transform(sc.transform(seed_vec.reshape(1, -1)))
        pred = float(ridge.predict(z)[0])
        return max(1, min(MAXD, int(round(pred)) + 1))

    # ── PHASE 3: benchmark tok/s ────────────────────────────────────────────────────── #
    def bench(prompts):
        torch.cuda.synchronize(); t0 = time.time(); ntok = 0
        for p in prompts:
            ids = encode(p)
            with torch.no_grad():
                out = model.eagenerate(ids, temperature=0.0, max_new_tokens=maxtok)
            ntok += (out[0].shape[0] if out[0].dim() == 1 else out[0].shape[1]) - ids.shape[1]
        torch.cuda.synchronize()
        return ntok / (time.time() - t0), ntok

    results = {"MAXD": MAXD, "seed_probe_corr": r, "fixed": {}, "adaptive": {}}
    warmup()
    # fixed-depth sweep (probe off, but still runs the wrapper codepath)
    for d in depths:
        LOG["probe_on"] = False; LOG["fixed_depth"] = d
        per_w = {}
        for w, prompts in workloads.items():
            tps, ntok = bench(prompts)
            per_w[w] = round(tps, 2)
            print(f"  FIXED depth={d} {w:10s} {tps:7.1f} tok/s", flush=True)
        results["fixed"][d] = per_w
    # adaptive (probe on)
    LOG["probe_on"] = True; LOG["fixed_depth"] = None
    results["adaptive_mean_depth"] = {}
    for w, prompts in workloads.items():
        LOG["adaptive_depths"].clear()
        tps, ntok = bench(prompts)
        results["adaptive"][w] = round(tps, 2)
        md = round(float(np.mean(LOG["adaptive_depths"])), 2) if LOG["adaptive_depths"] else None
        sd = round(float(np.std(LOG["adaptive_depths"])), 2) if LOG["adaptive_depths"] else None
        results["adaptive_mean_depth"][w] = {"mean": md, "std": sd}
        print(f"  ADAPTIVE (probe)  {w:10s} {tps:7.1f} tok/s  (mean depth={md}±{sd})", flush=True)

    # summarize: best fixed per workload vs adaptive
    summary = {}
    for w in workloads:
        best_d = max(results["fixed"], key=lambda d: results["fixed"][d][w])
        best_fixed = results["fixed"][best_d][w]
        adap = results["adaptive"][w]
        summary[w] = {"best_fixed_depth": best_d, "best_fixed_tps": best_fixed,
                      "adaptive_tps": adap, "gain_pct": round(100 * (adap / best_fixed - 1), 2)}
        print(f"  >>> {w:10s} best_fixed(d={best_d})={best_fixed} tok/s | "
              f"adaptive={adap} tok/s | gain={summary[w]['gain_pct']:+.2f}%", flush=True)
    results["summary"] = summary

    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/eagle_wallclock.json", "w") as f:
        json.dump({"target": BASE, "n": n, "maxtok": maxtok, **results}, f, indent=2)
    vol.commit()
    return summary


@app.local_entrypoint()
def main(n: int = 30, maxtok: int = 160, depths: str = "2,3,4,5"):
    ds = [int(x) for x in depths.split(",")]
    print(f"EAGLE-3 wall-clock: n={n} maxtok={maxtok} fixed-depths={ds}")
    print(json.dumps(run.remote(n, maxtok, ds), indent=2))
