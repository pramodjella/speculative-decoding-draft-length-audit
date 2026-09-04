"""Head-to-head controller benchmark on Llama-3.1-8B / 3.2-1B (custom harness).

Compares ALL controllers on identical setup (bf16, batch=1, greedy, KV cache, SDPA,
warmup excluded) with neural-draft signals (entropy + top1/top2 margin) available:
  fixed K=1/2/4/8 | SVIP entropy-threshold | eps-greedy | UCB (BanditSpec/UCBSpec) |
  EXP3 (BanditSpec/EXP3Spec) | History | TapOut-style | LinUCB(v1) | EntropyMarginLinUCB(ours)

Metrics per controller: tok/s, speedup vs best-fixed-K, acceptance, wasted/accepted,
mean accepted length. The harness is the validated greedy-spec logic (lossless).
"""
import os, sys, time, json
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers==4.46.3", "accelerate", "datasets", "numpy")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir("src", "/root/project/src")
)
app = modal.App("controller-benchmark")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

TGT = ["unsloth/Meta-Llama-3.1-8B-Instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct"]
DRF = ["unsloth/Llama-3.2-1B-Instruct"]
N = int(os.environ.get("CB_N", "20"))
MNT = int(os.environ.get("CB_MNT", "96"))


@app.function(image=image, gpu="A100", timeout=10800, volumes={"/root/out": vol})
def run():
    import traceback
    try:
        return _run()
    except Exception:
        print("FATAL:\n" + traceback.format_exc(), flush=True); raise


def _run():
    sys.path.insert(0, "/root/project/src")
    import torch, numpy as np
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from controllers import (EntropyThreshold, EpsilonGreedy, UCB,
                             AcceptanceHistoryController, LinUCBController)
    from controllers.linucb_v2 import EntropyMarginLinUCB
    from controllers.baselines_2025 import EXP3Spec, TapOutStyle

    def load_first(cands):
        for mid in cands:
            try:
                tok = AutoTokenizer.from_pretrained(mid)
                if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
                m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16,
                    device_map="cuda", attn_implementation="sdpa").eval()
                print("loaded", mid, flush=True); return m, tok
            except Exception as e:
                print("skip", mid, repr(e)[:100], flush=True)
        raise RuntimeError("no model loaded")

    target, tok = load_first(TGT)
    draft, _ = load_first(DRF)
    dev = next(target.parameters()).device
    vlog2 = float(np.log2(target.config.vocab_size))

    def take(x, n): return list(x)[:n]
    workloads = {
        "humaneval": take(load_dataset("openai/openai_humaneval", split="test")["prompt"], N),
        "gsm8k": take(load_dataset("openai/gsm8k", "main", split="test")["question"], N),
        "mt_bench": take(load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
                         .map(lambda x: {"p": x["prompt"][0]})["p"], N),
    }

    def make_controllers(seed=0):
        arms = (1, 2, 4, 8)
        return {
            "fixed_1": 1, "fixed_2": 2, "fixed_4": 4, "fixed_8": 8,
            "entropy_svip": EntropyThreshold(tau=0.8, max_len=8),
            "eps_greedy": EpsilonGreedy(eps=0.1, arms=arms, seed=seed),
            "ucb_banditspec": UCB(c=2.0, arms=arms),
            "exp3_banditspec": EXP3Spec(arms=arms, gamma=0.1, seed=seed),
            "history": AcceptanceHistoryController(window_size=10, arms=arms),
            "tapout_style": TapOutStyle(max_len=8, c=2.0),
            "linucb_v1": LinUCBController(arms=arms, alpha=1.0),
            "entropy_margin_linucb_OURS": EntropyMarginLinUCB(arms=arms, alpha=1.0, vocab_log2=vlog2),
        }

    @torch.no_grad()
    def baseline_ar(prompt):
        """Python greedy AR loop — same framework/overhead as spec_decode (honest baseline)."""
        ids = tok.encode(prompt, return_tensors="pt").to(dev)
        out = target(ids, use_cache=True)
        tkv = out.past_key_values
        x = int(out.logits[:, -1, :].argmax(-1))
        gen = [x]
        while len(gen) < MNT:
            out = target(torch.tensor([[x]], device=dev), past_key_values=tkv, use_cache=True)
            tkv = out.past_key_values
            x = int(out.logits[:, -1, :].argmax(-1))
            gen.append(x)
            if x == tok.eos_token_id:
                break
        return len(gen)

    @torch.no_grad()
    def hf_assisted_tps(prompts):
        """HF built-in speculative decoding (Yash's gut-check). Should track our harness ±5%."""
        nt = 0
        torch.cuda.synchronize(); s = time.perf_counter()
        for p in prompts:
            ids = tok.encode(p, return_tensors="pt").to(dev)
            out = target.generate(ids, max_new_tokens=MNT, do_sample=False,
                                  assistant_model=draft, pad_token_id=tok.eos_token_id)
            nt += out.shape[1] - ids.shape[1]
        torch.cuda.synchronize()
        return nt / (time.perf_counter() - s)

    def choose_K(ctrl, last_ent, last_mar, max_k=8):
        if isinstance(ctrl, int): return max(1, ctrl)
        if isinstance(ctrl, EntropyMarginLinUCB): return max(1, ctrl.choose(entropy=last_ent, margin=last_mar))
        if isinstance(ctrl, LinUCBController): return max(1, ctrl.choose(entropy=last_ent))
        if isinstance(ctrl, EntropyThreshold): return max_k
        if hasattr(ctrl, "choose"): return max(1, ctrl.choose())
        return 4

    def should_stop(ctrl, i, ent, mar):
        if hasattr(ctrl, "should_stop"): return ctrl.should_stop(i, ent, mar)
        if hasattr(ctrl, "tau"): return ent > ctrl.tau
        return False

    def update(ctrl, K, accepted, cyc):
        if isinstance(ctrl, int) or not hasattr(ctrl, "update"): return
        import inspect
        params = inspect.signature(ctrl.update).parameters
        if "cycle_time" in params:
            ctrl.update(K, accepted, cycle_time=cyc, K=K)
        elif "reward" in params:
            ctrl.update(K, (accepted + 1) / (K + 1))
        else:
            ctrl.update(K, accepted)

    def crop(past, n):
        """Remove last n tokens from KV cache. Handles DynamicCache and legacy tuple format."""
        if n <= 0:
            return past
        if hasattr(past, "crop"):                   # transformers DynamicCache
            past.crop(past.get_seq_length() - n)
            return past
        # legacy tuple: ((k0, v0), (k1, v1), ...) with k/v shape (B, H, T, D)
        return tuple((k[:, :, :-n, :], v[:, :, :-n, :]) for k, v in past)

    @torch.no_grad()
    def oracle_probe(prompt, max_k=8):
        """Draft max_k every step, record matched-prefix length m per step.
        From m we can post-hoc pick the optimal K per step (the ceiling)."""
        ids = tok.encode(prompt, return_tensors="pt").to(dev)
        to = target(ids, use_cache=True); tkv = to.past_key_values
        do = draft(ids, use_cache=True); dkv = do.past_key_values
        x = int(to.logits[:, -1, :].argmax(-1)); gen = [x]
        d = draft(torch.tensor([[x]], device=dev), past_key_values=dkv, use_cache=True)
        dkv = d.past_key_values; logits = d.logits[:, -1, :]
        ms = []
        while len(gen) < MNT:
            drafts = []
            for i in range(max_k):
                tokn = int(logits.argmax(-1)); drafts.append(tokn)
                d = draft(torch.tensor([[tokn]], device=dev), past_key_values=dkv, use_cache=True)
                dkv = d.past_key_values; logits = d.logits[:, -1, :]
            tin = torch.tensor([[x] + drafts], device=dev)
            to = target(tin, past_key_values=tkv, use_cache=True); tkv = to.past_key_values
            m, corr = 0, None
            for i in range(max_k):
                tp = int(to.logits[0, i, :].argmax(-1))
                if tp == drafts[i]: m += 1
                else: corr = tp; break
            if corr is None: corr = int(to.logits[0, max_k, :].argmax(-1))
            ms.append(m)
            disc = max_k - m
            if disc > 0: tkv = crop(tkv, disc); dkv = crop(dkv, disc)
            gen.extend(drafts[:m]); gen.append(corr)
            x = corr
            if x == tok.eos_token_id: break
            d = draft(torch.tensor([[x]], device=dev), past_key_values=dkv, use_cache=True)
            dkv = d.past_key_values; logits = d.logits[:, -1, :]
        return ms

    ARMS = (1, 2, 4, 8)
    def opt_k(m):
        """Optimal arm K for observed matched-prefix m: max committed-per-forward."""
        return max(ARMS, key=lambda K: (min(m, K) + 1) / (K + 1))

    @torch.no_grad()
    def spec_decode(prompt, ctrl):
        ids = tok.encode(prompt, return_tensors="pt").to(dev)
        if hasattr(ctrl, "set_max_steps"): ctrl.set_max_steps(MNT)
        if hasattr(ctrl, "reset_episode"): ctrl.reset_episode()
        to = target(ids, use_cache=True); tkv = to.past_key_values
        do = draft(ids, use_cache=True); dkv = do.past_key_values
        x = int(to.logits[:, -1, :].argmax(-1)); gen = [x]
        d = draft(torch.tensor([[x]], device=dev), past_key_values=dkv, use_cache=True)
        dkv = d.past_key_values; dlog = d.logits[:, -1, :]
        last_ent, last_mar = 0.0, 0.5
        n_acc = n_draft = n_steps = 0
        while len(gen) < MNT:
            t0 = time.perf_counter()
            K = choose_K(ctrl, last_ent, last_mar)
            drafts, logits = [], dlog
            for i in range(K):
                p = F.softmax(logits[0], dim=-1)
                ent = float(-(p * torch.log2(p + 1e-9)).sum())
                top2 = torch.topk(p, 2).values
                mar = float(top2[0] - top2[1])
                if i == 0: last_ent, last_mar = ent, mar
                tokn = int(logits.argmax(-1)); drafts.append(tokn)
                if should_stop(ctrl, i, ent, mar): break
                d = draft(torch.tensor([[tokn]], device=dev), past_key_values=dkv, use_cache=True)
                dkv = d.past_key_values; logits = d.logits[:, -1, :]
            Ka = len(drafts); n_draft += Ka
            tin = torch.tensor([[x] + drafts], device=dev)
            to = target(tin, past_key_values=tkv, use_cache=True); tkv = to.past_key_values
            acc, corr = 0, None
            for i in range(Ka):
                tp = int(to.logits[0, i, :].argmax(-1))
                if tp == drafts[i]: acc += 1
                else: corr = tp; break
            if corr is None: corr = int(to.logits[0, Ka, :].argmax(-1))
            disc = Ka - acc
            if disc > 0: tkv = crop(tkv, disc); dkv = crop(dkv, disc)
            gen.extend(drafts[:acc]); gen.append(corr)
            n_acc += acc; n_steps += 1
            cyc = time.perf_counter() - t0
            update(ctrl, Ka, acc, cyc)
            x = corr
            if x == tok.eos_token_id: break
            d = draft(torch.tensor([[x]], device=dev), past_key_values=dkv, use_cache=True)
            dkv = d.past_key_values; dlog = d.logits[:, -1, :]
        return {"tokens": len(gen), "acc": n_acc, "draft": n_draft, "steps": n_steps}

    SEEDS = [int(s) for s in os.environ.get("CB_SEEDS", "0,1,2").split(",")]
    raw = []          # one dict per (seed, workload, controller)
    oracle_rows = []  # one per workload (deterministic)

    for w, prompts in workloads.items():
        print(f"\n===== {w} =====", flush=True)

        # Honest baseline: Python AR loop (same framework as spec_decode)
        baseline_ar(prompts[0])                         # warmup
        torch.cuda.synchronize(); s = time.perf_counter(); nt = 0
        for p in prompts: nt += baseline_ar(p)
        torch.cuda.synchronize(); base_tps = nt / (time.perf_counter() - s)
        print(f"python_ar_baseline {base_tps:.1f} tok/s", flush=True)

        # Yash's gut-check: HF built-in assisted generation
        try:
            hf_tps = hf_assisted_tps(prompts)
            print(f"hf_assisted_gen {hf_tps:.1f} tok/s ({hf_tps/base_tps:.3f}x)", flush=True)
            raw.append({"seed": -1, "workload": w, "controller": "hf_assisted_gen",
                        "tps": hf_tps, "base_tps": base_tps, "speedup": hf_tps / base_tps,
                        "E": None, "acc": None})
        except Exception as e:
            print(f"hf_assisted_gen failed: {e}", flush=True)

        # Oracle ceiling (deterministic): probe match-prefix m per step, pick optimal K
        oc_committed = oc_fwd = 0
        kdist = {k: 0 for k in ARMS}
        for p in prompts:
            for m in oracle_probe(p):
                K = opt_k(m); kdist[K] += 1
                oc_committed += min(m, K) + 1
                oc_fwd += K + 1
        E_oracle = oc_committed / max(1, oc_fwd)
        oracle_rows.append({"workload": w, "E_oracle": round(E_oracle, 4),
                            "base_tps": base_tps, "opt_k_dist": kdist})
        print(f"ORACLE E={E_oracle:.4f}  opt_K_dist={kdist}", flush=True)

        for seed in SEEDS:
            ctrls = make_controllers(seed)
            for name, ctrl in ctrls.items():
                spec_decode(prompts[0], ctrl)               # warmup
                torch.cuda.synchronize(); s = time.perf_counter()
                tot_tok = tot_acc = tot_draft = tot_steps = 0
                for p in prompts:
                    r = spec_decode(p, ctrl)
                    tot_tok += r["tokens"]; tot_acc += r["acc"]; tot_draft += r["draft"]; tot_steps += r["steps"]
                torch.cuda.synchronize(); tps = tot_tok / (time.perf_counter() - s)
                committed = tot_acc + tot_steps           # accepted + 1 bonus/step
                fwd = tot_draft + tot_steps                # draft fwds + verifies
                E = committed / max(1, fwd)
                raw.append({"seed": seed, "workload": w, "controller": name,
                            "tps": tps, "base_tps": base_tps, "speedup": tps / base_tps,
                            "E": E, "acc": tot_acc / max(1, tot_draft)})
            print(f"  seed {seed} done", flush=True)

    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/controller_benchmark.json", "w") as f:
        json.dump({"target": "Llama-3.1-8B", "draft": "Llama-3.2-1B",
                   "seeds": SEEDS, "raw": raw, "oracle": oracle_rows}, f, indent=2)
    vol.commit()
    return {"raw": raw, "oracle": oracle_rows}


@app.local_entrypoint()
def main():
    import collections, statistics
    out = run.remote()
    raw, oracle = out["raw"], out["oracle"]
    E_oracle = {o["workload"]: o["E_oracle"] for o in oracle}

    print("\n===== Controller benchmark (Llama-3.1-8B / 3.2-1B, honest Python-AR baseline) =====")

    # aggregate over seeds: (workload, controller) -> list of speedups / E
    agg = collections.defaultdict(lambda: {"sp": [], "E": [], "acc": []})
    for r in raw:
        k = (r["workload"], r["controller"])
        agg[k]["sp"].append(r["speedup"])
        if r["E"] is not None: agg[k]["E"].append(r["E"])
        if r["acc"] is not None: agg[k]["acc"].append(r["acc"])

    def ms(xs):
        if not xs: return (None, None)
        return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)

    byw = collections.defaultdict(list)
    for (w, c) in agg: byw[w].append(c)

    for w in byw:
        ctrls = byw[w]
        spd = {c: ms(agg[(w, c)]["sp"]) for c in ctrls}
        eff = {c: ms(agg[(w, c)]["E"]) for c in ctrls}
        fixed = {c: spd[c][0] for c in ctrls if c.startswith("fixed_")}
        bestfix = max(fixed, key=fixed.get) if fixed else None
        Eo = E_oracle.get(w)
        print(f"\n--- {w}  (best fixed = {bestfix} @ {fixed[bestfix]:.3f}x | oracle E={Eo:.3f}) ---")
        print(f"  {'controller':28s} {'speedup(mean±sd)':>20s}  {'E':>6s}  {'%oracle':>7s}  {'vs_bestfix':>10s}")
        for c in sorted(ctrls, key=lambda c: -(spd[c][0] or 0)):
            sm, ss = spd[c]
            em, _ = eff[c]
            pct = f"{em/Eo*100:4.0f}%" if (em and Eo) else "  n/a"
            gap = (sm - fixed[bestfix]) / fixed[bestfix] * 100 if bestfix else 0.0
            estr = f"{em:.3f}" if em else " n/a "
            print(f"  {c:28s} {sm:.3f}±{ss:.3f}{'':>6s}  {estr}  {pct:>7s}  {gap:+9.1f}%")
