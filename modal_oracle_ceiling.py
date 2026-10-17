"""Corrected oracle ceiling for adaptive draft length (real wall-clock replay).

The earlier in-benchmark oracle weighted every forward equally — but a draft
forward is the 1B model and a verify is the 8B model (~8x cheaper draft). That
trivially collapsed the optimum to K=1. This fixes it two ways:

  1. Measure the REAL cost ratio r = c_draft / c_verify (median single-token fwds).
  2. Oracle is a REAL wall-clock replay, not an analytic model:
       pass A (probe): draft max_k, record matched-prefix m per step (deterministic).
       pass B (replay): re-run the SAME prompt choosing K* = argmax (min(m,K)+1)/(K*r+1)
                        per step, TIMED honestly. Greedy => pass A's m's apply exactly.

  Reported oracle tok/s is therefore a measured upper bound (perfect per-step K),
  directly comparable on the same GPU to python_ar baseline and best fixed-K.
"""
import os, sys, time, json
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers==4.46.3", "accelerate", "datasets", "numpy")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir("src", "/root/project/src")
)
app = modal.App("oracle-ceiling")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

TGT = ["unsloth/Meta-Llama-3.1-8B-Instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct"]
DRF = ["unsloth/Llama-3.2-1B-Instruct"]
N = int(os.environ.get("CB_N", "20"))
MNT = int(os.environ.get("CB_MNT", "96"))
ARMS = (1, 2, 4, 8)


@app.function(image=image, gpu="A100", timeout=3600, volumes={"/root/out": vol})
def run():
    import traceback
    try:
        return _run()
    except Exception:
        print("FATAL:\n" + traceback.format_exc(), flush=True); raise


def _run():
    sys.path.insert(0, "/root/project/src")
    import torch, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

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

    def take(x, n): return list(x)[:n]
    workloads = {
        "humaneval": take(load_dataset("openai/openai_humaneval", split="test")["prompt"], N),
        "gsm8k": take(load_dataset("openai/gsm8k", "main", split="test")["question"], N),
        "mt_bench": take(load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
                         .map(lambda x: {"p": x["prompt"][0]})["p"], N),
    }

    def crop(past, n):
        if n <= 0: return past
        if hasattr(past, "crop"):
            past.crop(past.get_seq_length() - n); return past
        return tuple((k[:, :, :-n, :], v[:, :, :-n, :]) for k, v in past)

    # ---- measure real cost ratio r = c_draft / c_verify ----
    @torch.no_grad()
    def time_fwd(model, reps=30):
        ids = tok.encode("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to(dev)
        o = model(ids, use_cache=True); kv = o.past_key_values
        x = int(o.logits[:, -1, :].argmax(-1))
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            o = model(torch.tensor([[x]], device=dev), past_key_values=kv, use_cache=True)
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
            kv = o.past_key_values; x = int(o.logits[:, -1, :].argmax(-1))
        ts.sort(); return ts[len(ts) // 2]                 # median

    c_draft = time_fwd(draft); c_verify = time_fwd(target)
    r = c_draft / c_verify
    print(f"c_draft={c_draft*1e3:.3f}ms  c_verify={c_verify*1e3:.3f}ms  r={r:.4f}", flush=True)

    def opt_k(m):
        return max(ARMS, key=lambda K: (min(m, K) + 1) / (K * r + 1))

    @torch.no_grad()
    def baseline_ar(prompt):
        ids = tok.encode(prompt, return_tensors="pt").to(dev)
        o = target(ids, use_cache=True); kv = o.past_key_values
        x = int(o.logits[:, -1, :].argmax(-1)); gen = [x]
        while len(gen) < MNT:
            o = target(torch.tensor([[x]], device=dev), past_key_values=kv, use_cache=True)
            kv = o.past_key_values; x = int(o.logits[:, -1, :].argmax(-1)); gen.append(x)
            if x == tok.eos_token_id: break
        return len(gen)

    @torch.no_grad()
    def probe(prompt, max_k=8):
        """Record matched-prefix m per step (deterministic under greedy)."""
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
            ms.append(m); disc = max_k - m
            if disc > 0: tkv = crop(tkv, disc); dkv = crop(dkv, disc)
            gen.extend(drafts[:m]); gen.append(corr); x = corr
            if x == tok.eos_token_id: break
            d = draft(torch.tensor([[x]], device=dev), past_key_values=dkv, use_cache=True)
            dkv = d.past_key_values; logits = d.logits[:, -1, :]
        return ms

    @torch.no_grad()
    def spec_replay(prompt, k_of_step):
        """Spec decode choosing K from a precomputed per-step schedule (or constant int).
        Timed honestly; returns (tokens, seconds)."""
        ids = tok.encode(prompt, return_tensors="pt").to(dev)
        to = target(ids, use_cache=True); tkv = to.past_key_values
        do = draft(ids, use_cache=True); dkv = do.past_key_values
        x = int(to.logits[:, -1, :].argmax(-1)); gen = [x]
        d = draft(torch.tensor([[x]], device=dev), past_key_values=dkv, use_cache=True)
        dkv = d.past_key_values; dlog = d.logits[:, -1, :]
        step = 0
        torch.cuda.synchronize(); t0 = time.perf_counter()
        while len(gen) < MNT:
            K = k_of_step(step)
            drafts, logits = [], dlog
            for i in range(K):
                tokn = int(logits.argmax(-1)); drafts.append(tokn)
                d = draft(torch.tensor([[tokn]], device=dev), past_key_values=dkv, use_cache=True)
                dkv = d.past_key_values; logits = d.logits[:, -1, :]
            tin = torch.tensor([[x] + drafts], device=dev)
            to = target(tin, past_key_values=tkv, use_cache=True); tkv = to.past_key_values
            acc, corr = 0, None
            for i in range(K):
                tp = int(to.logits[0, i, :].argmax(-1))
                if tp == drafts[i]: acc += 1
                else: corr = tp; break
            if corr is None: corr = int(to.logits[0, K, :].argmax(-1))
            disc = K - acc
            if disc > 0: tkv = crop(tkv, disc); dkv = crop(dkv, disc)
            gen.extend(drafts[:acc]); gen.append(corr); step += 1; x = corr
            if x == tok.eos_token_id: break
            d = draft(torch.tensor([[x]], device=dev), past_key_values=dkv, use_cache=True)
            dkv = d.past_key_values; dlog = d.logits[:, -1, :]
        torch.cuda.synchronize()
        return len(gen), time.perf_counter() - t0

    out = {"r": r, "c_draft_ms": c_draft * 1e3, "c_verify_ms": c_verify * 1e3, "workloads": {}}
    for w, prompts in workloads.items():
        print(f"\n===== {w} =====", flush=True)
        # honest baseline
        baseline_ar(prompts[0])
        torch.cuda.synchronize(); s = time.perf_counter(); nt = 0
        for p in prompts: nt += baseline_ar(p)
        base_tps = nt / (time.perf_counter() - s)

        # capture per-step match-run traces at WIDE max_k=32 for offline r-sweep.
        # greedy acceptance is position-local, so these traces let us simulate any
        # K-policy at any cost-ratio r on CPU (no more GPU runs needed).
        m_traces = [probe(p, max_k=32) for p in prompts]

        # oracle: probe -> optimal per-step schedule -> timed replay (arms only)
        schedules = [[opt_k(m) for m in probe(p)] for p in prompts]
        spec_replay(prompts[0], lambda s: schedules[0][s] if s < len(schedules[0]) else 1)  # warmup
        torch.cuda.synchronize(); s0 = time.perf_counter(); otok = 0
        kcount = {k: 0 for k in ARMS}
        for p, sched in zip(prompts, schedules):
            for k in sched: kcount[k] += 1
            tk, _ = spec_replay(p, lambda st, sc=sched: sc[st] if st < len(sc) else 1)
            otok += tk
        oracle_tps = otok / (time.perf_counter() - s0)

        # fixed-K reference on same instance
        fixed = {}
        for K in ARMS:
            spec_replay(prompts[0], lambda s, K=K: K)
            torch.cuda.synchronize(); sf = time.perf_counter(); ft = 0
            for p in prompts: tk, _ = spec_replay(p, lambda s, K=K: K); ft += tk
            fixed[K] = ft / (time.perf_counter() - sf)

        best_fixed = max(fixed, key=fixed.get)
        oc = {"base_tps": round(base_tps, 2), "oracle_tps": round(oracle_tps, 2),
              "oracle_speedup": round(oracle_tps / base_tps, 3),
              "fixed_tps": {k: round(v, 2) for k, v in fixed.items()},
              "fixed_speedup": {k: round(v / base_tps, 3) for k, v in fixed.items()},
              "best_fixed_K": best_fixed,
              "oracle_gain_over_bestfix_pct": round((oracle_tps / fixed[best_fixed] - 1) * 100, 1),
              "opt_k_dist": kcount,
              "m_traces": m_traces,                 # per-prompt match-run lengths (max_k=32)
              "probe_max_k": 32}
        out["workloads"][w] = oc
        print(json.dumps(oc, indent=2), flush=True)

    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/oracle_ceiling.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return out


@app.local_entrypoint()
def main():
    o = run.remote()
    print(f"\n===== Oracle ceiling (r=c_draft/c_verify={o['r']:.3f}) =====")
    for w, oc in o["workloads"].items():
        print(f"\n--- {w} ---")
        print(f"  baseline (python_ar)   {oc['base_tps']:>7} tok/s   1.000x")
        for k in (1, 2, 4, 8):
            star = "  <- best fixed" if k == oc["best_fixed_K"] else ""
            print(f"  fixed_{k:<2}               {oc['fixed_tps'][str(k)] if str(k) in oc['fixed_tps'] else oc['fixed_tps'][k]:>7} tok/s   {oc['fixed_speedup'][str(k)] if str(k) in oc['fixed_speedup'] else oc['fixed_speedup'][k]:.3f}x{star}")
        print(f"  ORACLE (perfect K)     {oc['oracle_tps']:>7} tok/s   {oc['oracle_speedup']:.3f}x")
        print(f"  => oracle headroom over best-fixed: {oc['oracle_gain_over_bestfix_pct']:+.1f}%")
        print(f"  => optimal-K distribution: {oc['opt_k_dist']}")
