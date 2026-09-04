"""
  pair A:  meta-llama/Llama-3.1-8B   target / meta-llama/Llama-3.2-1B  draft
  pair B:  facebook/opt-6.7b         target / facebook/opt-125m        draft
  pair C:  Qwen/Qwen3-8B            target / Qwen/Qwen3-0.6B           draft

For each pair, batch=1, bf16, SDPA, KV-cache on, warmup excluded, it reports:
  - vocab match (so HF assisted generation is valid)
  - acceptance rate, mean accepted length, wasted/accepted   (Yash's #1 number)
  - tok/s + speedup vs plain target.generate()   for: custom harness & HF assisted
  - the per-phase time breakdown (where the time goes)


"""
import os, time, json, contextlib
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    # latest transformers: Qwen3.5 (2026) needs a recent release; Llama-3.x & OPT
    # are still supported there too.
    .pip_install("torch", "transformers", "accelerate", "numpy")
    .env({"PYTHONUNBUFFERED": "1"})
)
app = modal.App("spec-compare-pairs")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

PAIRS = [
    {"name": "qwen3-8b / 0.6b",
     "target": ["Qwen/Qwen3-8B"],
     "draft":  ["Qwen/Qwen3-0.6B"]},
]
OUT_NAME = os.environ.get("SC_OUT", "spec_compare_qwen3.json")
K = int(os.environ.get("SC_K", "4"))
MNT = int(os.environ.get("SC_MNT", "96"))

PROMPTS = [
    "Write a python function to compute the nth Fibonacci number.",
    "Explain why the sky is blue in two sentences.",
    "What is 17 times 24? Show your work.",
    "def quicksort(arr):",
    "List three benefits of regular exercise.",
    "Summarize the theory of relativity briefly.",
    "Write a haiku about autumn leaves.",
    "What are the main causes of the French Revolution?",
    "Implement binary search in Python.",
    "Describe how a transformer neural network works.",
    "Translate 'good morning, how are you' into French.",
    "Give me three tips for writing clean code.",
    "What is the capital of Australia and why is it not Sydney?",
    "Write a SQL query to select all users older than 30.",
    "Explain recursion to a five year old.",
    "What happens during photosynthesis?",
]


@app.function(image=image, gpu="A100", timeout=3600, volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run():
    import traceback
    try:
        return _run()
    except Exception:
        print("FATAL:\n" + traceback.format_exc(), flush=True)
        raise


def _run():
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)

    # ---- timing helper (syncs per-section: only for the breakdown ratios) ----
    class Timer:
        def __init__(self, enabled): self.enabled, self.t = enabled, {}
        @contextlib.contextmanager
        def section(self, name):
            if not self.enabled:
                yield; return
            torch.cuda.synchronize(); s = time.perf_counter()
            yield
            torch.cuda.synchronize()
            self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - s)

    def try_load(cands):
        last = None
        for mid in cands:
            try:
                tok = AutoTokenizer.from_pretrained(mid)
                if tok.pad_token_id is None:
                    tok.pad_token_id = tok.eos_token_id
                m = AutoModelForCausalLM.from_pretrained(
                    mid, torch_dtype=torch.bfloat16, device_map="cuda",
                    attn_implementation="sdpa").eval()
                print(f"  loaded {mid}", flush=True)
                return m, tok, mid
            except Exception as e:
                last = repr(e)[:160]; print(f"  skip {mid}: {last}", flush=True)
        raise RuntimeError(f"none of {cands} loaded; last={last}")

    def crop_cache(past, n_remove):
        if n_remove <= 0 or past is None:
            return past
        if hasattr(past, "crop"):
            past.crop(past.get_seq_length() - n_remove)
            return past
        return tuple((k[:, :, :-n_remove, :], v[:, :, :-n_remove, :]) for k, v in past)

    @torch.no_grad()
    def target_only(tgt, tok, prompt):
        dev = next(tgt.parameters()).device
        ids = tok.encode(prompt, return_tensors="pt").to(dev)
        out = tgt.generate(ids, max_new_tokens=MNT, do_sample=False,  # do_sample=False == temperature 0
                           pad_token_id=tok.eos_token_id)
        return out[0, ids.shape[1]:].tolist()

    @torch.no_grad()
    def hf_assisted(tgt, drf, tok, prompt, k):
        dev = next(tgt.parameters()).device
        ids = tok.encode(prompt, return_tensors="pt").to(dev)
        tgt.generation_config.num_assistant_tokens = k
        tgt.generation_config.num_assistant_tokens_schedule = "constant"
        out = tgt.generate(ids, max_new_tokens=MNT, do_sample=False,  # greedy / temp 0
                           assistant_model=drf, pad_token_id=tok.eos_token_id)
        return out[0, ids.shape[1]:].tolist()

    @torch.no_grad()
    def custom_spec(tgt, drf, tok, prompt, k, timer=None):
        tm = timer or Timer(False)
        dev = next(tgt.parameters()).device
        ids = tok.encode(prompt, return_tensors="pt").to(dev)
        with tm.section("prefill"):
            t_out = tgt(ids, use_cache=True); target_kv = t_out.past_key_values
            t_logits = t_out.logits[:, -1, :]
            d_out = drf(ids, use_cache=True); draft_kv = d_out.past_key_values
        x = int(t_logits.argmax(-1))             # greedy first token (temp 0)
        generated = [x]
        with tm.section("draft_fwd"):
            d_out = drf(torch.tensor([[x]], device=dev), past_key_values=draft_kv, use_cache=True)
            draft_kv = d_out.past_key_values; d_logits = d_out.logits[:, -1, :]
        n_drafted = n_accepted = n_steps = 0
        while len(generated) < MNT:
            drafts, logits = [], d_logits
            for i in range(k):
                tokn = int(logits.argmax(-1))    # greedy draft (temp 0)
                drafts.append(tokn)
                with tm.section("draft_fwd"):
                    d_out = drf(torch.tensor([[tokn]], device=dev),
                                past_key_values=draft_kv, use_cache=True)
                    draft_kv = d_out.past_key_values; logits = d_out.logits[:, -1, :]
            n_drafted += k
            with tm.section("target_fwd"):
                t_in = torch.tensor([[x] + drafts], device=dev)
                t_out = tgt(t_in, past_key_values=target_kv, use_cache=True)
                target_kv = t_out.past_key_values
            with tm.section("verify"):
                accepted, correction = 0, None
                for i in range(k):
                    t_pred = int(t_out.logits[0, i, :].argmax(-1))   # greedy verify (temp 0)
                    if t_pred == drafts[i]:
                        accepted += 1
                    else:
                        correction = t_pred; break
                if correction is None:
                    correction = int(t_out.logits[0, k, :].argmax(-1))
            n_accepted += accepted; n_steps += 1
            discard = k - accepted
            with tm.section("crop"):
                if discard > 0:
                    target_kv = crop_cache(target_kv, discard)
                    draft_kv = crop_cache(draft_kv, discard)
            generated.extend(drafts[:accepted]); generated.append(correction)
            x = correction
            if x == tok.eos_token_id:
                break
            with tm.section("draft_fwd"):
                d_out = drf(torch.tensor([[x]], device=dev), past_key_values=draft_kv, use_cache=True)
                draft_kv = d_out.past_key_values; d_logits = d_out.logits[:, -1, :]
        stats = {"acceptance_rate": n_accepted / max(1, n_drafted),
                 "mean_accepted_len": n_accepted / max(1, n_steps),
                 "wasted_per_accepted": (n_drafted - n_accepted) / max(1, n_accepted),
                 "steps": n_steps}
        return generated[:MNT], stats

    def tok_per_s(fn, prompts):
        fn(prompts[0])                           # warmup excluded
        torch.cuda.synchronize(); t0 = time.perf_counter(); n = 0
        for p in prompts:
            out = fn(p)
            n += len(out[0]) if isinstance(out, tuple) else len(out)
        torch.cuda.synchronize()
        return n / (time.perf_counter() - t0)

    results = []
    for pair in PAIRS:
        print(f"\n########## {pair['name']} ##########", flush=True)
        print("loading target..."); tgt, tok, tgt_id = try_load(pair["target"])
        print("loading draft...");  drf, _, drf_id = try_load(pair["draft"])
        tv, dv = tgt.config.vocab_size, drf.config.vocab_size
        print(f"target={tgt_id} (vocab {tv})  draft={drf_id} (vocab {dv})  match={tv==dv}", flush=True)

        # equivalence (custom must equal target greedy)
        g_base = target_only(tgt, tok, PROMPTS[0])
        g_cust, _ = custom_spec(tgt, drf, tok, PROMPTS[0], K)
        n = min(len(g_base), len(g_cust))
        equiv = g_base[:n] == g_cust[:n]

        base = tok_per_s(lambda p: target_only(tgt, tok, p), PROMPTS)
        cust = tok_per_s(lambda p: custom_spec(tgt, drf, tok, p, K), PROMPTS)
        # HF built-in assisted generation (Yash's gut-check). Some newer stateful
        # architectures (e.g. Qwen3.5) reject this path; record it instead of crashing.
        hf, hf_err = None, None
        try:
            hf = tok_per_s(lambda p: hf_assisted(tgt, drf, tok, p, K), PROMPTS)
        except Exception as e:
            hf_err = repr(e)[:160]
            print(f"  hf_assisted unsupported: {hf_err}", flush=True)

        # acceptance stats + breakdown (single instrumented run)
        timer = Timer(enabled=True)
        _, stats = custom_spec(tgt, drf, tok, PROMPTS[0], K, timer=timer)
        breakdown = {kk: round(vv * 1000, 1) for kk, vv in
                     sorted(timer.t.items(), key=lambda x: -x[1])}

        rec = {
            "pair": pair["name"], "target": tgt_id, "draft": drf_id,
            "vocab_target": tv, "vocab_draft": dv, "vocab_match": tv == dv,
            "K": K, "max_new_tokens": MNT, "batch_size": 1,
            "temperature": 0, "dtype": "bfloat16", "attn": "sdpa",
            "equivalence_greedy": equiv,
            "acceptance_rate": round(stats["acceptance_rate"], 4),
            "mean_accepted_len": round(stats["mean_accepted_len"], 3),
            "wasted_per_accepted": round(stats["wasted_per_accepted"], 3),
            "tok_s_baseline": round(base, 2),
            "tok_s_custom": round(cust, 2),
            "tok_s_hf_assisted": round(hf, 2) if hf is not None else None,
            "speedup_custom": round(cust / base, 3),
            "speedup_hf_assisted": round(hf / base, 3) if hf is not None else None,
            "hf_assisted_error": hf_err,
            "time_breakdown_ms": breakdown,
        }
        results.append(rec)
        print(json.dumps(rec, indent=2), flush=True)
        del tgt, drf; torch.cuda.empty_cache()

    os.makedirs("/root/out", exist_ok=True)
    with open(f"/root/out/{OUT_NAME}", "w") as f:
        json.dump({"n_prompts": len(PROMPTS), "results": results}, f, indent=2)
    vol.commit()
    return results


@app.local_entrypoint()
def main():
    res = run.remote()
    print("\n" + "=" * 78)
    print(f"SPECULATIVE DECODING — A100, batch=1, temperature=0 (greedy), bf16, SDPA, K={K}, "
          f"max_new_tokens={MNT}")
    print("=" * 78)
    for r in res:
        print(f"\n### {r['pair']}")
        print(f"  target={r['target']}  draft={r['draft']}")
        print(f"  vocab match={r['vocab_match']} ({r['vocab_target']} vs {r['vocab_draft']})   "
              f"greedy-equivalent={r['equivalence_greedy']}")
        print(f"  acceptance_rate = {r['acceptance_rate']:.1%}   "
              f"mean_accepted_len = {r['mean_accepted_len']:.2f}   "
              f"wasted/accepted = {r['wasted_per_accepted']:.2f}")
        print(f"  {'method':22} {'tok/s':>8} {'speedup':>9}")
        print(f"  {'target.generate (base)':22} {r['tok_s_baseline']:>8} {'1.00x':>9}")
        print(f"  {'custom harness':22} {r['tok_s_custom']:>8} {str(r['speedup_custom'])+'x':>9}")
        if r['tok_s_hf_assisted'] is not None:
            print(f"  {'HF assisted (gut-check)':22} {r['tok_s_hf_assisted']:>8} {str(r['speedup_hf_assisted'])+'x':>9}")
        else:
            print(f"  {'HF assisted (gut-check)':22} {'UNSUPPORTED':>8}   (assisted-gen rejects this arch)")
        print(f"  where time goes (ms, syncs add overhead — ratios are the point):")
        for kk, vv in r["time_breakdown_ms"].items():
            print(f"      {kk:12} {vv:>9} ms")
    print("\n(saved to volume spec-dec-m5-results:/spec_compare_pairs.json)")
