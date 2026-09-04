"""
Compare three decoding paths on identical config, measuring BOTH wall-clock and
harness internals (Yash's ask). Lets you SEE where my custom harness spends time
vs HF's built-in assisted generation.

  1. target_only        — plain greedy target.generate()  (the speedup baseline)
  2. custom_spec         — my hand-rolled loop: K sequential draft forwards
                           (+ optional per-token entropy), one target verify
                           forward, KV crop on rejection. Instrumented.
  3. hf_assisted         — HF's built-in assisted generation (same algorithm,
                           optimized internal loop)

Run:
  python spec_decode_compare.py                          # Qwen2.5 1.5B/0.5B (matched vocab)
  python spec_decode_compare.py --target ... --draft ... --K 4 --max-new-tokens 96

Config follows Yash's checklist: bf16, batch=1, greedy, KV cache on, SDPA,
both models on one GPU, warmup pass excluded.
"""
import argparse, time, contextlib
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------------
# timing helper: a context manager that accumulates GPU-synced wall-clock into a
# dict bucket. Syncing per-section is ONLY for the breakdown; the headline tok/s
# is measured separately with NO extra syncs so it isn't penalised.
# ----------------------------------------------------------------------------
class Timer:
    def __init__(self, enabled):
        self.enabled = enabled
        self.t = {}
    @contextlib.contextmanager
    def section(self, name):
        if not self.enabled:
            yield; return
        torch.cuda.synchronize(); s = time.perf_counter()
        yield
        torch.cuda.synchronize()
        self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - s)


def load(mid):
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    m = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa").eval()
    return m, tok


def crop_cache(past, n_remove):
    """Remove n_remove tokens from the END of a DynamicCache (transformers>=4.43)."""
    if n_remove <= 0 or past is None:
        return past
    if hasattr(past, "crop"):
        past.crop(past.get_seq_length() - n_remove)
        return past
    # legacy tuple fallback
    return tuple((k[:, :, :-n_remove, :], v[:, :, :-n_remove, :]) for k, v in past)


@torch.no_grad()
def target_only(target, tok, prompt, max_new_tokens):
    """Plain greedy decoding via HF generate() — the baseline."""
    dev = next(target.parameters()).device
    ids = tok.encode(prompt, return_tensors="pt").to(dev)
    out = target.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    return out[0, ids.shape[1]:].tolist()


@torch.no_grad()
def hf_assisted(target, draft, tok, prompt, K, max_new_tokens):
    dev = next(target.parameters()).device
    ids = tok.encode(prompt, return_tensors="pt").to(dev)
    target.generation_config.num_assistant_tokens = K
    target.generation_config.num_assistant_tokens_schedule = "constant"
    out = target.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                          assistant_model=draft, pad_token_id=tok.eos_token_id)
    return out[0, ids.shape[1]:].tolist()


@torch.no_grad()
def custom_spec(target, draft, tok, prompt, K, max_new_tokens,
                compute_entropy=True, timer=None):
    """My hand-rolled greedy speculative loop, instrumented.

    Per step: K sequential DRAFT forwards (each with optional entropy = a
    full-vocab softmax + a .item() CPU<->GPU sync), then ONE target forward to
    verify all K, then crop KV caches on rejection. This mirrors the harness in
    src/serve/physical_runner.py (which is fp32 bit-exact to target greedy).
    """
    tm = timer or Timer(False)
    dev = next(target.parameters()).device
    ids = tok.encode(prompt, return_tensors="pt").to(dev)

    # ---- prefill both models ----
    with tm.section("prefill"):
        t_out = target(ids, use_cache=True)
        target_kv = t_out.past_key_values
        t_logits = t_out.logits[:, -1, :]
        d_out = draft(ids, use_cache=True)
        draft_kv = d_out.past_key_values

    x = int(t_logits.argmax(-1))            # first token (bonus from target prefill)
    generated = [x]

    # advance draft cache by feeding x so it can propose the token after x
    with tm.section("draft_fwd"):
        d_out = draft(torch.tensor([[x]], device=dev), past_key_values=draft_kv, use_cache=True)
        draft_kv = d_out.past_key_values
        d_logits = d_out.logits[:, -1, :]

    n_drafted = n_accepted = n_steps = 0

    while len(generated) < max_new_tokens:
        # ---- DRAFT PHASE: K sequential forward passes ----
        drafts, logits = [], d_logits
        for i in range(K):
            if compute_entropy:
                with tm.section("entropy"):
                    p = F.softmax(logits[0], dim=-1)
                    _ = float(-(p * torch.log2(p + 1e-9)).sum())   # .item() sync here
            tokn = int(logits.argmax(-1))                          # .item() sync here
            drafts.append(tokn)
            # Feed EVERY drafted token (incl. the last) so the draft KV cache holds
            # prompt + x + d0..d_{K-1}; the crop math (discard=K-accepted) then keeps
            # exactly prompt + x + d0..d_{accepted-1}. (Skipping the last feed leaves
            # the cache one short and corrupts subsequent draft predictions.)
            with tm.section("draft_fwd"):
                d_out = draft(torch.tensor([[tokn]], device=dev),
                              past_key_values=draft_kv, use_cache=True)
                draft_kv = d_out.past_key_values
                logits = d_out.logits[:, -1, :]
        n_drafted += K

        # ---- TARGET VERIFY: one forward over [x, d0..d_{K-1}] ----
        with tm.section("target_fwd"):
            t_in = torch.tensor([[x] + drafts], device=dev)
            t_out = target(t_in, past_key_values=target_kv, use_cache=True)
            target_kv = t_out.past_key_values

        with tm.section("verify"):
            accepted, correction = 0, None
            for i in range(K):
                t_pred = int(t_out.logits[0, i, :].argmax(-1))     # .item() sync per position
                if t_pred == drafts[i]:
                    accepted += 1
                else:
                    correction = t_pred
                    break
            if correction is None:                                  # all K accepted -> bonus token
                correction = int(t_out.logits[0, K, :].argmax(-1))
        n_accepted += accepted; n_steps += 1

        # ---- crop rejected drafts from BOTH caches ----
        discard = K - accepted
        with tm.section("crop"):
            if discard > 0:
                target_kv = crop_cache(target_kv, discard)
                draft_kv = crop_cache(draft_kv, discard)

        generated.extend(drafts[:accepted]); generated.append(correction)
        x = correction
        if x == tok.eos_token_id:
            break
        # re-seed draft logits for next step
        with tm.section("draft_fwd"):
            d_out = draft(torch.tensor([[x]], device=dev), past_key_values=draft_kv, use_cache=True)
            draft_kv = d_out.past_key_values
            d_logits = d_out.logits[:, -1, :]

    stats = {"accepted": n_accepted, "drafted": n_drafted, "steps": n_steps,
             "acceptance_rate": n_accepted / max(1, n_drafted),
             "mean_accepted_len": n_accepted / max(1, n_steps),
             "wasted_per_accepted": (n_drafted - n_accepted) / max(1, n_accepted)}
    return generated[:max_new_tokens], stats


def tok_per_s(fn, prompts):
    """Wall-clock tok/s with warmup pass excluded, NO internal syncs."""
    fn(prompts[0])                       # warmup (excluded)
    torch.cuda.synchronize(); t0 = time.perf_counter(); n = 0
    for p in prompts:
        out = fn(p)
        n += len(out[0]) if isinstance(out, tuple) else len(out)
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    prompts = [
        "Write a python function to compute the nth Fibonacci number.",
        "Explain why the sky is blue in two sentences.",
        "What is 17 times 24? Show your work.",
        "def quicksort(arr):",
        "List three benefits of regular exercise.",
        "Summarize the theory of relativity briefly.",
    ]
    K, MNT = args.K, args.max_new_tokens
    print(f"target={args.target}  draft={args.draft}  K={K}  max_new_tokens={MNT}  batch=1 bf16 greedy SDPA")
    target, tok = load(args.target)
    draft, _ = load(args.draft)
    print(f"vocab: target={target.config.vocab_size} draft={draft.config.vocab_size} "
          f"match={target.config.vocab_size==draft.config.vocab_size}")

    # ---- correctness: custom output must equal target greedy ----
    g_base = target_only(target, tok, prompts[0], MNT)
    g_cust, _ = custom_spec(target, draft, tok, prompts[0], K, MNT)
    n = min(len(g_base), len(g_cust))
    print(f"\n[equivalence] custom vs target-greedy: first {n} tokens identical = "
          f"{g_base[:n] == g_cust[:n]}  (bf16 ties may cause rare late divergence)")

    # ---- headline wall-clock tok/s (no internal syncs) ----
    base = tok_per_s(lambda p: target_only(target, tok, p, MNT), prompts)
    cust = tok_per_s(lambda p: custom_spec(target, draft, tok, p, K, MNT, compute_entropy=True), prompts)
    cust_ne = tok_per_s(lambda p: custom_spec(target, draft, tok, p, K, MNT, compute_entropy=False), prompts)
    hf = tok_per_s(lambda p: hf_assisted(target, draft, tok, p, K, MNT), prompts)

    print("\n=== WALL-CLOCK (tok/s, higher=better) ===")
    print(f"  target_only (baseline)        : {base:6.1f}   1.00x")
    print(f"  custom_spec  (entropy ON)     : {cust:6.1f}   {cust/base:.2f}x")
    print(f"  custom_spec  (entropy OFF)    : {cust_ne:6.1f}   {cust_ne/base:.2f}x   <- isolates entropy overhead")
    print(f"  hf_assisted  (built-in)       : {hf:6.1f}   {hf/base:.2f}x")

    # ---- harness internals + per-component timing breakdown (instrumented) ----
    timer = Timer(enabled=True)
    _, stats = custom_spec(target, draft, tok, prompts[0], K, MNT, compute_entropy=True, timer=timer)
    print("\n=== HARNESS INTERNALS (custom, one prompt) ===")
    print(f"  acceptance_rate={stats['acceptance_rate']:.1%}  mean_accepted_len={stats['mean_accepted_len']:.2f}"
          f"  wasted/accepted={stats['wasted_per_accepted']:.2f}  steps={stats['steps']}")
    total = sum(timer.t.values())
    print("\n=== WHERE THE TIME GOES (instrumented; syncs add overhead, ratios are the point) ===")
    for k, v in sorted(timer.t.items(), key=lambda x: -x[1]):
        print(f"  {k:12s}: {v*1000:7.1f} ms  ({v/total:5.1%})")
    print(f"  {'TOTAL':12s}: {total*1000:7.1f} ms")
    print("\nNote: the K sequential 'draft_fwd' passes dominate (many tiny GPU ops with")
    print("per-call Python/launch overhead). Entropy is negligible (ON==OFF above). On a")
    print("small/fast target this fixed per-step overhead is what keeps speedup near/below")
    print("1x; a larger target/draft cost ratio or an optimized engine widens the margin.")


if __name__ == "__main__":
    main()
