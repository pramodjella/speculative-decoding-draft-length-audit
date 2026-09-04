"""E0.2 — Lossless / output-equivalence proof.

Speculative decoding with greedy verification must reproduce the target model's
own greedy decoding *bit-for-bit*. This module runs both the honest Python
autoregressive baseline and the speculative runner on the same prompts and
asserts their generated token-id sequences match on the overlapping prefix.

A passing result is the paper's "free speedup" guarantee: the controller changes
only speed, never the output.
"""
import argparse


def compare_sequences(baseline_ids, spec_ids):
    """Compare two token-id sequences on their overlapping prefix.

    Returns (is_equal, n_compared, first_mismatch_idx_or_-1, n_token_matches).
    """
    n = min(len(baseline_ids), len(spec_ids))
    n_match = sum(1 for i in range(n) if baseline_ids[i] == spec_ids[i])
    first = next((i for i in range(n) if baseline_ids[i] != spec_ids[i]), -1)
    return (first == -1), n, first, n_match


def verify(target_model, draft_model, tokenizer, prompts, max_new_tokens=64, controllers=None):
    """Run equivalence checks across a set of controllers and prompts.

    Returns a dict: {policy_name: {"passed": int, "total": int, "mismatches": [...]}}.
    """
    from serve.physical_runner import PhysicalSpeculativeRunner, PythonAutoregressiveBaseline
    from controllers import EntropyThreshold, EpsilonGreedy, UCB

    if controllers is None:
        controllers = {
            "fixed_4": 4,
            "entropy": EntropyThreshold(tau=0.8, max_len=8),
            "ucb": UCB(c=2.0, arms=(1, 2, 3, 4, 5, 6, 7, 8)),
            "epsilon_greedy": EpsilonGreedy(eps=0.1, arms=(1, 2, 4, 8)),
        }

    baseline = PythonAutoregressiveBaseline(target_model, tokenizer)

    # Cache baseline outputs once per prompt
    baseline_out = [baseline.generate(p, max_new_tokens=max_new_tokens)["generated_ids"] for p in prompts]

    report = {}
    for name, ctrl in controllers.items():
        runner = PhysicalSpeculativeRunner(target_model, draft_model, tokenizer, controller=ctrl)
        passed, mismatches = 0, []
        tok_total, tok_match = 0, 0
        for i, p in enumerate(prompts):
            spec_ids = runner.generate(p, max_new_tokens=max_new_tokens)["generated_ids"]
            ok, n_cmp, first, n_tok_match = compare_sequences(baseline_out[i], spec_ids)
            tok_total += n_cmp
            tok_match += n_tok_match
            if ok:
                passed += 1
            else:
                mismatches.append({"prompt_idx": i, "n_compared": n_cmp, "first_mismatch": first})
        tok_rate = tok_match / tok_total if tok_total else 0.0
        report[name] = {
            "passed": passed, "total": len(prompts),
            "token_match_rate": round(tok_rate, 4),
            "tokens_matched": tok_match, "tokens_total": tok_total,
            "mismatches": mismatches,
        }
        status = "PASS" if passed == len(prompts) else "PARTIAL"
        print(f"  [{status}] {name}: {passed}/{len(prompts)} prompts exact, "
              f"{tok_rate:.2%} per-token match")
    return report


def _cli():
    import os
    import sys
    import json
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", default="results/equivalence_report.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.target)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    dev = "auto" if torch.cuda.is_available() else None
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=dtype, device_map=dev).eval()
    draft = AutoModelForCausalLM.from_pretrained(args.draft, torch_dtype=dtype, device_map=dev).eval()

    prompts = [
        "Write a python function to compute the nth Fibonacci number.",
        "Explain why the sky is blue in two sentences.",
        "What is 17 times 24?",
        "Translate to French: Good morning, how are you?",
        "Summarize the theory of relativity briefly.",
        "List three benefits of regular exercise.",
        "def quicksort(arr):",
        "The capital of Australia is",
        "Solve: a train travels 60 miles in 1.5 hours, what is its speed?",
        "Write a haiku about autumn.",
    ][: args.n]

    print(f"Verifying output equivalence: {args.target} (target) vs {args.draft} (draft)")
    report = verify(target, draft, tok, prompts, max_new_tokens=args.max_new_tokens)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    all_pass = all(r["passed"] == r["total"] for r in report.values())
    report["_summary"] = {"all_pass": all_pass}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nOverall: {'ALL PASS — speculative decoding is lossless' if all_pass else 'MISMATCH DETECTED'}")
    print(f"Report: {args.out}")


if __name__ == "__main__":
    _cli()
