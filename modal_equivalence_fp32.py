"""fp32 equivalence proof (E0.2): show speculative decoding is EXACTLY lossless.

The bf16 sweep shows occasional divergence from the target's greedy output. The
diagnostic traced every divergence to exact bf16 logit ties (Δ=0.0) broken
differently by the cached vs batched numerical paths — a precision artifact, not
an algorithmic change. In fp32 those exact ties essentially vanish, so this run
should report ~100% per-token match, proving the implementation is lossless and
the bf16 divergence is purely numerical.

Runs the small 1.5B/0.5B pair in fp32 on an L40 (cheap, ~8 GB needed), so it can
run alongside the main A100 sweep without contention.
"""
import os
import modal

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers==4.44.2", "accelerate", "numpy")
    .add_local_dir("src", "/root/project/src")
)

app = modal.App("adaptive-draft-equivalence-fp32")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)


@app.function(image=image, gpu="L40S", timeout=3600,
              volumes={"/root/project/results": vol})
def run_fp32_equivalence():
    os.chdir("/root/project")
    import sys, json
    sys.path.insert(0, "/root/project/src")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verify_equivalence import verify
    from controllers import EntropyThreshold, EpsilonGreedy, UCB, AcceptanceHistoryController, LinUCBController

    target_id = "Qwen/Qwen2.5-1.5B-Instruct"
    draft_id = "Qwen/Qwen2.5-0.5B-Instruct"

    def load(mid):
        tok = AutoTokenizer.from_pretrained(mid)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32, device_map="cuda").eval()
        return m, tok

    target, tok = load(target_id)
    draft, _ = load(draft_id)

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
        "What are the first 5 prime numbers?",
        "Describe the water cycle in one paragraph.",
    ]

    controllers = {
        "fixed_4": 4,
        "entropy": EntropyThreshold(tau=0.8, max_len=8),
        "ucb": UCB(c=2.0, arms=(1, 2, 3, 4, 5, 6, 7, 8)),
        "epsilon_greedy": EpsilonGreedy(eps=0.1, arms=(1, 2, 4, 8), seed=0),
        "history": AcceptanceHistoryController(window_size=10, arms=(1, 2, 4, 8)),
        "linucb": LinUCBController(arms=(1, 4, 8), alpha=1.0),
    }

    print(f"=== fp32 equivalence proof: {target_id} vs {draft_id} ===")
    report = verify(target, draft, tok, prompts, max_new_tokens=48, controllers=controllers)
    all_pass = all(r["passed"] == r["total"] for r in report.values())
    min_tok = min(r["token_match_rate"] for r in report.values())
    report["_summary"] = {"dtype": "float32", "all_prompts_exact": all_pass,
                          "min_token_match_rate": min_tok}

    os.makedirs("results", exist_ok=True)
    with open("results/m5_equivalence_fp32.json", "w") as f:
        json.dump(report, f, indent=2)
    vol.commit()
    print(f"\nfp32: all_prompts_exact={all_pass}  min_token_match={min_tok:.4%}")
    return report


@app.local_entrypoint()
def main():
    res = run_fp32_equivalence.remote()
    print("\n=== fp32 equivalence summary ===")
    for k, v in res.items():
        if k == "_summary":
            print("SUMMARY:", v)
        else:
            print(f"  {k}: {v['passed']}/{v['total']} prompts exact, "
                  f"{v['token_match_rate']:.2%} per-token match")
