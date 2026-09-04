"""Run the full 13-policy physical evaluation sweep on 7B/1.5B models on an A100 VM.

Loads Qwen2.5-7B-Instruct (Target) and Qwen2.5-1.5B-Instruct (Draft),
runs all 4 workloads, and saves the output to results/physical_roadmap_metrics_7b.csv.
To keep the run under 2 hours, GSM8K is evaluated on the first 150 prompts.
"""
import os
import gc
import csv
import time
import torch
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from serve.physical_runner import PhysicalSpeculativeRunner
from controllers import EntropyThreshold, EpsilonGreedy, UCB, AcceptanceHistoryController, LinUCBController


def load_hf_model(model_id, dtype="bfloat16", device="auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, dtype),
        device_map=device,
    )
    model.eval()
    return model, tokenizer


class TargetOnlyRunner:
    """Autoregressive baseline using HuggingFace .generate() with built-in KV cache."""

    def __init__(self, target_model, tokenizer):
        self.target_model = target_model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 128) -> dict:
        device = next(self.target_model.parameters()).device
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        outputs = self.target_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.time() - t0

        n_gen = outputs.shape[1] - input_ids.shape[1]
        return {
            "generated_tokens": n_gen,
            "latency_s": latency,
            "tokens_per_sec": n_gen / latency if latency > 0 else 0,
        }


def _load_checkpoint(metrics_file, workload_name):
    """Load completed runs from the checkpoint CSV."""
    completed_runs = set()
    completed_baselines = {}

    if os.path.exists(metrics_file):
        try:
            df = pd.read_csv(metrics_file)
            for _, row in df.iterrows():
                if row["workload"] == workload_name:
                    if row["policy"] == "baseline":
                        completed_baselines[int(row["prompt_idx"])] = row["avg_latency"]
                    else:
                        completed_runs.add((int(row["prompt_idx"]), row["policy"]))
        except Exception as e:
            print(f"  Warning: could not read checkpoint: {e}")

    return completed_runs, completed_baselines


def _ensure_headers(metrics_file, taxonomy_file):
    """Write CSV headers if the files don't exist yet."""
    if not os.path.exists(metrics_file):
        with open(metrics_file, "w", newline="") as f:
            csv.writer(f).writerow(
                ["workload", "prompt_idx", "policy", "avg_latency", "net_speedup", "wasted_tokens_per_accepted"]
            )
    if not os.path.exists(taxonomy_file):
        with open(taxonomy_file, "w", newline="") as f:
            csv.writer(f).writerow(
                ["workload", "prompt_idx", "policy", "error_type", "K", "accepted_or_entropy"]
            )


def _make_policies():
    """Create fresh controller instances for all ablation policies."""
    return {
        # A1: fixed baselines
        "fixed_1": 1,
        "fixed_2": 2,
        "fixed_4": 4,
        "fixed_8": 8,
        # A2: adaptive controllers
        "entropy": EntropyThreshold(tau=0.8, max_len=8),
        "epsilon_greedy": EpsilonGreedy(eps=0.1, arms=(1, 2, 3, 4, 5, 6, 7, 8)),
        "ucb": UCB(c=2.0, arms=(1, 2, 3, 4, 5, 6, 7, 8)),
        "history": AcceptanceHistoryController(window_size=10, arms=(1, 2, 3, 4, 5, 6, 7, 8)),
        # A4: coarse candidate set
        "ucb_coarse": UCB(c=2.0, arms=(1, 4, 8)),
        # Contextual bandit controllers (ours)
        "linucb": LinUCBController(arms=(1, 4, 8), alpha=1.0),
        "linucb_explore": LinUCBController(arms=(1, 4, 8), alpha=2.0),
        "linucb_fine": LinUCBController(arms=(1, 2, 4, 8), alpha=1.0),
    }


def run_workload(
    workload_name, prompts, target_model, draft_model, tokenizer, metrics_file, taxonomy_file
):
    print(f"\n{'='*60}")
    print(f"  Workload: {workload_name} ({len(prompts)} prompts)")
    print(f"{'='*60}")

    completed_runs, completed_baselines = _load_checkpoint(metrics_file, workload_name)
    target_runner = TargetOnlyRunner(target_model, tokenizer)

    # ── 1. Baseline (target-only autoregressive) ──
    n_baseline_remaining = sum(1 for i in range(len(prompts)) if i not in completed_baselines)
    print(f"  Baseline: {len(completed_baselines)} done, {n_baseline_remaining} remaining")

    for idx in tqdm(range(len(prompts)), desc="Baseline Target"):
        if idx in completed_baselines:
            continue

        res = target_runner.generate(prompts[idx], max_new_tokens=64)
        completed_baselines[idx] = res["latency_s"]

        with open(metrics_file, "a", newline="") as f:
            csv.writer(f).writerow([workload_name, idx, "baseline", res["latency_s"], 1.0, 0.0])

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_baseline = sum(completed_baselines.values()) / max(1, len(completed_baselines))
    print(f"  Baseline avg latency: {avg_baseline:.3f}s")

    # ── 2. Speculative policies ──
    policies = _make_policies()

    for policy_name, controller in policies.items():
        n_done = sum(1 for i in range(len(prompts)) if (i, policy_name) in completed_runs)
        n_remaining = len(prompts) - n_done
        print(f"  Policy {policy_name}: {n_done} done, {n_remaining} remaining")

        if n_remaining == 0:
            continue

        runner = PhysicalSpeculativeRunner(target_model, draft_model, tokenizer, controller=controller)

        for idx in tqdm(range(len(prompts)), desc=f"Policy {policy_name}"):
            if (idx, policy_name) in completed_runs:
                continue

            res = runner.generate(prompts[idx], max_new_tokens=64)

            spec_latency = res["latency_s"]
            baseline_latency = completed_baselines.get(idx, spec_latency)
            net_speedup = baseline_latency / spec_latency if spec_latency > 0 else 0
            total_accepted = res["avg_accepted"] * max(1, res["steps"])
            wasted_per_accepted = res["total_wasted"] / max(1, total_accepted)

            with open(metrics_file, "a", newline="") as f:
                csv.writer(f).writerow(
                    [workload_name, idx, policy_name, spec_latency, net_speedup, wasted_per_accepted]
                )

            # Error taxonomy
            if "step_logs" in res:
                for step in res["step_logs"]:
                    if step["K"] >= 6 and step["accepted"] <= 1:
                        with open(taxonomy_file, "a", newline="") as f:
                            csv.writer(f).writerow(
                                [workload_name, idx, policy_name, "Over-Drafting", step["K"], step["accepted"]]
                            )
                    elif step["K"] == 1 and step["entropy"] < 0.2:
                        with open(taxonomy_file, "a", newline="") as f:
                            csv.writer(f).writerow(
                                [workload_name, idx, policy_name, "Under-Drafting", step["K"], step["entropy"]]
                            )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    target_id = "Qwen/Qwen2.5-7B-Instruct"
    draft_id = "Qwen/Qwen2.5-0.5B-Instruct"
    metrics_file = "results/physical_roadmap_metrics_7b_0.5b.csv"
    taxonomy_file = "results/physical_error_taxonomy_7b_0.5b.csv"

    os.makedirs("results", exist_ok=True)
    _ensure_headers(metrics_file, taxonomy_file)

    # ── Load models ONCE ──
    print("Loading target model...")
    target_model, tokenizer = load_hf_model(target_id, dtype="bfloat16", device="auto")
    print("Loading draft model...")
    draft_model, _ = load_hf_model(draft_id, dtype="bfloat16", device="auto")

    print(f"\nTarget: {target_id}")
    print(f"Draft:  {draft_id}")
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory used by models: {mem:.2f} GB")

    # ── Load datasets ──
    print("\nLoading datasets...")
    humaneval_prompts = load_dataset("openai/openai_humaneval", split="test")["prompt"]
    
    gsm8k_ds = load_dataset("openai/gsm8k", "main", split="test")
    # Select first 150 prompts for GSM8K to ensure it fits in A100 runtime
    gsm8k_prompts = gsm8k_ds.select(range(min(150, len(gsm8k_ds))))["question"]
    
    mt_bench_prompts = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train").map(
        lambda x: {"prompt": x["prompt"][0]}
    )["prompt"]

    # Synthetic spec_bench
    spec_prompts = [
        "Translate to French: The house is blue.",
        "Write a python script to reverse a list.",
        "What is 15 * 12?",
        "Summarize the plot of Inception.",
        "Explain quantum computing.",
    ] * 20

    datasets = {
        "humaneval": humaneval_prompts,
        "gsm8k": gsm8k_prompts,
        "mt_bench": mt_bench_prompts,
        "spec_bench": spec_prompts,
    }

    # ── Run all workloads ──
    for name, prompts in datasets.items():
        run_workload(name, prompts, target_model, draft_model, tokenizer, metrics_file, taxonomy_file)

    print("\n" + "=" * 60)
    print("  ALL 7B PHYSICAL EVALUATIONS COMPLETE!")
    print(f"  Metrics: {metrics_file}")
    print(f"  Taxonomy: {taxonomy_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
