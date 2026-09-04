"""Run physical evaluation for LinUCB policies only.

Appends results to the existing physical_roadmap_metrics.csv so we can
compare LinUCB directly against the 9 existing policies.

Policies evaluated:
  - linucb:         LinUCBController(arms=(1,4,8), alpha=1.0)
  - linucb_explore: LinUCBController(arms=(1,4,8), alpha=2.0)
  - linucb_fine:    LinUCBController(arms=(1,2,4,8), alpha=1.0)
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
from controllers import LinUCBController


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


def _load_existing_baselines(metrics_file, workload_name):
    """Load baseline latencies from existing CSV."""
    baselines = {}
    if os.path.exists(metrics_file):
        try:
            df = pd.read_csv(metrics_file)
            mask = (df["workload"] == workload_name) & (df["policy"] == "baseline")
            for _, row in df[mask].iterrows():
                baselines[int(row["prompt_idx"])] = row["avg_latency"]
        except Exception as e:
            print(f"  Warning reading baselines: {e}")
    return baselines


def _load_completed(metrics_file, workload_name):
    """Load already-completed (prompt_idx, policy) pairs."""
    completed = set()
    if os.path.exists(metrics_file):
        try:
            df = pd.read_csv(metrics_file)
            mask = df["workload"] == workload_name
            for _, row in df[mask].iterrows():
                if row["policy"] != "baseline":
                    completed.add((int(row["prompt_idx"]), row["policy"]))
        except Exception as e:
            print(f"  Warning: {e}")
    return completed


def main():
    target_id = "Qwen/Qwen2.5-1.5B"
    draft_id = "Qwen/Qwen2.5-0.5B"
    metrics_file = "results/physical_roadmap_metrics.csv"
    taxonomy_file = "results/physical_error_taxonomy.csv"

    os.makedirs("results", exist_ok=True)

    # Ensure headers exist
    if not os.path.exists(metrics_file):
        with open(metrics_file, "w", newline="") as f:
            csv.writer(f).writerow(
                ["workload", "prompt_idx", "policy", "avg_latency", "net_speedup", "wasted_tokens_per_accepted"]
            )

    # ── Load models ONCE ──
    print("Loading target model...")
    target_model, tokenizer = load_hf_model(target_id, dtype="bfloat16", device="auto")
    print("Loading draft model...")
    draft_model, _ = load_hf_model(draft_id, dtype="bfloat16", device="auto")

    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory used: {mem:.2f} GB")

    # ── LinUCB policies to evaluate ──
    linucb_policies = {
        "linucb": LinUCBController(arms=(1, 4, 8), alpha=1.0),
        "linucb_explore": LinUCBController(arms=(1, 4, 8), alpha=2.0),
        "linucb_fine": LinUCBController(arms=(1, 2, 4, 8), alpha=1.0),
    }

    # ── Load datasets ──
    print("\nLoading datasets...")
    datasets = {
        "humaneval": load_dataset("openai_humaneval", split="test"),
        "gsm8k": load_dataset("gsm8k", "main", split="test").map(lambda x: {"prompt": x["question"]}),
        "mt_bench": load_dataset("HuggingFaceH4/mt_bench_prompts", split="train").map(
            lambda x: {"prompt": x["prompt"][0]}
        ),
    }
    # Synthetic spec_bench
    spec_prompts = [
        "Translate to French: The house is blue.",
        "Write a python script to reverse a list.",
        "What is 15 * 12?",
        "Summarize the plot of Inception.",
        "Explain quantum computing.",
    ] * 20
    datasets["spec_bench"] = {"prompt": spec_prompts}

    # ── Run each workload ──
    for wl_name, ds in datasets.items():
        prompts = ds["prompt"]
        baselines = _load_existing_baselines(metrics_file, wl_name)
        completed = _load_completed(metrics_file, wl_name)

        if not baselines:
            print(f"\n  WARNING: No baselines found for {wl_name} — skipping!")
            print(f"  Run the full roadmap first to get baseline latencies.")
            continue

        print(f"\n{'='*60}")
        print(f"  Workload: {wl_name} ({len(prompts)} prompts, {len(baselines)} baselines)")
        print(f"{'='*60}")

        for policy_name, controller in linucb_policies.items():
            n_done = sum(1 for i in range(len(prompts)) if (i, policy_name) in completed)
            n_remaining = len(prompts) - n_done
            print(f"  Policy {policy_name}: {n_done} done, {n_remaining} remaining")

            if n_remaining == 0:
                continue

            # Create fresh runner with this controller
            runner = PhysicalSpeculativeRunner(
                target_model, draft_model, tokenizer, controller=controller
            )

            for idx in tqdm(range(len(prompts)), desc=f"{policy_name}@{wl_name}"):
                if (idx, policy_name) in completed:
                    continue

                res = runner.generate(prompts[idx], max_new_tokens=64)

                spec_latency = res["latency_s"]
                baseline_latency = baselines.get(idx, spec_latency)
                net_speedup = baseline_latency / spec_latency if spec_latency > 0 else 0
                total_accepted = res["avg_accepted"] * max(1, res["steps"])
                wasted_per_accepted = res["total_wasted"] / max(1, total_accepted)

                with open(metrics_file, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [wl_name, idx, policy_name, spec_latency, net_speedup, wasted_per_accepted]
                    )

                # Error taxonomy
                if "step_logs" in res:
                    for step in res["step_logs"]:
                        if step["K"] >= 6 and step["accepted"] <= 1:
                            with open(taxonomy_file, "a", newline="") as f:
                                csv.writer(f).writerow(
                                    [wl_name, idx, policy_name, "Over-Drafting", step["K"], step["accepted"]]
                                )
                        elif step["K"] == 1 and step["entropy"] < 0.2:
                            with open(taxonomy_file, "a", newline="") as f:
                                csv.writer(f).writerow(
                                    [wl_name, idx, policy_name, "Under-Drafting", step["K"], step["entropy"]]
                                )

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("  LinUCB PHYSICAL EVALUATION COMPLETE!")
    print(f"  Results appended to: {metrics_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
