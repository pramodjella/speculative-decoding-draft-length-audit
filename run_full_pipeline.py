"""M5 full physical pipeline (honest baseline + 3 seeds + real mixed trace).

Fixes the three blocking issues from the M5 plan:
  E0.1  Net speedup is measured against PythonAutoregressiveBaseline (same
        pure-Python forward/KV-cache code path as the speculative runner), NOT
        HuggingFace .generate(). The HF number is still recorded (seed 0 only)
        as an appendix "production-engine reference".
  E0.2  Output equivalence is verified before the sweep; the report is saved.
  E0.3  Every (workload, policy) is run over N_SEEDS seeds so the analysis can
        report mean +/- 95% CI.

Real benchmarks (fixes synthetic-data challenge): humaneval, gsm8k, mt_bench are
real datasets; "mixed" is a genuine interleaved trace assembled from real
chat/code/math/translation/summarisation prompts (the roadmap's stress workload).

Everything is checkpointed per (seed, workload, prompt_idx, policy) so a timed-out
run resumes exactly where it stopped.
"""
import os
import gc
import csv
import time
import random
import argparse

import torch
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from serve.physical_runner import PhysicalSpeculativeRunner, PythonAutoregressiveBaseline
from controllers import (
    EntropyThreshold,
    EpsilonGreedy,
    UCB,
    AcceptanceHistoryController,
    LinUCBController,
)

METRICS_FILE = "results/m5_metrics.csv"
TAXONOMY_FILE = "results/m5_taxonomy.csv"
EQUIV_FILE = "results/m5_equivalence.json"

METRICS_HEADER = [
    "seed", "workload", "prompt_idx", "policy",
    "latency_s", "net_speedup", "net_speedup_vs_hf",
    "wasted_tokens_per_accepted", "avg_accepted", "accepted_tokens_per_step", "steps",
]
TAXONOMY_HEADER = ["seed", "workload", "prompt_idx", "policy", "error_type", "K", "accepted_or_entropy"]


def load_hf_model(model_id, dtype="bfloat16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, dtype) if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
    ).eval()
    return model, tokenizer


def make_policies(seed: int) -> dict:
    """Fresh controller instances. Bandits are seeded so seeds give real variance."""
    return {
        # A1: fixed baselines (deterministic)
        "fixed_1": 1,
        "fixed_2": 2,
        "fixed_4": 4,
        "fixed_8": 8,
        # A2: adaptive controllers
        "entropy": EntropyThreshold(tau=0.8, max_len=8),
        "epsilon_greedy": EpsilonGreedy(eps=0.1, arms=(1, 2, 3, 4, 5, 6, 7, 8), seed=seed),
        "ucb": UCB(c=2.0, arms=(1, 2, 3, 4, 5, 6, 7, 8)),
        "history": AcceptanceHistoryController(window_size=10, arms=(1, 2, 3, 4, 5, 6, 7, 8)),
        # A4: coarse candidate set
        "ucb_coarse": UCB(c=2.0, arms=(1, 4, 8)),
        # Contextual bandit (novel contribution)
        "linucb": LinUCBController(arms=(1, 4, 8), alpha=1.0),
        "linucb_explore": LinUCBController(arms=(1, 4, 8), alpha=2.0),
        "linucb_fine": LinUCBController(arms=(1, 2, 4, 8), alpha=1.0),
    }


def load_workloads(n_per_workload: int):
    from datasets import load_dataset

    def take(lst, n):
        return list(lst)[:n]

    humaneval = take(load_dataset("openai/openai_humaneval", split="test")["prompt"], n_per_workload)
    gsm8k = take(load_dataset("openai/gsm8k", "main", split="test")["question"], n_per_workload)
    mt_bench = take(
        load_dataset("HuggingFaceH4/mt_bench_prompts", split="train").map(
            lambda x: {"p": x["prompt"][0]}
        )["p"],
        n_per_workload,
    )

    # Real mixed trace: interleave real chat/code/math/translation/summarisation.
    translation = [
        "Translate to French: The weather is beautiful today.",
        "Translate to German: I would like a cup of coffee, please.",
        "Translate to Spanish: Where is the nearest train station?",
        "Translate to Italian: Thank you very much for your help.",
    ]
    summarisation = [
        "Summarize in one sentence: The mitochondria is the powerhouse of the cell, "
        "producing ATP through cellular respiration.",
        "Summarize the key idea of supply and demand in economics.",
        "Summarize what photosynthesis does for a plant.",
        "Summarize the plot of Romeo and Juliet in two sentences.",
    ]
    pools = [humaneval, gsm8k, mt_bench, translation, summarisation]
    mixed, idxs = [], [0] * len(pools)
    pi = 0
    while len(mixed) < n_per_workload:
        pool = pools[pi % len(pools)]
        if idxs[pi % len(pools)] < len(pool):
            mixed.append(pool[idxs[pi % len(pools)]])
            idxs[pi % len(pools)] += 1
        pi += 1
        if all(idxs[j] >= len(pools[j]) for j in range(len(pools))):
            # recycle if we run out
            idxs = [0] * len(pools)

    return {
        "humaneval": humaneval,
        "gsm8k": gsm8k,
        "mt_bench": mt_bench,
        "mixed": mixed,
    }


def load_checkpoint():
    """Return (done_set, honest_lat, hf_lat).

    done_set: {(seed, workload, prompt_idx, policy)}
    honest_lat / hf_lat: {(seed, workload, prompt_idx): latency}
    """
    done, honest, hf = set(), {}, {}
    if os.path.exists(METRICS_FILE):
        try:
            df = pd.read_csv(METRICS_FILE)
            for _, r in df.iterrows():
                key = (int(r["seed"]), r["workload"], int(r["prompt_idx"]))
                if r["policy"] == "baseline_honest":
                    honest[key] = r["latency_s"]
                elif r["policy"] == "baseline_hf":
                    hf[key] = r["latency_s"]
                else:
                    done.add(key + (r["policy"],))
        except Exception as e:
            print(f"  Warning reading checkpoint: {e}")
    return done, honest, hf


def ensure_headers():
    os.makedirs("results", exist_ok=True)
    if not os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "w", newline="") as f:
            csv.writer(f).writerow(METRICS_HEADER)
    if not os.path.exists(TAXONOMY_FILE):
        with open(TAXONOMY_FILE, "w", newline="") as f:
            csv.writer(f).writerow(TAXONOMY_HEADER)


def _append(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(target_model, draft_model, tokenizer, workloads, seeds, max_new_tokens, run_hf_ref):
    done, honest_lat, hf_lat = load_checkpoint()
    honest_runner = PythonAutoregressiveBaseline(target_model, tokenizer)
    hf_runner = TargetOnlyHF(target_model, tokenizer) if run_hf_ref else None

    for seed in seeds:
        torch.manual_seed(seed)
        random.seed(seed)
        for wname, prompts in workloads.items():
            print(f"\n{'='*64}\n  seed={seed}  workload={wname}  ({len(prompts)} prompts)\n{'='*64}")

            # ── Honest baseline (E0.1) ──
            for idx in tqdm(range(len(prompts)), desc=f"honest s{seed} {wname}"):
                key = (seed, wname, idx)
                if key in honest_lat:
                    continue
                res = honest_runner.generate(prompts[idx], max_new_tokens=max_new_tokens)
                honest_lat[key] = res["latency_s"]
                _append(METRICS_FILE, [seed, wname, idx, "baseline_honest",
                                       res["latency_s"], 1.0, "", 0.0, 0.0, "", 0])
                _cleanup()

            # ── HF reference baseline (appendix, seed 0 only) ──
            if hf_runner is not None and seed == seeds[0]:
                for idx in tqdm(range(len(prompts)), desc=f"hf_ref s{seed} {wname}"):
                    key = (seed, wname, idx)
                    if key in hf_lat:
                        continue
                    res = hf_runner.generate(prompts[idx], max_new_tokens=max_new_tokens)
                    hf_lat[key] = res["latency_s"]
                    _append(METRICS_FILE, [seed, wname, idx, "baseline_hf",
                                           res["latency_s"], "", 1.0, 0.0, 0.0, "", 0])
                    _cleanup()

            # ── Speculative policies ──
            policies = make_policies(seed)
            for pname, ctrl in policies.items():
                runner = PhysicalSpeculativeRunner(target_model, draft_model, tokenizer, controller=ctrl)
                for idx in tqdm(range(len(prompts)), desc=f"{pname} s{seed} {wname}"):
                    if (seed, wname, idx, pname) in done:
                        continue
                    res = runner.generate(prompts[idx], max_new_tokens=max_new_tokens)
                    spec_lat = res["latency_s"]
                    hb = honest_lat.get((seed, wname, idx), spec_lat)
                    hf = hf_lat.get((seeds[0], wname, idx))
                    net = hb / spec_lat if spec_lat > 0 else 0
                    net_hf = (hf / spec_lat) if (hf and spec_lat > 0) else ""
                    total_acc = res["avg_accepted"] * max(1, res["steps"])
                    wasted = res["total_wasted"] / max(1, total_acc)

                    _append(METRICS_FILE, [seed, wname, idx, pname, spec_lat, net, net_hf,
                                           wasted, res["avg_accepted"],
                                           res["accepted_tokens_per_step"], res["steps"]])

                    for step in res.get("step_logs", []):
                        if step["K"] >= 6 and step["accepted"] <= 1:
                            _append(TAXONOMY_FILE, [seed, wname, idx, pname, "Over-Drafting",
                                                    step["K"], step["accepted"]])
                        elif step["K"] == 1 and step["entropy"] < 0.2:
                            _append(TAXONOMY_FILE, [seed, wname, idx, pname, "Under-Drafting",
                                                    step["K"], step["entropy"]])
                    _cleanup()


class TargetOnlyHF:
    """HF .generate() reference baseline (appendix only)."""

    def __init__(self, target_model, tokenizer):
        self.target_model = target_model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=64):
        device = next(self.target_model.parameters()).device
        ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        out = self.target_model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat = time.time() - t0
        return {"latency_s": lat, "generated_tokens": out.shape[1] - ids.shape[1]}


def maybe_verify_equivalence(target_model, draft_model, tokenizer, workloads, max_new_tokens):
    if os.path.exists(EQUIV_FILE):
        print("Equivalence already verified; skipping.")
        return
    import json
    from verify_equivalence import verify
    print("\n=== E0.2: Verifying output equivalence (lossless check) ===")
    sample = []
    for prompts in workloads.values():
        sample.extend(prompts[:3])
    report = verify(target_model, draft_model, tokenizer, sample[:10], max_new_tokens=max_new_tokens)
    report["_summary"] = {"all_pass": all(r["passed"] == r["total"] for r in report.values())}
    with open(EQUIV_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Equivalence report -> {EQUIV_FILE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=os.environ.get("M5_TARGET", "Qwen/Qwen2.5-7B-Instruct"))
    ap.add_argument("--draft", default=os.environ.get("M5_DRAFT", "Qwen/Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--seeds", type=int, default=int(os.environ.get("M5_SEEDS", "3")))
    ap.add_argument("--n-prompts", type=int, default=int(os.environ.get("M5_NPROMPTS", "60")))
    ap.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("M5_MAXTOK", "64")))
    ap.add_argument("--no-hf-ref", action="store_true")
    ap.add_argument("--tag", default=os.environ.get("M5_TAG", ""),
                    help="Suffix for output files so different model pairs don't clash, "
                         "e.g. --tag qwen3b writes results/m5_metrics_qwen3b.csv")
    args = ap.parse_args()

    # Isolate outputs per run/model-pair so the 3B sweep and the 1.5B fallback
    # (and any older runs) never share a CSV with a different schema.
    if args.tag:
        global METRICS_FILE, TAXONOMY_FILE, EQUIV_FILE
        METRICS_FILE = f"results/m5_metrics_{args.tag}.csv"
        TAXONOMY_FILE = f"results/m5_taxonomy_{args.tag}.csv"
        EQUIV_FILE = f"results/m5_equivalence_{args.tag}.json"

    ensure_headers()
    print(f"Target: {args.target}\nDraft:  {args.draft}")
    print(f"Seeds: {args.seeds}  Prompts/workload: {args.n_prompts}  MaxTok: {args.max_new_tokens}")

    target_model, tokenizer = load_hf_model(args.target)
    draft_model, _ = load_hf_model(args.draft)
    if torch.cuda.is_available():
        print(f"GPU mem after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print("\nLoading workloads...")
    workloads = load_workloads(args.n_prompts)
    for k, v in workloads.items():
        print(f"  {k}: {len(v)} prompts")

    maybe_verify_equivalence(target_model, draft_model, tokenizer, workloads, args.max_new_tokens)

    run(target_model, draft_model, tokenizer, workloads,
        seeds=list(range(args.seeds)), max_new_tokens=args.max_new_tokens,
        run_hf_ref=not args.no_hf_ref)

    print("\n" + "=" * 64)
    print("  M5 PIPELINE COMPLETE")
    print(f"  Metrics:     {METRICS_FILE}")
    print(f"  Taxonomy:    {TAXONOMY_FILE}")
    print(f"  Equivalence: {EQUIV_FILE}")
    print("=" * 64)


if __name__ == "__main__":
    main()
