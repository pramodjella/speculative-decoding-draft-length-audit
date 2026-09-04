import pandas as pd
import torch
import time
import os
from tqdm import tqdm
from datasets import load_dataset
from src.models import load_hf_model
from src.serve.physical_runner import PhysicalSpeculativeRunner
from src.controllers.entropy import EntropyThreshold
from src.controllers.epsilon_greedy import EpsilonGreedy
from src.controllers.ucb import UCB
from src.controllers.history import AcceptanceHistoryController

def get_prompts(workload, num_samples=5):
    prompts = []
    if workload == "mt_bench":
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        for i in range(min(num_samples, len(ds))):
            prompts.append(ds[i]["prompt"][0])
    elif workload == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        for i in range(min(num_samples, len(ds))):
            prompts.append(ds[i]["prompt"])
    elif workload == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        for i in range(min(num_samples, len(ds))):
            prompts.append(ds[i]["question"])
    elif workload == "spec_bench":
        prompts = [
            "Translate to French: The house is blue.", 
            "Write a python script to reverse a list.", 
            "What is 15 * 12?",
            "Summarize the plot of Inception.",
            "Explain quantum computing."
        ]
        prompts = (prompts * 5)[:num_samples]
    return prompts

def main():
    target_id = "Qwen/Qwen2.5-1.5B-Instruct"
    draft_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print("Loading target model...")
    target_model, tokenizer = load_hf_model(target_id, dtype="bfloat16", device="auto")
    print("Loading draft model...")
    draft_model, _ = load_hf_model(draft_id, dtype="bfloat16", device="auto")
    
    workloads = ["humaneval", "gsm8k", "mt_bench", "spec_bench"]
    
    # We redefine policies dynamically per workload to reset bandit state
    def get_policies():
        return {
            "fixed_2": 2,
            "fixed_4": 4,
            "entropy": EntropyThreshold(tau=0.8, max_len=8),
            "epsilon_greedy": EpsilonGreedy(arms=(1, 2, 4, 8)),
            "ucb": UCB(arms=(1, 2, 4, 8)),
            "history": AcceptanceHistoryController(arms=(1, 2, 4, 8))
        }
    
    results = []
    NUM_SAMPLES = 5  # Sub-sampled to run in a reasonable time locally
    MAX_NEW_TOKENS = 64
    
    for wl in workloads:
        print(f"\n--- Workload: {wl} ---")
        prompts = get_prompts(wl, num_samples=NUM_SAMPLES)
        policies = get_policies()
        
        for p_name, ctrl in policies.items():
            print(f"  Policy: {p_name}")
            runner = PhysicalSpeculativeRunner(target_model, draft_model, tokenizer, controller=ctrl)
            
            wl_accepted = []
            wl_speedup = []
            wl_wasted = []
            
            for prompt in tqdm(prompts, leave=False):
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # 1. Plain Target Decoding (Baseline Latency)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                target_start = time.time()
                input_ids = tokenizer(formatted_prompt, return_tensors="pt").to(target_model.device)
                with torch.no_grad():
                    target_model.generate(**input_ids, max_new_tokens=MAX_NEW_TOKENS)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                target_time = max(time.time() - target_start, 0.001)
                
                # 2. Speculative Decoding
                res = runner.generate(formatted_prompt, max_new_tokens=MAX_NEW_TOKENS)
                
                spec_time = max(res["latency_s"], 0.001)
                speedup = target_time / spec_time
                wasted_per_acc = res["total_wasted"] / max(1, res["steps"] * res["avg_accepted"])
                
                wl_accepted.append(res["avg_accepted"])
                wl_speedup.append(speedup)
                wl_wasted.append(wasted_per_acc)
                
            results.append({
                "workload": wl,
                "policy": p_name,
                "mean_accepted_length": round(sum(wl_accepted) / len(wl_accepted), 3),
                "net_speedup": round(sum(wl_speedup) / len(wl_speedup), 3),
                "wasted_tokens_per_accepted": round(sum(wl_wasted) / len(wl_wasted), 3)
            })
            
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/physical_metrics_all.csv", index=False)
    print("\nSaved physical results to results/physical_metrics_all.csv")
    print(df)

if __name__ == "__main__":
    main()
