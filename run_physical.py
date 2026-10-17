import argparse
import json
import os
import torch
from src.models import load_hf_model
from src.serve.physical_runner import PhysicalSpeculativeRunner
from src.controllers.entropy import EntropyThreshold
from src.controllers.history import AcceptanceHistoryController
from src.controllers.ucb import UCB

def main():
    parser = argparse.ArgumentParser(description="Run physical speculative decoding pipeline.")
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Target model ID")
    parser.add_argument("--draft-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Draft model ID")
    parser.add_argument("--controller", type=str, choices=["constant", "entropy", "history", "ucb"], default="entropy")
    parser.add_argument("--prompt", type=str, default="Explain the theory of relativity in simple terms.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=str, default="results/physical_run.json")
    
    args = parser.parse_args()
    
    print(f"Loading target model: {args.target_model}...")
    target_model, target_tokenizer = load_hf_model(args.target_model, dtype="bfloat16", device="auto")
    
    print(f"Loading draft model: {args.draft_model}...")
    draft_model, _ = load_hf_model(args.draft_model, dtype="bfloat16", device="auto")
    
    # Initialize controller
    if args.controller == "constant":
        controller = 4
    elif args.controller == "entropy":
        controller = EntropyThreshold(tau=0.8, max_len=8)
    elif args.controller == "history":
        controller = AcceptanceHistoryController(arms=(1, 2, 4, 8))
    elif args.controller == "ucb":
        controller = UCB(arms=(1, 2, 4, 8))
    else:
        controller = 4
        
    print(f"Running speculative decoding with {args.controller} controller...")
    runner = PhysicalSpeculativeRunner(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=target_tokenizer,
        controller=controller
    )
    
    # Prompt formatting for Instruct models
    messages = [{"role": "user", "content": args.prompt}]
    formatted_prompt = target_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    result = runner.generate(formatted_prompt, max_new_tokens=args.max_new_tokens)
    
    print("\n--- Generated Text ---")
    print(result["text"])
    print("----------------------\n")
    
    print(f"Metrics:")
    print(f"  Tokens Generated: {result['generated_tokens']}")
    print(f"  Speculative Steps: {result['steps']}")
    print(f"  Latency: {result['latency_s']:.2f} s")
    print(f"  Tokens / sec: {result['tokens_per_sec']:.2f}")
    print(f"  Avg Accepted/Step: {result['avg_accepted']:.2f}")
    print(f"  Total Wasted Draft Tokens: {result['total_wasted']}")
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
