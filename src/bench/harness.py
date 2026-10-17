import numpy as np
import pandas as pd
from src.serve.simulator import simulate_step, T_TARGET, T_DRAFT, ACCEPT_PROBS
from src.controllers import EntropyThreshold, EpsilonGreedy, UCB, AcceptanceHistoryController, OracleController
import copy

def run_experiment_stream(
    workload: str,
    policy_name: str,
    controller_fn,
    num_steps: int = 1000,
    batch_size: int = 1,
    seed: int = 42
) -> list[dict]:
    """
    Runs a stream of speculative decoding steps under a specific policy and batch size.
    
    Args:
        workload: Name of workload (e.g. 'humaneval') or 'mixed' for interleaved.
        policy_name: Label for the policy (e.g. 'fixed_3', 'ucb').
        controller_fn: A callable that returns a fresh controller instance or an int.
        num_steps: Number of generation steps.
        batch_size: Batch size B.
        seed: Seed for reproducibility.
        
    Returns:
        A list of dictionaries representing the metrics for each step.
    """
    rng = np.random.default_rng(seed)
    
    # Instantiate controllers. If batch_size > 1, we can have per-request controllers.
    # For fixed length, controller is just an int.
    is_fixed = isinstance(controller_fn(), int)
    
    if is_fixed:
        controllers = [controller_fn() for _ in range(batch_size)]
    else:
        # We use independent controller instances per sequence in the batch to model per-request adaptation
        controllers = [controller_fn() for _ in range(batch_size)]
        
    # We also have an OracleController if the policy is oracle
    is_oracle = (policy_name.lower() == "oracle")
    if is_oracle:
        controllers = [OracleController(t_draft=T_DRAFT, t_target=T_TARGET) for _ in range(batch_size)]
        
    step_records = []
    
    for step in range(num_steps):
        # Determine workload for each sequence in the batch for this step
        if workload == "mixed":
            # Interleave workloads randomly per sequence in the batch
            batch_workloads = rng.choice(["humaneval", "gsm8k", "mt_bench", "spec_bench"], size=batch_size)
        else:
            batch_workloads = [workload] * batch_size
            
        # Simulate each request in the batch to get chosen lengths and potential decisions
        batch_results = []
        for b in range(batch_size):
            wl = batch_workloads[b]
            ctrl = controllers[b]
            
            # Temporary simulator run to determine K and token-by-token acceptances
            # To ensure proper sequence generation, we simulate the step
            res = simulate_step(wl, ctrl, step, rng)
            batch_results.append(res)
            
        # Batched Speculative Decoding Execution:
        # The actual draft length executed for the batch is the max of chosen lengths in the batch
        chosen_lengths = [r["chosen_length"] for r in batch_results]
        K_batch = max(chosen_lengths)
        
        # Calculate actual batch latency
        # We draft up to K_batch tokens in parallel, and target verifies them
        actual_spec_latency = K_batch * T_DRAFT + T_TARGET
        
        # Sum up tokens and evaluate batch performance
        total_accepted = 0
        total_ar_latency = 0.0
        total_wasted = 0
        
        for b in range(batch_size):
            res = batch_results[b]
            total_accepted += res["accepted_count"]
            total_ar_latency += res["ar_latency"]
            # Wasted tokens for request b is K_batch - accepted_count
            # (since we drafted K_batch tokens for the batch)
            total_wasted += (K_batch - res["accepted_count"])
            
        mean_accepted = total_accepted / batch_size
        batch_speedup = total_ar_latency / (batch_size * actual_spec_latency)
        wasted_per_accepted = total_wasted / max(1, total_accepted)
        
        step_records.append({
            "step": step,
            "workload": workload,
            "policy": policy_name,
            "batch_size": batch_size,
            "batch_length": K_batch,
            "mean_accepted_length": round(mean_accepted, 3),
            "net_speedup": round(batch_speedup, 3),
            "wasted_tokens_per_accepted": round(wasted_per_accepted, 3),
            "total_accepted": total_accepted,
            "total_wasted": total_wasted,
        })
        
    return step_records
