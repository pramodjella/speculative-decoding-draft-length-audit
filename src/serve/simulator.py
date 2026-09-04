import math
import numpy as np
from controllers.linucb import LinUCBController

# Latencies (ms)
T_TARGET = 25.0
T_DRAFT = 5.0

# Base conditional acceptance probabilities per draft index
ACCEPT_PROBS = {
    "humaneval": [0.95, 0.92, 0.88, 0.85, 0.80, 0.75, 0.70, 0.65],
    "gsm8k":     [0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15],
    "mt_bench":  [0.80, 0.70, 0.55, 0.45, 0.35, 0.20, 0.10, 0.05],
    "spec_bench":[0.85, 0.78, 0.70, 0.62, 0.52, 0.42, 0.32, 0.22]
}

def simulate_step(workload: str, controller, step_idx: int, rng: np.random.Generator) -> dict:
    """
    Simulates a single speculative decoding step.
    
    Args:
        workload: name of the workload (e.g., 'humaneval')
        controller: a controller object or an int (fixed draft length)
        step_idx: the sequence step index
        rng: numpy random generator
        
    Returns:
        A dictionary with step metrics and history.
    """
    probs = ACCEPT_PROBS[workload]
    max_len = len(probs)
    
    # 1. Generate step difficulty multiplier
    z_t = rng.lognormal(mean=0.0, sigma=0.25)
    
    # 2. Generate token-by-token entropies and target acceptances
    entropies = []
    accept_booleans = []
    
    for i in range(max_len):
        p_base = probs[i]
        noise = rng.normal(loc=0.0, scale=0.05)
        h_val = -math.log2(p_base) * z_t + noise
        h_val = max(0.05, h_val)
        entropies.append(h_val)
        
        p_accept = min(0.99, max(0.01, 2 ** (-h_val)))
        accept_booleans.append(rng.uniform() < p_accept)
        
    # 3. Determine draft length choice K
    if isinstance(controller, int):
        K = controller
    elif hasattr(controller, "choose_for_step"):
        K = controller.choose_for_step(accept_booleans)
    elif hasattr(controller, "tau"):
        K = controller.choose(entropies)
    elif isinstance(controller, LinUCBController):
        # Contextual bandit: pass first entropy as context signal
        K = controller.choose(entropy=entropies[0] if entropies else 0.0)
    else:
        K = controller.choose()
        
    K = min(max_len, max(1, K))
    
    # 4. Run the speculative step using chosen length K
    accepted_count = 0
    for i in range(K):
        if accept_booleans[i]:
            accepted_count += 1
        else:
            break
            
    wasted_tokens = K - accepted_count
    spec_latency = K * T_DRAFT + T_TARGET
    ar_latency = (accepted_count + 1) * T_TARGET
    
    step_res = {
        "workload": workload,
        "step_idx": step_idx,
        "chosen_length": K,
        "accepted_count": accepted_count,
        "spec_latency": spec_latency,
        "ar_latency": ar_latency,
        "wasted_tokens": wasted_tokens,
        "entropies": entropies[:K],
        "acceptances": accept_booleans[:K]
    }
    
    # Update controller if it exists
    if not isinstance(controller, int) and hasattr(controller, "update"):
        reward = (accepted_count + 1) * T_TARGET / spec_latency
        import inspect
        sig = inspect.signature(controller.update)
        if "reward" in sig.parameters:
            controller.update(K, reward)
        else:
            controller.update(K, accepted_count)
            
    return step_res
