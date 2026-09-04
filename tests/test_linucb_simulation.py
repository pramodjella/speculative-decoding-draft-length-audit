"""Quick simulation smoke test for LinUCB integration."""
import sys, os

# Ensure we can import both 'src' package and 'controllers' module
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "src"))

from src.bench.harness import run_experiment_stream
from controllers import LinUCBController, UCB, EpsilonGreedy, EntropyThreshold

def test_policy(name, ctrl_fn, workload="humaneval", steps=100):
    records = run_experiment_stream(workload, name, ctrl_fn, num_steps=steps, batch_size=1, seed=42)
    speedups = [r["net_speedup"] for r in records]
    wasted = [r["wasted_tokens_per_accepted"] for r in records]
    avg_sp = sum(speedups) / len(speedups)
    avg_wa = sum(wasted) / len(wasted)
    k_choices = [r["batch_length"] for r in records[:15]]
    print(f"  {name:20s} | speedup={avg_sp:.3f} | wasted={avg_wa:.3f} | K_sample={k_choices}")
    return avg_sp, avg_wa

print("=" * 80)
print("  LinUCB Simulation Comparison (humaneval, B=1, 100 steps)")
print("=" * 80)

results = {}
# Baselines
results["fixed_4"] = test_policy("fixed_4", lambda: 4)
results["ucb"] = test_policy("ucb", lambda: UCB(c=0.5))
results["ucb_coarse"] = test_policy("ucb_coarse", lambda: UCB(c=2.0, arms=(1, 4, 8)))
results["entropy"] = test_policy("entropy", lambda: EntropyThreshold(tau=1.0, max_len=8))
results["epsilon_greedy"] = test_policy("eps_greedy", lambda: EpsilonGreedy(eps=0.1, seed=42))

# Our novel controllers
results["linucb"] = test_policy("linucb", lambda: LinUCBController(arms=(1, 4, 8), alpha=1.0))
results["linucb_explore"] = test_policy("linucb_explore", lambda: LinUCBController(arms=(1, 4, 8), alpha=2.0))
results["linucb_fine"] = test_policy("linucb_fine", lambda: LinUCBController(arms=(1, 2, 4, 8), alpha=1.0))

print("\n" + "=" * 80)
print("  Cross-workload test (LinUCB vs UCB_coarse vs fixed_4)")
print("=" * 80)
for wl in ["humaneval", "gsm8k", "mt_bench", "spec_bench"]:
    sp_fixed, _ = test_policy(f"fixed_4({wl})", lambda: 4, workload=wl)
    sp_ucb, _ = test_policy(f"ucb_coarse({wl})", lambda: UCB(c=2.0, arms=(1, 4, 8)), workload=wl)
    sp_lin, _ = test_policy(f"linucb({wl})", lambda: LinUCBController(arms=(1, 4, 8), alpha=1.0), workload=wl)
    d1 = sp_lin - sp_ucb
    d2 = sp_lin - sp_fixed
    print(f"    -> {wl}: LinUCB vs UCB_coarse: {d1:+.3f} | LinUCB vs fixed_4: {d2:+.3f}")

print("\nSimulation smoke test COMPLETE")
