"""Smoke and integration tests for speculative decoding controllers and simulation backbone."""
import numpy as np
import pytest
from src.controllers import EntropyThreshold, EpsilonGreedy, UCB, AcceptanceHistoryController, OracleController
from src.serve.simulator import simulate_step
from src.bench.harness import run_experiment_stream

def test_controllers_initialization():
    """Test that all controllers initialize with expected defaults."""
    entropy = EntropyThreshold(tau=1.0)
    assert entropy.tau == 1.0
    
    eps_greedy = EpsilonGreedy(eps=0.1, seed=42)
    assert eps_greedy.eps == 0.1
    
    ucb = UCB(c=0.5)
    assert ucb.c == 0.5
    
    history = AcceptanceHistoryController()
    assert history.window_size == 5
    
    oracle = OracleController()
    assert len(oracle.arms) == 6

def test_controllers_choice_and_update():
    """Test that controllers can make choices and accept feedback/updates."""
    # EntropyThreshold
    entropy = EntropyThreshold(tau=1.0)
    assert entropy.choose([0.2, 0.5, 1.2, 0.4]) == 2 # stops at 1.2
    assert entropy.choose([0.2, 0.5, 0.8, 0.9]) == 4 # does not stop
    
    # EpsilonGreedy
    eg = EpsilonGreedy(arms=(1, 2, 4), eps=0.0, seed=42) # eps=0 means strict exploitation
    eg._val = {1: 1.0, 2: 2.0, 4: 0.5}
    assert eg.choose() == 2
    eg.update(2, 0.1)
    
    # UCB
    ucb = UCB(arms=(1, 2, 4), c=1.0)
    # The first 3 choices will try each arm once (N=0)
    assert ucb.choose() == 1
    ucb.update(1, 1.0)
    assert ucb.choose() == 2
    ucb.update(2, 1.5)
    assert ucb.choose() == 4
    ucb.update(4, 2.0)
    
    # AcceptanceHistory
    history = AcceptanceHistoryController(arms=(1, 2, 3), window_size=2, upper_threshold=0.8, lower_threshold=0.4)
    # starts in middle (index 1 -> arm 2)
    assert history.choose() == 2
    # record high acceptance rate: 2/2 = 1.0
    history.update(2, 2)
    history.update(2, 2)
    # moving average is 1.0 > 0.8 -> current_idx increases to index 2 (arm 3)
    assert history.choose() == 3
    
    # Oracle
    oracle = OracleController(arms=(1, 2, 4), t_draft=5.0, t_target=25.0)
    # First failure is at index 2 (meaning tokens at 0, 1 are accepted, 2 is rejected)
    # Let's test what K oracle chooses
    assert oracle.choose_for_step([True, True, False, False]) == 2

def test_simulator_step():
    """Test that simulate_step executes and returns correct structure."""
    rng = np.random.default_rng(42)
    
    # Test with fixed draft length
    res = simulate_step("humaneval", 4, 0, rng)
    assert res["workload"] == "humaneval"
    assert res["chosen_length"] == 4
    assert "accepted_count" in res
    assert "spec_latency" in res
    assert "ar_latency" in res
    assert "wasted_tokens" in res
    assert len(res["entropies"]) == 4
    assert len(res["acceptances"]) == 4

    # Test with EntropyThreshold controller
    entropy_ctrl = EntropyThreshold(tau=0.5)
    res = simulate_step("gsm8k", entropy_ctrl, 1, rng)
    assert res["chosen_length"] >= 1
    
    # Test with Bandit controller
    ucb_ctrl = UCB(c=1.0)
    res = simulate_step("mt_bench", ucb_ctrl, 2, rng)
    assert res["chosen_length"] in ucb_ctrl.arms

def test_harness_stream():
    """Test running a short simulation stream via the harness."""
    records = run_experiment_stream(
        workload="spec_bench",
        policy_name="ucb",
        controller_fn=lambda: UCB(c=0.5),
        num_steps=10,
        batch_size=2,
        seed=42
    )
    assert len(records) == 10
    for r in records:
        assert r["workload"] == "spec_bench"
        assert r["batch_size"] == 2
        assert "net_speedup" in r
