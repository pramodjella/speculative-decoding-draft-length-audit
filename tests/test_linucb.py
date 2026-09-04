"""Unit tests for LinUCBController.

Validates:
  1. Initialization and basic interface
  2. Context feature computation
  3. Arm selection responds to context
  4. Learning from reward feedback
  5. Overhead measurement (must be < 100μs)
  6. Episode reset behavior
  7. Sherman-Morrison inverse consistency
"""
import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from controllers.linucb import LinUCBController


def test_initialization():
    """LinUCB initializes with correct arm set and identity matrices."""
    ctrl = LinUCBController(arms=(1, 4, 8), alpha=1.0)
    assert ctrl.arms == (1, 4, 8)
    assert ctrl.d == 4
    assert ctrl.alpha == 1.0
    assert ctrl._t == 0
    for a in ctrl.arms:
        assert np.allclose(ctrl._A[a], np.eye(4))
        assert np.allclose(ctrl._b[a], np.zeros(4))
        assert np.allclose(ctrl._A_inv[a], np.eye(4))
    print("✓ test_initialization passed")


def test_choose_returns_valid_arm():
    """choose() always returns an arm from the candidate set."""
    ctrl = LinUCBController(arms=(1, 4, 8))
    ctrl.set_max_steps(64)
    ctrl.reset_episode()
    for _ in range(50):
        K = ctrl.choose(entropy=np.random.uniform(0, 5))
        assert K in (1, 4, 8), f"Got invalid arm {K}"
        # Provide feedback
        ctrl.update(K, reward=np.random.uniform(0, 1))
    print("✓ test_choose_returns_valid_arm passed")


def test_context_features_range():
    """All context features should be in [0, 1]."""
    ctrl = LinUCBController(arms=(1, 4, 8))
    ctrl.set_max_steps(64)
    ctrl.reset_episode()

    for step in range(20):
        entropy = np.random.uniform(0, 10)
        _ = ctrl.choose(entropy=entropy)
        ctx = ctrl._last_context
        assert ctx is not None
        for i, name in enumerate(["entropy", "accept", "progress", "volatility"]):
            assert 0.0 <= ctx[i] <= 1.01, f"Feature {name} = {ctx[i]} out of range at step {step}"
        ctrl.update(ctrl.arms[0], reward=0.5)
    print("✓ test_context_features_range passed")


def test_learns_from_high_entropy():
    """After many high-entropy steps with K=8 failing, LinUCB should prefer K=1."""
    ctrl = LinUCBController(arms=(1, 4, 8), alpha=0.5)
    ctrl.set_max_steps(200)
    ctrl.reset_episode()

    # Simulate: high entropy → K=8 gets bad reward, K=1 gets good reward
    for _ in range(100):
        _ = ctrl.choose(entropy=8.0)  # High entropy
        # Punish large arms
        ctrl.update(8, reward=0.1)
        _ = ctrl.choose(entropy=8.0)
        ctrl.update(1, reward=0.9)  # Reward small arm

    # After learning, with high entropy the controller should prefer small K
    ctrl.reset_episode()
    ctrl.set_max_steps(200)
    choices = []
    for _ in range(20):
        K = ctrl.choose(entropy=8.0)
        choices.append(K)
        ctrl.update(K, reward=0.5)

    avg_K = sum(choices) / len(choices)
    print(f"  Average K after high-entropy training: {avg_K:.1f}")
    assert avg_K < 5.0, f"Expected K < 5 for high entropy, got {avg_K}"
    print("✓ test_learns_from_high_entropy passed")


def test_learns_from_low_entropy():
    """After many low-entropy steps with K=8 succeeding, LinUCB should prefer K=8."""
    ctrl = LinUCBController(arms=(1, 4, 8), alpha=0.5)
    ctrl.set_max_steps(200)
    ctrl.reset_episode()

    # Simulate: low entropy → K=8 gets great reward
    for _ in range(100):
        _ = ctrl.choose(entropy=0.5)  # Low entropy
        ctrl.update(8, reward=0.9)    # Reward large arm
        _ = ctrl.choose(entropy=0.5)
        ctrl.update(1, reward=0.3)    # Punish small arm

    # After learning, with low entropy the controller should prefer large K
    ctrl.reset_episode()
    ctrl.set_max_steps(200)
    choices = []
    for _ in range(20):
        K = ctrl.choose(entropy=0.5)
        choices.append(K)
        ctrl.update(K, reward=0.5)

    avg_K = sum(choices) / len(choices)
    print(f"  Average K after low-entropy training: {avg_K:.1f}")
    assert avg_K > 3.0, f"Expected K > 3 for low entropy, got {avg_K}"
    print("✓ test_learns_from_low_entropy passed")


def test_overhead_under_threshold():
    """choose() + update() must complete in < 100μs (our I4 requirement)."""
    ctrl = LinUCBController(arms=(1, 4, 8))
    ctrl.set_max_steps(128)
    ctrl.reset_episode()

    # Warmup
    for _ in range(10):
        K = ctrl.choose(entropy=2.0)
        ctrl.update(K, reward=0.5)

    # Timed benchmark
    n_trials = 1000
    t0 = time.perf_counter()
    for i in range(n_trials):
        K = ctrl.choose(entropy=float(i % 10))
        ctrl.update(K, reward=0.5)
    elapsed = time.perf_counter() - t0

    avg_us = (elapsed / n_trials) * 1e6
    print(f"  Average choose() + update() overhead: {avg_us:.2f} μs")
    assert avg_us < 100.0, f"Overhead {avg_us:.2f} μs exceeds 100μs threshold"
    print("✓ test_overhead_under_threshold passed")


def test_episode_reset():
    """reset_episode() clears per-episode state but preserves learned params."""
    ctrl = LinUCBController(arms=(1, 4, 8), alpha=1.0)
    ctrl.set_max_steps(64)
    ctrl.reset_episode()

    # Run some steps to accumulate state
    for _ in range(20):
        K = ctrl.choose(entropy=3.0)
        ctrl.update(K, reward=0.5)

    # Save learned params
    A_before = {a: ctrl._A[a].copy() for a in ctrl.arms}

    # Reset episode
    ctrl.reset_episode()

    # Per-episode state should be cleared
    assert len(ctrl._entropy_history) == 0
    assert len(ctrl._accept_history) == 0
    assert ctrl._current_step == 0
    assert ctrl._last_context is None

    # Learned params should be preserved
    for a in ctrl.arms:
        assert np.allclose(ctrl._A[a], A_before[a]), "Learned params should survive reset"

    print("✓ test_episode_reset passed")


def test_sherman_morrison_consistency():
    """Verify that cached A_inv stays consistent with actual inverse of A."""
    ctrl = LinUCBController(arms=(1, 4, 8), alpha=1.0)
    ctrl.set_max_steps(128)
    ctrl.reset_episode()

    for _ in range(50):
        K = ctrl.choose(entropy=np.random.uniform(0, 8))
        ctrl.update(K, reward=np.random.uniform(0, 1))

    for a in ctrl.arms:
        actual_inv = np.linalg.inv(ctrl._A[a])
        assert np.allclose(ctrl._A_inv[a], actual_inv, atol=1e-6), \
            f"Sherman-Morrison inverse drifted for arm {a}"

    print("✓ test_sherman_morrison_consistency passed")


def test_get_stats():
    """get_stats() returns diagnostic information."""
    ctrl = LinUCBController(arms=(1, 4, 8))
    ctrl.set_max_steps(64)
    ctrl.reset_episode()

    for _ in range(10):
        K = ctrl.choose(entropy=2.0)
        ctrl.update(K, reward=0.5)

    stats = ctrl.get_stats()
    assert "t" in stats
    assert stats["t"] == 10
    for a in ctrl.arms:
        assert f"theta_{a}" in stats
        assert f"n_{a}" in stats
    print("✓ test_get_stats passed")


if __name__ == "__main__":
    print("=" * 60)
    print("  LinUCBController Unit Tests")
    print("=" * 60)

    test_initialization()
    test_choose_returns_valid_arm()
    test_context_features_range()
    test_learns_from_high_entropy()
    test_learns_from_low_entropy()
    test_overhead_under_threshold()
    test_episode_reset()
    test_sherman_morrison_consistency()
    test_get_stats()

    print("\n" + "=" * 60)
    print("  ALL 9 TESTS PASSED ✓")
    print("=" * 60)
