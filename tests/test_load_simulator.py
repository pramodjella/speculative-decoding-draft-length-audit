"""Locks the core invariants of the load+content thesis instrument."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
from serve.load_simulator import step_speedup, oracle_K, ContentStream, ACCEPT_PROBS
from controllers import ContextLinUCB, NightjarStyle


def _opt_fixed_K(B, arms=(1, 2, 4, 8)):
    # expected-acceptance optimal K at batch B on the easy 'humaneval' profile
    p = ACCEPT_PROBS["humaneval"]
    best_k, best_s = arms[0], -1.0
    for K in arms:
        exp_acc, cum = 0.0, 1.0
        for i in range(K):
            cum *= p[i]; exp_acc += cum
        s = step_speedup(B, K, round(exp_acc))
        if s > best_s:
            best_s, best_k = s, K
    return best_k


def test_optimal_K_decreases_with_batch():
    # The whole thesis premise: optimal fixed K shrinks as batch (load) grows.
    ks = [_opt_fixed_K(B) for B in [1, 8, 32, 64]]
    assert ks[0] >= ks[1] >= ks[2] >= ks[3]
    assert ks[0] >= 4 and ks[-1] == 1


def test_long_K_drops_below_parity_at_high_batch():
    # At high batch, long speculation must be able to hurt (speedup < 1x).
    assert step_speedup(64, 8, 8) < 1.0
    assert step_speedup(1, 8, 8) > 1.5


def test_contextlinucb_feature_dims():
    content = ContextLinUCB(features=("entropy", "accept"))
    load = ContextLinUCB(features=("batch", "load"))
    combined = ContextLinUCB(features=("entropy", "accept", "batch", "load"),
                             interactions=(("batch", "accept"), ("batch", "entropy")))
    assert content.d == 3            # 2 feats + bias
    assert load.d == 3
    assert combined.d == 7           # 4 feats + 2 interactions + bias
    # only interactions whose operands are active features are kept
    dropped = ContextLinUCB(features=("entropy",), interactions=(("batch", "accept"),))
    assert dropped.d == 2            # 1 feat + bias, interaction dropped


def test_controllers_pick_valid_arms():
    rng = np.random.default_rng(0)
    for ctrl in [ContextLinUCB(), NightjarStyle()]:
        ctrl.set_max_steps(50)
        ctrl.reset_episode() if hasattr(ctrl, "reset_episode") else None
        stream = ContentStream("gsm8k")
        for _ in range(50):
            ent, acc = stream.step(rng)
            B = int(rng.choice([1, 8, 32]))
            K = ctrl.choose_k(entropy=ent[0], batch=B)
            assert K in ctrl.arms
            ctrl.update(K, 0.5, accepted=1, K=K)


def test_oracle_is_upper_bound_per_step():
    rng = np.random.default_rng(1)
    stream = ContentStream("spec_bench")
    arms = (1, 2, 4, 8)
    for _ in range(200):
        ent, acc = stream.step(rng)
        B = int(rng.choice([1, 8, 32]))
        ok = oracle_K(B, acc, arms)
        from serve.load_simulator import accepted_given_K
        os_ = step_speedup(B, ok, accepted_given_K(acc, ok))
        for K in arms:
            assert os_ >= step_speedup(B, K, accepted_given_K(acc, K)) - 1e-9
