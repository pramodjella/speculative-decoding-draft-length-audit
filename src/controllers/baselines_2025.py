"""Faithful-ish reimplementations of 2025 adaptive-SD baselines for head-to-head.

Comparable at batch=1, draft-length / stopping decisions:
  - EXP3Spec      : BanditSpec's adversarial variant (EXP3 over draft lengths).
                    (UCBSpec ~ our existing UCB controller.)
  - TapOutStyle   : TapOut's "MAB over parameter-free stopping heuristics" — a UCB
                    bandit selects WHICH stop-rule to apply this step; the rule
                    then decides per-token stop/continue. Context-free (the rule
                    choice does not condition on per-step features).

NOT reproduced head-to-head (different paradigm/setting; cite as related):
  - SpecKV (offline supervised MLP), Nightjar (serving-load-aware MAB).

All expose: choose()->K (max for the step); optional should_stop(i, entropy, margin);
update(K, accepted, cycle_time=None).
"""
import math
import numpy as np


class EXP3Spec:
    """EXP3 adversarial bandit over draft lengths (BanditSpec-style)."""
    def __init__(self, arms=(1, 2, 4, 8), gamma=0.1, seed=0):
        self.arms = list(arms)
        self.gamma = gamma
        self.w = {a: 1.0 for a in self.arms}
        self._p = {a: 1.0 / len(self.arms) for a in self.arms}
        self._last = self.arms[0]
        self._rng = np.random.default_rng(seed)

    def choose(self, *_, **__):
        W = sum(self.w.values()); n = len(self.arms)
        self._p = {a: (1 - self.gamma) * self.w[a] / W + self.gamma / n for a in self.arms}
        self._last = int(self._rng.choice(self.arms, p=[self._p[a] for a in self.arms]))
        return self._last

    def update(self, arm, accepted, cycle_time=None, K=None):
        k = K or arm
        reward = (accepted + 1) / cycle_time if (cycle_time and cycle_time > 0) else (accepted + 1) / (k + 1)
        reward = min(max(reward if cycle_time is None else reward / 200.0, 0.0), 1.0)  # ~[0,1]
        if arm in self.w:
            est = reward / max(1e-6, self._p[arm])
            self.w[arm] *= math.exp(self.gamma * est / len(self.arms))


class TapOutStyle:
    """UCB MAB over a set of parameter-free STOPPING heuristics (TapOut-style).

    Arms are heuristics; the chosen heuristic decides per-token stop/continue this
    step. Context-free: the arm selection uses only per-arm reward statistics.
    """
    HEURISTICS = [
        ("fixed", 8),         # never stop early (draft full max)
        ("fixed", 2),         # short fixed
        ("ent", 1.0),         # stop when entropy > 1.0
        ("ent", 2.0),         # stop when entropy > 2.0
        ("margin", 0.3),      # stop when top1-top2 margin < 0.3 (low confidence)
    ]

    def __init__(self, max_len=8, c=2.0):
        self.max_len = max_len
        self.c = c
        self.arms = list(range(len(self.HEURISTICS)))
        self._val = {a: 0.0 for a in self.arms}
        self._n = {a: 0 for a in self.arms}
        self._t = 0
        self._cur = 0

    def choose(self, *_, **__):
        self._t += 1
        for a in self.arms:
            if self._n[a] == 0:
                self._cur = a; break
        else:
            self._cur = max(self.arms, key=lambda a: self._val[a] + self.c * math.sqrt(math.log(self._t) / self._n[a]))
        return self.max_len      # draft up to max; should_stop enforces the heuristic

    def should_stop(self, i, entropy, margin):
        kind, thr = self.HEURISTICS[self._cur]
        if kind == "fixed":
            return (i + 1) >= thr
        if kind == "ent":
            return entropy > thr
        if kind == "margin":
            return margin < thr
        return False

    def update(self, arm_k, accepted, cycle_time=None, K=None):
        k = K or arm_k
        reward = (accepted + 1) / cycle_time if (cycle_time and cycle_time > 0) else (accepted + 1) / (k + 1)
        if cycle_time:
            reward = min(reward / 200.0, 1.0)
        a = self._cur
        self._n[a] += 1
        self._val[a] += (reward - self._val[a]) / self._n[a]
