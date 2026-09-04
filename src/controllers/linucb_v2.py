"""EntropyMarginLinUCB — improved contextual bandit for adaptive draft length.

Positioning (from the 2025-26 landscape review):
  - Online + CONTEXTUAL bandit (vs context-free BanditSpec/TapOut/Nightjar): the
    decision conditions on per-step draft signals, not just arm statistics.
  - Bandit-light, training-free (vs RL: LTD/PPOW).
  - Uses draft-quality features incl. the **top-1/top-2 logit margin**, which the
    survey flags as essentially absent across prior work.
  - **Throughput-aware reward** (vs acceptance-length proxy): reward is accepted
    tokens per unit *wall-clock* of the draft-verify cycle, addressing LTD's
    critique that proxy metrics ignore true time cost.

Context vector x_t (d=4), each ~[0,1]:
  [0] entropy   : draft next-token entropy / log2(vocab)        (hard -> short)
  [1] margin    : top1-top2 probability gap                     (confident -> long)
  [2] accept    : rolling acceptance rate (window=10)           (high -> long)
  [3] progress  : generation position / max_steps               (position effect)

LinUCB (Li et al., WWW'10): per-arm A_k, b_k; choose argmax theta_k^T x + alpha*sqrt(x^T A_k^-1 x);
Sherman-Morrison rank-1 update. Round-robin warm-up. Regret O(d sqrt(T log T)).
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


def top1_top2_margin(probs) -> float:
    """top-1 minus top-2 probability from a probability vector (numpy or torch->np)."""
    p = np.asarray(probs, dtype=np.float64)
    if p.size < 2:
        return 1.0
    idx = np.argpartition(p, -2)[-2:]
    a, b = np.sort(p[idx])[::-1]
    return float(a - b)


@dataclass
class EntropyMarginLinUCB:
    arms: tuple = (1, 2, 4, 8)
    alpha: float = 1.0                 # exploration
    d: int = 4
    warmup_rounds: int = 2
    vocab_log2: float = 17.0           # ~log2(128256) for Llama; set per model
    window: int = 10
    reward_clip: float = 1.0

    _A: dict = field(default_factory=dict, init=False)
    _b: dict = field(default_factory=dict, init=False)
    _Ainv: dict = field(default_factory=dict, init=False)
    _n: dict = field(default_factory=dict, init=False)
    _t: int = field(default=0, init=False)
    _accept_hist: list = field(default_factory=list, init=False)
    _step: int = field(default=0, init=False)
    _max_steps: int = field(default=128, init=False)
    _last_x: Optional[np.ndarray] = field(default=None, init=False)
    _rmax: float = field(default=1e-6, init=False)   # running max reward (for normalisation)

    def __post_init__(self):
        for a in self.arms:
            self._A[a] = np.eye(self.d)
            self._b[a] = np.zeros(self.d)
            self._Ainv[a] = np.eye(self.d)
            self._n[a] = 0

    def set_max_steps(self, m):
        self._max_steps = max(1, m)

    def reset_episode(self):
        self._accept_hist.clear(); self._step = 0; self._last_x = None

    def _context(self, entropy, margin):
        h_ent = min(max(entropy / self.vocab_log2, 0.0), 1.0)
        h_mar = min(max(margin, 0.0), 1.0)
        h_acc = (sum(self._accept_hist[-self.window:]) / len(self._accept_hist[-self.window:])
                 if self._accept_hist else 0.5)
        h_prog = min(self._step / self._max_steps, 1.0)
        return np.array([h_ent, h_mar, h_acc, h_prog])

    def choose(self, entropy: float = 0.0, margin: float = 0.5) -> int:
        """Pick draft length K from current draft signals (called at step start)."""
        self._t += 1; self._step += 1
        x = self._context(entropy, margin); self._last_x = x

        mn = min(self._n[a] for a in self.arms)
        if mn < self.warmup_rounds:                      # round-robin warm-up
            return min(self.arms, key=lambda a: self._n[a])

        best, best_s = self.arms[0], -1e18
        for a in self.arms:
            theta = self._Ainv[a] @ self._b[a]
            score = float(theta @ x) + self.alpha * float(np.sqrt(x @ self._Ainv[a] @ x))
            if score > best_s:
                best_s, best = score, a
        return best

    def update(self, arm: int, accepted: int, cycle_time: Optional[float] = None, K: int = None):
        """Update with a THROUGHPUT reward = (accepted+1)/cycle_time, normalised.

        cycle_time = measured wall-clock (s) of this draft+verify step. If absent,
        fall back to a compute-cost proxy (accepted+1)/(K+1) so the controller still
        runs in environments without per-step timing.
        """
        if arm not in self._A or self._last_x is None:
            return
        x = self._last_x
        if cycle_time and cycle_time > 0:
            raw = (accepted + 1) / cycle_time            # tokens / second this step
        else:
            raw = (accepted + 1) / ((K or arm) + 1)      # proxy fallback
        self._rmax = max(self._rmax, raw)
        reward = min(raw / self._rmax, self.reward_clip)  # ~[0,1], throughput-relative

        self._n[arm] += 1
        self._A[arm] += np.outer(x, x)
        self._b[arm] += reward * x
        Ai = self._Ainv[arm]; Aix = Ai @ x
        self._Ainv[arm] = Ai - np.outer(Aix, Aix) / (1.0 + float(x @ Aix))

        rate = accepted / max(1, (K or arm))
        self._accept_hist.append(rate)

    def get_stats(self):
        return {"t": self._t, "rmax": self._rmax,
                **{f"theta_{a}": (self._Ainv[a] @ self._b[a]).round(3).tolist() for a in self.arms},
                **{f"n_{a}": self._n[a] for a in self.arms}}
