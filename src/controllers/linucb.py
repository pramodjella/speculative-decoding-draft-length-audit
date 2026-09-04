"""EntropyLinUCB: Contextual bandit for adaptive draft-length selection.

This is the core novel contribution — a LinUCB controller that conditions
draft-length selection on a 4-dimensional context vector:
  1. Draft entropy (normalized)     — current token difficulty
  2. Recent acceptance rate          — rolling draft quality signal
  3. Generation progress             — position-dependent difficulty
  4. Entropy volatility              — stability of current span

Unlike BanditSpec (ICML'25) which is context-free, and SVIP (EMNLP'25)
which uses entropy as a fixed threshold, EntropyLinUCB learns a *policy*
that maps state → arm via linear UCB with provable O(d√(T log T)) regret.

Reference: Li et al., "A Contextual-Bandit Approach to Personalized News
Article Recommendation", WWW 2010.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LinUCBController:
    """Contextual LinUCB bandit for speculative decoding draft length.

    Arms: candidate draft lengths (default {1, 4, 8} — our compact set finding).
    Context: 4D vector [entropy, accept_rate, progress, volatility].

    At each step:
      1. Observe context x_t from the current generation state
      2. For each arm k, compute UCB score: θ_k^T x_t + α√(x_t^T A_k^{-1} x_t)
      3. Select K_t = argmax_k UCB(k)
      4. After verification, update chosen arm's parameters

    Includes round-robin warm-up to ensure all arms are explored before
    exploitation begins — standard practice for LinUCB (Li et al., 2010).

    Theoretical guarantee:
      Regret(T) = O(d √(T ln(T/δ))) with probability 1-δ
      where d=4 is context dimension.
    """

    arms: tuple = (1, 4, 8)
    alpha: float = 1.0          # Exploration parameter
    d: int = 4                  # Context dimension (fixed at 4)
    warmup_rounds: int = 2      # Round-robin cycles through all arms before UCB kicks in

    # Internal state (managed by __post_init__)
    _A: dict = field(default_factory=dict, init=False)       # A_k: d×d matrix per arm
    _b: dict = field(default_factory=dict, init=False)       # b_k: d-vector per arm
    _A_inv: dict = field(default_factory=dict, init=False)   # Cached A_k^{-1}
    _t: int = field(default=0, init=False)                   # Total steps
    _arm_counts: dict = field(default_factory=dict, init=False)  # Per-arm pull counts

    # Context tracking state
    _entropy_history: list = field(default_factory=list, init=False)
    _accept_history: list = field(default_factory=list, init=False)
    _current_step: int = field(default=0, init=False)
    _max_steps: int = field(default=128, init=False)

    # Last context used (for update)
    _last_context: Optional[np.ndarray] = field(default=None, init=False)

    def __post_init__(self):
        for a in self.arms:
            self._A[a] = np.eye(self.d)
            self._b[a] = np.zeros(self.d)
            self._A_inv[a] = np.eye(self.d)
            self._arm_counts[a] = 0

    def set_max_steps(self, max_steps: int):
        """Set the max generation length for progress feature computation."""
        self._max_steps = max(1, max_steps)

    def reset_episode(self):
        """Reset per-episode tracking (call at start of each new prompt).

        Preserves learned model parameters (A, b, A_inv) across episodes.
        Only clears per-episode context features (entropy/accept history).
        """
        self._entropy_history.clear()
        self._accept_history.clear()
        self._current_step = 0
        self._last_context = None

    def _build_context(self, entropy: float = 0.0) -> np.ndarray:
        """Build the 4D context vector from current generation state.

        Features:
          [0] h_entropy:    Normalized draft entropy of last token (/ log2(vocab))
          [1] h_accept:     Rolling acceptance rate over last 10 steps
          [2] h_progress:   Current step / max steps ∈ [0, 1]
          [3] h_volatility: Std dev of entropy over last 5 steps
        """
        # Feature 1: Normalized entropy (approximate, assuming vocab ~32K → log2 ≈ 15)
        h_entropy = min(entropy / 15.0, 1.0)

        # Feature 2: Rolling acceptance rate (window=10)
        if len(self._accept_history) > 0:
            window = self._accept_history[-10:]
            h_accept = sum(window) / len(window)
        else:
            h_accept = 0.5  # Prior: neutral before any data

        # Feature 3: Generation progress
        h_progress = min(self._current_step / self._max_steps, 1.0)

        # Feature 4: Entropy volatility (std of last 5 entropies)
        if len(self._entropy_history) >= 2:
            recent = self._entropy_history[-5:]
            h_volatility = float(np.std(recent)) / 15.0  # Normalized
        else:
            h_volatility = 0.5  # Prior: moderate uncertainty

        return np.array([h_entropy, h_accept, h_progress, h_volatility])

    def _in_warmup(self) -> bool:
        """Check if we're still in the round-robin warm-up phase."""
        min_pulls = min(self._arm_counts.get(a, 0) for a in self.arms)
        return min_pulls < self.warmup_rounds

    def choose(self, entropy: float = 0.0) -> int:
        """Select draft length K given current entropy.

        During warm-up: round-robin through arms to collect initial data.
        After warm-up: LinUCB with contextual exploration.

        Args:
            entropy: Draft model's next-token entropy at current position.

        Returns:
            Selected draft length K from the arm set.
        """
        self._t += 1
        self._current_step += 1

        # Build context vector
        x = self._build_context(entropy)
        self._last_context = x

        # Record entropy for volatility tracking
        self._entropy_history.append(entropy)

        # ── Warm-up phase: round-robin ──
        if self._in_warmup():
            # Pick the arm with fewest pulls
            best_arm = min(self.arms, key=lambda a: self._arm_counts[a])
            return best_arm

        # ── Exploitation phase: LinUCB ──
        best_arm = self.arms[0]
        best_score = -float('inf')

        for a in self.arms:
            # θ_k = A_k^{-1} b_k
            theta = self._A_inv[a] @ self._b[a]

            # UCB = θ^T x + α √(x^T A^{-1} x)
            pred = float(theta @ x)
            uncertainty = float(np.sqrt(x @ self._A_inv[a] @ x))
            score = pred + self.alpha * uncertainty

            if score > best_score:
                best_score = score
                best_arm = a

        return best_arm

    def update(self, arm: int, reward: float, accepted: int = -1, K: int = -1):
        """Update the chosen arm's parameters after observing reward.

        Args:
            arm: The draft length that was used (chosen arm).
            reward: Observed reward (e.g., (accepted+1)/(K+1) from physical runner
                    or (accepted+1)*T_TARGET/spec_latency from simulator).
            accepted: Number of accepted tokens (optional, for accept rate tracking).
            K: Draft length used (optional, same as arm typically).
        """
        if arm not in self._A or self._last_context is None:
            return

        x = self._last_context

        # Track arm pull count
        self._arm_counts[arm] = self._arm_counts.get(arm, 0) + 1

        # A_k ← A_k + x x^T
        self._A[arm] += np.outer(x, x)

        # b_k ← b_k + r · x
        self._b[arm] += reward * x

        # Update cached inverse using Sherman-Morrison formula:
        # (A + xx^T)^{-1} = A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
        A_inv = self._A_inv[arm]
        Ax = A_inv @ x
        denom = 1.0 + float(x @ Ax)
        self._A_inv[arm] = A_inv - np.outer(Ax, Ax) / denom

        # Track acceptance rate for context features
        if accepted >= 0 and K > 0:
            rate = accepted / K
        else:
            # Infer from reward: in physical runner reward=(accepted+1)/(K+1)
            # In simulator reward=(accepted+1)*T_TARGET/spec_latency
            # Clamp to [0, 1] as approximate acceptance rate
            rate = min(max(reward, 0.0), 1.0)
        self._accept_history.append(rate)

    def get_stats(self) -> dict:
        """Return diagnostic stats for logging."""
        stats = {"t": self._t}
        for a in self.arms:
            theta = self._A_inv[a] @ self._b[a]
            stats[f"theta_{a}"] = theta.tolist()
            stats[f"n_{a}"] = self._arm_counts.get(a, 0)
        return stats
