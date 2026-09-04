"""Load+content contextual controllers for draft-length selection (thesis core).

The verified-open contribution (see novelty_memo.md): a contextual bandit whose
context combines per-step draft-CONTENT signals (entropy, acceptance history) with
serving-LOAD signals (batch size, load factor). No published method uses both
(content-only: TapOut/SpecKV/AdaEAGLE; load-only: Nightjar/SpecuStream/BASS).

`ContextLinUCB` is a single LinUCB whose **active feature set is configurable**, so
the head-to-head is a clean ablation of the SAME algorithm:
    content_only = ContextLinUCB(features=("entropy", "accept"))
    load_only    = ContextLinUCB(features=("batch", "load"))
    combined     = ContextLinUCB(features=("entropy", "accept", "batch", "load"))
A constant bias feature is always included so arms have learnable baselines.

`NightjarStyle` is a published-method baseline: a per-batch-bucket UCB over draft
length (load-aware, content-blind), in the spirit of Nightjar (arXiv:2512.22420).
"""
import math
import numpy as np

# Normalization constants for features (simulator-scale, vocab-free entropies).
_ENT_SCALE = 4.0     # sim entropies ~0.05..~3.5
_LOG2_BMAX = 6.0     # batch up to 64 -> log2 = 6


class ContextLinUCB:
    """LinUCB over draft lengths with a configurable context.

    Args:
        arms: candidate draft lengths.
        alpha: exploration parameter.
        features: subset of {"entropy","accept","progress","batch","load"} to use.
        warmup_rounds: round-robin cycles before UCB exploitation.
    """

    ALL_FEATURES = ("entropy", "accept", "progress", "batch", "load")

    def __init__(self, arms=(1, 2, 4, 8), alpha=1.0,
                 features=("entropy", "accept", "batch", "load"),
                 interactions=(), warmup_rounds=2):
        self.arms = tuple(arms)
        self.alpha = alpha
        self.features = tuple(f for f in self.ALL_FEATURES if f in features)
        if not self.features:
            raise ValueError("ContextLinUCB needs at least one feature")
        # Cross terms (e.g. ("batch","accept")) let the linear model represent
        # CONJUNCTIONS like "draft long only when load is low AND the span is easy" —
        # which a purely additive content+load model cannot capture. Only keep
        # interactions whose both operands are active features.
        self.interactions = tuple(
            (a, b) for (a, b) in interactions if a in self.features and b in self.features
        )
        self.warmup_rounds = warmup_rounds
        self.d = len(self.features) + len(self.interactions) + 1  # + bias

        self._A = {a: np.eye(self.d) for a in self.arms}
        self._b = {a: np.zeros(self.d) for a in self.arms}
        self._A_inv = {a: np.eye(self.d) for a in self.arms}
        self._n = {a: 0 for a in self.arms}
        self._t = 0
        self._accept_hist = []
        self._step = 0
        self._max_steps = 128
        self._last_x = None

    # -- episode lifecycle (parity with LinUCBController) --
    def set_max_steps(self, m):
        self._max_steps = max(1, m)

    def reset_episode(self):
        self._accept_hist.clear()
        self._step = 0
        self._last_x = None

    def _build_context(self, entropy, batch, load):
        vals = {
            "entropy": min(entropy / _ENT_SCALE, 1.0),
            "accept": (sum(self._accept_hist[-10:]) / len(self._accept_hist[-10:]))
                      if self._accept_hist else 0.5,
            "progress": min(self._step / self._max_steps, 1.0),
            "batch": min(math.log2(max(1, batch)) / _LOG2_BMAX, 1.0),
            "load": load if load is not None else min(batch / 64.0, 1.0),
        }
        x = [vals[f] for f in self.features]
        x += [vals[a] * vals[b] for (a, b) in self.interactions]
        x += [1.0]  # trailing bias
        return np.array(x, dtype=float)

    def _in_warmup(self):
        return min(self._n.values()) < self.warmup_rounds

    def choose_k(self, entropy=0.0, batch=1, load=None) -> int:
        self._t += 1
        self._step += 1
        x = self._build_context(entropy, batch, load)
        self._last_x = x

        if self._in_warmup():
            return min(self.arms, key=lambda a: self._n[a])

        best_a, best_s = self.arms[0], -float("inf")
        for a in self.arms:
            theta = self._A_inv[a] @ self._b[a]
            score = float(theta @ x) + self.alpha * float(np.sqrt(x @ self._A_inv[a] @ x))
            if score > best_s:
                best_s, best_a = score, a
        return best_a

    def update(self, arm, reward, accepted=-1, K=-1):
        if arm not in self._A or self._last_x is None:
            return
        x = self._last_x
        self._n[arm] += 1
        self._A[arm] += np.outer(x, x)
        self._b[arm] += reward * x
        A_inv = self._A_inv[arm]
        Ax = A_inv @ x
        self._A_inv[arm] = A_inv - np.outer(Ax, Ax) / (1.0 + float(x @ Ax))
        if accepted >= 0 and K > 0:
            self._accept_hist.append(accepted / K)


class NightjarStyle:
    """Load-aware, content-blind baseline: per-batch-bucket UCB over draft length.

    Mirrors Nightjar's core idea (learn optimal speculative length per batch size,
    no per-token content signal). Context = batch-size bucket only.
    """

    def __init__(self, arms=(1, 2, 4, 8), c=2.0):
        self.arms = tuple(arms)
        self.c = c
        self._val = {}   # (bucket, arm) -> mean reward
        self._n = {}     # (bucket, arm) -> count
        self._t = {}     # bucket -> total pulls
        self._last = None

    @staticmethod
    def _bucket(batch):
        return int(round(math.log2(max(1, batch))))  # 1,2,4,8,16,32,64 -> 0..6

    def set_max_steps(self, m):
        pass

    def reset_episode(self):
        pass

    def choose_k(self, entropy=0.0, batch=1, load=None) -> int:
        bk = self._bucket(batch)
        self._t[bk] = self._t.get(bk, 0) + 1
        # explore any unpulled arm in this bucket first
        for a in self.arms:
            if self._n.get((bk, a), 0) == 0:
                self._last = (bk, a)
                return a
        t = self._t[bk]
        best_a = max(self.arms, key=lambda a:
                     self._val[(bk, a)] + self.c * math.sqrt(math.log(t) / self._n[(bk, a)]))
        self._last = (bk, best_a)
        return best_a

    def update(self, arm, reward, accepted=-1, K=-1):
        if self._last is None:
            return
        bk, a = self._last
        self._n[(bk, a)] = self._n.get((bk, a), 0) + 1
        self._val[(bk, a)] = self._val.get((bk, a), 0.0) + \
            (reward - self._val.get((bk, a), 0.0)) / self._n[(bk, a)]
