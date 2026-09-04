import random
from dataclasses import dataclass, field

@dataclass
class EpsilonGreedy:
    """Treat each candidate draft length as a bandit arm; explore with prob eps."""
    arms: tuple = (1, 2, 3, 4, 6, 8)
    eps: float = 0.1
    seed: int = 0
    _val: dict = field(default_factory=dict, init=False)
    _n: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self._val = {a: 0.0 for a in self.arms}
        self._n = {a: 0 for a in self.arms}

    def choose(self, *_):
        if self._rng.random() < self.eps:
            return self._rng.choice(self.arms)
        return max(self.arms, key=lambda a: self._val[a])

    def update(self, arm: int, reward: float):
        if arm not in self._n:
            return
        self._n[arm] += 1
        self._val[arm] += (reward - self._val[arm]) / self._n[arm]
