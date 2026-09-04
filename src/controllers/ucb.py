import math
from dataclasses import dataclass, field

@dataclass
class UCB:
    """BanditSpec-style UCB1 over draft lengths."""
    arms: tuple = (1, 2, 3, 4, 6, 8)
    c: float = 1.0
    _val: dict = field(default_factory=dict, init=False)
    _n: dict = field(default_factory=dict, init=False)
    _t: int = field(default=0, init=False)

    def __post_init__(self):
        self._val = {a: 0.0 for a in self.arms}
        self._n = {a: 0 for a in self.arms}
        self._t = 0

    def choose(self, *_):
        self._t += 1
        for a in self.arms:
            if self._n[a] == 0:
                return a
        return max(self.arms, key=lambda a: self._val[a] + self.c * math.sqrt(math.log(self._t) / self._n[a]))

    def update(self, arm: int, reward: float):
        if arm not in self._n:
            return
        self._n[arm] += 1
        self._val[arm] += (reward - self._val[arm]) / self._n[arm]
