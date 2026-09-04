class EntropyThreshold:
    """SVIP-style: draft until the draft model's next-token entropy exceeds tau."""
    def __init__(self, tau: float = 1.0, max_len: int = 8):
        self.tau = tau
        self.max_len = max_len

    def choose(self, entropies: list[float]) -> int:
        n = 0
        for e in entropies[:self.max_len]:
            if e > self.tau:
                break
            n += 1
        return max(1, n)
