class AcceptanceHistoryController:
    """Adjusts draft length dynamically based on recent token acceptance rate."""
    def __init__(self, arms: tuple = (1, 2, 3, 4, 6, 8), window_size: int = 5, upper_threshold: float = 0.85, lower_threshold: float = 0.5):
        self.arms = arms
        self.window_size = window_size
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.history = []
        self.current_idx = len(arms) // 2  # Start in the middle of candidate arms
        
    def choose(self, *_):
        return self.arms[self.current_idx]
        
    def update(self, arm: int, accepted: int):
        # We record the acceptance rate of this step: accepted / arm
        rate = accepted / max(1, arm)
        self.history.append(rate)
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        # If we have enough history, update the current arm choice
        if len(self.history) == self.window_size:
            avg_rate = sum(self.history) / len(self.history)
            if avg_rate > self.upper_threshold:
                # Move to next larger arm
                self.current_idx = min(len(self.arms) - 1, self.current_idx + 1)
            elif avg_rate < self.lower_threshold:
                # Move to next smaller arm
                self.current_idx = max(0, self.current_idx - 1)
