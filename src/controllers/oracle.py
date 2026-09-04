class OracleController:
    """Choose the absolute best draft length based on true acceptance outcomes for the step."""
    def __init__(self, arms: tuple = (1, 2, 3, 4, 6, 8), t_draft: float = 5.0, t_target: float = 25.0):
        self.arms = arms
        self.t_draft = t_draft
        self.t_target = t_target
        
    def choose_for_step(self, acceptances: list[bool]) -> int:
        # We find where the first rejection is (0-indexed)
        first_rej = len(acceptances)
        for i, acc in enumerate(acceptances):
            if not acc:
                first_rej = i
                break
                
        best_arm = self.arms[0]
        best_speedup = -1.0
        
        for K in self.arms:
            accepted = min(K, first_rej)
            speedup = (accepted + 1) * self.t_target / (K * self.t_draft + self.t_target)
            if speedup > best_speedup:
                best_speedup = speedup
                best_arm = K
                
        return best_arm
