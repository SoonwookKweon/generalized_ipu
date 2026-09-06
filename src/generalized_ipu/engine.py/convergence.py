import numpy as np

class ConvergenceEvaluator:
    def __init__(self, relative_gap: float = 0.01, absolute_diff: float = 1.0):
        self.relative_gap = relative_gap
        self.absolute_diff = absolute_diff

    def evaluate(self, current_sums: np.ndarray, target_sums: np.ndarray) -> tuple[bool, float]:
        valid = target_sums > 0
        diffs = np.abs(current_sums[valid] - target_sums[valid])
        gaps = diffs / target_sums[valid]
        
        max_gap = float(np.max(gaps)) if len(gaps) > 0 else 0.0
        is_converged = max_gap <= self.relative_gap
        return is_converged, max_gap