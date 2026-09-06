import numpy as np

class WeightBoundController:
    def __init__(self, weight_floor: float = 1e-5, min_ratio: float = 0.1, max_ratio: float = 10.0):
        self.weight_floor = weight_floor
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def apply_bounds(self, old_weights: np.ndarray, new_weights: np.ndarray) -> np.ndarray:
        ratios = new_weights / np.maximum(old_weights, 1e-10)
        clipped_ratios = np.clip(ratios, self.min_ratio, self.max_ratio)
        bounded_weights = old_weights * clipped_ratios
        
        # 하한선 (Weight Floor) 적용
        bounded_weights = np.maximum(bounded_weights, self.weight_floor)
        return bounded_weights