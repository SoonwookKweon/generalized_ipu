import numpy as np
from .base import AbstractConstraint

class TensorTarget(AbstractConstraint):
    def __init__(self, name: str, level: str, target_matrix: np.ndarray):
        super().__init__(name, level)
        self.target_matrix = target_matrix
        self.flat_target = target_matrix.flatten()

    def compute_ratio(self, current_sum: np.ndarray) -> np.ndarray:
        # 0 나누기 방지
        valid_mask = current_sum > 0
        ratios = np.ones_like(current_sum, dtype=np.float64)
        ratios[valid_mask] = self.flat_target[valid_mask] / current_sum[valid_mask]
        return ratios

class TensorFlattener:
    @staticmethod
    def flatten_nd_array(tensor: np.ndarray) -> np.ndarray:
        return tensor.flatten()