import numpy as np
from scipy.sparse import csr_matrix
from .bounds import WeightBoundController
from .convergence import ConvergenceEvaluator
from ..matrix.scaling import SparseScalingKernel

class GeneralizedIPUEngine:
    def __init__(self, matrix: csr_matrix, targets: np.ndarray, max_iterations: int = 100,
                 relative_gap: float = 0.01, weight_floor: float = 1e-5):
        self.matrix = matrix
        self.targets = targets
        self.max_iterations = max_iterations
        self.bound_controller = WeightBoundController(weight_floor=weight_floor)
        self.evaluator = ConvergenceEvaluator(relative_gap=relative_gap)

    def fit(self, initial_weights: np.ndarray = None) -> np.ndarray:
        n_samples = self.matrix.shape[0]
        weights = initial_weights.copy() if initial_weights is not None else np.ones(n_samples, dtype=np.float64)

        for iteration in range(self.max_iterations):
            current_sums = SparseScalingKernel.compute_weighted_sum(self.matrix, weights)
            
            is_converged, max_gap = self.evaluator.evaluate(current_sums, self.targets)
            if is_converged:
                break

            valid_mask = current_sums > 0
            ratios = np.ones_like(current_sums)
            ratios[valid_mask] = self.targets[valid_mask] / current_sums[valid_mask]

            updated_weights = SparseScalingKernel.update_weights(weights, self.matrix, ratios)
            weights = self.bound_controller.apply_bounds(weights, updated_weights)

        return weights