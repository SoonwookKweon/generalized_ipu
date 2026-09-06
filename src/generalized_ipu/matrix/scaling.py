import numpy as np
from scipy.sparse import csr_matrix

class SparseScalingKernel:
    @staticmethod
    def compute_weighted_sum(matrix: csr_matrix, weights: np.ndarray) -> np.ndarray:
        # S = A^T * w (희소 행렬-벡터 곱)
        return matrix.T.dot(weights)

    @staticmethod
    def update_weights(weights: np.ndarray, matrix: csr_matrix, ratios: np.ndarray, eta: float = 1.0) -> np.ndarray:
        # w_i^(t+1) = w_i^(t) * [1 + eta * sum_k((r_k - 1) * a_ik / sum_m a_im)]
        row_sums = np.array(matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0 # 0 나누기 방지
        
        delta = ratios - 1.0
        scaled_delta = matrix.dot(delta) / row_sums
        
        new_weights = weights * (1.0 + eta * scaled_delta)
        return new_weights