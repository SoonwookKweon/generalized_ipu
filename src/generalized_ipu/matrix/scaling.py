"""반복 루프가 사용하는 희소 연산 커널.

한 회차는 희소 행렬-벡터 곱 두 번으로 끝난다. 가중합 계산에 A^T w 를, 가중치
갱신에 A (r - 1) 을 쓴다. 따라서 반복 1회의 연산 복잡도는 비영 성분 수에
비례한다.
"""

import numpy as np
from scipy.sparse import csr_matrix

ROW_SUM_FLOOR = 1.0


class SparseScalingKernel:
    """반복 루프가 사용하는 희소 연산.

    상태를 가지지 않는 정적 메서드와, 전치 행렬을 재사용하는 인스턴스 형태를
    함께 제공한다. 회차 수가 많으면 인스턴스 형태가 유리하다.
    """

    def __init__(self, matrix: csr_matrix):
        self.matrix = matrix.tocsr()
        self.transposed = self.matrix.T.tocsr()
        row_sums = np.asarray(self.matrix.sum(axis=1)).ravel()
        self.row_sums = np.where(row_sums == 0.0, ROW_SUM_FLOOR, row_sums)
        self.inactive_units = np.flatnonzero(row_sums == 0.0)

    # ------------------------------------------------------------- 인스턴스

    def weighted_sums(self, weights: np.ndarray) -> np.ndarray:
        """가중합 S = A^T w 를 계산한다."""
        return self.transposed.dot(np.asarray(weights, dtype="float64"))

    def next_weights(
        self, weights: np.ndarray, ratios: np.ndarray, eta: float = 1.0
    ) -> np.ndarray:
        """동시 갱신식을 적용한다.

        w_i(t+1) = w_i(t) * (1 + eta * sum_k[(r_k - 1) a_ik] / sum_k[a_ik])
        """
        weights = np.asarray(weights, dtype="float64")
        adjustment = self.matrix.dot(np.asarray(ratios, dtype="float64") - 1.0) / self.row_sums
        return weights * (1.0 + eta * adjustment)

    # ------------------------------------------------------------- 정적

    @staticmethod
    def compute_weighted_sum(matrix: csr_matrix, weights: np.ndarray) -> np.ndarray:
        """전치 행렬을 보관하지 않고 가중합을 한 번 계산한다."""
        return matrix.T.dot(np.asarray(weights, dtype="float64"))

    @staticmethod
    def update_weights(
        weights: np.ndarray, matrix: csr_matrix, ratios: np.ndarray, eta: float = 1.0
    ) -> np.ndarray:
        """행 합을 보관하지 않고 동시 갱신식을 한 번 적용한다."""
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        row_sums = np.where(row_sums == 0.0, ROW_SUM_FLOOR, row_sums)
        adjustment = matrix.dot(np.asarray(ratios, dtype="float64") - 1.0) / row_sums
        return np.asarray(weights, dtype="float64") * (1.0 + eta * adjustment)
