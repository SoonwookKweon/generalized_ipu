import numpy as np
from scipy.sparse import csr_matrix

class StructuralZeroMask:
    @staticmethod
    def apply_mask(matrix: csr_matrix, invalid_mask: np.ndarray) -> csr_matrix:
        # 구조적 영 위치의 희소 행렬 요소를 0으로 덮어씀 [cite: 1]
        matrix_csr = matrix.tocsr().copy()
        if invalid_mask.any():
            matrix_csr[invalid_mask, :] = 0
            matrix_csr.eliminate_zeros()
        return matrix_csr

class ZeroCellResolver:
    @staticmethod
    def apply_epsilon_smoothing(matrix: csr_matrix, invalid_mask: np.ndarray, epsilon: float = 1e-5) -> csr_matrix:
        # 표본 영(Sampling Zero) 유효 위치에만 분모 0 방지용 epsilon 주입 [cite: 2]
        dense = matrix.toarray()
        valid_mask = ~invalid_mask
        zero_cells = (dense == 0) & valid_mask[:, None]
        dense[zero_cells] += epsilon
        return csr_matrix(dense)