"""구조적 영과 표본 영의 구별 처리.

두 유형은 판정 근거와 처리 방식이 모두 다르다. 구조적 영은 도메인 규칙으로
판정하며 모집단에서의 참값이 0이므로 색인에서 원천 제외하고 목표를 0으로
강제한다. 표본 영은 표본 관측 결과로 판정하며 참값이 양수이므로 미세 의사
빈도를 주입한 뒤 정상적으로 조정한다.

두 유형을 혼동하면 양방향의 오류가 생긴다. 구조적 영을 표본 영으로 오인하면
불가능한 개체에 가중치가 배분되고, 표본 영을 구조적 영으로 오인하면 실재하는
범주가 영구히 0으로 고정된다.
"""

from typing import List, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

DEFAULT_EPSILON = 1e-5


class StructuralZeroMask:
    """구조적 영 제약 열을 속성 행렬에서 원천 제외한다.

    마스크의 축은 제약 열(길이 K)이며, 유효한 쪽이 True 인 validity_mask 를 받는다.
    """

    @staticmethod
    def check_axis(validity_mask: np.ndarray, n_columns: int) -> np.ndarray:
        """마스크가 제약 열의 축을 따르는지 확인하고 논리 배열로 변환한다."""
        mask = np.asarray(validity_mask, dtype=bool)
        if mask.size != n_columns:
            raise ValueError(
                f"유효 마스크의 길이({mask.size})가 제약 열 수({n_columns})와 다릅니다. "
                "마스크는 기본 단위가 아니라 제약 열의 축을 따릅니다."
            )
        return mask

    @staticmethod
    def select_valid_columns(
        matrix: csr_matrix, validity_mask: np.ndarray
    ) -> Tuple[csr_matrix, np.ndarray]:
        """유효 제약 열만 남긴 행렬과 남은 열의 원래 색인을 반환한다."""
        mask = StructuralZeroMask.check_axis(validity_mask, matrix.shape[1])
        kept = np.flatnonzero(mask)
        return matrix[:, kept].tocsr(), kept

    @staticmethod
    def enforce_zero_targets(
        lower: np.ndarray,
        upper: np.ndarray,
        validity_mask: np.ndarray,
        policy: str = "force_zero",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """구조적 영 위치에 양수 목표가 들어온 경우를 처리한다.

        policy 가 'force_zero' 이면 0으로 강제하고, 'error' 이면 예외를 발생시킨다.
        """
        lower = np.asarray(lower, dtype="float64").copy()
        upper = np.asarray(upper, dtype="float64").copy()
        mask = StructuralZeroMask.check_axis(validity_mask, lower.size)

        offending = np.flatnonzero((~mask) & (upper > 0))
        if offending.size:
            if policy == "error":
                raise ValueError(
                    f"구조적 영 제약 열에 양수 목표가 입력되었습니다: 열 색인 {offending.tolist()}"
                )
            if policy != "force_zero":
                raise ValueError(f"알 수 없는 policy 입니다: {policy!r}")
            lower[offending] = 0.0
            upper[offending] = 0.0
        return lower, upper


class ZeroCellResolver:
    """표본 영 제약 열에 의사 빈도를 주입한다."""

    @staticmethod
    def detect(
        matrix: csr_matrix,
        weights: np.ndarray,
        lower: np.ndarray,
        validity_mask: np.ndarray = None,
    ) -> np.ndarray:
        """가중합이 0이면서 하한이 양수인 유효 제약 열의 색인을 반환한다."""
        weighted_sums = matrix.T.dot(np.asarray(weights, dtype="float64"))
        candidate = (weighted_sums <= 0) & (np.asarray(lower, dtype="float64") > 0)
        if validity_mask is not None:
            candidate &= StructuralZeroMask.check_axis(validity_mask, matrix.shape[1])
        return np.flatnonzero(candidate)

    @staticmethod
    def apply_epsilon_smoothing(
        matrix: csr_matrix,
        columns: np.ndarray,
        epsilon: float = DEFAULT_EPSILON,
        rows: np.ndarray = None,
    ) -> csr_matrix:
        """지정한 제약 열에만 epsilon 을 주입한다.

        희소 표현을 유지하기 위해 조밀 변환 없이 좌표 형식으로 증분 행렬을 만들어
        더한다. rows 를 주면 해당 기본 단위에만 주입하고, 생략하면 모든 기본 단위에
        주입한다.
        """
        columns = np.asarray(columns, dtype=np.int64)
        if columns.size == 0:
            return matrix.tocsr()
        if epsilon <= 0:
            raise ValueError(f"epsilon 은 양수여야 합니다: {epsilon}")

        n_rows, n_cols = matrix.shape
        row_index = np.arange(n_rows) if rows is None else np.asarray(rows, dtype=np.int64)
        if row_index.size == 0:
            raise ValueError("epsilon 을 주입할 기본 단위가 없습니다.")

        repeated_rows = np.repeat(row_index, columns.size)
        tiled_cols = np.tile(columns, row_index.size)
        increment = coo_matrix(
            (np.full(repeated_rows.size, float(epsilon)), (repeated_rows, tiled_cols)),
            shape=(n_rows, n_cols),
        )
        return (matrix + increment.tocsr()).tocsr()

    @staticmethod
    def resolve(
        matrix: csr_matrix,
        weights: np.ndarray,
        lower: np.ndarray,
        validity_mask: np.ndarray = None,
        epsilon: float = DEFAULT_EPSILON,
    ) -> Tuple[csr_matrix, List[int]]:
        """표본 영을 탐지하고 평활을 적용한 행렬과 대상 열 목록을 반환한다."""
        columns = ZeroCellResolver.detect(matrix, weights, lower, validity_mask)
        smoothed = ZeroCellResolver.apply_epsilon_smoothing(matrix, columns, epsilon)
        return smoothed, columns.tolist()
