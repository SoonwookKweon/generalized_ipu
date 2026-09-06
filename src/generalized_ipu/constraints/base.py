"""모든 제약 객체가 공유하는 추상 인터페이스.

제약은 이름, 대상 계층, 제약 열 이름, 열별 목표 구간의 네 요소로 구성된다.
갱신 비율과 구간 위반량의 계산은 `interval` 모듈의 함수에 위임하므로, 하위
클래스는 제약 열을 어떻게 구성하는지만 정의하면 된다.
"""

from abc import ABC
from typing import List, Sequence

import numpy as np


class AbstractConstraint(ABC):
    """모든 제약 객체의 기반 클래스.

    제약은 이름, 대상 계층, 제약 열 이름, 그리고 열별 목표 구간 (lower, upper) 로
    구성된다. 스칼라 목표는 lower == upper 인 퇴화 구간으로 보관한다.
    """

    def __init__(
        self,
        name: str,
        level: str,
        column_names: Sequence[str],
        lower: np.ndarray,
        upper: np.ndarray,
    ):
        lower = np.asarray(lower, dtype="float64")
        upper = np.asarray(upper, dtype="float64")
        column_names = list(column_names)

        if lower.shape != upper.shape:
            raise ValueError(f"'{name}' 제약의 하한과 상한의 길이가 다릅니다.")
        if len(column_names) != lower.size:
            raise ValueError(
                f"'{name}' 제약의 열 이름 수({len(column_names)})와 목표 수({lower.size})가 다릅니다."
            )
        if np.any(lower > upper):
            raise ValueError(f"'{name}' 제약에 하한이 상한보다 큰 열이 있습니다.")
        if np.any(lower < 0):
            raise ValueError(f"'{name}' 제약에 음수 하한이 있습니다.")

        self.name = name
        self.level = level
        self.column_names: List[str] = column_names
        self.lower = lower
        self.upper = upper

    @property
    def n_columns(self) -> int:
        """이 제약이 차지하는 제약 열의 수를 반환한다."""
        return self.lower.size

    def compute_ratio(self, weighted_sums: np.ndarray) -> np.ndarray:
        """가중합을 받아 제약 열별 갱신 비율을 반환한다."""
        from .interval import compute_update_ratios

        return compute_update_ratios(weighted_sums, self.lower, self.upper)

    def bound_violations(self, weighted_sums: np.ndarray) -> np.ndarray:
        """가중합을 받아 제약 열별 구간 위반량을 반환한다."""
        from .interval import bound_violations

        return bound_violations(weighted_sums, self.lower, self.upper)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, level={self.level!r}, "
            f"n_columns={self.n_columns})"
        )
