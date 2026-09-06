"""제약 객체의 등록과 제약 열 배치의 관리.

등록소는 제약 열의 순서에 대한 단일 기준점이다. 등록 순서가 그대로 속성 행렬의
열 순서이며, 목표 배열과 갱신 비율 배열도 같은 순서로 이어 붙여진다. 속성 행렬을
구축한 블록의 순서와 제약의 등록 순서가 어긋나면 목표가 엉뚱한 열에 부과되므로,
`validate_against` 로 열 수의 일치를 반드시 확인한다.
"""

from typing import Dict, Iterator, List, Tuple

import numpy as np

from .base import AbstractConstraint


class ConstraintRegistry:
    """제약 객체를 등록하고 제약 열의 배치를 관리하는 단일 기준점."""

    def __init__(self):
        self._constraints: List[AbstractConstraint] = []
        self._slices: Dict[str, slice] = {}

    # ------------------------------------------------------------------ 등록

    def register(self, constraint: AbstractConstraint) -> "ConstraintRegistry":
        """제약 하나를 등록하고 이어지는 열 구간을 배정한다."""
        if constraint.name in self._slices:
            raise ValueError(f"제약 '{constraint.name}' 이 이미 등록되어 있습니다.")
        start = self.n_columns
        self._slices[constraint.name] = slice(start, start + constraint.n_columns)
        self._constraints.append(constraint)
        return self

    def extend(self, constraints) -> "ConstraintRegistry":
        """여러 제약을 나열 순서대로 등록한다."""
        for constraint in constraints:
            self.register(constraint)
        return self

    # ------------------------------------------------------------------ 조회

    def get_all(self) -> List[AbstractConstraint]:
        """등록된 제약을 등록 순서대로 반환한다."""
        return list(self._constraints)

    def get(self, name: str) -> AbstractConstraint:
        """이름으로 제약을 찾는다."""
        for constraint in self._constraints:
            if constraint.name == name:
                return constraint
        raise KeyError(f"등록되지 않은 제약입니다: {name!r}")

    def slice_of(self, name: str) -> slice:
        """제약이 차지하는 열 구간을 반환한다."""
        if name not in self._slices:
            raise KeyError(f"등록되지 않은 제약입니다: {name!r}")
        return self._slices[name]

    def owner_of_column(self, column: int) -> str:
        """제약 열 색인이 어느 제약에 속하는지 판정한다."""
        for name, span in self._slices.items():
            if span.start <= column < span.stop:
                return name
        raise IndexError(f"제약 열 색인 {column} 이 범위를 벗어났습니다.")

    def __iter__(self) -> Iterator[AbstractConstraint]:
        return iter(self._constraints)

    def __len__(self) -> int:
        return len(self._constraints)

    # ------------------------------------------------------------------ 집계

    @property
    def n_columns(self) -> int:
        """등록된 제약 열의 총 수 K 를 반환한다."""
        return sum(constraint.n_columns for constraint in self._constraints)

    @property
    def column_names(self) -> List[str]:
        """'제약명::열이름' 형식의 제약 열 이름을 열 순서대로 반환한다."""
        names: List[str] = []
        for constraint in self._constraints:
            names.extend(f"{constraint.name}::{column}" for column in constraint.column_names)
        return names

    @property
    def column_owners(self) -> List[str]:
        """제약 열마다 소속 제약의 이름을 열 순서대로 반환한다."""
        owners: List[str] = []
        for constraint in self._constraints:
            owners.extend([constraint.name] * constraint.n_columns)
        return owners

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """등록 순서대로 이어 붙인 하한 배열과 상한 배열을 반환한다."""
        if not self._constraints:
            return np.empty(0, dtype="float64"), np.empty(0, dtype="float64")
        lower = np.concatenate([constraint.lower for constraint in self._constraints])
        upper = np.concatenate([constraint.upper for constraint in self._constraints])
        return lower, upper

    def compute_ratios(self, weighted_sums: np.ndarray) -> np.ndarray:
        """제약별로 갱신 비율을 산출하여 이어 붙인다."""
        weighted_sums = np.asarray(weighted_sums, dtype="float64")
        if weighted_sums.size != self.n_columns:
            raise ValueError(
                f"가중합의 길이({weighted_sums.size})가 등록된 제약 열 수({self.n_columns})와 다릅니다."
            )
        ratios = np.empty(self.n_columns, dtype="float64")
        for constraint in self._constraints:
            span = self._slices[constraint.name]
            ratios[span] = constraint.compute_ratio(weighted_sums[span])
        return ratios

    def validate_against(self, n_matrix_columns: int) -> bool:
        """속성 행렬의 열 수와 등록된 제약 열 수가 일치하는지 확인한다."""
        if n_matrix_columns != self.n_columns:
            raise ValueError(
                f"속성 행렬의 열 수({n_matrix_columns})와 등록된 제약 열 수({self.n_columns})가 "
                "일치하지 않습니다."
            )
        return True
