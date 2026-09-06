"""N차원 교차표 목표의 정의와 제약 열로의 평탄화.

평탄화 순서는 numpy 의 C 순서이며 마지막 차원이 가장 빠르게 변한다. 속성 행렬의
해당 블록도 같은 순서로 구축되어야 목표가 올바른 열에 대응한다.
"""

from itertools import product
from typing import List, Mapping, Sequence, Tuple

import numpy as np

from .base import AbstractConstraint
from .interval import normalize_targets


class TensorFlattener:
    """N차원 교차표를 제약 열의 나열로 평탄화한다.

    평탄화 순서는 numpy 의 C 순서(마지막 차원이 가장 빠르게 변함)이며, 속성 행렬의
    열 구성도 동일한 순서를 따라야 한다.
    """

    @staticmethod
    def flatten(tensor: np.ndarray) -> np.ndarray:
        """텐서를 C 순서로 1차원 배열에 편다."""
        return np.asarray(tensor).reshape(-1, order="C")

    @staticmethod
    def cell_names(dims: Mapping[str, Sequence]) -> List[str]:
        """차원별 범주 목록으로부터 평탄화 순서에 대응하는 셀 이름을 생성한다."""
        dim_names = list(dims.keys())
        categories = [list(dims[name]) for name in dim_names]
        return [
            "|".join(f"{name}={value}" for name, value in zip(dim_names, combination))
            for combination in product(*categories)
        ]

    @staticmethod
    def multi_indices(shape: Tuple[int, ...]) -> np.ndarray:
        """평탄화된 각 열의 다중 색인을 (n_cells, n_dims) 배열로 반환한다."""
        return np.array(list(np.ndindex(*shape)), dtype=np.int64)


class TensorTarget(AbstractConstraint):
    """N차원 결합 교차표 목표.

    dims 는 차원 이름에서 범주 목록으로의 대응이며, 선언 순서가 텐서의 축 순서이다.
    상한 텐서를 함께 주면 구간 목표가 된다.
    """

    def __init__(
        self,
        name: str,
        level: str,
        dims: Mapping[str, Sequence],
        target: np.ndarray,
        upper: np.ndarray = None,
    ):
        target = np.asarray(target, dtype="float64")
        expected_shape = tuple(len(list(values)) for values in dims.values())
        if target.shape != expected_shape:
            raise ValueError(
                f"'{name}' 제약의 목표 텐서 형태 {target.shape} 가 "
                f"차원 정의 {expected_shape} 와 일치하지 않습니다."
            )
        if upper is not None:
            upper = np.asarray(upper, dtype="float64")
            if upper.shape != target.shape:
                raise ValueError(f"'{name}' 제약의 상한 텐서 형태가 목표 텐서와 다릅니다.")

        self.dims = {key: list(values) for key, values in dims.items()}
        self.shape = expected_shape
        self.target_tensor = target

        flat_lower = TensorFlattener.flatten(target)
        flat_upper = flat_lower if upper is None else TensorFlattener.flatten(upper)
        column_names = TensorFlattener.cell_names(self.dims)
        super().__init__(name, level, column_names, flat_lower, flat_upper)

    @property
    def dim_names(self) -> List[str]:
        """축 순서대로 차원 이름을 반환한다."""
        return list(self.dims.keys())

    def multi_index_of(self, column: int) -> Tuple:
        """평탄화된 제약 열이 대응하는 차원별 범주 조합을 반환한다."""
        indices = np.unravel_index(column, self.shape, order="C")
        return tuple(
            self.dims[name][int(index)] for name, index in zip(self.dim_names, indices)
        )

    def category_frame(self):
        """제약 열별 범주 조합을 표로 반환한다. 논리 규칙 평가의 입력으로 쓴다."""
        import pandas as pd

        records = [self.multi_index_of(column) for column in range(self.n_columns)]
        return pd.DataFrame(records, columns=self.dim_names)


def interval_tensor_target(
    name: str,
    level: str,
    dims: Mapping[str, Sequence],
    intervals: Sequence,
) -> TensorTarget:
    """셀별 목표가 스칼라와 구간으로 섞여 있는 경우의 생성 보조 함수."""
    lower, upper = normalize_targets(list(intervals))
    shape = tuple(len(list(values)) for values in dims.values())
    return TensorTarget(name, level, dims, lower.reshape(shape), upper.reshape(shape))
