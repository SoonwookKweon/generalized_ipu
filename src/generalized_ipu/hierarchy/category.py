"""범주 위계의 등록과 매핑 행렬 생성.

하나의 속성에 대해 가장 세분화된 범주를 소범주, 소범주들을 배타적으로 묶은
상위 범주를 대범주라 한다. 매핑 행렬 M 은 대범주가 소범주를 포섭하면 1, 아니면
0인 K_coarse x K_fine 이진 행렬이며, 대범주 단위의 가중합은 S_coarse = M S_fine
으로 얻는다.
"""

from typing import Dict, List, Sequence

import numpy as np
from scipy.sparse import csr_matrix


class CategoryHierarchy:
    """한 속성에 대한 소범주와 대범주의 포섭 관계를 등록한다.

    소범주는 정확히 하나의 대범주에 속해야 한다. 등록 순서가 그대로 매핑 행렬의
    행 순서(대범주)와 열 순서(소범주)가 된다.
    """

    def __init__(self, name: str):
        self.name = name
        self.mapping: Dict[str, List[str]] = {}
        self.fine_categories: List[str] = []
        self._fine_index: Dict[str, int] = {}

    def add_mapping(self, coarse: str, fines: Sequence[str]) -> "CategoryHierarchy":
        """대범주 하나와 그것이 포섭하는 소범주 목록을 등록한다.

        대범주가 중복되거나 소범주가 둘 이상의 대범주에 포섭되면 예외를 발생시킨다.
        """
        if coarse in self.mapping:
            raise ValueError(f"대범주 '{coarse}' 가 이미 등록되어 있습니다.")
        fines = list(fines)
        if not fines:
            raise ValueError(f"대범주 '{coarse}' 에 소범주가 지정되지 않았습니다.")

        self.mapping[coarse] = fines
        for fine in fines:
            if fine in self._fine_index:
                owner = next(c for c, f in self.mapping.items() if fine in f and c != coarse)
                raise ValueError(
                    f"소범주 '{fine}' 가 대범주 '{owner}' 와 '{coarse}' 에 중복 포섭되었습니다."
                )
            self._fine_index[fine] = len(self.fine_categories)
            self.fine_categories.append(fine)
        return self

    @property
    def coarse_categories(self) -> List[str]:
        """등록 순서대로 대범주 이름 목록을 반환한다."""
        return list(self.mapping.keys())

    def validate(self, fine_categories: Sequence[str] = None) -> bool:
        """정합성 조건(각 소범주가 정확히 하나의 대범주에 속함)을 검사한다.

        fine_categories 를 주면 원자 스키마 전체와 대조하여 누락 소범주도 걸러낸다.
        """
        if fine_categories is None:
            return True
        missing = [f for f in fine_categories if f not in self._fine_index]
        if missing:
            raise ValueError(
                f"'{self.name}' 위계에서 어느 대범주에도 속하지 않은 소범주가 있습니다: {missing}"
            )
        unknown = [f for f in self._fine_index if f not in set(fine_categories)]
        if unknown:
            raise ValueError(
                f"'{self.name}' 위계에 원자 스키마에 없는 소범주가 등록되었습니다: {unknown}"
            )
        return True


class MappingMatrix:
    """범주 위계를 이진 집계 행렬로 변환한다."""

    @staticmethod
    def build_binary_matrix(
        hierarchy: CategoryHierarchy, fine_categories: Sequence[str] = None
    ) -> csr_matrix:
        """M (K_coarse x K_fine) 을 생성한다.

        행 순서는 대범주 등록 순서, 열 순서는 fine_categories 이며 생략 시
        위계에 등록된 소범주 순서를 사용한다. 열 합이 1이 아닌 소범주가 있으면
        포섭 누락 또는 대범주 중첩이므로 예외를 발생시킨다.
        """
        fine_list = list(fine_categories) if fine_categories is not None else hierarchy.fine_categories
        hierarchy.validate(fine_list)

        index_of = {name: position for position, name in enumerate(fine_list)}
        coarse_list = hierarchy.coarse_categories

        rows: List[int] = []
        cols: List[int] = []
        for row, coarse in enumerate(coarse_list):
            for fine in hierarchy.mapping[coarse]:
                rows.append(row)
                cols.append(index_of[fine])

        data = np.ones(len(rows), dtype="float64")
        matrix = csr_matrix(
            (data, (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
            shape=(len(coarse_list), len(fine_list)),
        )

        column_sums = np.asarray(matrix.sum(axis=0)).ravel()
        if not np.all(column_sums == 1.0):
            offending = [fine_list[i] for i in np.flatnonzero(column_sums != 1.0)]
            raise ValueError(
                f"매핑 행렬의 열 합이 1이 아닌 소범주가 있습니다: {offending}"
            )
        return matrix
