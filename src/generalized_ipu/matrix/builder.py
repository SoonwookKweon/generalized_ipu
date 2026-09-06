"""블록 명세로부터 통합 희소 속성 행렬을 구축한다.

속성 행렬 A 의 성분 a_ik 는 기본 단위 i 가 가중치 1을 가질 때 제약 열 k 의
가중합에 기여하는 수량이다. 이 모듈은 서로 다른 계층과 표현 방식을 가진 여러
제약표를 각각 블록으로 구축한 뒤 가로로 이어 붙여 하나의 A 를 만든다.

블록 명세는 네 종류이다.

| 명세 | 제약 열의 의미 | 대상 계층 |
| --- | --- | --- |
| `ColumnBlock` | 속성 열의 수량 합계 | 임의 |
| `CrossTabBlock` | N차원 결합 분포의 셀 빈도 | 기본 계층 또는 그 하위 |
| `ShareBlock` | 상위 계층 개체의 수 | 기본 계층의 상위 |
| `CoarseBlock` | 이미 구축된 소범주 블록의 대범주 집계 | 원천 블록을 따름 |

블록의 구축 순서가 그대로 제약 열의 배치 순서가 되며, `ConstraintRegistry` 의
등록 순서와 일치해야 한다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from ..hierarchy.category import CategoryHierarchy, MappingMatrix
from ..hierarchy.mapper import AggregationMapper
from ..hierarchy.tree import HierarchyTree


# ---------------------------------------------------------------- 블록 명세


@dataclass
class ColumnBlock:
    """계층의 속성 열을 그대로 제약 열로 삼는 블록.

    cols 의 각 열이 제약 열 하나가 되며, 대상 계층이 기본 계층이 아니면 집계
    사상을 거쳐 기본 단위 좌표계로 옮겨진다.
    """

    name: str
    level: str
    cols: Sequence[str]


@dataclass
class CrossTabBlock:
    """N차원 결합 분포를 제약 열로 평탄화하는 블록.

    dims 의 선언 순서가 텐서의 축 순서이며, 제약 열은 C 순서 평탄화를 따른다.
    weight_col 을 주면 행을 1이 아니라 그 열의 값으로 계상한다.
    """

    name: str
    level: str
    dims: Mapping[str, Sequence]
    weight_col: str = None


@dataclass
class ShareBlock:
    """상위 계층 개체의 수를 기본 단위 좌표계에서 세는 블록.

    상위 계층 개체 하나에 기본 단위 m개가 소속되면 각 기본 단위에 1/m 을 부여하는
    분수 귀속을 사용한다. 거처를 상위 계층, 가구를 기본 계층으로 두었을 때
    "거처 수"나 "가구 수 계급별 거처 수"를 제약하는 데 쓴다. 같은 목적으로
    `ColumnBlock` 을 쓰면 상위 계층의 값이 소속 기본 단위마다 복제되어 중복
    계상되므로, 개체 수를 세는 제약에는 이 블록을 사용한다.

    dims 를 생략하면 상위 계층 개체 수 전체를 세는 한 개의 제약 열을 만든다.
    """

    name: str
    level: str
    dims: Optional[Mapping[str, Sequence]] = None


@dataclass
class CoarseBlock:
    """이미 구축된 소범주 블록을 대범주로 집계하는 블록.

    source 는 먼저 구축된 블록의 이름이며, 그 블록의 제약 열이 소범주가 된다.
    구조적 영으로 제외된 소범주는 위계에서 자동으로 걷어낸다.
    """

    name: str
    source: str
    hierarchy: CategoryHierarchy


BlockSpec = "ColumnBlock | CrossTabBlock | ShareBlock | CoarseBlock"


# ---------------------------------------------------------------- 산출물


@dataclass
class AttributeMatrix:
    """속성 행렬과 그 열의 의미를 함께 담는다.

    column_names 의 순서는 행렬의 열 순서이며, block_slices 는 블록 이름에서 그
    블록이 차지하는 열 구간으로의 대응이다. base_keys 는 행 순서에 대응하는 기본
    단위의 주키로, 결과를 원본 자료에 되붙일 때 사용한다.
    """

    matrix: csr_matrix
    column_names: List[str]
    block_slices: Dict[str, slice] = field(default_factory=dict)
    base_keys: pd.Series = None

    def __post_init__(self):
        if self.matrix.shape[1] != len(self.column_names):
            raise ValueError(
                f"속성 행렬의 열 수({self.matrix.shape[1]})와 "
                f"열 이름 수({len(self.column_names)})가 다릅니다."
            )

    @property
    def n_units(self) -> int:
        """기본 단위의 수를 반환한다."""
        return self.matrix.shape[0]

    @property
    def n_columns(self) -> int:
        """제약 열의 수를 반환한다."""
        return self.matrix.shape[1]

    @property
    def nnz(self) -> int:
        """비영 성분의 수를 반환한다. 반복 1회의 연산 복잡도를 좌우한다."""
        return int(self.matrix.nnz)

    def block(self, name: str) -> csr_matrix:
        """블록 이름으로 해당 열 구간의 부분 행렬을 잘라 반환한다."""
        return self.matrix[:, self.block_slices[name]]

    def column_frame(self) -> pd.DataFrame:
        """제약 열마다 소속 블록과 열 이름을 적은 표를 반환한다."""
        owners: List[str] = [""] * self.n_columns
        for block_name, span in self.block_slices.items():
            for column in range(span.start, span.stop):
                owners[column] = block_name
        return pd.DataFrame({"block": owners, "column": self.column_names})


# ---------------------------------------------------------------- 구축기


class UnifiedSparseMatrixBuilder:
    """계층, 다차원 교차표, 범주 위계를 결합한 통합 희소 속성 행렬을 구축한다."""

    def __init__(self, tree: HierarchyTree, validate: bool = True):
        if validate:
            tree.validate_integrity()
        self.tree = tree

    def build(
        self,
        blocks: Sequence[BlockSpec],
        validity_masks: Mapping[str, np.ndarray] = None,
    ) -> AttributeMatrix:
        """블록 명세를 순서대로 구축하여 가로로 결합한다.

        validity_masks 는 블록 이름에서 그 블록의 유효 마스크로의 대응이며, 유효한
        쪽이 True 이다. 구조적 영 제약 열은 결합 이전에 제외되므로 최종 행렬에
        색인이 할당되지 않는다.
        """
        validity_masks = dict(validity_masks or {})
        built: Dict[str, Tuple[csr_matrix, List[str]]] = {}
        order: List[str] = []

        for block in blocks:
            if block.name in built:
                raise ValueError(f"블록 '{block.name}' 이 이미 정의되어 있습니다.")

            if isinstance(block, ColumnBlock):
                matrix, names = self._build_column_block(block)
            elif isinstance(block, CrossTabBlock):
                matrix, names = self._build_crosstab_block(block)
            elif isinstance(block, ShareBlock):
                matrix, names = self._build_share_block(block)
            elif isinstance(block, CoarseBlock):
                matrix, names = self._build_coarse_block(block, built)
            else:
                raise TypeError(f"알 수 없는 블록 명세입니다: {type(block).__name__}")

            mask = validity_masks.get(block.name)
            if mask is not None:
                matrix, names = self._drop_invalid_columns(block.name, matrix, names, mask)

            built[block.name] = (matrix, names)
            order.append(block.name)

        return self._assemble(built, order)

    # -------------------------------------------------------------- 블록 구축

    def _build_column_block(self, block: ColumnBlock) -> Tuple[csr_matrix, List[str]]:
        """속성 열을 집계 사상으로 옮겨 블록을 만든다."""
        frame = AggregationMapper.aggregate_to_base(self.tree, block.level, block.cols)
        matrix = csr_matrix(frame.to_numpy(dtype="float64"))
        matrix.eliminate_zeros()
        names = [f"{block.level}.{column}" for column in block.cols]
        return matrix, names

    def _build_crosstab_block(self, block: CrossTabBlock) -> Tuple[csr_matrix, List[str]]:
        """결합 분포를 평탄화하여 블록을 만든다."""
        matrix, names = AggregationMapper.crosstab_to_base(
            self.tree, block.level, block.dims, block.weight_col
        )
        return matrix, list(names)

    def _build_share_block(self, block: ShareBlock) -> Tuple[csr_matrix, List[str]]:
        """상위 계층 개체 수를 분수 귀속으로 세는 블록을 만든다."""
        matrix, names = AggregationMapper.ancestor_share_to_base(
            self.tree, block.level, block.dims
        )
        return matrix, list(names)

    def _build_coarse_block(
        self, block: CoarseBlock, built: Mapping[str, Tuple[csr_matrix, List[str]]]
    ) -> Tuple[csr_matrix, List[str]]:
        """원천 블록의 소범주 열을 매핑 행렬로 대범주에 집계한다."""
        if block.source not in built:
            raise KeyError(
                f"대범주 블록 '{block.name}' 이 참조하는 소범주 블록 '{block.source}' 이 "
                "먼저 구축되지 않았습니다."
            )
        fine_matrix, fine_names = built[block.source]
        hierarchy = self._restrict_hierarchy(block.hierarchy, fine_names)
        mapping = MappingMatrix.build_binary_matrix(hierarchy, fine_names)
        coarse_matrix = (fine_matrix @ mapping.T).tocsr()
        names = list(hierarchy.coarse_categories)
        return coarse_matrix, names

    @staticmethod
    def _restrict_hierarchy(
        hierarchy: CategoryHierarchy, available: Sequence[str]
    ) -> CategoryHierarchy:
        """구조적 영으로 제외된 소범주를 위계에서 걷어낸다."""
        available_set = set(available)
        restricted = CategoryHierarchy(hierarchy.name)
        for coarse, fines in hierarchy.mapping.items():
            retained = [fine for fine in fines if fine in available_set]
            if retained:
                restricted.add_mapping(coarse, retained)
        missing = available_set - set(restricted.fine_categories)
        if missing:
            raise ValueError(
                f"'{hierarchy.name}' 위계가 포섭하지 않는 소범주가 있습니다: {sorted(missing)}"
            )
        return restricted

    @staticmethod
    def _drop_invalid_columns(
        block_name: str, matrix: csr_matrix, names: List[str], mask: np.ndarray
    ) -> Tuple[csr_matrix, List[str]]:
        """유효 마스크가 False 인 제약 열을 블록에서 제외한다."""
        mask = np.asarray(mask, dtype=bool)
        if mask.size != matrix.shape[1]:
            raise ValueError(
                f"블록 '{block_name}' 의 유효 마스크 길이({mask.size})가 "
                f"제약 열 수({matrix.shape[1]})와 다릅니다."
            )
        kept = np.flatnonzero(mask)
        return matrix[:, kept].tocsr(), [names[index] for index in kept]

    # -------------------------------------------------------------- 결합

    def _assemble(
        self, built: Mapping[str, Tuple[csr_matrix, List[str]]], order: Sequence[str]
    ) -> AttributeMatrix:
        """구축된 블록들을 선언 순서대로 이어 붙이고 열 구간을 기록한다."""
        matrices = [built[name][0] for name in order]
        column_names: List[str] = []
        block_slices: Dict[str, slice] = {}

        cursor = 0
        for name in order:
            names = built[name][1]
            block_slices[name] = slice(cursor, cursor + len(names))
            column_names.extend(names)
            cursor += len(names)

        unified = hstack(matrices, format="csr") if matrices else csr_matrix((0, 0))
        base_node = self.tree.base_node
        return AttributeMatrix(
            matrix=unified,
            column_names=column_names,
            block_slices=block_slices,
            base_keys=base_node.data[base_node.primary_key].reset_index(drop=True),
        )
