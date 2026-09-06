"""임의 계층의 속성을 기본 단위 좌표계로 옮기는 집계 사상.

집계 사상은 대상 계층과 기본 계층의 관계에 따라 세 갈래로 나뉜다. 하위 계층은
부모키를 따라 속성 값을 합산하여 올리고, 상위 계층은 속성 값을 그에 속한 모든
기본 단위에 방송하며, 기본 계층 자신은 항등 사상이다.

상위 계층의 개체 수를 세는 경우는 방송과 구별된다. 방송은 같은 값을 모든 기본
단위에 복제하므로 가중합이 소속 기본 단위 수만큼 중복 계상된다. 상위 계층 개체
하나가 가중합에서 한 번만 세어지도록 하려면 분수 귀속(fractional attribution)을
써야 하며, 이를 `ancestor_share_to_base` 가 담당한다.
"""

from itertools import product
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from .tree import HierarchyTree

COUNT_COLUMN = "__count__"


class BaseUnitSelector:
    """기본 가중치 부여 단위를 확정하고 그 좌표계를 제공한다.

    생성 시점에 계층 트리의 무결성을 검사하므로, 이 객체가 만들어졌다는 것은
    기본 계층의 행 순서를 좌표계로 신뢰할 수 있다는 뜻이다.
    """

    def __init__(self, tree: HierarchyTree):
        tree.validate_integrity()
        self.tree = tree
        self.base_node = tree.base_node

    @property
    def n_units(self) -> int:
        """기본 단위의 수를 반환한다."""
        return len(self.base_node.data)

    @property
    def keys(self) -> pd.Series:
        """기본 단위의 주키를 행 순서대로 반환한다."""
        return self.base_node.data[self.base_node.primary_key].reset_index(drop=True)


class AggregationMapper:
    """임의 계층의 속성을 기본 계층의 행 좌표계로 옮긴다.

    모든 반환값의 행 수와 순서는 기본 계층 자료와 동일하다. 계층 관계 판정은
    `HierarchyTree.relation_to_base` 에 위임한다.
    """

    @staticmethod
    def aggregate_to_base(
        tree: HierarchyTree,
        target_level: str,
        attribute_cols: Sequence[str],
    ) -> pd.DataFrame:
        """대상 계층의 속성 열을 기본 단위 행 순서의 표로 반환한다.

        반환 표의 행 수와 순서는 기본 계층 자료와 동일하며, 열은 attribute_cols 이다.
        """
        attribute_cols = list(attribute_cols)
        base_node = tree.base_node
        relation = tree.relation_to_base(target_level)

        if relation == "base":
            frame = base_node.data[attribute_cols].reset_index(drop=True)
            return frame.astype("float64")

        if relation == "descendant":
            return AggregationMapper._aggregate_upward(tree, target_level, attribute_cols)

        return AggregationMapper._broadcast_downward(tree, target_level, attribute_cols)

    # -------------------------------------------------------------- 하위 → 기본

    @staticmethod
    def _aggregate_upward(
        tree: HierarchyTree,
        target_level: str,
        attribute_cols: List[str],
        source: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """하위 계층의 속성 수량을 경로를 따라 누적 합산하여 기본 계층까지 올린다.

        source 를 주면 대상 계층의 자료 표 대신 그 표를 사용한다. 집계할 열을
        원본에 두지 않고 파생시키는 경우에 쓴다.
        """
        path = tree.path_from_base(target_level)  # [base, ..., target]
        node = tree.nodes[target_level]
        data = node.data if source is None else source
        frame = data[[node.parent_key] + attribute_cols].copy()
        key = node.parent_key

        for level in reversed(path[1:-1]):
            intermediate = tree.nodes[level]
            grouped = frame.groupby(key, dropna=False)[attribute_cols].sum()
            frame = intermediate.data[[intermediate.primary_key, intermediate.parent_key]].merge(
                grouped, left_on=intermediate.primary_key, right_index=True, how="left"
            )
            frame[attribute_cols] = frame[attribute_cols].fillna(0.0)
            frame = frame[[intermediate.parent_key] + attribute_cols]
            key = intermediate.parent_key

        grouped = frame.groupby(key, dropna=False)[attribute_cols].sum()
        base_node = tree.base_node
        result = (
            base_node.data[[base_node.primary_key]]
            .merge(grouped, left_on=base_node.primary_key, right_index=True, how="left")
            .drop(columns=[base_node.primary_key])
            .fillna(0.0)
            .reset_index(drop=True)
        )
        return result[attribute_cols].astype("float64")

    @staticmethod
    def descendant_count_to_base(tree: HierarchyTree, target_level: str) -> pd.Series:
        """기본 단위별로 소속된 하위 계층 개체의 수를 반환한다.

        기본 계층이 가구이고 대상 계층이 가구원이면 이 값이 가구원 수, 곧 가구
        규모이다. 소속된 하위 개체가 없는 기본 단위의 값은 0이며, 대상이 기본
        계층 자신이면 모든 값이 1이다.
        """
        relation = tree.relation_to_base(target_level)
        n_units = len(tree.base_node.data)
        if relation == "base":
            return pd.Series(np.ones(n_units, dtype="float64"), name=COUNT_COLUMN)
        if relation != "descendant":
            raise ValueError(
                f"'{target_level}' 은 기본 계층 '{tree.base_level}' 의 하위 계층이 아니므로 "
                "개체 수를 셀 수 없습니다."
            )

        node = tree.nodes[target_level]
        source = node.data[[node.parent_key]].copy()
        source[COUNT_COLUMN] = 1.0
        frame = AggregationMapper._aggregate_upward(
            tree, target_level, [COUNT_COLUMN], source=source
        )
        return frame[COUNT_COLUMN].rename(COUNT_COLUMN)

    @staticmethod
    def size_class_labels(
        counts: Sequence[float],
        boundaries: Sequence[float],
        labels: Sequence[str] = None,
    ) -> pd.Series:
        """개체 수를 계급 이름으로 변환한다.

        boundaries 는 오름차순 하한의 나열이며 마지막 계급은 상한을 두지 않는다.
        boundaries=(1, 2, 3, 4) 는 '1', '2', '3', '4+' 네 계급을 뜻한다. 최소 하한
        미만의 값이 있으면 계급 정의가 자료를 덮지 못한다는 뜻이므로 예외를
        발생시킨다.
        """
        counts = np.asarray(counts, dtype="float64")
        bounds = np.asarray(boundaries, dtype="float64")
        if bounds.size == 0:
            raise ValueError("계급 하한이 비어 있습니다.")
        if np.any(np.diff(bounds) <= 0):
            raise ValueError(f"계급 하한은 오름차순이어야 합니다: {boundaries!r}")

        below = counts < bounds[0]
        if np.any(below):
            raise ValueError(
                f"최소 계급 하한 {bounds[0]:g} 미만인 값이 {int(below.sum())}건 있습니다."
            )

        names = AggregationMapper._default_size_labels(bounds) if labels is None else list(labels)
        if len(names) != bounds.size:
            raise ValueError(f"계급 이름 수({len(names)})와 계급 수({bounds.size})가 다릅니다.")

        position = np.searchsorted(bounds, counts, side="right") - 1
        return pd.Series([names[index] for index in position], name="size_class")

    @staticmethod
    def _default_size_labels(bounds: np.ndarray) -> List[str]:
        """계급 하한으로부터 '1', '2-3', '4+' 형태의 이름을 생성한다."""
        names: List[str] = []
        for index, low in enumerate(bounds):
            if index == bounds.size - 1:
                names.append(f"{low:g}+")
            elif bounds[index + 1] - low == 1:
                names.append(f"{low:g}")
            else:
                names.append(f"{low:g}-{bounds[index + 1] - 1:g}")
        return names

    # -------------------------------------------------------------- 상위 → 기본

    @staticmethod
    def ancestor_key_of_base(tree: HierarchyTree, target_level: str) -> pd.Series:
        """각 기본 단위가 소속된 상위 계층 개체의 주키를 기본 단위 행 순서로 반환한다.

        대상이 기본 계층 자신이면 기본 단위의 주키를 그대로 반환한다.
        """
        base_node = tree.base_node
        if target_level == tree.base_level:
            return base_node.data[base_node.primary_key].reset_index(drop=True)

        if target_level not in tree.ancestors(tree.base_level):
            raise ValueError(f"'{target_level}' 이 기본 계층의 상위 계층이 아닙니다.")

        keys = base_node.data[base_node.parent_key].reset_index(drop=True)
        current = tree.resolve_parent_level(tree.base_level)
        while current != target_level:
            node = tree.nodes[current]
            lookup = pd.Series(
                node.data[node.parent_key].to_numpy(), index=node.data[node.primary_key]
            )
            keys = keys.map(lookup)
            current = tree.resolve_parent_level(current)
        return keys.reset_index(drop=True)

    @staticmethod
    def _broadcast_frame(
        tree: HierarchyTree, target_level: str, attribute_cols: List[str]
    ) -> pd.DataFrame:
        """상위 계층의 열을 기본 단위 행 순서로 복제한다. 자료형을 바꾸지 않는다."""
        keys = AggregationMapper.ancestor_key_of_base(tree, target_level)
        node = tree.nodes[target_level]
        lookup = node.data.set_index(node.primary_key)[attribute_cols]
        return lookup.reindex(keys.to_numpy()).reset_index(drop=True)

    @staticmethod
    def _broadcast_downward(
        tree: HierarchyTree, target_level: str, attribute_cols: List[str]
    ) -> pd.DataFrame:
        """상위 계층의 속성 값을 그에 속한 모든 기본 단위에 방송한다."""
        frame = AggregationMapper._broadcast_frame(tree, target_level, list(attribute_cols))
        return frame.fillna(0.0).astype("float64")

    @staticmethod
    def ancestor_share_to_base(
        tree: HierarchyTree,
        target_level: str,
        dims: Optional[Mapping[str, Sequence]] = None,
    ) -> Tuple[csr_matrix, List[str]]:
        """상위 계층 개체를 기본 단위 좌표계에서 한 번만 세는 분수 귀속 행렬을 만든다.

        상위 계층 개체 하나에 기본 단위 m개가 소속되면 각 기본 단위에 1/m 을
        부여한다. 그 개체에 속한 기본 단위의 가중치가 모두 같으면 해당 열의
        가중합은 상위 계층 개체 수와 일치한다. 가중치가 서로 다르면 가중치의
        평균으로 계상되므로, 상위 계층 제약은 한 상위 개체에 속한 기본 단위들의
        가중치가 크게 벌어지지 않는다는 전제 위에서 해석해야 한다.

        dims 를 주면 상위 계층의 변수 조합별로 열을 나누고, 생략하면 상위 계층
        개체 수 전체를 세는 한 개의 열을 만든다.
        """
        relation = tree.relation_to_base(target_level)
        if relation == "descendant":
            raise ValueError(
                f"'{target_level}' 은 기본 계층 '{tree.base_level}' 의 하위 계층이므로 "
                "분수 귀속의 대상이 아닙니다. aggregate_to_base 를 사용하십시오."
            )

        keys = AggregationMapper.ancestor_key_of_base(tree, target_level)
        sizes = keys.map(keys.value_counts())
        share = 1.0 / sizes.to_numpy(dtype="float64")

        if dims is None:
            matrix = csr_matrix(share.reshape(-1, 1))
            matrix.eliminate_zeros()
            return matrix, [f"{target_level}.count"]

        dim_names = list(dims.keys())
        categories = [list(dims[name]) for name in dim_names]
        frame = AggregationMapper._broadcast_frame(tree, target_level, dim_names)
        codes, valid, n_cells = AggregationMapper._encode_cells(frame, dim_names, categories)

        values = np.where(valid, share, 0.0)
        matrix = coo_matrix(
            (values, (np.arange(share.size, dtype=np.int64), codes)),
            shape=(share.size, n_cells),
        ).tocsr()
        matrix.eliminate_zeros()
        return matrix, AggregationMapper.cell_names(dim_names, categories)

    # -------------------------------------------------------------- 교차 집계

    @staticmethod
    def crosstab_to_base(
        tree: HierarchyTree,
        target_level: str,
        dims: Mapping[str, Sequence],
        weight_col: str = None,
    ) -> Tuple[csr_matrix, List[str]]:
        """대상 계층의 N개 변수 결합 분포를 기본 단위별 셀 수량 희소 행렬로 반환한다.

        열 순서는 dims 의 선언 순서를 축으로 하는 C 순서 평탄화와 일치한다.
        조밀 중간 산출물을 만들지 않고 좌표 형식으로 직접 구축한다. 대상 계층은
        기본 계층 자신이거나 그 직속 하위 계층이어야 한다.
        """
        node = tree.nodes[target_level]
        dim_names = list(dims.keys())
        categories = [list(dims[name]) for name in dim_names]
        codes, valid, n_cells = AggregationMapper._encode_cells(node.data, dim_names, categories)

        cell_frame = pd.DataFrame({"__cell__": codes})
        cell_frame["__count__"] = (
            1.0 if weight_col is None else node.data[weight_col].to_numpy(dtype="float64")
        )
        cell_frame.loc[~valid, "__count__"] = 0.0

        if target_level == tree.base_level:
            cell_frame["__unit__"] = np.arange(len(node.data))
            n_units = len(tree.base_node.data)
            unit_index = cell_frame["__unit__"]
        else:
            cell_frame["__unit__"] = node.data[node.parent_key].to_numpy()
            base_node = tree.base_node
            position = pd.Series(
                np.arange(len(base_node.data)), index=base_node.data[base_node.primary_key]
            )
            unit_index = cell_frame["__unit__"].map(position)
            n_units = len(base_node.data)
            valid_rows = unit_index.notna().to_numpy()
            cell_frame = cell_frame[valid_rows]
            unit_index = unit_index[valid_rows]

        block = coo_matrix(
            (
                cell_frame["__count__"].to_numpy(dtype="float64"),
                (
                    unit_index.to_numpy(dtype=np.int64),
                    cell_frame["__cell__"].to_numpy(dtype=np.int64),
                ),
            ),
            shape=(n_units, n_cells),
        ).tocsr()
        block.eliminate_zeros()
        return block, AggregationMapper.cell_names(dim_names, categories)

    @staticmethod
    def _encode_cells(
        frame: pd.DataFrame, dim_names: Sequence[str], categories: Sequence[Sequence]
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """행별 범주 조합을 C 순서 평탄화 색인으로 변환한다.

        선언되지 않은 범주 값이나 결측을 가진 행은 valid 가 False 이며, 그 행의
        색인 값은 의미를 가지지 않는다.
        """
        shape = tuple(len(values) for values in categories)
        codes = np.zeros(len(frame), dtype=np.int64)
        valid = np.ones(len(frame), dtype=bool)
        for axis, (name, values) in enumerate(zip(dim_names, categories)):
            lookup = {value: index for index, value in enumerate(values)}
            mapped = frame[name].map(lookup)
            valid &= mapped.notna().to_numpy()
            codes = codes * shape[axis] + mapped.fillna(0).to_numpy(dtype=np.int64)
        return codes, valid, int(np.prod(shape)) if shape else 0

    @staticmethod
    def cell_names(dim_names: Sequence[str], categories: Sequence[Sequence]) -> List[str]:
        """C 순서 평탄화에 대응하는 셀 이름을 생성한다."""
        return [
            "|".join(f"{name}={value}" for name, value in zip(dim_names, combination))
            for combination in product(*categories)
        ]
