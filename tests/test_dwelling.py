"""거처-가구-가구원 삼계층에서 다가구주택과 n인가구 제약을 다루는 시험."""

import numpy as np
import pytest

from generalized_ipu import (
    BoundChecker,
    ConstraintRegistry,
    CrossTabBlock,
    GeneralizedIPUEngine,
    ShareBlock,
    UnifiedSparseMatrixBuilder,
)
from generalized_ipu.hierarchy.mapper import AggregationMapper

# 거처별 가구 수 계급. d1 은 2가구, d2 는 1가구, d3 는 3가구를 포섭한다.
DWELLING_DIMS = {"hh_class": ["1", "2", "3+"]}
SIZE_DIMS = {"size_class": ["1", "2", "3+"]}


def test_ancestor_key_walks_up_to_the_dwelling(dwelling_tree):
    keys = AggregationMapper.ancestor_key_of_base(dwelling_tree, "dwelling")
    assert list(keys) == ["d1", "d1", "d2", "d3", "d3", "d3"]


def test_share_attribution_gives_reciprocal_of_sibling_count(dwelling_tree):
    matrix, names = AggregationMapper.ancestor_share_to_base(dwelling_tree, "dwelling")

    assert names == ["dwelling.count"]
    assert matrix.toarray().ravel() == pytest.approx(
        [0.5, 0.5, 1.0, 1 / 3, 1 / 3, 1 / 3]
    )


def test_share_block_counts_each_dwelling_exactly_once(dwelling_tree):
    builder = UnifiedSparseMatrixBuilder(dwelling_tree)
    attribute = builder.build(
        [ShareBlock(name="dwelling_class", level="dwelling", dims=DWELLING_DIMS)]
    )

    weights = np.ones(attribute.n_units)
    weighted_sums = attribute.matrix.T.dot(weights)

    assert attribute.column_names == [
        "hh_class=1",
        "hh_class=2",
        "hh_class=3+",
    ]
    # 각 계급에 거처가 정확히 하나씩 있으므로 모든 열의 가중합이 1이다.
    assert weighted_sums == pytest.approx([1.0, 1.0, 1.0])


def test_broadcast_block_would_double_count_the_dwelling(dwelling_tree):
    """방송은 상위 계층 개체를 소속 기본 단위 수만큼 중복 계상한다.

    분수 귀속을 따로 두는 근거이다.
    """
    dwelling_tree.nodes["dwelling"].data["unit"] = 1.0

    broadcast = AggregationMapper.aggregate_to_base(dwelling_tree, "dwelling", ["unit"])
    assert float(broadcast["unit"].sum()) == pytest.approx(6.0)

    matrix, _ = AggregationMapper.ancestor_share_to_base(dwelling_tree, "dwelling")
    assert float(matrix.sum()) == pytest.approx(3.0)  # 실제 거처 수


def test_descendant_count_gives_household_size(dwelling_tree):
    counts = AggregationMapper.descendant_count_to_base(dwelling_tree, "person")
    assert list(counts) == [2.0, 1.0, 3.0, 2.0, 1.0, 2.0]


def test_size_class_labels_bin_the_counts(dwelling_tree):
    counts = AggregationMapper.descendant_count_to_base(dwelling_tree, "person")
    labels = AggregationMapper.size_class_labels(counts, boundaries=(1, 2, 3))
    assert list(labels) == ["2", "1", "3+", "2", "1", "2"]


def test_size_class_labels_reject_counts_below_the_lowest_class():
    with pytest.raises(ValueError, match="최소 계급 하한"):
        AggregationMapper.size_class_labels([0.0, 2.0], boundaries=(1, 2))


def test_size_class_default_labels_span_ranges():
    labels = AggregationMapper.size_class_labels([1.0, 3.0, 6.0], boundaries=(1, 3, 5))
    assert list(labels) == ["1-2", "3-4", "5+"]


def test_n_person_household_block_counts_households_by_size(dwelling_tree):
    counts = AggregationMapper.descendant_count_to_base(dwelling_tree, "person")
    household = dwelling_tree.nodes["household"]
    household.data["size_class"] = AggregationMapper.size_class_labels(
        counts, boundaries=(1, 2, 3)
    ).to_numpy()

    builder = UnifiedSparseMatrixBuilder(dwelling_tree)
    attribute = builder.build(
        [CrossTabBlock(name="hh_size", level="household", dims=SIZE_DIMS)]
    )

    weighted_sums = attribute.matrix.T.dot(np.ones(attribute.n_units))
    # 1인가구 2호(h2, h5), 2인가구 3호(h1, h4, h6), 3인 이상 1호(h3)
    assert weighted_sums == pytest.approx([2.0, 3.0, 1.0])


def test_joint_dwelling_and_size_constraints_converge(dwelling_tree):
    counts = AggregationMapper.descendant_count_to_base(dwelling_tree, "person")
    household = dwelling_tree.nodes["household"]
    household.data["size_class"] = AggregationMapper.size_class_labels(
        counts, boundaries=(1, 2, 3)
    ).to_numpy()

    builder = UnifiedSparseMatrixBuilder(dwelling_tree)
    attribute = builder.build(
        [
            ShareBlock(name="dwelling_class", level="dwelling", dims=DWELLING_DIMS),
            CrossTabBlock(name="hh_size", level="household", dims=SIZE_DIMS),
        ]
    )

    registry = ConstraintRegistry()
    registry.register(
        BoundChecker(
            "dwelling_class",
            "dwelling",
            {"hh_class=1": 2.0, "hh_class=2": 2.0, "hh_class=3+": 2.0},
        )
    )
    registry.register(
        BoundChecker(
            "hh_size",
            "household",
            {"size_class=1": 4.0, "size_class=2": 6.0, "size_class=3+": 2.0},
        )
    )
    registry.validate_against(attribute.n_columns)

    engine = GeneralizedIPUEngine.from_registry(
        attribute.matrix, registry, max_iterations=200, relative_gap=1e-4
    )
    result = engine.fit()

    assert result.converged
    # 거처 계급별 가중합이 목표를 재현한다.
    assert result.weighted_sums[:3] == pytest.approx([2.0, 2.0, 2.0], rel=1e-3)
    assert result.weighted_sums[3:] == pytest.approx([4.0, 6.0, 2.0], rel=1e-3)
