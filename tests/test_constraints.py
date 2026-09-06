import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from generalized_ipu import (
    BoundChecker,
    ConstraintRegistry,
    IntervalTarget,
    LogicalRuleParser,
    StructuralZeroMask,
    TensorTarget,
    ZeroCellResolver,
)
from generalized_ipu.constraints.interval import bound_violations, compute_update_ratios


def test_scalar_target_is_degenerate_interval():
    target = IntervalTarget.from_value(100.0)
    assert target.lower == target.upper == 100.0
    assert target.is_scalar


def test_update_ratio_is_piecewise():
    lower = np.array([100.0, 100.0, 100.0])
    upper = np.array([200.0, 200.0, 200.0])
    sums = np.array([50.0, 150.0, 400.0])

    ratios = compute_update_ratios(sums, lower, upper)
    assert ratios[0] == pytest.approx(2.0)
    assert ratios[1] == 1.0
    assert ratios[2] == pytest.approx(0.5)


def test_bound_violation_is_zero_inside_interval():
    lower = np.array([100.0, 100.0, 100.0])
    upper = np.array([200.0, 200.0, 200.0])
    sums = np.array([50.0, 150.0, 400.0])

    violations = bound_violations(sums, lower, upper)
    assert list(violations) == [50.0, 0.0, 200.0]


def test_zero_weighted_sum_leaves_ratio_unchanged():
    ratios = compute_update_ratios(
        np.array([0.0]), np.array([10.0]), np.array([10.0])
    )
    assert ratios[0] == 1.0


def test_bound_checker_rejects_inverted_interval():
    with pytest.raises(ValueError):
        BoundChecker("c", "household", {"a": (10.0, 5.0)})


def test_tensor_target_flattens_in_c_order():
    dims = {"age_group": ["child", "adult"], "sex": ["M", "F"]}
    target = np.array([[1.0, 2.0], [3.0, 4.0]])
    constraint = TensorTarget("age_sex", "person", dims, target)

    assert constraint.column_names == [
        "age_group=child|sex=M",
        "age_group=child|sex=F",
        "age_group=adult|sex=M",
        "age_group=adult|sex=F",
    ]
    assert list(constraint.lower) == [1.0, 2.0, 3.0, 4.0]
    assert constraint.multi_index_of(1) == ("child", "F")


def test_validity_mask_is_column_axis_and_true_for_valid():
    dims = {"age_group": ["child", "adult"], "license": [0, 1]}
    constraint = TensorTarget("t", "person", dims, np.ones((2, 2)))
    parser = LogicalRuleParser(["age_group == 'child' and license == 1"])

    mask = parser.build_validity_mask(constraint.category_frame())
    assert mask.size == constraint.n_columns
    assert list(mask) == [True, False, True, True]


def test_structural_zero_columns_are_dropped_by_column_axis():
    matrix = csr_matrix(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    mask = np.array([True, False, True])

    reduced, kept = StructuralZeroMask.select_valid_columns(matrix, mask)
    assert reduced.shape == (2, 2)
    assert list(kept) == [0, 2]


def test_mask_with_wrong_axis_is_rejected():
    matrix = csr_matrix(np.zeros((5, 3)))
    with pytest.raises(ValueError, match="제약 열의 축"):
        StructuralZeroMask.select_valid_columns(matrix, np.ones(5, dtype=bool))


def test_positive_target_on_structural_zero_is_forced_to_zero():
    lower = np.array([10.0, 20.0])
    upper = np.array([10.0, 20.0])
    mask = np.array([True, False])

    new_lower, new_upper = StructuralZeroMask.enforce_zero_targets(lower, upper, mask)
    assert list(new_lower) == [10.0, 0.0]
    assert list(new_upper) == [10.0, 0.0]


def test_epsilon_smoothing_keeps_matrix_sparse():
    dense = np.zeros((100, 50))
    dense[:, 0] = 1.0
    matrix = csr_matrix(dense)
    lower = np.zeros(50)
    lower[7] = 5.0
    weights = np.ones(100)

    columns = ZeroCellResolver.detect(matrix, weights, lower)
    assert list(columns) == [7]

    smoothed = ZeroCellResolver.apply_epsilon_smoothing(matrix, columns)
    assert smoothed.nnz == matrix.nnz + 100
    assert smoothed[:, 7].sum() == pytest.approx(100 * 1e-5)


def test_epsilon_is_not_injected_into_structural_zero():
    matrix = csr_matrix(np.zeros((10, 3)))
    lower = np.array([1.0, 1.0, 1.0])
    mask = np.array([True, False, True])

    columns = ZeroCellResolver.detect(matrix, np.ones(10), lower, mask)
    assert list(columns) == [0, 2]


def test_registry_assigns_column_slices():
    registry = ConstraintRegistry()
    registry.register(BoundChecker("car", "household", {"car": 5.0}))
    registry.register(
        TensorTarget("age_sex", "person", {"age": ["c", "a"]}, np.array([3.0, 4.0]))
    )

    assert registry.n_columns == 3
    assert registry.slice_of("age_sex") == slice(1, 3)
    assert registry.owner_of_column(2) == "age_sex"
    assert registry.column_names[0] == "car::car"

    lower, upper = registry.bounds()
    assert list(lower) == [5.0, 3.0, 4.0]
    assert list(upper) == [5.0, 3.0, 4.0]


def test_registry_rejects_duplicate_name():
    registry = ConstraintRegistry()
    registry.register(BoundChecker("car", "household", {"car": 5.0}))
    with pytest.raises(ValueError, match="이미 등록"):
        registry.register(BoundChecker("car", "household", {"car": 7.0}))


def test_registry_computes_ratios_per_constraint():
    registry = ConstraintRegistry()
    registry.register(BoundChecker("a", "household", {"x": (10.0, 20.0)}))
    registry.register(BoundChecker("b", "household", {"y": 40.0}))

    ratios = registry.compute_ratios(np.array([15.0, 20.0]))
    assert ratios[0] == 1.0
    assert ratios[1] == pytest.approx(2.0)
