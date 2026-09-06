import numpy as np
import pandas as pd
import pytest

from generalized_ipu import MissingMarginEstimator, marginal_from_series
from generalized_ipu.constraints.missing import balance_from_classes


def test_single_missing_cell_is_determined_by_total():
    estimator = MissingMarginEstimator({"a": 30.0, "b": 50.0, "c": None})
    estimator.add_balance("total", 100.0)

    intervals = estimator.estimate()
    assert intervals["c"].lower == pytest.approx(20.0)
    assert intervals["c"].upper == pytest.approx(20.0)
    assert intervals["c"].is_scalar


def test_two_missing_cells_share_the_residual():
    estimator = MissingMarginEstimator({"a": 30.0, "b": None, "c": None})
    estimator.add_balance("total", 100.0)

    intervals = estimator.estimate()
    assert intervals["b"].lower == pytest.approx(0.0)
    assert intervals["b"].upper == pytest.approx(70.0)
    assert intervals["c"].upper == pytest.approx(70.0)


def test_prior_upper_bound_tightens_the_other_cell():
    estimator = MissingMarginEstimator({"a": 30.0, "b": (0.0, 4.0), "c": None})
    estimator.add_balance("total", 100.0)

    intervals = estimator.estimate()
    assert intervals["c"].lower == pytest.approx(66.0)
    assert intervals["c"].upper == pytest.approx(70.0)


def test_interval_total_widens_the_estimate():
    estimator = MissingMarginEstimator({"a": 30.0, "b": None})
    estimator.add_balance("total", (90.0, 110.0))

    intervals = estimator.estimate()
    assert intervals["b"].lower == pytest.approx(60.0)
    assert intervals["b"].upper == pytest.approx(80.0)


def test_observed_cells_stay_at_their_published_value():
    estimator = MissingMarginEstimator({"a": 30.0, "b": None})
    estimator.add_balance("total", 100.0)

    intervals = estimator.estimate()
    assert intervals["a"].lower == intervals["a"].upper == pytest.approx(30.0)


def test_missing_cell_without_balance_has_infinite_upper():
    estimator = MissingMarginEstimator({"a": 30.0, "b": None})

    intervals = estimator.estimate()
    assert intervals["b"].lower == 0.0
    assert not np.isfinite(intervals["b"].upper)


def test_coefficient_balance_links_household_size_to_person_total():
    # 1인가구 40호, 2인가구 미상, 3인가구 10호이고 총 가구원 수가 100명이다.
    estimator = MissingMarginEstimator({"size1": 40.0, "size2": None, "size3": 10.0})
    estimator.add_balance(
        "persons", 100.0, balance_from_classes(["size1", "size2", "size3"], [1.0, 2.0, 3.0])
    )

    intervals = estimator.estimate()
    assert intervals["size2"].lower == pytest.approx(15.0)
    assert intervals["size2"].upper == pytest.approx(15.0)


def test_two_balances_narrow_each_other():
    # s1 + s2 + 10 = 100 과 s1 + 2*s2 + 30 = 190 이 함께 주어지면 해가 유일하다.
    # 두 균형식이 셀을 공유하므로 구간은 유한 회차에 닫히지 않고 점근적으로 좁혀진다.
    estimator = MissingMarginEstimator({"size1": None, "size2": None, "size3": 10.0})
    estimator.add_balance("households", 100.0)
    estimator.add_balance(
        "persons", 190.0, balance_from_classes(["size1", "size2", "size3"], [1.0, 2.0, 3.0])
    )

    intervals = estimator.estimate()
    assert intervals["size1"].lower == pytest.approx(20.0, abs=1e-6)
    assert intervals["size1"].upper == pytest.approx(20.0, abs=1e-6)
    assert intervals["size2"].lower == pytest.approx(70.0, abs=1e-6)
    assert intervals["size2"].upper == pytest.approx(70.0, abs=1e-6)


def test_contradictory_inputs_are_rejected():
    estimator = MissingMarginEstimator({"a": 80.0, "b": 50.0})
    estimator.add_balance("total", 100.0)

    with pytest.raises(ValueError, match="모순"):
        estimator.estimate()


def test_negative_coefficient_is_rejected():
    estimator = MissingMarginEstimator({"a": None, "b": None})
    with pytest.raises(ValueError, match="양수"):
        estimator.add_balance("total", 100.0, {"a": 1.0, "b": -1.0})


def test_balance_referring_to_unknown_cell_is_rejected():
    estimator = MissingMarginEstimator({"a": None})
    with pytest.raises(ValueError, match="등록되지 않은 셀"):
        estimator.add_balance("total", 100.0, {"a": 1.0, "z": 1.0})


def test_to_constraint_keeps_declaration_order():
    estimator = MissingMarginEstimator({"a": 30.0, "b": None, "c": None})
    estimator.add_balance("total", 100.0)

    constraint = estimator.to_constraint("margin", "household")
    assert constraint.column_names == ["a", "b", "c"]
    assert constraint.n_columns == 3
    assert constraint.lower[0] == pytest.approx(30.0)
    assert constraint.upper[1] == pytest.approx(70.0)


def test_balance_hierarchy_groups_all_cells_into_one_coarse_category():
    estimator = MissingMarginEstimator({"a": 30.0, "b": None})
    estimator.add_balance("total", 100.0)

    hierarchy = estimator.balance_hierarchy("total")
    assert hierarchy.coarse_categories == ["total"]
    assert hierarchy.mapping["total"] == ["a", "b"]

    constraint = estimator.balance_constraint("total", "margin_total", "household")
    assert constraint.lower[0] == pytest.approx(100.0)


def test_balance_hierarchy_rejects_non_unit_coefficients():
    estimator = MissingMarginEstimator({"a": None, "b": None})
    estimator.add_balance("persons", 100.0, {"a": 1.0, "b": 2.0})

    with pytest.raises(ValueError, match="계수가 1이 아니므로"):
        estimator.balance_hierarchy("persons")


def test_allocate_splits_residual_by_given_shares():
    estimator = MissingMarginEstimator({"a": 40.0, "b": None, "c": None})
    estimator.add_balance("total", 100.0)

    allocated = estimator.allocate("total", shares={"b": 3.0, "c": 1.0})
    assert allocated["b"] == pytest.approx(45.0)
    assert allocated["c"] == pytest.approx(15.0)


def test_allocate_defaults_to_equal_split():
    estimator = MissingMarginEstimator({"a": 40.0, "b": None, "c": None})
    estimator.add_balance("total", 100.0)

    allocated = estimator.allocate("total")
    assert allocated["b"] == pytest.approx(30.0)
    assert allocated["c"] == pytest.approx(30.0)


def test_to_frame_reports_status_and_width():
    estimator = MissingMarginEstimator({"a": 30.0, "b": None})
    estimator.add_balance("total", 100.0)

    frame = estimator.to_frame()
    assert list(frame["status"]) == ["observed", "missing"]
    assert frame.loc[0, "width"] == pytest.approx(0.0)
    assert frame.loc[1, "width"] == pytest.approx(0.0)


def test_marginal_from_series_treats_nan_as_missing():
    series = pd.Series({"a": 30.0, "b": np.nan, "c": 20.0})
    cells = marginal_from_series(series)

    assert cells["b"] is None
    assert cells["a"] == pytest.approx(30.0)

    estimator = MissingMarginEstimator(cells)
    estimator.add_balance("total", 100.0)
    assert estimator.missing_cells == ["b"]
    assert estimator.estimate()["b"].lower == pytest.approx(50.0)
