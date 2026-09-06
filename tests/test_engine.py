import numpy as np
import pytest
from scipy.sparse import csr_matrix

from generalized_ipu import GeneralizedIPUEngine
from generalized_ipu.engine import CONVERGED, ITERATION_LIMIT, ConvergenceEvaluator


def simple_matrix() -> csr_matrix:
    return csr_matrix(
        np.array(
            [
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )


def test_scalar_targets_converge():
    engine = GeneralizedIPUEngine(
        simple_matrix(), lower=np.array([20.0, 30.0, 25.0]), relative_gap=1e-4, max_iterations=500
    )
    result = engine.fit()

    assert result.converged
    assert result.termination_reason == CONVERGED
    assert result.weighted_sums == pytest.approx([20.0, 30.0, 25.0], rel=1e-3)


def test_interval_target_stops_inside_the_interval():
    lower = np.array([10.0, 10.0, 5.0])
    upper = np.array([30.0, 30.0, 40.0])
    engine = GeneralizedIPUEngine(
        simple_matrix(), lower=lower, upper=upper, relative_gap=0.01, max_iterations=200
    )
    result = engine.fit()

    # 수렴은 허용 한계 기준이므로, 구간 이탈량이 한계 안에 있는지로 확인한다.
    assert result.converged
    assert result.report.max_relative_gap <= 0.01
    assert np.all(result.weighted_sums >= lower * 0.99)
    assert np.all(result.weighted_sums <= upper * 1.01)


def test_initial_weights_inside_interval_terminate_without_iteration():
    engine = GeneralizedIPUEngine(
        simple_matrix(), lower=np.array([0.0, 0.0, 0.0]), upper=np.array([100.0, 100.0, 100.0])
    )
    result = engine.fit()

    assert result.n_iterations == 0
    assert result.converged


def test_iteration_limit_is_reported_separately():
    # 양립 불가능한 목표: 열 0과 열 1의 합이 열 2와 모순된다.
    lower = np.array([10.0, 10.0, 1000.0])
    engine = GeneralizedIPUEngine(simple_matrix(), lower=lower, max_iterations=5)
    result = engine.fit()

    assert not result.converged
    assert result.termination_reason == ITERATION_LIMIT
    assert result.n_iterations == 5


def test_tracker_records_every_iteration():
    engine = GeneralizedIPUEngine(
        simple_matrix(), lower=np.array([20.0, 30.0, 25.0]), max_iterations=10
    )
    result = engine.fit()

    frame = result.tracker.to_frame()
    assert len(frame) == result.n_iterations + 1
    assert frame["iteration"].tolist() == list(range(result.n_iterations + 1))
    assert frame["max_relative_gap"].iloc[-1] <= frame["max_relative_gap"].iloc[0]


def test_relaxation_factor_is_exposed():
    engine = GeneralizedIPUEngine(
        simple_matrix(), lower=np.array([20.0, 30.0, 25.0]), eta=0.5, max_iterations=500
    )
    result = engine.fit()
    assert result.converged


def test_invalid_relaxation_factor_is_rejected():
    with pytest.raises(ValueError):
        GeneralizedIPUEngine(simple_matrix(), lower=np.zeros(3), eta=1.5)


def test_target_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="열 수"):
        GeneralizedIPUEngine(simple_matrix(), lower=np.zeros(2))


def test_positive_sum_against_zero_target_is_a_violation():
    evaluator = ConvergenceEvaluator(relative_gap=0.01)
    report = evaluator.evaluate(np.array([5.0]), np.array([0.0]), np.array([0.0]))

    assert not report.converged
    assert report.n_violations == 1


def test_absolute_tolerance_admits_small_target_columns():
    evaluator = ConvergenceEvaluator(relative_gap=0.001, absolute_diff=1.0)
    report = evaluator.evaluate(np.array([2.0]), np.array([1.0]), np.array([1.0]))

    assert report.converged
    assert report.max_absolute_diff == 1.0


def test_bound_controller_limits_change_ratio():
    engine = GeneralizedIPUEngine(
        simple_matrix(), lower=np.array([1000.0, 1000.0, 1000.0]), max_ratio=2.0, max_iterations=1
    )
    result = engine.fit()

    assert np.all(result.weights <= 2.0 + 1e-12)
    assert engine.bound_controller.last_statistics.n_ratio_clipped == 4


def test_weights_stay_positive_under_conflicting_targets():
    lower = np.array([0.0, 100.0, 0.0])
    upper = np.array([0.0, 100.0, 0.0])
    engine = GeneralizedIPUEngine(simple_matrix(), lower=lower, upper=upper, max_iterations=50)
    result = engine.fit()

    assert np.all(result.weights > 0)
