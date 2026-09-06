"""반복 갱신 루프의 주도.

한 회차 안에서 모든 제약 열의 갱신 비율을 동일한 가중치로 계산한 뒤 한 번에
반영하는 동시 갱신(Jacobi 방식)을 사용한다. 제약 열의 순서에 결과가 의존하지
않고 한 회차를 희소 행렬-벡터 곱 두 번으로 마칠 수 있다. 순차 갱신보다 수렴이
느리고 완화 계수가 클 때 진동할 수 있으므로, 수렴 특성을 논할 때는 이 성질을
전제로 삼는다.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.sparse import csr_matrix

from ..constraints.interval import compute_update_ratios, normalize_targets
from ..constraints.registry import ConstraintRegistry
from ..diagnostics.tracker import ConvergenceTracker
from ..matrix.scaling import SparseScalingKernel
from .bounds import WeightBoundController
from .convergence import ConvergenceEvaluator, ConvergenceReport

CONVERGED = "converged"
ITERATION_LIMIT = "iteration_limit_reached"


@dataclass
class IPUResult:
    """반복의 최종 산출물."""

    weights: np.ndarray
    converged: bool
    termination_reason: str
    n_iterations: int
    report: ConvergenceReport
    weighted_sums: np.ndarray
    tracker: ConvergenceTracker

    @property
    def max_relative_gap(self) -> float:
        """최종 회차의 최대 상대 격차를 반환한다."""
        return self.report.max_relative_gap

    def __repr__(self) -> str:
        return (
            f"IPUResult(termination_reason={self.termination_reason!r}, "
            f"n_iterations={self.n_iterations}, "
            f"max_relative_gap={self.report.max_relative_gap:.6g}, "
            f"n_violations={self.report.n_violations})"
        )


class GeneralizedIPUEngine:
    """가중치의 반복 갱신을 주도한다.

    목표는 하한과 상한의 쌍으로 받는다. upper 를 생략하면 스칼라 목표로 보아
    lower 와 동일하게 둔다.
    """

    def __init__(
        self,
        matrix: csr_matrix,
        lower: np.ndarray,
        upper: np.ndarray = None,
        max_iterations: int = 100,
        relative_gap: float = 0.01,
        absolute_diff: float = 0.0,
        weight_floor: float = 1e-5,
        min_ratio: float = 0.1,
        max_ratio: float = 10.0,
        eta: float = 1.0,
        registry: Optional[ConstraintRegistry] = None,
        tracker: Optional[ConvergenceTracker] = None,
    ):
        if not 0 < eta <= 1.0:
            raise ValueError(f"완화 계수는 (0, 1] 범위여야 합니다: {eta}")
        if max_iterations < 1:
            raise ValueError("max_iterations 는 1 이상이어야 합니다.")

        self.kernel = SparseScalingKernel(matrix)
        self.matrix = self.kernel.matrix

        lower = np.asarray(lower, dtype="float64")
        upper = lower.copy() if upper is None else np.asarray(upper, dtype="float64")
        if lower.size != self.matrix.shape[1]:
            raise ValueError(
                f"목표의 길이({lower.size})가 속성 행렬의 열 수({self.matrix.shape[1]})와 다릅니다."
            )
        if np.any(lower > upper):
            raise ValueError("하한이 상한보다 큰 제약 열이 있습니다.")

        self.lower = lower
        self.upper = upper
        self.max_iterations = max_iterations
        self.eta = eta
        self.registry = registry
        if registry is not None:
            registry.validate_against(self.matrix.shape[1])

        self.bound_controller = WeightBoundController(
            weight_floor=weight_floor, min_ratio=min_ratio, max_ratio=max_ratio
        )
        self.evaluator = ConvergenceEvaluator(
            relative_gap=relative_gap, absolute_diff=absolute_diff
        )
        self.tracker = tracker if tracker is not None else ConvergenceTracker()

    # ------------------------------------------------------------- 생성 보조

    @classmethod
    def from_registry(
        cls, matrix: csr_matrix, registry: ConstraintRegistry, **kwargs
    ) -> "GeneralizedIPUEngine":
        """제약 등록소의 목표를 그대로 받아 엔진을 만든다.

        등록소를 넘기면 갱신 비율을 제약별로 나누어 계산하므로, 제약마다 다른
        비율 규칙을 두는 확장이 가능해진다.
        """
        lower, upper = registry.bounds()
        return cls(matrix, lower, upper, registry=registry, **kwargs)

    @classmethod
    def from_targets(cls, matrix: csr_matrix, targets, **kwargs) -> "GeneralizedIPUEngine":
        """스칼라와 구간이 섞인 목표 나열로부터 생성한다."""
        lower, upper = normalize_targets(list(targets))
        return cls(matrix, lower, upper, **kwargs)

    # ------------------------------------------------------------- 반복

    def fit(self, initial_weights: np.ndarray = None) -> IPUResult:
        """수렴하거나 최대 반복에 도달할 때까지 가중치를 갱신한다.

        초기 가중치를 생략하면 모두 1로 둔다. 0회차의 진단도 기록하므로 진단
        기록의 행 수는 실제 반복 횟수보다 하나 많다.
        """
        weights = self._initial_weights(initial_weights)
        weighted_sums = self.kernel.weighted_sums(weights)
        report = self.evaluator.evaluate(weighted_sums, self.lower, self.upper)
        self._record(0, report)

        iteration = 0
        while not report.converged and iteration < self.max_iterations:
            iteration += 1
            ratios = self._compute_ratios(weighted_sums)
            candidate = self.kernel.next_weights(weights, ratios, eta=self.eta)
            weights = self.bound_controller.apply_bounds(weights, candidate)

            weighted_sums = self.kernel.weighted_sums(weights)
            report = self.evaluator.evaluate(weighted_sums, self.lower, self.upper)
            self._record(iteration, report)

        return IPUResult(
            weights=weights,
            converged=report.converged,
            termination_reason=CONVERGED if report.converged else ITERATION_LIMIT,
            n_iterations=iteration,
            report=report,
            weighted_sums=weighted_sums,
            tracker=self.tracker,
        )

    # ------------------------------------------------------------- 내부

    def _initial_weights(self, initial_weights: Optional[np.ndarray]) -> np.ndarray:
        """초기 가중치를 검증하여 복사본으로 반환한다."""
        n_units = self.matrix.shape[0]
        if initial_weights is None:
            return np.ones(n_units, dtype="float64")
        weights = np.array(initial_weights, dtype="float64", copy=True)
        if weights.size != n_units:
            raise ValueError(
                f"초기 가중치의 길이({weights.size})가 기본 단위 수({n_units})와 다릅니다."
            )
        if np.any(weights <= 0):
            raise ValueError("초기 가중치는 모두 양수여야 합니다.")
        return weights

    def _compute_ratios(self, weighted_sums: np.ndarray) -> np.ndarray:
        """등록소가 있으면 제약별로, 없으면 전체 목표로 갱신 비율을 계산한다."""
        if self.registry is not None:
            return self.registry.compute_ratios(weighted_sums)
        return compute_update_ratios(weighted_sums, self.lower, self.upper)

    def _record(self, iteration: int, report: ConvergenceReport) -> None:
        """수렴 진단과 경계 제어 개입 정도를 진단 기록에 남긴다."""
        statistics = self.bound_controller.last_statistics
        self.tracker.log(
            iteration=iteration,
            report=report,
            n_ratio_clipped=statistics.n_ratio_clipped,
            n_floor_clipped=statistics.n_floor_clipped,
        )

    @property
    def inactive_units(self) -> List[int]:
        """어떤 제약에도 관여하지 않아 가중치가 갱신되지 않는 기본 단위."""
        return self.kernel.inactive_units.tolist()
