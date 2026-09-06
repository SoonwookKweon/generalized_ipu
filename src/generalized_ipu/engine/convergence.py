"""구간 위반량을 기준으로 하는 수렴 판정.

수렴은 모든 제약 열의 격차가 허용 한계 이하가 된 상태이며, 최대 반복 도달은
수렴이 아니다. 두 종료 사유는 진단 보고서에서 반드시 구분하여 표기한다.
"""

from dataclasses import dataclass

import numpy as np

from ..constraints.interval import bound_violations

RELATIVE_SCALE_FLOOR = 1.0


@dataclass
class ConvergenceReport:
    """한 회차의 수렴 진단 결과."""

    converged: bool
    max_relative_gap: float
    max_absolute_diff: float
    violation_rate: float
    n_violations: int

    def as_dict(self) -> dict:
        """진단 기록에 넘기기 위한 사전 형태로 변환한다."""
        return {
            "converged": self.converged,
            "max_relative_gap": self.max_relative_gap,
            "max_absolute_diff": self.max_absolute_diff,
            "violation_rate": self.violation_rate,
            "n_violations": self.n_violations,
        }


class ConvergenceEvaluator:
    """구간 위반량을 기준으로 수렴 여부를 판정한다.

    제약 열은 절대 격차와 상대 격차 중 하나만 충족해도 만족한 것으로 본다.
    absolute_diff 를 0 으로 두면 상대 격차만으로 판정한다.
    """

    def __init__(self, relative_gap: float = 0.01, absolute_diff: float = 0.0):
        if relative_gap < 0 or absolute_diff < 0:
            raise ValueError("허용 한계는 0 이상이어야 합니다.")
        self.relative_gap = relative_gap
        self.absolute_diff = absolute_diff

    def evaluate(
        self, weighted_sums: np.ndarray, lower: np.ndarray, upper: np.ndarray = None
    ) -> ConvergenceReport:
        """가중합과 목표 구간으로부터 한 회차의 수렴 진단을 산출한다.

        상대 격차의 분모는 max(L_k, 1) 이다. 목표가 0에 가까운 제약 열에서 분모가
        0으로 수렴하여 상대 격차가 발산하는 것을 막는다.
        """
        lower = np.asarray(lower, dtype="float64")
        upper = lower if upper is None else np.asarray(upper, dtype="float64")

        violations = bound_violations(weighted_sums, lower, upper)
        scale = np.maximum(lower, RELATIVE_SCALE_FLOOR)
        relative = violations / scale

        satisfied = (violations <= self.absolute_diff) | (relative <= self.relative_gap)
        n_columns = violations.size
        n_violations = int(np.count_nonzero(~satisfied))

        return ConvergenceReport(
            converged=n_violations == 0,
            max_relative_gap=float(relative.max()) if n_columns else 0.0,
            max_absolute_diff=float(violations.max()) if n_columns else 0.0,
            violation_rate=(n_violations / n_columns) if n_columns else 0.0,
            n_violations=n_violations,
        )
