"""제약 열별 적합도 보고서의 생성.

수렴 판정이 하나의 논리값으로 압축한 정보를 제약 열 단위로 되펼친다. 미수렴
시에는 어느 제약 열이 얼마나 어긋났는지가 진단의 출발점이 된다.
"""

from typing import Sequence

import numpy as np
import pandas as pd

RELATIVE_SCALE_FLOOR = 1.0


class FitReportGenerator:
    """제약 열별 적합도를 표로 산출한다."""

    @staticmethod
    def generate_report(
        weighted_sums: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray = None,
        column_names: Sequence[str] = None,
        column_owners: Sequence[str] = None,
        relative_gap: float = 0.0,
        absolute_diff: float = 0.0,
    ) -> pd.DataFrame:
        """구간 위반량 기준의 적합도 표를 내림차순으로 반환한다.

        `within_bounds` 는 가중합이 목표 구간 안에 있는지를 엄격히 판정하고,
        `satisfied` 는 ConvergenceEvaluator 와 같은 규칙으로 허용 한계를 반영한다.
        허용 한계를 모두 0으로 두면 두 열이 일치한다.
        """
        weighted_sums = np.asarray(weighted_sums, dtype="float64")
        lower = np.asarray(lower, dtype="float64")
        upper = lower if upper is None else np.asarray(upper, dtype="float64")

        below = lower - weighted_sums
        above = np.where(np.isfinite(upper), weighted_sums - upper, -np.inf)
        violations = np.maximum(np.maximum(below, above), 0.0)
        relative = violations / np.maximum(lower, RELATIVE_SCALE_FLOOR)

        if column_names is None:
            column_names = [f"column_{index}" for index in range(weighted_sums.size)]

        report = pd.DataFrame(
            {
                "column": list(column_names),
                "lower": lower,
                "upper": upper,
                "estimated": weighted_sums,
                "bound_violation": violations,
                "relative_gap": relative,
                "within_bounds": violations == 0.0,
                "satisfied": (violations <= absolute_diff) | (relative <= relative_gap),
            }
        )
        if column_owners is not None:
            report.insert(0, "constraint", list(column_owners))

        return report.sort_values(
            by=["relative_gap", "bound_violation"], ascending=False
        ).reset_index(drop=True)

    @staticmethod
    def summarize_by_constraint(report: pd.DataFrame) -> pd.DataFrame:
        """제약별 요약 집계를 반환한다."""
        if "constraint" not in report.columns:
            raise KeyError("보고서에 'constraint' 열이 없습니다. column_owners 를 지정하십시오.")
        grouped = report.groupby("constraint", sort=False)
        return pd.DataFrame(
            {
                "n_columns": grouped.size(),
                "n_outside_bounds": grouped["within_bounds"].apply(lambda s: int((~s).sum())),
                "n_unsatisfied": grouped["satisfied"].apply(lambda s: int((~s).sum())),
                "max_relative_gap": grouped["relative_gap"].max(),
                "max_bound_violation": grouped["bound_violation"].max(),
                "total_target": grouped["lower"].sum(),
                "total_estimated": grouped["estimated"].sum(),
            }
        ).reset_index()

    @staticmethod
    def worst_columns(report: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """적합도가 가장 나쁜 제약 열을 반환한다."""
        return report.head(n)
