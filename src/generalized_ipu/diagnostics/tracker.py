"""회차별 수렴 지표의 누적과 이력 진단.

미수렴을 보고할 때는 알고리즘의 실패로 단정하지 않는다. 주어진 제약 집합이
실현 불가능할 가능성을 우선 진단 가설로 제시하며, `diagnose` 가 이력의 형태로부터
그 가설을 좁힌다.
"""

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class IterationRecord:
    """한 회차의 진단 기록."""

    iteration: int
    max_relative_gap: float
    max_absolute_diff: float
    violation_rate: float
    n_violations: int
    n_ratio_clipped: int = 0
    n_floor_clipped: int = 0


class ConvergenceTracker:
    """회차별 수렴 지표를 누적한다."""

    def __init__(self):
        self.records: List[IterationRecord] = []

    def log(
        self,
        iteration: int,
        report,
        n_ratio_clipped: int = 0,
        n_floor_clipped: int = 0,
    ) -> IterationRecord:
        """한 회차의 수렴 진단과 경계 제어 통계를 기록한다."""
        record = IterationRecord(
            iteration=iteration,
            max_relative_gap=report.max_relative_gap,
            max_absolute_diff=report.max_absolute_diff,
            violation_rate=report.violation_rate,
            n_violations=report.n_violations,
            n_ratio_clipped=n_ratio_clipped,
            n_floor_clipped=n_floor_clipped,
        )
        self.records.append(record)
        return record

    @property
    def history(self) -> List[float]:
        """회차별 최대 상대 격차."""
        return [record.max_relative_gap for record in self.records]

    def to_frame(self) -> pd.DataFrame:
        """회차별 기록을 표로 반환한다. 기록이 없으면 빈 표를 반환한다."""
        if not self.records:
            return pd.DataFrame(
                columns=[
                    "iteration",
                    "max_relative_gap",
                    "max_absolute_diff",
                    "violation_rate",
                    "n_violations",
                    "n_ratio_clipped",
                    "n_floor_clipped",
                ]
            )
        return pd.DataFrame([record.__dict__ for record in self.records])

    def diagnose(self, tolerance: float = 1e-12) -> str:
        """이력의 형태로부터 진단 소견을 반환한다."""
        gaps = self.history
        if len(gaps) < 3:
            return "회차가 부족하여 판정할 수 없습니다."

        recent = gaps[-min(10, len(gaps)) :]
        decreasing = all(later <= earlier + tolerance for earlier, later in zip(recent, recent[1:]))
        if gaps[-1] <= tolerance:
            return "정상 수렴"
        if decreasing:
            span = recent[0] - recent[-1]
            if span <= tolerance:
                return "정체: 완화 계수 과소 또는 제약 간 상충이 의심됩니다."
            return "감소 중"
        if gaps[-1] > gaps[0]:
            return "발산: 갱신식 또는 목표 정규화의 결함이 의심됩니다."
        return "진동: 과잉 제약 또는 완화 계수 과대가 의심됩니다."
