"""최종 가중치 분포의 건전성 진단.

제약 적합도가 양호하더라도 소수의 기본 단위가 전체 가중치의 대부분을 차지하면
합성 결과는 표본의 다양성을 잃는다. 적합도 지표만으로는 이 상태를 탐지할 수
없으므로 분포 자체를 따로 진단한다.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class WeightDistributionReport:
    """최종 가중치 분포의 건전성 지표."""

    n_units: int
    total_weight: float
    minimum: float
    maximum: float
    mean: float
    std: float
    quantiles: Dict[str, float] = field(default_factory=dict)
    coefficient_of_variation: float = 0.0
    normalized_entropy: float = 0.0
    effective_sample_size: float = 0.0
    top_1_percent_share: float = 0.0
    top_5_percent_share: float = 0.0
    n_at_floor: int = 0
    max_expansion_ratio: Optional[float] = None
    min_expansion_ratio: Optional[float] = None

    def to_series(self) -> pd.Series:
        """분위수를 개별 항목으로 펼친 계열로 변환한다."""
        flat = {key: value for key, value in self.__dict__.items() if key != "quantiles"}
        flat.update({f"q{key}": value for key, value in self.quantiles.items()})
        return pd.Series(flat)


class WeightDistributionAnalyzer:
    """가중치 분포를 분석한다.

    제약 적합도가 양호하더라도 소수의 기본 단위가 전체 가중치의 대부분을 차지하면
    합성 결과는 표본의 다양성을 잃는다. 적합도만으로는 이 상태를 탐지할 수 없다.
    """

    @staticmethod
    def analyze(
        weights: np.ndarray,
        initial_weights: np.ndarray = None,
        weight_floor: float = None,
    ) -> WeightDistributionReport:
        """가중치 분포의 집중도와 유효 표본 수를 산출한다.

        initial_weights 를 주면 기본 단위별 확대 배율의 범위를 함께 보고하고,
        weight_floor 를 주면 하한에 걸린 기본 단위 수를 센다.
        """
        weights = np.asarray(weights, dtype="float64")
        if weights.size == 0:
            raise ValueError("가중치가 비어 있습니다.")

        total = float(weights.sum())
        sorted_descending = np.sort(weights)[::-1]

        report = WeightDistributionReport(
            n_units=int(weights.size),
            total_weight=total,
            minimum=float(weights.min()),
            maximum=float(weights.max()),
            mean=float(weights.mean()),
            std=float(weights.std(ddof=0)),
            quantiles={
                "01": float(np.quantile(weights, 0.01)),
                "25": float(np.quantile(weights, 0.25)),
                "50": float(np.quantile(weights, 0.50)),
                "75": float(np.quantile(weights, 0.75)),
                "99": float(np.quantile(weights, 0.99)),
            },
        )

        mean = report.mean
        report.coefficient_of_variation = float(report.std / mean) if mean > 0 else 0.0
        report.normalized_entropy = WeightDistributionAnalyzer._normalized_entropy(weights)
        report.effective_sample_size = float(total**2 / np.square(weights).sum())
        report.top_1_percent_share = WeightDistributionAnalyzer._top_share(
            sorted_descending, total, 0.01
        )
        report.top_5_percent_share = WeightDistributionAnalyzer._top_share(
            sorted_descending, total, 0.05
        )

        if weight_floor is not None:
            report.n_at_floor = int(np.count_nonzero(weights <= weight_floor))

        if initial_weights is not None:
            initial = np.asarray(initial_weights, dtype="float64")
            if initial.size != weights.size:
                raise ValueError("초기 가중치와 최종 가중치의 길이가 다릅니다.")
            ratios = weights / np.where(initial == 0.0, np.nan, initial)
            report.max_expansion_ratio = float(np.nanmax(ratios))
            report.min_expansion_ratio = float(np.nanmin(ratios))

        return report

    # ------------------------------------------------------------------ 내부

    @staticmethod
    def _normalized_entropy(weights: np.ndarray) -> float:
        """가중치 분포의 엔트로피를 균등 분포 대비 비율로 환산한다."""
        total = weights.sum()
        if total <= 0 or weights.size < 2:
            return 0.0
        share = weights / total
        share = share[share > 0]
        entropy = float(-(share * np.log(share)).sum())
        return entropy / np.log(weights.size)

    @staticmethod
    def _top_share(sorted_descending: np.ndarray, total: float, fraction: float) -> float:
        """상위 일정 비율의 기본 단위가 차지하는 가중치 몫을 구한다."""
        if total <= 0:
            return 0.0
        count = max(1, int(np.ceil(sorted_descending.size * fraction)))
        return float(sorted_descending[:count].sum() / total)

    @staticmethod
    def flag_concerns(
        report: WeightDistributionReport,
        min_effective_ratio: float = 0.5,
        max_top_share: float = 0.5,
    ) -> "list[str]":
        """주의가 필요한 항목을 문자열 목록으로 반환한다."""
        concerns = []
        if report.effective_sample_size < report.n_units * min_effective_ratio:
            concerns.append(
                f"유효 표본 수 {report.effective_sample_size:.1f} 이 "
                f"기본 단위 수 {report.n_units} 의 {min_effective_ratio:.0%} 에 못 미칩니다."
            )
        if report.top_5_percent_share > max_top_share:
            concerns.append(
                f"상위 5% 기본 단위가 전체 가중치의 {report.top_5_percent_share:.1%} 를 차지합니다."
            )
        if report.n_at_floor:
            concerns.append(f"가중치 하한에 걸린 기본 단위가 {report.n_at_floor}개입니다.")
        return concerns
