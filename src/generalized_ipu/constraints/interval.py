"""목표 구간의 표현과 구간 기반 갱신 비율 계산.

라이브러리의 모든 목표는 구간 [L_k, U_k] 로 정규화된다. 고정 스칼라 목표는
L_k = U_k 인 퇴화 구간으로 취급하므로, 내부에는 스칼라 목표라는 별도의 자료형이
존재하지 않는다. 상한이 없는 목표는 U_k 를 무한대로 두어 표현한다.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple, Union

import numpy as np

from .base import AbstractConstraint

TargetLike = Union[float, int, Tuple[float, float], "IntervalTarget"]


@dataclass(frozen=True)
class IntervalTarget:
    """목표 구간 [lower, upper]. 스칼라 목표는 lower == upper 인 퇴화 구간이다."""

    lower: float
    upper: float

    def __post_init__(self):
        if self.lower > self.upper:
            raise ValueError(f"하한 {self.lower} 이 상한 {self.upper} 보다 큽니다.")
        if self.lower < 0:
            raise ValueError(f"음수 하한은 허용되지 않습니다: {self.lower}")

    @classmethod
    def from_value(cls, value: TargetLike) -> "IntervalTarget":
        """스칼라, 두 원소 나열, 구간 객체 중 어느 형태든 구간으로 정규화한다."""
        if isinstance(value, IntervalTarget):
            return value
        if isinstance(value, (tuple, list, np.ndarray)):
            if len(value) != 2:
                raise ValueError(f"구간 목표는 (하한, 상한) 두 값이어야 합니다: {value!r}")
            return cls(float(value[0]), float(value[1]))
        return cls(float(value), float(value))

    @property
    def is_scalar(self) -> bool:
        """퇴화 구간, 곧 고정 스칼라 목표인지 판정한다."""
        return self.lower == self.upper


def normalize_targets(targets: Sequence[TargetLike]) -> Tuple[np.ndarray, np.ndarray]:
    """목표 나열을 하한 배열과 상한 배열의 쌍으로 정규화한다."""
    intervals = [IntervalTarget.from_value(value) for value in targets]
    lower = np.array([interval.lower for interval in intervals], dtype="float64")
    upper = np.array([interval.upper for interval in intervals], dtype="float64")
    return lower, upper


def compute_update_ratios(
    weighted_sums: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """구간별 갱신 비율을 벡터 연산으로 산출한다.

    가중합이 구간 안에 있으면 비율이 1이 되어 가중치를 변경하지 않는다. 이는
    과잉 제약 상황에서 양립 불가능한 제약들이 가중치를 번갈아 끌어당기며 만드는
    진동을 억제하기 위한 설계이다.

    가중합이 0 이하인 열은 비율을 정의할 수 없으므로 1.0 으로 둔다. 이런 열은
    `ZeroCellResolver` 가 표본 영으로 판정하여 별도로 처리한다.
    """
    weighted_sums = np.asarray(weighted_sums, dtype="float64")
    ratios = np.ones_like(weighted_sums, dtype="float64")

    positive = weighted_sums > 0
    below = positive & (weighted_sums < lower)
    above = positive & (weighted_sums > upper) & np.isfinite(upper)

    ratios[below] = lower[below] / weighted_sums[below]
    ratios[above] = upper[above] / weighted_sums[above]
    return ratios


def bound_violations(
    weighted_sums: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """구간 위반량 v_k = max(L_k - S_k, S_k - U_k, 0) 을 산출한다.

    가중합이 목표 구간 안에 있으면 0이며, 수렴 판정과 적합도 보고의 공통 기준이다.
    """
    weighted_sums = np.asarray(weighted_sums, dtype="float64")
    below = lower - weighted_sums
    above = np.where(np.isfinite(upper), weighted_sums - upper, -np.inf)
    return np.maximum(np.maximum(below, above), 0.0)


class BoundChecker(AbstractConstraint):
    """구간 목표를 가지는 제약.

    targets 는 제약 열 이름에서 목표로의 대응이며, 값은 스칼라와 구간을 섞어
    줄 수 있다. 선언 순서가 그대로 제약 열의 순서가 된다.
    """

    def __init__(self, name: str, level: str, targets: Mapping[str, TargetLike]):
        column_names = list(targets.keys())
        lower, upper = normalize_targets([targets[key] for key in column_names])
        super().__init__(name, level, column_names, lower, upper)

    @classmethod
    def from_arrays(
        cls,
        name: str,
        level: str,
        column_names: Sequence[str],
        lower: np.ndarray,
        upper: np.ndarray = None,
    ) -> "BoundChecker":
        """하한과 상한 배열로부터 직접 생성한다. upper 를 생략하면 스칼라 목표로 본다."""
        instance = cls.__new__(cls)
        upper = lower if upper is None else upper
        AbstractConstraint.__init__(instance, name, level, column_names, lower, upper)
        return instance

    def is_within(self, weighted_sums: np.ndarray) -> np.ndarray:
        """제약 열별로 가중합이 목표 구간 안에 있는지 판정한다."""
        return self.bound_violations(weighted_sums) == 0.0
