"""주변표의 결측 셀에 대한 목표 구간 추정.

공표된 주변표에는 값이 없는 셀이 흔히 존재한다. 비공개 처리, 반올림 공표,
표본 오차로 인한 미공표가 그 원인이다. 이런 셀을 그대로 버리면 그 범주가 제약
없이 방치되고, 임의의 점 추정으로 채우면 근거 없는 정보를 주입하게 된다. 본
모듈은 셋째 길을 택한다. 결측 셀에 목표 구간 [L_k, U_k] 를 부여하고, 그 구간을
관측된 셀과 총계가 함께 만족해야 하는 균형식으로부터 연역한다.

## 균형식

균형식(`LinearBalance`)은 셀 값들이 만족해야 하는 선형 등식 또는 부등식이다.

    sum_k c_k x_k in [T_lo, T_hi]

계수 c_k 를 모두 1로 두면 주변표의 총계 조건이 된다. 계수를 달리 두면 표
사이의 정합 조건도 같은 틀로 표현된다. 예를 들어 n인가구 수 H_n 과 총 가구원
수 P 사이에는 sum_n n * H_n = P 가 성립하고, 거처의 가구 수 계급별 거처 수
D_m 과 총 가구 수 H 사이에는 sum_m m * D_m = H 가 성립한다. 이 관계들을
균형식으로 등록하면 한 표의 결측이 다른 표의 관측값으로 좁혀진다.

## 구간 전파

균형식 하나와 셀별 현재 구간이 주어지면, 셀 k 의 구간은 다음으로 좁혀진다.

    U_k <- min(U_k, (T_hi - sum_{j != k} c_j L_j) / c_k)
    L_k <- max(L_k, (T_lo - sum_{j != k} c_j U_j) / c_k)

균형식이 하나뿐이면 한 번의 순회로 최적 구간에 도달한다. 여러 균형식이 셀을
공유하면 한 균형식에서 좁혀진 구간이 다른 균형식의 입력이 되므로, 더 이상
변화가 없을 때까지 순회를 반복한다. 이때 구간은 유한 회차에 닫히지 않고
점근적으로 좁혀질 수 있다. 순회마다 구간이 단조 감소하므로 절차 자체는 항상
정지하며, 어느 정도까지 좁힐지는 tolerance 와 max_passes 가 결정한다. 좁힌 결과
L_k > U_k 가 되면 주어진 관측값과 총계가 서로 모순된다는 뜻이므로 예외를
발생시킨다.

## 산출물의 사용

`to_constraint` 는 셀별 구간을 그대로 `BoundChecker` 로 만든다. 결측 셀은 넓은
구간을, 관측 셀은 퇴화 구간을 가지므로, 반복 갱신은 관측된 값을 정확히
재현하면서 결측 셀은 구간 안에서 자유롭게 움직인다. 다만 셀별 구간만으로는
"결측 셀들의 합이 잔차와 같다"는 결합 조건이 표현되지 않는다. 이 조건까지
부과하려면 `balance_hierarchy` 로 대범주 위계를 얻어 `CoarseBlock` 을 추가하고,
`balance_constraint` 가 만든 총계 제약을 함께 등록한다.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..hierarchy.category import CategoryHierarchy
from .interval import BoundChecker, IntervalTarget, TargetLike

DEFAULT_TOLERANCE = 1e-9
DEFAULT_MAX_PASSES = 100


@dataclass(frozen=True)
class LinearBalance:
    """셀 값들이 만족해야 하는 선형 균형식.

    coefficients 는 셀 이름에서 계수 c_k 로의 대응이며 계수는 모두 양수여야 한다.
    구간 전파의 단조성이 계수의 부호에 의존하기 때문이다. total 은 균형식의 값이
    놓일 구간 [T_lo, T_hi] 이며, 총계가 정확히 알려진 경우에는 퇴화 구간이 된다.
    """

    name: str
    coefficients: Mapping[str, float]
    total: IntervalTarget

    def __post_init__(self):
        if not self.coefficients:
            raise ValueError(f"균형식 '{self.name}' 에 셀이 지정되지 않았습니다.")
        nonpositive = [cell for cell, value in self.coefficients.items() if float(value) <= 0]
        if nonpositive:
            raise ValueError(
                f"균형식 '{self.name}' 의 계수는 모두 양수여야 합니다. "
                f"위반한 셀: {nonpositive}"
            )

    @property
    def cells(self) -> List[str]:
        """균형식이 포함하는 셀 이름을 선언 순서대로 반환한다."""
        return list(self.coefficients.keys())

    @property
    def is_unit(self) -> bool:
        """모든 계수가 1인 단순 합 균형식인지 판정한다."""
        return all(float(value) == 1.0 for value in self.coefficients.values())


class MissingMarginEstimator:
    """주변표의 결측 셀에 대한 목표 구간을 균형식으로부터 연역한다.

    cells 는 셀 이름에서 관측값으로의 대응이다. 값의 형태가 셀의 상태를 규정한다.

    | 입력 형태 | 뜻 | 초기 구간 |
    | --- | --- | --- |
    | 스칼라 | 정확히 공표된 값 | 퇴화 구간 [v, v] |
    | (하한, 상한) | 반올림 공표나 비공개 상한 | 주어진 구간 |
    | None | 값이 없는 결측 셀 | default_bounds |

    default_bounds 의 기정값은 [0, inf) 이며, 이 상태의 셀은 균형식이 있어야만
    유한한 상한을 얻는다. 비공개 처리가 "5 미만"과 같은 상한을 함의하는 경우에는
    (0.0, 4.0) 처럼 셀별로 직접 주는 편이 더 좁은 구간을 만든다.
    """

    def __init__(
        self,
        cells: Mapping[str, Optional[TargetLike]],
        default_bounds: TargetLike = (0.0, float("inf")),
    ):
        if not cells:
            raise ValueError("셀이 하나도 지정되지 않았습니다.")

        self.default_bounds = IntervalTarget.from_value(default_bounds)
        self.cell_names: List[str] = list(cells.keys())
        self.observed: Dict[str, Optional[IntervalTarget]] = {
            name: (None if value is None else IntervalTarget.from_value(value))
            for name, value in cells.items()
        }
        self.balances: List[LinearBalance] = []

    # ------------------------------------------------------------------ 등록

    def add_balance(
        self,
        name: str,
        total: TargetLike,
        coefficients: Mapping[str, float] = None,
    ) -> "MissingMarginEstimator":
        """균형식을 등록하고 자기 자신을 반환하여 연쇄 호출을 허용한다.

        coefficients 를 생략하면 등록된 모든 셀의 계수를 1로 두어 주변표 총계
        조건이 된다. 일부 셀만 참여하는 균형식이나 계수가 1이 아닌 정합 조건은
        coefficients 를 명시하여 표현한다.
        """
        if any(balance.name == name for balance in self.balances):
            raise ValueError(f"균형식 '{name}' 이 이미 등록되어 있습니다.")

        if coefficients is None:
            coefficients = {cell: 1.0 for cell in self.cell_names}
        unknown = [cell for cell in coefficients if cell not in self.observed]
        if unknown:
            raise ValueError(f"균형식 '{name}' 이 등록되지 않은 셀을 참조합니다: {unknown}")

        self.balances.append(
            LinearBalance(
                name=name,
                coefficients={cell: float(value) for cell, value in coefficients.items()},
                total=IntervalTarget.from_value(total),
            )
        )
        return self

    # ------------------------------------------------------------------ 조회

    @property
    def missing_cells(self) -> List[str]:
        """값이 공표되지 않은 셀의 이름을 반환한다."""
        return [name for name in self.cell_names if self.observed[name] is None]

    @property
    def observed_cells(self) -> List[str]:
        """값이 공표된 셀의 이름을 반환한다. 구간으로 공표된 셀도 포함한다."""
        return [name for name in self.cell_names if self.observed[name] is not None]

    # ------------------------------------------------------------------ 추정

    def estimate(
        self,
        max_passes: int = DEFAULT_MAX_PASSES,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> Dict[str, IntervalTarget]:
        """모든 균형식을 만족하도록 셀별 구간을 좁혀 반환한다.

        더 이상 좁혀지지 않거나 max_passes 회에 도달하면 멈춘다. 좁히는 도중
        하한이 상한을 넘어서면 관측값과 총계가 모순된다는 뜻이므로 예외를
        발생시킨다.
        """
        lower, upper = self._initial_bounds()

        for _ in range(max_passes):
            changed = False
            for balance in self.balances:
                changed |= self._propagate(balance, lower, upper, tolerance)
            self._check_feasible(lower, upper, tolerance)
            if not changed:
                break

        return {
            name: IntervalTarget(float(lower[index]), float(upper[index]))
            for index, name in enumerate(self.cell_names)
        }

    def _initial_bounds(self) -> "tuple[np.ndarray, np.ndarray]":
        """관측 상태로부터 셀별 초기 구간 배열을 만든다."""
        lower = np.empty(len(self.cell_names), dtype="float64")
        upper = np.empty(len(self.cell_names), dtype="float64")
        for index, name in enumerate(self.cell_names):
            observed = self.observed[name]
            interval = self.default_bounds if observed is None else observed
            lower[index] = interval.lower
            upper[index] = interval.upper
        return lower, upper

    def _propagate(
        self,
        balance: LinearBalance,
        lower: np.ndarray,
        upper: np.ndarray,
        tolerance: float,
    ) -> bool:
        """균형식 하나로 참여 셀의 구간을 좁히고 변화가 있었는지 반환한다."""
        position = {name: index for index, name in enumerate(self.cell_names)}
        indices = np.array([position[cell] for cell in balance.cells], dtype=np.int64)
        weights = np.array(
            [balance.coefficients[cell] for cell in balance.cells], dtype="float64"
        )

        weighted_lower = weights * lower[indices]
        weighted_upper = weights * upper[indices]

        rest_lower = float(np.sum(weighted_lower)) - weighted_lower
        rest_upper = self._rest_upper(weighted_upper)

        candidate_upper = (balance.total.upper - rest_lower) / weights
        candidate_lower = np.where(
            np.isfinite(rest_upper), (balance.total.lower - rest_upper) / weights, -np.inf
        )

        new_lower = np.maximum(lower[indices], candidate_lower)
        new_upper = np.minimum(upper[indices], candidate_upper)

        changed = bool(
            np.any(new_lower - lower[indices] > tolerance)
            or np.any(upper[indices] - new_upper > tolerance)
        )
        lower[indices] = new_lower
        upper[indices] = new_upper
        return changed

    @staticmethod
    def _rest_upper(weighted_upper: np.ndarray) -> np.ndarray:
        """각 셀을 제외한 나머지 셀 상한의 합을 구한다.

        나머지 중에 상한이 무한한 셀이 하나라도 있으면 그 합은 무한대이며, 해당
        셀의 하한은 이 균형식으로 좁혀지지 않는다. 무한대끼리의 뺄셈을 피하기
        위해 유한한 성분의 합과 무한한 성분의 개수를 나누어 계산한다.
        """
        finite = np.isfinite(weighted_upper)
        finite_total = float(np.sum(weighted_upper[finite]))
        n_infinite = int(np.count_nonzero(~finite))

        remaining_infinite = n_infinite - (~finite).astype(np.int64)
        excluded = np.where(finite, weighted_upper, 0.0)
        return np.where(remaining_infinite > 0, np.inf, finite_total - excluded)

    def _check_feasible(
        self, lower: np.ndarray, upper: np.ndarray, tolerance: float
    ) -> None:
        """하한이 상한을 넘어선 셀이 있으면 실현 불가로 판정한다."""
        offending = np.flatnonzero(lower - upper > tolerance)
        if offending.size:
            names = [self.cell_names[index] for index in offending]
            raise ValueError(
                "관측값과 균형식이 서로 모순되어 구간을 좁힐 수 없습니다. "
                f"하한이 상한을 넘어선 셀: {names}"
            )

    # ------------------------------------------------------------------ 산출

    def allocate(
        self,
        balance_name: str,
        shares: Mapping[str, float] = None,
        **estimate_kwargs,
    ) -> Dict[str, float]:
        """균형식의 잔차를 결측 셀에 배분한 점 추정을 반환한다.

        shares 는 결측 셀에 대한 상대 비율이며, 생략하면 균등 배분한다. 표본에서
        관측된 비율을 넘기면 표본 분포를 따르는 배분이 된다. 배분 결과는 추정
        구간 안으로 잘라내므로, 배분 뒤의 합이 잔차와 정확히 일치하지 않을 수
        있다. 점 추정은 보고와 초기값 설정에 쓰고, 반복 갱신의 목표로는 구간을
        사용한다.
        """
        balance = self._balance(balance_name)
        intervals = self.estimate(**estimate_kwargs)

        targets = [cell for cell in balance.cells if self.observed[cell] is None]
        if not targets:
            return {}

        known = sum(
            balance.coefficients[cell] * self._point_of(self.observed[cell])
            for cell in balance.cells
            if self.observed[cell] is not None
        )
        residual = self._point_of(balance.total) - known

        if shares is None:
            weights = {cell: 1.0 for cell in targets}
        else:
            weights = {cell: float(shares.get(cell, 0.0)) for cell in targets}
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError(f"균형식 '{balance_name}' 의 배분 비율 합이 0 이하입니다.")

        allocated: Dict[str, float] = {}
        for cell in targets:
            raw = residual * weights[cell] / (total_weight * balance.coefficients[cell])
            interval = intervals[cell]
            allocated[cell] = float(min(max(raw, interval.lower), interval.upper))
        return allocated

    def to_constraint(self, name: str, level: str, **estimate_kwargs) -> BoundChecker:
        """셀별 추정 구간을 그대로 제약 객체로 만든다.

        제약 열의 순서는 cells 의 선언 순서이며, 속성 행렬의 해당 블록도 같은
        순서를 따라야 한다.
        """
        intervals = self.estimate(**estimate_kwargs)
        return BoundChecker(
            name, level, {cell: intervals[cell] for cell in self.cell_names}
        )

    def balance_constraint(self, balance_name: str, name: str, level: str) -> BoundChecker:
        """균형식의 총계 자체를 하나의 제약 열로 만든다.

        셀별 구간만으로는 셀들의 합이 총계와 같아야 한다는 결합 조건이 표현되지
        않는다. 이 제약을 `balance_hierarchy` 가 만든 대범주 블록과 함께 등록하면
        결합 조건까지 부과된다.
        """
        balance = self._balance(balance_name)
        return BoundChecker(name, level, {balance.name: balance.total})

    def balance_hierarchy(
        self, balance_name: str, hierarchy_name: str = None
    ) -> CategoryHierarchy:
        """균형식이 포함하는 셀들을 하나의 대범주로 묶은 범주 위계를 반환한다.

        `CoarseBlock` 에 넘기면 셀들의 합을 세는 제약 열이 만들어진다. 매핑 행렬은
        이진 행렬이므로 계수가 모두 1인 균형식에만 적용할 수 있다.
        """
        balance = self._balance(balance_name)
        if not balance.is_unit:
            raise ValueError(
                f"균형식 '{balance_name}' 의 계수가 1이 아니므로 이진 매핑 행렬로 "
                "표현할 수 없습니다. 계수를 반영한 속성 열을 ColumnBlock 으로 "
                "구축하십시오."
            )
        hierarchy = CategoryHierarchy(hierarchy_name or balance.name)
        hierarchy.add_mapping(balance.name, balance.cells)
        return hierarchy

    def to_frame(self, **estimate_kwargs) -> pd.DataFrame:
        """셀별 관측 상태와 추정 구간을 표로 정리한다.

        `width` 는 상한과 하한의 차이로, 그 셀에 남은 불확실성의 크기이다.
        """
        intervals = self.estimate(**estimate_kwargs)
        records = []
        for name in self.cell_names:
            interval = intervals[name]
            records.append(
                {
                    "cell": name,
                    "status": "observed" if self.observed[name] is not None else "missing",
                    "lower": interval.lower,
                    "upper": interval.upper,
                    "width": interval.upper - interval.lower,
                }
            )
        return pd.DataFrame(records)

    # ------------------------------------------------------------------ 내부

    def _balance(self, balance_name: str) -> LinearBalance:
        """이름으로 등록된 균형식을 찾는다."""
        for balance in self.balances:
            if balance.name == balance_name:
                return balance
        raise KeyError(f"등록되지 않은 균형식입니다: {balance_name!r}")

    @staticmethod
    def _point_of(interval: IntervalTarget) -> float:
        """구간의 대표값으로 중점을 취한다. 상한이 무한하면 하한을 쓴다."""
        if not np.isfinite(interval.upper):
            return float(interval.lower)
        return float((interval.lower + interval.upper) / 2.0)


def marginal_from_series(
    values: pd.Series, missing_as: TargetLike = None
) -> Dict[str, Optional[TargetLike]]:
    """결측을 담은 주변표 계열을 `MissingMarginEstimator` 의 입력 형태로 변환한다.

    NaN 인 항목을 결측 셀로 본다. missing_as 를 주면 결측 셀의 초기 구간을 그
    값으로 두고, 생략하면 None 으로 두어 추정기의 default_bounds 를 따르게 한다.
    """
    result: Dict[str, Optional[TargetLike]] = {}
    for name, value in values.items():
        result[str(name)] = missing_as if pd.isna(value) else float(value)
    return result


def balance_from_classes(
    class_labels: Sequence[str], class_sizes: Sequence[float]
) -> Dict[str, float]:
    """계급별 개체 수 제약과 총 개체 수를 잇는 균형식의 계수를 만든다.

    n인가구 수와 총 가구원 수 사이의 sum_n n * H_n = P, 가구 수 계급별 거처 수와
    총 가구 수 사이의 sum_m m * D_m = H 가 이 형태이다. 상한이 없는 계급('4+' 등)
    은 대표값을 class_sizes 에 직접 지정해야 하며, 그 대표값의 오차는 균형식의
    총계를 구간으로 주어 흡수한다.
    """
    labels = list(class_labels)
    sizes = [float(value) for value in class_sizes]
    if len(labels) != len(sizes):
        raise ValueError(f"계급 이름 수({len(labels)})와 대표값 수({len(sizes)})가 다릅니다.")
    return dict(zip(labels, sizes))
