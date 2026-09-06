"""가중치의 발산과 소멸을 억제하는 경계 제어.

절단의 대상은 가중치의 절대 수준이 아니라 한 회차의 변화 배율이다. 전자를
수준 제한, 후자를 배율 제한이라 하며 라이브러리는 배율 제한과 가중치 하한만
제공한다. 같은 방향의 조정이 여러 회차에 걸쳐 누적되면 절대 수준은 여전히
크게 변할 수 있다.
"""

from dataclasses import dataclass

import numpy as np

DENOMINATOR_FLOOR = 1e-10


@dataclass
class BoundStatistics:
    """한 회차에서 경계 제어가 개입한 정도."""

    n_ratio_clipped: int = 0
    n_floor_clipped: int = 0


class WeightBoundController:
    """회차별 변화 배율을 제한하고 가중치 하한을 적용한다.

    절단 대상은 가중치의 절대 수준이 아니라 한 회차의 변화 배율이다. 여러 회차에
    걸쳐 같은 방향의 조정이 누적되면 절대 수준은 여전히 크게 변할 수 있다.
    """

    def __init__(
        self,
        weight_floor: float = 1e-5,
        min_ratio: float = 0.1,
        max_ratio: float = 10.0,
    ):
        if min_ratio <= 0 or max_ratio <= 0:
            raise ValueError("변화 배율 한계는 양수여야 합니다.")
        if min_ratio > max_ratio:
            raise ValueError("min_ratio 가 max_ratio 보다 큽니다.")
        if weight_floor < 0:
            raise ValueError("weight_floor 는 0 이상이어야 합니다.")

        self.weight_floor = weight_floor
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.last_statistics = BoundStatistics()

    def apply_bounds(self, old_weights: np.ndarray, new_weights: np.ndarray) -> np.ndarray:
        """변화 배율을 절단하고 가중치 하한을 적용한 가중치를 반환한다.

        개입 정도는 `last_statistics` 에 기록하여 진단 기록에 넘긴다.
        """
        old_weights = np.asarray(old_weights, dtype="float64")
        new_weights = np.asarray(new_weights, dtype="float64")

        ratios = new_weights / np.maximum(old_weights, DENOMINATOR_FLOOR)
        clipped = np.clip(ratios, self.min_ratio, self.max_ratio)
        n_ratio_clipped = int(np.count_nonzero(clipped != ratios))

        bounded = old_weights * clipped
        below_floor = bounded < self.weight_floor
        n_floor_clipped = int(np.count_nonzero(below_floor))
        bounded = np.maximum(bounded, self.weight_floor)

        self.last_statistics = BoundStatistics(
            n_ratio_clipped=n_ratio_clipped, n_floor_clipped=n_floor_clipped
        )
        return bounded
