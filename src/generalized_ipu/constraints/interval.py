from dataclasses import dataclass
from typing import Tuple, Union
import numpy as np
from .base import AbstractConstraint

@dataclass
class IntervalTarget:
    # 단일 값(float) 또는 구간 (Min, Max)
    bounds: Union[float, Tuple[float, float]]

class BoundChecker(AbstractConstraint):
    def __init__(self, name: str, level: str, targets: dict):
        super().__init__(name, level)
        self.targets = targets # {category_name: IntervalTarget or (L, U)}

    def compute_ratio(self, current_sum: np.ndarray) -> np.ndarray:
        # Bounded Piecewise 스케일링 함수
        ratios = np.ones_like(current_sum, dtype=np.float64)
        for i, (cat, target) in enumerate(self.targets.items()):
            s = current_sum[i]
            if s <= 0:
                continue
            if isinstance(target, tuple):
                low, high = target
                if s < low:
                    ratios[i] = low / s
                elif s > high:
                    ratios[i] = high / s
                else:
                    ratios[i] = 1.0
            else:
                ratios[i] = target / s
        return ratios