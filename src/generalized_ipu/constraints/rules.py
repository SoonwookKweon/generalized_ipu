"""도메인 규칙으로부터 구조적 영 제약 열을 식별한다.

평가의 대상은 표본 자료의 행이 아니라 제약 열이 나타내는 범주 조합이다. 규칙은
현실에 존재할 수 없는 조합을 기술하며, 그 조합에 대응하는 제약 열은 속성 행렬
구축 단계에서 색인 자체를 배정받지 않는다.
"""

from typing import List, Sequence

import numpy as np
import pandas as pd


class LogicalRuleParser:
    """도메인 규칙으로부터 제약 열의 유효 마스크를 산출한다.

    규칙은 불가능한 조합을 기술하는 문자열이며, 평가의 대상은 표본 자료의 행이
    아니라 제약 열이 나타내는 범주 조합이다. 반환값은 유효한 쪽이 True 인
    validity_mask 이다.
    """

    def __init__(self, rules: Sequence[str]):
        self.rules: List[str] = list(rules)

    def build_validity_mask(self, category_frame: pd.DataFrame) -> np.ndarray:
        """제약 열별 범주 조합 표를 받아 길이 K 의 유효 마스크를 반환한다.

        category_frame 의 각 행이 제약 열 하나에 대응하며, 행 순서는 속성 행렬의
        열 순서와 같아야 한다.
        """
        impossible = np.zeros(len(category_frame), dtype=bool)
        for rule in self.rules:
            try:
                condition = category_frame.eval(rule)
            except Exception as error:
                raise ValueError(f"규칙 파싱 오류 '{rule}': {error}") from error

            condition = np.asarray(condition)
            if condition.dtype != bool:
                raise ValueError(f"규칙 '{rule}' 의 평가 결과가 논리값이 아닙니다.")
            if condition.shape != impossible.shape:
                raise ValueError(
                    f"규칙 '{rule}' 의 평가 결과 길이({condition.shape})가 "
                    f"제약 열 수({impossible.size})와 다릅니다."
                )
            impossible |= condition
        return ~impossible

    def structural_zero_columns(self, category_frame: pd.DataFrame) -> np.ndarray:
        """구조적 영으로 판정된 제약 열의 색인을 반환한다."""
        return np.flatnonzero(~self.build_validity_mask(category_frame))

    def describe(self, category_frame: pd.DataFrame) -> pd.DataFrame:
        """규칙별로 몇 개의 제약 열이 걸러지는지 정리한 표를 반환한다."""
        rows = []
        for rule in self.rules:
            condition = np.asarray(category_frame.eval(rule))
            rows.append({"rule": rule, "n_columns_excluded": int(condition.sum())})
        return pd.DataFrame(rows)
