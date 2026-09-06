import pandas as pd
import numpy as np

class LogicalRuleParser:
    def __init__(self, rules: list):
        self.rules = rules # 예: ["age < 15 and driving_license == 1"]

    def parse_and_apply(self, df: pd.DataFrame) -> np.ndarray:
        # 1: 구조적 영(불가능한 조합), 0: 유효 조합
        invalid_mask = np.zeros(len(df), dtype=bool)
        for rule in self.rules:
            try:
                condition = df.eval(rule)
                invalid_mask = invalid_mask | condition
            except Exception as e:
                raise ValueError(f"규칙 파싱 오류 '{rule}': {e}")
        return invalid_mask