"""계층별 자료의 적재와 자료형 정규화.

자료의 의미 검증은 `hierarchy` 패키지가 담당한다. 이 모듈은 계층 사이의 결합이
조용히 실패하지 않도록 키 열의 자료형을 통일하는 데까지만 관여한다.
"""

from typing import Mapping, Sequence

import pandas as pd

from ..hierarchy.tree import HierarchyTree


class DataLoader:
    """계층별 자료를 적재한다. 자료의 의미 검증은 hierarchy 패키지가 담당한다."""

    @staticmethod
    def load_csv(
        file_path: str,
        key_cols: Sequence[str] = (),
        attribute_cols: Sequence[str] = (),
        key_dtype: str = "string",
        **kwargs,
    ) -> pd.DataFrame:
        """CSV 를 적재하고 키 열의 자료형을 통일한다.

        계층마다 키 열이 정수와 문자열로 다르게 읽히면 결합이 조용히 실패하므로,
        키 열은 명시적으로 같은 자료형으로 변환한다. 속성 열의 결측은 0으로 채운다.
        """
        frame = pd.read_csv(file_path, **kwargs)
        return DataLoader.normalize(frame, key_cols, attribute_cols, key_dtype)

    @staticmethod
    def load_parquet(
        file_path: str,
        key_cols: Sequence[str] = (),
        attribute_cols: Sequence[str] = (),
        key_dtype: str = "string",
        **kwargs,
    ) -> pd.DataFrame:
        """Parquet 을 적재하고 키 열의 자료형을 통일한다."""
        frame = pd.read_parquet(file_path, **kwargs)
        return DataLoader.normalize(frame, key_cols, attribute_cols, key_dtype)

    @staticmethod
    def normalize(
        frame: pd.DataFrame,
        key_cols: Sequence[str] = (),
        attribute_cols: Sequence[str] = (),
        key_dtype: str = "string",
    ) -> pd.DataFrame:
        """키 열의 자료형을 통일하고 속성 열을 수치형으로 변환한다.

        키 열의 결측은 참조 무결성을 깨뜨리므로 예외로 처리하고, 속성 열의
        결측은 0으로 채운다.
        """
        frame = frame.copy()
        for column in key_cols:
            if column not in frame.columns:
                raise KeyError(f"키 열 '{column}' 이 자료에 없습니다.")
            if frame[column].isna().any():
                raise ValueError(f"키 열 '{column}' 에 결측값이 있습니다.")
            frame[column] = frame[column].astype(key_dtype)
        for column in attribute_cols:
            if column not in frame.columns:
                raise KeyError(f"속성 열 '{column}' 이 자료에 없습니다.")
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        return frame

    @staticmethod
    def build_tree(base_level: str, levels: Mapping[str, Mapping]) -> HierarchyTree:
        """계층 설정 사전으로부터 HierarchyTree 를 구성하고 무결성을 검증한다.

        levels 의 각 항목은 data, primary_key, parent_key, parent_level 을 가진다.
        """
        tree = HierarchyTree(base_level)
        for level_name, config in levels.items():
            tree.add_level(
                level_name,
                config["data"],
                config["primary_key"],
                config.get("parent_key"),
                config.get("parent_level"),
            )
        tree.validate_integrity()
        return tree
