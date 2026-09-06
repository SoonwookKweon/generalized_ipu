"""최종 가중치의 결합과 결과 내보내기.

기본 계층에 부여된 가중치를 원본 자료에 되붙이고, 필요하면 하위 계층으로
전파하거나 정수로 변환한다.
"""

from typing import Optional

import numpy as np
import pandas as pd

from ..hierarchy.tree import HierarchyTree


class DatasetExporter:
    """최종 가중치를 원본 자료에 결합하여 내보낸다."""

    @staticmethod
    def attach_weights(
        frame: pd.DataFrame,
        weights: np.ndarray,
        weight_column: str = "ipu_weight",
        keys: pd.Series = None,
        key_column: str = None,
    ) -> pd.DataFrame:
        """가중치를 표에 붙인다.

        keys 와 key_column 을 주면 주키를 기준으로 결합하고, 생략하면 행 순서가
        일치한다고 보고 그대로 붙인다. 순서 전제가 어긋나면 오류 없이 잘못된
        가중치가 부여되므로 주키 결합을 권장한다.
        """
        weights = np.asarray(weights, dtype="float64")

        if keys is not None:
            if key_column is None:
                raise ValueError("keys 를 지정하면 key_column 도 지정해야 합니다.")
            lookup = pd.DataFrame({key_column: np.asarray(keys), weight_column: weights})
            merged = frame.merge(lookup, on=key_column, how="left")
            if merged[weight_column].isna().any():
                missing = int(merged[weight_column].isna().sum())
                raise ValueError(f"가중치를 찾지 못한 행이 {missing}건 있습니다.")
            return merged

        if len(frame) != weights.size:
            raise ValueError(
                f"표의 행 수({len(frame)})와 가중치의 길이({weights.size})가 다릅니다."
            )
        result = frame.copy()
        result[weight_column] = weights
        return result

    @staticmethod
    def propagate_to_level(
        tree: HierarchyTree,
        target_level: str,
        weights: np.ndarray,
        weight_column: str = "ipu_weight",
    ) -> pd.DataFrame:
        """기본 계층의 가중치를 하위 계층 자료로 전파한다."""
        base_node = tree.base_node
        if tree.relation_to_base(target_level) != "descendant":
            raise ValueError(f"'{target_level}' 은 기본 계층의 하위 계층이 아닙니다.")

        lookup = pd.DataFrame(
            {
                base_node.primary_key: base_node.data[base_node.primary_key].to_numpy(),
                weight_column: np.asarray(weights, dtype="float64"),
            }
        )
        node = tree.nodes[target_level]
        merged = node.data.merge(
            lookup, left_on=node.parent_key, right_on=base_node.primary_key, how="left"
        )
        if merged[weight_column].isna().any():
            raise ValueError("상위 기본 단위를 찾지 못한 하위 계층 행이 있습니다.")
        return merged

    @staticmethod
    def export(
        frame: pd.DataFrame,
        output_path: str,
        fmt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """표를 파일로 저장한다. fmt 를 생략하면 확장자로 판정한다."""
        fmt = (fmt or output_path.rsplit(".", 1)[-1]).lower()
        if fmt == "csv":
            frame.to_csv(output_path, index=False, **kwargs)
        elif fmt in {"parquet", "pq"}:
            frame.to_parquet(output_path, index=False, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 저장 형식입니다: {fmt}")
        return output_path

    @staticmethod
    def export_with_weights(
        frame: pd.DataFrame,
        weights: np.ndarray,
        output_path: str,
        weight_column: str = "ipu_weight",
        keys: pd.Series = None,
        key_column: str = None,
    ) -> str:
        """가중치를 붙인 표를 파일로 저장하고 저장 경로를 반환한다."""
        attached = DatasetExporter.attach_weights(
            frame, weights, weight_column=weight_column, keys=keys, key_column=key_column
        )
        return DatasetExporter.export(attached, output_path)

    @staticmethod
    def integerize(weights: np.ndarray, random_state: int = None) -> np.ndarray:
        """가중치를 확률적 반올림으로 정수화한다.

        소수부를 성공 확률로 삼는 베르누이 추출을 사용하므로 총합의 기댓값이
        보존된다.
        """
        weights = np.asarray(weights, dtype="float64")
        if np.any(weights < 0):
            raise ValueError("음수 가중치는 정수화할 수 없습니다.")
        generator = np.random.default_rng(random_state)
        floor = np.floor(weights)
        remainder = weights - floor
        return (floor + (generator.random(weights.size) < remainder)).astype(np.int64)
