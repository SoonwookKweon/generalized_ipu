import pandas as pd
from .tree import HierarchyTree

class BaseUnitSelector:
    def __init__(self, tree: HierarchyTree):
        self.tree = tree
        self.base_node = tree.nodes[tree.base_level]

class AggregationMapper:
    @staticmethod
    def aggregate_to_base(tree: HierarchyTree, target_level: str, attribute_cols: list) -> pd.DataFrame:
        base_node = tree.nodes[tree.base_level]
        target_node = tree.nodes[target_level]
        
        if target_level == tree.base_level:
            return base_node.data[[base_node.primary_key] + attribute_cols]
            
        # 하위 계층에서 상위(Base) 계층으로 속성 수량(Count) 집계
        grouped = target_node.data.groupby(target_node.parent_key)[attribute_cols].sum().reset_index()
        merged = pd.merge(
            base_node.data[[base_node.primary_key]],
            grouped,
            left_on=base_node.primary_key,
            right_on=target_node.parent_key,
            how='left'
        ).fillna(0)
        
        return merged