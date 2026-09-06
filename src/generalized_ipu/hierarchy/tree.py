from typing import Dict, Optional
import pandas as pd

class HierarchyNode:
    def __init__(self, name: str, data: pd.DataFrame, primary_key: str, parent_key: Optional[str] = None):
        self.name = name
        self.data = data.copy()
        self.primary_key = primary_key
        self.parent_key = parent_key

class HierarchyTree:
    def __init__(self, base_level: str):
        self.base_level = base_level
        self.nodes: Dict[str, HierarchyNode] = {}
        
    def add_level(self, level_name: str, data: pd.DataFrame, primary_key: str, parent_key: Optional[str] = None):
        node = HierarchyNode(level_name, data, primary_key, parent_key)
        self.nodes[level_name] = node
        
    def validate_integrity(self) -> bool:
        if self.base_level not in self.nodes:
            raise ValueError(f"Base level '{self.base_level}' 이 HierarchyTree에 존재하지 않습니다.")
            
        for name, node in self.nodes.items():
            if node.parent_key and node.parent_key not in node.data.columns:
                raise KeyError(f"'{name}' 노드에 parent_key '{node.parent_key}' 가 존재하지 않습니다.")
        return True