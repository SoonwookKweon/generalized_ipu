import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from ..hierarchy.tree import HierarchyTree
from ..hierarchy.mapper import AggregationMapper

class UnifiedSparseMatrixBuilder:
    def __init__(self, tree: HierarchyTree):
        self.tree = tree

    def build_matrix(self, constraint_configs: list) -> csr_matrix:
        sparse_blocks = []
        for config in constraint_configs:
            level = config['level']
            cols = config['cols']
            
            aggregated_df = AggregationMapper.aggregate_to_base(self.tree, level, cols)
            feature_matrix = aggregated_df[cols].values
            sparse_blocks.append(csr_matrix(feature_matrix))
            
        unified_matrix = hstack(sparse_blocks, format='csr')
        return unified_matrix