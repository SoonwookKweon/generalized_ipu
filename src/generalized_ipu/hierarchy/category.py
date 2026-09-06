from typing import Dict, List
import numpy as np
from scipy.sparse import csr_matrix

class CategoryHierarchy:
    def __init__(self, name: str):
        self.name = name
        self.mapping: Dict[str, List[str]] = {}
        self.fine_categories: List[str] = []
        
    def add_mapping(self, coarse: str, fines: List[str]):
        self.mapping[coarse] = fines
        for f in fines:
            if f not in self.fine_categories:
                self.fine_categories.append(f)

class MappingMatrix:
    @staticmethod
    def build_binary_matrix(hierarchy: CategoryHierarchy) -> csr_matrix:
        coarse_list = list(hierarchy.mapping.keys())
        fine_list = hierarchy.fine_categories
        
        matrix = np.zeros((len(coarse_list), len(fine_list)), dtype=np.float64)
        for i, coarse in enumerate(coarse_list):
            for fine in hierarchy.mapping[coarse]:
                j = fine_list.index(fine)
                matrix[i, j] = 1.0
                
        return csr_matrix(matrix)