from abc import ABC, abstractmethod
import numpy as np

class AbstractConstraint(ABC):
    def __init__(self, name: str, level: str):
        self.name = name
        self.level = level
        
    @abstractmethod
    def compute_ratio(self, current_sum: np.ndarray) -> np.ndarray:
        pass