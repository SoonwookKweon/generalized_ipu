from typing import List
from .base import AbstractConstraint

class ConstraintRegistry:
    def __init__(self):
        self.constraints: List[AbstractConstraint] = []

    def register(self, constraint: AbstractConstraint):
        self.constraints.append(constraint)

    def get_all( me ) -> List[AbstractConstraint]:
        return self.constraints