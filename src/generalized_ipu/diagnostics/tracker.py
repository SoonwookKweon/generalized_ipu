from typing import List

class ConvergenceTracker:
    def __init__(self):
        self.history: List[float] = []

    def log(self, max_gap: float):
        self.history.append(max_gap)