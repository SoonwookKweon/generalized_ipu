import numpy as np
import pandas as pd

class FitReportGenerator:
    @staticmethod
    def generate_report(current_sums: np.ndarray, target_sums: np.ndarray, feature_names: list) -> pd.DataFrame:
        diffs = current_sums - target_sums
        gaps = np.where(target_sums > 0, np.abs(diffs) / target_sums, 0.0)
        
        report_df = pd.DataFrame({
            'feature': feature_names,
            'target': target_sums,
            'estimated': current_sums,
            'abs_diff': np.abs(diffs),
            'relative_gap': gaps
        }).sort_values(by='relative_gap', ascending=False)
        
        return report_df