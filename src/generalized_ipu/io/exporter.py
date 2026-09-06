import pandas as pd

class DatasetExporter:
    @staticmethod
    def export_with_weights(df: pd.DataFrame, weights: pd.Series, output_path: str):
        result_df = df.copy()
        result_df['ipu_weight'] = weights
        result_df.to_csv(output_path, index=False)