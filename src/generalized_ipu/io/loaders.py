import pandas as pd

class DataLoader:
    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)