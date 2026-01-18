import pandas as pd


def read_csv(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    return pd.read_csv(path, encoding=encoding)
