import numpy as np
import pandas as pd


def load_and_preprocess_tabular(
    df: pd.DataFrame,
    date_col: str,
    datetime_col: str,
    target_col: str,
    sentinel_values: list,
    max_missing_feature_ratio: float,
    use_abs_target: bool,
):
    df = df.copy()

    # datetime
    df[datetime_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # sentinel to NaN
    if sentinel_values:
        df = df.replace(sentinel_values, np.nan)

    exclude_cols = [date_col, datetime_col]
    potential_features = [
        c for c in df.columns
        if c not in exclude_cols + [target_col]
        and df[c].dtype in ["float64", "int64", "int32", "float32"]
    ]

    feature_cols = [c for c in potential_features if df[c].isnull().mean() < max_missing_feature_ratio]

    # fill NaN with median for features and target
    for c in feature_cols + [target_col]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    if use_abs_target:
        df[target_col] = np.abs(df[target_col].astype(float))

    return df, feature_cols
