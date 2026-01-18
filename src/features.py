import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df[datetime_col].dt.hour
    df["dayofweek"] = df[datetime_col].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    return df


def add_target_lags(df: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_target_rolling(df: pd.DataFrame, target_col: str, windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"{target_col}_rolling_mean_{w}"] = df[target_col].rolling(w, min_periods=1).mean()
        df[f"{target_col}_rolling_std_{w}"] = df[target_col].rolling(w, min_periods=1).std()
    return df


def add_interactions(df: pd.DataFrame, interactions: list[dict]) -> pd.DataFrame:
    """
    supports:
      op=mul -> cols[0] * cols[1]
      op=pdyn_absbz -> Pdyn * abs(Bz)
    """
    df = df.copy()
    for item in interactions or []:
        name = item.get("name")
        cols = item.get("cols", [])
        op = item.get("op")
        if not name or len(cols) < 2:
            continue
        if op == "mul":
            if all(c in df.columns for c in cols[:2]):
                df[name] = df[cols[0]] * df[cols[1]]
        elif op == "pdyn_absbz":
            if all(c in df.columns for c in cols[:2]):
                df[name] = df[cols[0]] * np.abs(df[cols[1]])
    return df


def build_feature_table(
    df: pd.DataFrame,
    base_features: list[str],
    datetime_col: str,
    target_col: str,
    time_features: bool,
    target_lags: list[int],
    rolling_windows: list[int],
    interactions: list[dict],
):
    df_feat = df.copy()

    if time_features:
        df_feat = add_time_features(df_feat, datetime_col)

    if target_lags:
        df_feat = add_target_lags(df_feat, target_col, target_lags)

    if rolling_windows:
        df_feat = add_target_rolling(df_feat, target_col, rolling_windows)

    df_feat = add_interactions(df_feat, interactions)

    # collect additional features that actually exist
    extra = []
    for c in ["hour", "dayofweek", "hour_sin", "hour_cos"]:
        if c in df_feat.columns:
            extra.append(c)

    # interactions are named explicitly
    for item in interactions or []:
        name = item.get("name")
        if name and name in df_feat.columns:
            extra.append(name)

    # lag & rolling
    for lag in target_lags or []:
        c = f"{target_col}_lag_{lag}"
        if c in df_feat.columns:
            extra.append(c)

    for w in rolling_windows or []:
        c1 = f"{target_col}_rolling_mean_{w}"
        c2 = f"{target_col}_rolling_std_{w}"
        if c1 in df_feat.columns:
            extra.append(c1)
        if c2 in df_feat.columns:
            extra.append(c2)

    all_features = list(base_features) + extra

    # match your original behavior: drop rows containing NaN after feature creation
    df_feat = df_feat.dropna().reset_index(drop=True)

    return df_feat, all_features
