import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from typing import List, Tuple


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


def compute_akasofu_epsilon(
    df: pd.DataFrame,
    *,
    v_col: str = "V",          # km/s
    b_col: str = "B",          # nT
    by_col: str = "By(GSM)",   # nT
    bz_col: str = "Bz(GSM)",   # nT
) -> np.ndarray:
    """
    Compute Akasofu epsilon coupling function.

    Implementation follows your current logic:
      - V in km/s -> m/s
      - B in nT -> T
      - theta = arctan2(|By|, Bz)
      - epsilon = V * B^2 * sin^4(theta/2) * l0^2, l0 = 7 * R_E

    Returns: numpy array (same length as df), NaN where inputs missing.
    """
    # constants
    R_E = 6_371_000.0  # m
    l0 = 7.0 * R_E

    required = [v_col, b_col, by_col, bz_col]
    for c in required:
        if c not in df.columns:
            # keep behavior robust: return zeros (or NaN) if missing
            return np.zeros(len(df), dtype=float)

    V_m_s = df[v_col].astype(float).to_numpy() * 1000.0
    B_T = df[b_col].astype(float).to_numpy() * 1e-9
    By = df[by_col].astype(float).to_numpy()
    Bz = df[bz_col].astype(float).to_numpy()

    theta = np.arctan2(np.abs(By), Bz)
    sin4 = np.sin(theta / 2.0) ** 4
    eps = V_m_s * (B_T ** 2) * sin4 * (l0 ** 2)
    return eps


def compute_symh_derivatives(
    df: pd.DataFrame,
    *,
    symh_col: str = "SYM/H",
    dt_seconds: float = 60.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute first and second derivatives of SYM-H using finite differences.

    - diff1: forward difference with shift(-1) and current (consistent with your plotting naming)
    - diff2: second derivative central difference:
        (SYM(t+dt) - 2*SYM(t) + SYM(t-dt)) / dt^2

    Returns: (diff1, diff2) as pandas Series (aligned with df index).
    """
    if symh_col not in df.columns:
        z = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
        return z, z

    sym = df[symh_col].astype(float)

    # diff1: you can change definition later; this is a common choice:
    diff1 = (sym - sym.shift(1)) / dt_seconds

    # diff2: central difference (your current code)
    diff2 = (sym.shift(-1) - 2.0 * sym + sym.shift(1)) / (dt_seconds ** 2)

    return diff1, diff2


def build_feature_table_physics(
    df: pd.DataFrame,
    base_features: List[str],
    datetime_col: str,
    targets: List[str],
    *,
    # time features
    add_time: bool = True,

    # physics features
    add_akasofu: bool = True,
    akasofu_name: str = "Akasofu",
    add_symh_diff1: bool = True,
    symh_diff1_name: str = "SYM/H_diff1",
    add_symh_diff2: bool = True,
    symh_diff2_name: str = "SYM/H_diff2",
    symh_col: str = "SYM/H",
    dt_seconds: float = 60.0,

    # lags / rolling
    target_lags: List[int] = [1, 2, 3, 6, 12],
    important_features: List[str] = ["Bz(GSM)", "V", "AE", "SYM/H", "Pdyn"],
    important_lags: List[int] = [1, 2, 3],
    rolling_windows: List[int] = [3, 6, 12],

    # interactions
    add_interactions: bool = True,
    add_epsilon_bz_interaction: bool = True,
    epsilon_bz_name: str = "epsilon_Bz_interaction",

    # rolling for physics
    physics_rolling_windows: List[int] = [3, 6, 12],

    dropna: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Repository version of your "create_advanced_features + prepare_features" with physics drivers.

    Adds:
      - time features
      - akasofu epsilon (akasofu_name)
      - SYM/H derivatives: diff1 and diff2 (names configurable)
      - lag features for targets and important solar-wind indices
      - rolling mean/std for targets + physics features
      - interactions: Bz*V, Pdyn*abs(Bz), epsilon*abs(Bz)

    Returns: (df_processed, all_features)
    """
    dfp = df.copy()
    engineered: List[str] = []

    # ---- time features ----
    if add_time:
        dfp["hour"] = dfp[datetime_col].dt.hour
        dfp["dayofweek"] = dfp[datetime_col].dt.dayofweek
        dfp["hour_sin"] = np.sin(2 * np.pi * dfp["hour"] / 24.0)
        dfp["hour_cos"] = np.cos(2 * np.pi * dfp["hour"] / 24.0)
        engineered += ["hour", "dayofweek", "hour_sin", "hour_cos"]

    # ---- physics features ----
    if add_akasofu:
        dfp[akasofu_name] = compute_akasofu_epsilon(dfp)
        engineered.append(akasofu_name)

    diff1, diff2 = compute_symh_derivatives(dfp, symh_col=symh_col, dt_seconds=dt_seconds)

    if add_symh_diff1:
        dfp[symh_diff1_name] = diff1
        engineered.append(symh_diff1_name)

    if add_symh_diff2:
        dfp[symh_diff2_name] = diff2
        engineered.append(symh_diff2_name)

    # ---- lags: targets ----
    for t in targets:
        if t not in dfp.columns:
            continue
        for lag in target_lags:
            name = f"{t}_lag_{lag}"
            dfp[name] = dfp[t].shift(lag)
            engineered.append(name)

    # ---- lags: important features (+ optionally include new physics columns) ----
    imp = list(important_features)
    # include physics names if created
    for extra in [akasofu_name, symh_diff2_name]:
        if extra in dfp.columns and extra not in imp:
            imp.append(extra)

    for c in imp:
        if c not in dfp.columns:
            continue
        for lag in important_lags:
            name = f"{c}_lag_{lag}"
            dfp[name] = dfp[c].shift(lag)
            engineered.append(name)

    # ---- rolling: targets ----
    for t in targets:
        if t not in dfp.columns:
            continue
        for w in rolling_windows:
            m = f"{t}_rolling_mean_{w}"
            s = f"{t}_rolling_std_{w}"
            dfp[m] = dfp[t].rolling(window=w, min_periods=1).mean()
            dfp[s] = dfp[t].rolling(window=w, min_periods=1).std()
            engineered += [m, s]

    # ---- rolling: physics ----
    for c in [akasofu_name, symh_diff2_name]:
        if c not in dfp.columns:
            continue
        for w in physics_rolling_windows:
            m = f"{c}_rolling_mean_{w}"
            s = f"{c}_rolling_std_{w}"
            dfp[m] = dfp[c].rolling(window=w, min_periods=1).mean()
            dfp[s] = dfp[c].rolling(window=w, min_periods=1).std()
            engineered += [m, s]

    # ---- interactions ----
    if add_interactions:
        if all(col in dfp.columns for col in ["Bz(GSM)", "V"]):
            dfp["Bz_V_interaction"] = dfp["Bz(GSM)"] * dfp["V"]
            engineered.append("Bz_V_interaction")

        if all(col in dfp.columns for col in ["Pdyn", "Bz(GSM)"]):
            dfp["Pdyn_Bz_interaction"] = dfp["Pdyn"] * np.abs(dfp["Bz(GSM)"])
            engineered.append("Pdyn_Bz_interaction")

    if add_epsilon_bz_interaction and (akasofu_name in dfp.columns) and ("Bz(GSM)" in dfp.columns):
        dfp[epsilon_bz_name] = dfp[akasofu_name] * np.abs(dfp["Bz(GSM)"])
        engineered.append(epsilon_bz_name)

    # ---- only keep real columns, avoid duplicates ----
    engineered = [c for c in engineered if c in dfp.columns]
    all_features = list(base_features)
    for c in engineered:
        if c not in all_features:
            all_features.append(c)

    if dropna:
        dfp = dfp.dropna().reset_index(drop=True)

    return dfp, all_features
