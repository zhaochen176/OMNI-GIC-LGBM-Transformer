import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def smape(y_true, y_pred, eps: float = 0.0) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE), returned in percent.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    if eps > 0:
        denom = np.maximum(denom, eps)
    out = np.where(denom == 0, 0.0, 2.0 * np.abs(y_pred - y_true) / denom)
    return float(np.mean(out) * 100.0)


def evaluate_regression(y_true, y_pred) -> dict:
    """
    Compute MAE, RMSE, R2, and sMAPE(%).
    Note: R2 is undefined for <2 samples; returns NaN in that case.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "sMAPE(%)": float(smape(y_true, y_pred)),
    }

    if len(y_true) < 2:
        out["R2"] = float("nan")
    else:
        out["R2"] = float(r2_score(y_true, y_pred))

    return out


def binned_metrics(y_true, y_pred, edges: list[float]) -> pd.DataFrame:
    """
    Bin-wise metrics by target amplitude.

    edges: e.g. [0, 2, 3, 5] -> bins:
      [0,2), [2,3), [3,5), [5, inf), plus an ALL row.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    bins = list(edges) + [np.inf]
    labels = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if np.isfinite(hi):
            labels.append(f"{lo}-{hi}")
        else:
            labels.append(f">={lo}")

    rows = []
    for i, lab in enumerate(labels):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_true >= lo) & (y_true < hi) if np.isfinite(hi) else (y_true >= lo)

        if int(mask.sum()) == 0:
            rows.append({
                "bin": lab,
                "count": 0,
                "MAE": np.nan,
                "RMSE": np.nan,
                "R2": np.nan,
                "sMAPE(%)": np.nan
            })
            continue

        yt, yp = y_true[mask], y_pred[mask]
        m = evaluate_regression(yt, yp)
        rows.append({"bin": lab, "count": int(mask.sum()), **m})

    # ALL
    m_all = evaluate_regression(y_true, y_pred)
    rows.append({"bin": "ALL", "count": int(len(y_true)), **m_all})

    return pd.DataFrame(rows)
