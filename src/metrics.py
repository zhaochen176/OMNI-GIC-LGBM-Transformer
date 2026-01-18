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


import numpy as np
import pandas as pd

def binned_metrics(y_true, y_pred, edges):
    """
    Bin-wise regression metrics with NaN/Inf-safe filtering.
    edges: e.g., [0, 10, 20, 30] -> bins [0-10), [10-20), [20-30), [>=30], plus ALL
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rows = []
    # build bins
    bins = list(edges) + [np.inf]
    labels = []
    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]
        if np.isinf(right):
            labels.append(f">={int(left)}")
        else:
            labels.append(f"{int(left)}-{int(right)}")
    labels.append("ALL")

    for i, lab in enumerate(labels):
        if lab == "ALL":
            mask = np.ones_like(y_true, dtype=bool)
        else:
            left = bins[i]
            right = bins[i + 1]
            if np.isinf(right):
                mask = (y_true >= left)
            else:
                mask = (y_true >= left) & (y_true < right)

        yt = y_true[mask]
        yp = y_pred[mask]

        # ---- critical: drop NaN/Inf ----
        finite = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[finite]
        yp = yp[finite]

        if yt.size < 2:
            rows.append({
                "bin": lab,
                "count": int(yt.size),
                "MAE": np.nan,
                "RMSE": np.nan,
                "R2": np.nan,
                "sMAPE(%)": np.nan,
            })
            continue

        m = evaluate_regression(yt, yp)
        rows.append({
            "bin": lab,
            "count": int(yt.size),
            **m
        })

    return pd.DataFrame(rows)
