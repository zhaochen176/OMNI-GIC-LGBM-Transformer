from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from src.io import read_csv
from src.preprocess import load_and_preprocess_tabular
from src.features import build_feature_table
from src.models.hybrid.lgbm_stage import train_lgbm
from src.metrics import evaluate_regression, binned_metrics
from src.plotting import plot_series
from src.utils import ensure_dir


def run_experiment(cfg: dict, logger):
    # ---- paths ----
    data_path = cfg["paths"]["data_path"]
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    run_name = cfg["project"]["run_name"]
    eval_cfg = cfg.get("evaluation", {})


    models_dir = outputs_dir / "models"
    metrics_dir = outputs_dir / "metrics"
    figures_dir = outputs_dir / "figures"
    ensure_dir(models_dir)
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)

    # ---- data config ----
    date_col = cfg["data"]["date_col"]
    datetime_col = cfg["data"]["datetime_col"]
    target_col = cfg["data"]["target_col"]

    df = read_csv(data_path)

    df, base_features = load_and_preprocess_tabular(
        df=df,
        date_col=date_col,
        datetime_col=datetime_col,
        target_col=target_col,
        sentinel_values=cfg["data"].get("sentinel_values", []),
        max_missing_feature_ratio=float(cfg["data"].get("max_missing_feature_ratio", 0.5)),
        use_abs_target=bool(cfg["data"].get("use_abs_target", True)),
    )
    logger.info("Loaded data: %s (rows=%d, base_features=%d)", data_path, len(df), len(base_features))

    # ---- feature build (keeps your logic) ----
    df_feat, all_features = build_feature_table(
        df=df,
        base_features=base_features,
        datetime_col=datetime_col,
        target_col=target_col,
        time_features=bool(cfg["features"].get("time_features", True)),
        target_lags=[int(x) for x in cfg["features"].get("target_lags", [])],
        rolling_windows=[int(x) for x in cfg["features"].get("rolling_windows", [])],
        interactions=cfg["features"].get("interactions", []),
    )
    logger.info("Feature table ready: rows=%d, features=%d", len(df_feat), len(all_features))

    # ---- time-based split (same as your split_idx logic) ----
    test_size = float(cfg["data"].get("test_size", 0.2))
    split_idx = int(len(df_feat) * (1 - test_size))

    X = df_feat[all_features].values
    y = df_feat[target_col].values
    dt = pd.to_datetime(df_feat[datetime_col]).values

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dt_test = dt[split_idx:]

    # ---- scaling (same as your StandardScaler) ----
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---- train LightGBM (same params & early stopping) ----
    lgbm_cfg = cfg["model"]
    model = train_lgbm(
        X_train=X_train_s,
        y_train=y_train,
        X_valid=X_test_s,   # keep your “valid = test” behavior for now
        y_valid=y_test,
        params=lgbm_cfg["params"],
        num_boost_round=int(lgbm_cfg.get("num_boost_round", 1000)),
        early_stopping_rounds=int(lgbm_cfg.get("early_stopping_rounds", 50)),
    )

    # ---- predict (your code predicts ALL using scaler.transform(X)) ----
    X_all_s = scaler.transform(X)
    y_pred_all = model.predict(X_all_s)
    y_pred_test = y_pred_all[split_idx:]

    # ---- evaluation ----
    overall_test = evaluate_regression(y_test, y_pred_test)
    logger.info("Test metrics: MAE=%.4f RMSE=%.4f R2=%.4f sMAPE=%.2f%%",
                overall_test["MAE"], overall_test["RMSE"], overall_test["R2"], overall_test["sMAPE(%)"])

    eval_cfg = cfg.get("evaluation", {})
    edges = [float(x) for x in eval_cfg.get("bins", [0, 2, 3, 5])]

    df_bin = binned_metrics(y, y_pred_all, edges=edges)

    # ---- save artifacts ----
    # model + scaler
    joblib.dump({"model": model, "scaler": scaler, "features": all_features},
                models_dir / f"{run_name}_lgbm.joblib")

    # metrics
    df_bin.to_csv(metrics_dir / f"{run_name}_binned_metrics.csv", index=False)

    # predictions
    if bool(eval_cfg.get("make_plot", True)):


        out_pred = df_feat[[datetime_col, target_col]].copy()
        out_pred["y_pred"] = y_pred_all
        out_pred.to_csv(metrics_dir / f"{run_name}_predictions_all.csv", index=False)

    # figure
    if bool(eval_cfg.get("make_plot", True)):

        plot_series(
            datetime=dt_test,
            y_true=y_test,
            y_pred=y_pred_test,
            out_path=figures_dir / f"{run_name}_test_plot.png",
            title=f"{target_col} Prediction (LightGBM)",
        )

    logger.info("Saved: model/metrics/figures under %s", str(outputs_dir))
