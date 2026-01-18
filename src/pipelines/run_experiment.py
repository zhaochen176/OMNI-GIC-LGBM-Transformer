from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

from src.io import read_csv
from src.preprocess import load_and_preprocess_tabular
from src.features import build_feature_table_physics
from src.models.hybrid.lgbm_stage import train_lgbm
from src.metrics import evaluate_regression, binned_metrics
from src.plotting import plot_series
from src.utils import ensure_dir
from src.models.hybrid.hybrid_stage2 import train_lgbm_then_transformer

# baselines
from src.models.baselines.registry import get_baseline
import src.models.baselines.rf  # noqa: F401 (register)
import src.models.baselines.cnn  # noqa: F401 (register)
import src.models.baselines.lstm  # noqa: F401 (register)

from src.datasets import make_sliding_windows


def _model_suffix(name: str) -> str:
    return name.lower().strip()


def _save_outputs(
    *,
    models_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    run_name: str,
    model_name: str,
    model_obj,
    scaler,
    features: list[str],
    df_feat: pd.DataFrame,
    datetime_col: str,
    target_col: str,
    y_pred_all: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    dt_test: np.ndarray,
    edges: list[float],
    eval_cfg: dict,
    logger,
):
    suffix = _model_suffix(model_name)

    # ---- save model bundle ----
    joblib.dump(
        {"model": model_obj, "scaler": scaler, "features": features},
        models_dir / f"{run_name}_{suffix}.joblib",
    )

    # ---- metrics (overall test) ----
    m_test = evaluate_regression(y_test, y_pred_test)
    pd.DataFrame([{
        "model": model_name,
        **m_test
    }]).to_csv(metrics_dir / f"{run_name}_{suffix}_metrics_test.csv", index=False)

    # ---- binned metrics (ALL by default, consistent with your current behavior) ----
    df_bin = binned_metrics(df_feat[target_col].values, y_pred_all, edges=edges)
    df_bin.to_csv(metrics_dir / f"{run_name}_{suffix}_binned_metrics_all.csv", index=False)

    # ---- predictions ----
    if bool(eval_cfg.get("save_predictions", True)):
        out_pred = df_feat[[datetime_col, target_col]].copy()
        out_pred["y_pred"] = y_pred_all
        out_pred.to_csv(metrics_dir / f"{run_name}_{suffix}_predictions_all.csv", index=False)

    # ---- figure ----
    if bool(eval_cfg.get("make_plot", True)):
        plot_series(
            datetime=dt_test,
            y_true=y_test,
            y_pred=y_pred_test,
            out_path=figures_dir / f"{run_name}_{suffix}_test_plot.png",
            title=f"{target_col} Prediction ({model_name})",
        )

    logger.info(
        "%s Test metrics: MAE=%.4f RMSE=%.4f R2=%.4f sMAPE=%.2f%%",
        model_name, m_test["MAE"], m_test["RMSE"], m_test["R2"], m_test["sMAPE(%)"]
    )


def run_experiment(cfg: dict, logger):
    # ---- paths ----
    data_path = cfg["paths"]["data_path"]
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    run_name = cfg["project"]["run_name"]
    models_dir = outputs_dir / "models"
    metrics_dir = outputs_dir / "metrics"
    figures_dir = outputs_dir / "figures"
    ensure_dir(models_dir)
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)

    eval_cfg = cfg.get("evaluation", {})
    edges = [float(x) for x in eval_cfg.get("bins", [0, 2, 3, 5])]

    # ---- data config ----
    date_col = cfg["data"]["date_col"]
    datetime_col = cfg["data"]["datetime_col"]
    target_col = cfg["data"]["target_col"]

    # ---- load ----
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

    # ---- features (physics version, as you are currently using) ----
    df_feat, all_features = build_feature_table_physics(
        df=df,
        base_features=base_features,
        datetime_col=datetime_col,
        targets=[target_col],
    )
    logger.info("Feature table ready: rows=%d, features=%d", len(df_feat), len(all_features))

    # ---- split ----
    test_size = float(cfg["data"].get("test_size", 0.2))
    split_idx = int(len(df_feat) * (1 - test_size))

    X = df_feat[all_features].values
    y = df_feat[target_col].values
    dt = pd.to_datetime(df_feat[datetime_col]).values

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dt_test = dt[split_idx:]

    # ---- scaling (fit on train only) ----
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_all_s = scaler.transform(X)

    # ---- model selection ----
    model_name = cfg.get("model", {}).get("name", "lgbm").lower().strip()
    if model_name not in ["lgbm", "rf", "cnn", "lstm", "all", "hybrid_stage2"]:
        raise ValueError("model.name must be one of: lgbm, rf, cnn, lstm, all")

    # ---- baselines config ----
    baselines_cfg = cfg.get("baselines", {})
    seq_len = int(baselines_cfg.get("sequence", {}).get("seq_len", 30))  # default 30 as you requested

    # ---- helper: prepare sequence data for CNN/LSTM ----
    def make_seq_split(X_all_scaled, y_all, split_idx_local, seq_len_local):
        """
        Create sliding windows and align train/test by time.
        - train windows end at t <= split_idx-1
        - test windows end at t >= split_idx
        """
        Xw, yw = make_sliding_windows(X_all_scaled, y_all, seq_len_local)
        # window i ends at time t = i + seq_len - 1
        end_t = np.arange(len(yw)) + (seq_len_local - 1)

        train_mask = end_t < split_idx_local
        test_mask = end_t >= split_idx_local

        Xw_train, yw_train = Xw[train_mask], yw[train_mask]
        Xw_test, yw_test = Xw[test_mask], yw[test_mask]

        # for plotting: corresponding datetimes for test points
        dt_test_points = dt[end_t[test_mask]]
        return Xw_train, yw_train, Xw_test, yw_test, dt_test_points, end_t

    # =========================
    # 1) LightGBM
    # =========================
    if model_name in ["lgbm", "all"]:
        lgbm_cfg = cfg["model"]
        lgbm_model = train_lgbm(
            X_train=X_train_s,
            y_train=y_train,
            X_valid=X_test_s,   # keep “valid=test” behavior
            y_valid=y_test,
            params=lgbm_cfg["params"],
            num_boost_round=int(lgbm_cfg.get("num_boost_round", 1000)),
            early_stopping_rounds=int(lgbm_cfg.get("early_stopping_rounds", 50)),
        )

        y_pred_all = lgbm_model.predict(X_all_s)
        y_pred_test = y_pred_all[split_idx:]

        _save_outputs(
            models_dir=models_dir,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
            run_name=run_name,
            model_name="LightGBM",
            model_obj=lgbm_model,
            scaler=scaler,
            features=all_features,
            df_feat=df_feat,
            datetime_col=datetime_col,
            target_col=target_col,
            y_pred_all=y_pred_all,
            y_test=y_test,
            y_pred_test=y_pred_test,
            dt_test=dt_test,
            edges=edges,
            eval_cfg=eval_cfg,
            logger=logger,
        )


            # =========================
    # LightGBM + Transformer (global attention) residual learning
    # =========================
    if model_name in ["hybrid_stage2"]:
        tf_cfg = cfg.get("transformer", {})  # put transformer config under "transformer"
        lgbm_cfg = cfg["model"]

        bundle, y_pred_test_h, y_pred_all_h = train_lgbm_then_transformer(
            X_train=X_train_s,
            y_train=y_train,
            X_test=X_test_s,
            y_test=y_test,
            X_all=X_all_s,
            lgbm_cfg=lgbm_cfg,
            tf_cfg=tf_cfg,
        )

        # save keras separately (optional but recommended)
        tf_model = bundle.get("tf_model", None)
        if tf_model is not None:
            tf_model.save(models_dir / f"{run_name}_transformer.keras")

        # avoid joblib serializing keras model
        bundle["tf_model"] = None

        _save_outputs(
            models_dir=models_dir,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
            run_name=run_name,
            model_name="LGBM+Transformer",
            model_obj=bundle,
            scaler=scaler,
            features=all_features,
            df_feat=df_feat,
            datetime_col=datetime_col,
            target_col=target_col,
            y_pred_all=y_pred_all_h,
            y_test=y_test,
            y_pred_test=y_pred_test_h,
            dt_test=dt_test,
            edges=edges,
            eval_cfg=eval_cfg,
            logger=logger,
        )


    # =========================
    # 2) Random Forest
    # =========================
    if model_name in ["rf", "all"]:
        rf_params = baselines_cfg.get("rf", {})
        train_predict_rf = get_baseline("rf")
        rf_model, y_pred_test_rf = train_predict_rf(X_train_s, y_train, X_test_s, rf_params)

        # build all preds
        y_pred_all_rf = rf_model.predict(X_all_s)

        _save_outputs(
            models_dir=models_dir,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
            run_name=run_name,
            model_name="RF",
            model_obj=rf_model,
            scaler=scaler,
            features=all_features,
            df_feat=df_feat,
            datetime_col=datetime_col,
            target_col=target_col,
            y_pred_all=y_pred_all_rf,
            y_test=y_test,
            y_pred_test=y_pred_test_rf,
            dt_test=dt_test,
            edges=edges,
            eval_cfg=eval_cfg,
            logger=logger,
        )

    # =========================
    # 3) CNN
    # =========================
    if model_name in ["cnn", "all"]:
        Xw_train, yw_train, Xw_test, yw_test, dt_test_pts, end_t = make_seq_split(
            X_all_s, y, split_idx, seq_len
        )

        # keep valid=test behavior for now (consistent with your pipeline philosophy)
        cnn_cfg = baselines_cfg.get("cnn", {})
        train_cfg = baselines_cfg.get("torch_train", {})

        train_predict_cnn = get_baseline("cnn")
        cnn_model, y_pred_test_cnn = train_predict_cnn(
            Xw_train, yw_train, Xw_test, yw_test, Xw_test, cnn_cfg, train_cfg
        )

        # predictions_all: align to original timeline by filling NaN for first seq_len-1
        y_pred_all_cnn = np.full(len(y), np.nan, dtype=float)
        # predictions correspond to window ends
        Xw_all, yw_all = make_sliding_windows(X_all_s, y, seq_len)
        from src.train.torch_trainer import predict as torch_predict
        y_pred_w_all = torch_predict(cnn_model, Xw_all)
        y_pred_all_cnn[seq_len - 1:] = y_pred_w_all

        _save_outputs(
            models_dir=models_dir,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
            run_name=run_name,
            model_name="CNN",
            model_obj=cnn_model,
            scaler=scaler,
            features=all_features,
            df_feat=df_feat,
            datetime_col=datetime_col,
            target_col=target_col,
            y_pred_all=y_pred_all_cnn,
            y_test=yw_test,                # test targets aligned to windows
            y_pred_test=y_pred_test_cnn,
            dt_test=dt_test_pts,
            edges=edges,
            eval_cfg=eval_cfg,
            logger=logger,
        )

    # =========================
    # 4) LSTM
    # =========================
    if model_name in ["lstm", "all"]:
        Xw_train, yw_train, Xw_test, yw_test, dt_test_pts, end_t = make_seq_split(
            X_all_s, y, split_idx, seq_len
        )

        lstm_cfg = baselines_cfg.get("lstm", {})
        train_cfg = baselines_cfg.get("torch_train", {})

        train_predict_lstm = get_baseline("lstm")
        lstm_model, y_pred_test_lstm = train_predict_lstm(
            Xw_train, yw_train, Xw_test, yw_test, Xw_test, lstm_cfg, train_cfg
        )

        y_pred_all_lstm = np.full(len(y), np.nan, dtype=float)
        Xw_all, yw_all = make_sliding_windows(X_all_s, y, seq_len)
        from src.train.torch_trainer import predict as torch_predict
        y_pred_w_all = torch_predict(lstm_model, Xw_all)
        y_pred_all_lstm[seq_len - 1:] = y_pred_w_all

        _save_outputs(
            models_dir=models_dir,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
            run_name=run_name,
            model_name="LSTM",
            model_obj=lstm_model,
            scaler=scaler,
            features=all_features,
            df_feat=df_feat,
            datetime_col=datetime_col,
            target_col=target_col,
            y_pred_all=y_pred_all_lstm,
            y_test=yw_test,
            y_pred_test=y_pred_test_lstm,
            dt_test=dt_test_pts,
            edges=edges,
            eval_cfg=eval_cfg,
            logger=logger,
        )

    logger.info("Saved: model/metrics/figures under %s", str(outputs_dir))
