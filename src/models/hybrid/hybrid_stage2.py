from __future__ import annotations

import numpy as np

from src.models.hybrid.lgbm_stage import train_lgbm
from src.models.hybrid.transformer_stage import (
    train_transformer_on_residual,
    predict_residual_rolling,
)


def train_lgbm_then_transformer(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_all: np.ndarray,
    lgbm_cfg: dict,
    tf_cfg: dict,
):
    """
    Step②: train LGBM on train, get residuals on train, then train Transformer on residual sequence.
    Inference: y_hat = y_lgbm + r_hat
    """
    # ---- Stage-1: LGBM ----
    # Use a small tail of train as valid for LGBM early stopping
    n = len(X_train)
    split = int(n * 0.8)
    X_tr, X_val = X_train[:split], X_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]

    lgbm_model = train_lgbm(
        X_train=X_tr, y_train=y_tr,
        X_valid=X_val, y_valid=y_val,
        params=lgbm_cfg["params"],
        num_boost_round=int(lgbm_cfg.get("num_boost_round", 1000)),
        early_stopping_rounds=int(lgbm_cfg.get("early_stopping_rounds", 50)),
    )

    # Base predictions
    y_train_base = lgbm_model.predict(X_train)
    y_test_base = lgbm_model.predict(X_test)
    y_all_base = lgbm_model.predict(X_all)

    # ---- Stage-2: Transformer on train residuals ----
    residual_train = (y_train - y_train_base).astype(np.float32)

    tf_model, tf_meta = train_transformer_on_residual(
        X_train=X_train,
        residual_train=residual_train,
        cfg=tf_cfg,
        verbose=0,
    )

    # Rolling residual prediction on test and all
    r_test = predict_residual_rolling(model=tf_model, meta=tf_meta, X=X_test)
    r_all = predict_residual_rolling(model=tf_model, meta=tf_meta, X=X_all)

    # Combine (note: first seq_len points have NaN residual; keep base there)
    y_test_hybrid = y_test_base.copy()
    m = np.isfinite(r_test)
    y_test_hybrid[m] = y_test_base[m] + r_test[m]

    y_all_hybrid = y_all_base.copy()
    m2 = np.isfinite(r_all)
    y_all_hybrid[m2] = y_all_base[m2] + r_all[m2]

    bundle = {
        "lgbm_model": lgbm_model,
        "tf_model": tf_model,
        "tf_meta": tf_meta,
    }
    return bundle, y_test_hybrid, y_all_hybrid
