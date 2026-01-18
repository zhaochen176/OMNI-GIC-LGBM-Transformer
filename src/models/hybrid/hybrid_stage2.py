from __future__ import annotations

import numpy as np

from src.models.hybrid.lgbm_stage import train_lgbm
from src.models.hybrid.transformer_stage import (
    train_transformer_on_residual,
    predict_residual_rolling,
)


def _topk_indices_from_lgbm(lgbm_model, k: int) -> list[int]:
    """Return indices of top-k features by LightGBM gain importance."""
    imp = np.asarray(lgbm_model.feature_importance(importance_type="gain"), dtype=float)
    if k <= 0 or k >= len(imp):
        return list(range(len(imp)))
    return list(np.argsort(-imp)[:k])


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
    Step②:
      1) Train LGBM on train (with internal val split for early stopping)
      2) Compute residuals on train
      3) Train Transformer on residual sequence using top-k important features
      4) Inference: y_hat = y_lgbm + r_hat

    New config passthrough (for Step③ readiness):
      - tf_cfg["attention"]: "global" | "sparse"
      - tf_cfg["window_size"]: int (sparse only)
      - tf_cfg["num_global_tokens"]: int (sparse only)

    Notes:
      - The attention selection is implemented inside train_transformer_on_residual(),
        which reads tf_cfg["attention"] and chooses TransformerBlock vs SparseAttentionBlock.
      - This file only needs to pass tf_cfg through unchanged and record metadata.
    """
    # ---- Stage-1: LGBM ----
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

    # top-k important features (reduce Transformer input dimension)
    top_k = int(tf_cfg.get("top_k_features", 15))
    feat_idx = _topk_indices_from_lgbm(lgbm_model, top_k)

    X_train_imp = X_train[:, feat_idx]
    X_test_imp = X_test[:, feat_idx]
    X_all_imp = X_all[:, feat_idx]

    # IMPORTANT: tf_cfg is passed through as-is.
    # train_transformer_on_residual will read tf_cfg["attention"] and build the right block.
    tf_model, tf_meta = train_transformer_on_residual(
        X_train=X_train_imp,
        residual_train=residual_train,
        cfg=tf_cfg,
        verbose=int(tf_cfg.get("verbose", 0)),
    )

    # Rolling residual prediction on test and all (NaN for first seq_len points)
    r_test = predict_residual_rolling(model=tf_model, meta=tf_meta, X=X_test_imp)
    r_all = predict_residual_rolling(model=tf_model, meta=tf_meta, X=X_all_imp)

    # Combine (first seq_len points keep base)
    y_test_hybrid = y_test_base.copy()
    m = np.isfinite(r_test)
    y_test_hybrid[m] = y_test_base[m] + r_test[m]

    y_all_hybrid = y_all_base.copy()
    m2 = np.isfinite(r_all)
    y_all_hybrid[m2] = y_all_base[m2] + r_all[m2]

    # Record config in metadata for reproducibility (optional but useful in artifacts)
    attn_cfg = {
        "attention": str(tf_cfg.get("attention", "global")).lower().strip(),
        "window_size": int(tf_cfg.get("window_size", 8)),
        "num_global_tokens": int(tf_cfg.get("num_global_tokens", 2)),
        "top_k_features": top_k,
    }

    bundle = {
        "lgbm_model": lgbm_model,
        "tf_model": tf_model,
        "tf_meta": tf_meta,  # includes seq_len, attention, etc (from transformer_stage.py)
        "tf_feature_indices": feat_idx,
        "tf_attention_cfg": attn_cfg,
    }
    return bundle, y_test_hybrid, y_all_hybrid


