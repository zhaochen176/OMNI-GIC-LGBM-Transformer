# src/models/hybrid/transformer_stage.py
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.models.hybrid.sparse_attention import SparseAttentionBlock


class TransformerBlock(tf.keras.layers.Layer):
    """Global self-attention Transformer encoder block."""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEncoding(tf.keras.layers.Layer):
    """Sin/Cos positional encoding."""
    def __init__(self, position: int, d_model: int):
        super().__init__()
        self.pos_encoding = self._positional_encoding(position, d_model)

    def _get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def _positional_encoding(self, position: int, d_model: int):
        angle_rads = self._get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def _select_features(X: np.ndarray, feature_indices: list[int] | None) -> np.ndarray:
    if feature_indices is None:
        return X
    return X[:, feature_indices]


def make_residual_sequences(X: np.ndarray, residual: np.ndarray, seq_len: int):
    """
    X: [N, F] (scaled features), residual: [N]
    Build pairs: X_seq[t] = X[t-seq_len:t], y_seq[t] = residual[t]
    for t = seq_len..N-1
    """
    X_seq = np.asarray([X[t - seq_len:t] for t in range(seq_len, len(X))], dtype=np.float32)
    y_seq = np.asarray([residual[t] for t in range(seq_len, len(X))], dtype=np.float32)
    return X_seq, y_seq


def build_transformer_from_cfg(*, seq_len: int, n_features: int, cfg: dict) -> Model:
    """
    Build residual regressor with either:
      - global attention (TransformerBlock)
      - sparse attention (SparseAttentionBlock: local window + global tokens)
    Controlled by cfg["attention"] in train_transformer_on_residual().
    """
    d_model = int(cfg.get("d_model", 64))
    num_heads = int(cfg.get("num_heads", 4))
    ff_dim = int(cfg.get("ff_dim", 64))
    num_layers = int(cfg.get("num_layers", 2))
    dropout = float(cfg.get("dropout", 0.1))
    lr = float(cfg.get("lr", 1e-3))

    attn_type = str(cfg.get("attention", "global")).lower().strip()
    window_size = int(cfg.get("window_size", 8))
    num_global_tokens = int(cfg.get("num_global_tokens", 2))

    inputs = Input(shape=(seq_len, n_features))

    x = Dense(d_model, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = PositionalEncoding(seq_len, d_model)(x)

    for _ in range(num_layers):
        if attn_type == "sparse":
            x = SparseAttentionBlock(
                embed_dim=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                window_size=window_size,
                num_global_tokens=num_global_tokens,
                dropout_rate=dropout,
            )(x)
        else:
            x = TransformerBlock(d_model, num_heads, ff_dim, dropout)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(16, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


def train_transformer_on_residual(
    *,
    X_train: np.ndarray,          # [Ntr, F] scaled
    residual_train: np.ndarray,   # [Ntr]
    cfg: dict,
    verbose: int = 0,
):
    """
    Train Transformer (global or sparse attention) on residual sequences.

    Required behavior:
      - reads cfg["attention"] to select:
          * "global" -> TransformerBlock
          * "sparse" -> SparseAttentionBlock

    cfg supports:
      - attention: "global" (default) | "sparse"
      - window_size (sparse only, default 8)
      - num_global_tokens (sparse only, default 2)
      - sequence_length (default 24)
      - d_model, num_heads, ff_dim, num_layers, dropout, lr
      - batch_size, epochs, validation_split, patience
      - lr_factor, lr_patience, min_lr
    """
    seq_len = int(cfg.get("sequence_length", 24))

    # Defensive: ensure finite residuals
    residual_train = np.asarray(residual_train, dtype=np.float32)
    residual_train = np.where(np.isfinite(residual_train), residual_train, 0.0).astype(np.float32)

    X_seq, y_seq = make_residual_sequences(X_train, residual_train, seq_len)
    if len(X_seq) < 50:
        return None, {"seq_len": seq_len, "n_features": X_train.shape[1]}

    # ---- select attention type here (as requested) ----
    # build_transformer_from_cfg reads cfg["attention"] internally
    model = build_transformer_from_cfg(
        seq_len=seq_len,
        n_features=X_train.shape[1],
        cfg=cfg,
    )

    callbacks = [
        EarlyStopping(patience=int(cfg.get("patience", 20)), restore_best_weights=True),
        ReduceLROnPlateau(
            factor=float(cfg.get("lr_factor", 0.5)),
            patience=int(cfg.get("lr_patience", 10)),
            min_lr=float(cfg.get("min_lr", 1e-6)),
        ),
    ]

    model.fit(
        X_seq, y_seq,
        batch_size=int(cfg.get("batch_size", 32)),
        epochs=int(cfg.get("epochs", 100)),
        validation_split=float(cfg.get("validation_split", 0.2)),
        callbacks=callbacks,
        verbose=verbose
    )

    meta = {
        "seq_len": seq_len,
        "n_features": X_train.shape[1],
        "attention": str(cfg.get("attention", "global")).lower().strip(),
        "window_size": int(cfg.get("window_size", 8)),
        "num_global_tokens": int(cfg.get("num_global_tokens", 2)),
    }
    return model, meta


def predict_residual_rolling(
    *,
    model,
    meta: dict,
    X: np.ndarray,  # [N, F] scaled (already selected if needed)
):
    """
    Return residual_pred [N], with NaN for t < seq_len.
    Uses batched window prediction (fast).
    """
    seq_len = int(meta["seq_len"])
    out = np.full(len(X), np.nan, dtype=float)
    if model is None:
        return out
    if len(X) <= seq_len:
        return out

    # Build all windows: [N-seq_len, seq_len, F]
    Xw = np.asarray([X[t - seq_len:t] for t in range(seq_len, len(X))], dtype=np.float32)
    yhat = model.predict(Xw, verbose=0).reshape(-1).astype(float)
    out[seq_len:] = yhat
    return out
