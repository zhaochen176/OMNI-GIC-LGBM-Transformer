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


def build_transformer(
    *,
    seq_len: int,
    n_features: int,
    d_model: int = 64,
    num_heads: int = 4,
    ff_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
) -> Model:
    """Transformer to regress residual at t from a length=seq_len window."""
    inputs = Input(shape=(seq_len, n_features))

    x = Dense(d_model, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = PositionalEncoding(seq_len, d_model)(x)

    for _ in range(num_layers):
        x = TransformerBlock(d_model, num_heads, ff_dim, dropout)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(16, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


def make_residual_sequences(
    X: np.ndarray, residual: np.ndarray, seq_len: int
):
    """
    X: [N, F] (scaled features), residual: [N]
    Build pairs: X_seq[t] = X[t-seq_len:t], y_seq[t] = residual[t]
    for t = seq_len..N-1
    """
    X_seq, y_seq = [], []
    for t in range(seq_len, len(X)):
        X_seq.append(X[t - seq_len:t])
        y_seq.append(residual[t])
    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def train_transformer_on_residual(
    *,
    X_train: np.ndarray,          # [Ntr, F] scaled
    residual_train: np.ndarray,   # [Ntr]
    cfg: dict,
    verbose: int = 0,
):
    seq_len = int(cfg.get("sequence_length", 24))
    X_seq, y_seq = make_residual_sequences(X_train, residual_train, seq_len)

    if len(X_seq) < 50:
        return None, {"seq_len": seq_len}

    model = build_transformer(
        seq_len=seq_len,
        n_features=X_train.shape[1],
        d_model=int(cfg.get("d_model", 64)),
        num_heads=int(cfg.get("num_heads", 4)),
        ff_dim=int(cfg.get("ff_dim", 64)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        lr=float(cfg.get("lr", 1e-3)),
    )

    callbacks = [
        EarlyStopping(patience=int(cfg.get("patience", 20)), restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6),
    ]

    model.fit(
        X_seq, y_seq,
        batch_size=int(cfg.get("batch_size", 32)),
        epochs=int(cfg.get("epochs", 100)),
        validation_split=float(cfg.get("validation_split", 0.2)),
        callbacks=callbacks,
        verbose=verbose
    )
    return model, {"seq_len": seq_len}


def predict_residual_rolling(
    *,
    model,
    meta: dict,
    X: np.ndarray,            # [N, F] scaled
):
    """Return residual_pred [N], with NaN for t < seq_len."""
    seq_len = int(meta["seq_len"])
    out = np.full(len(X), np.nan, dtype=float)
    if model is None:
        return out

    for t in range(seq_len, len(X)):
        seq_in = X[t - seq_len:t].reshape(1, seq_len, X.shape[1]).astype(np.float32)
        out[t] = float(model.predict(seq_in, verbose=0)[0][0])
    return out
