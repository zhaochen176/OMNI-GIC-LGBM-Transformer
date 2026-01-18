from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention


def _gather_last_k(x: tf.Tensor, k: int) -> tf.Tensor:
    """Take last-k tokens along time axis. x: [B, T, D] -> [B, k, D]"""
    t = tf.shape(x)[1]
    k = tf.minimum(k, t)
    return x[:, t - k:t, :]


def _local_attention_one_step(
    mha: MultiHeadAttention,
    x: tf.Tensor,
    t_idx: int,
    window: int,
) -> tf.Tensor:
    """
    Compute local attention output for position t_idx using [t-window+1 .. t] keys.
    x: [B, T, D]
    return: [B, 1, D]
    """
    t = tf.shape(x)[1]
    w = tf.minimum(window, t_idx + 1)
    start = t_idx + 1 - w
    # q: [B,1,D], kv: [B,w,D]
    q = x[:, t_idx:t_idx + 1, :]
    kv = x[:, start:t_idx + 1, :]
    return mha(q, kv)


class SparseAttentionBlock(tf.keras.layers.Layer):
    """
    Sparse attention encoder block:
      - Local window attention for each token (causal local)
      - Global tokens attend to whole sequence (optional)
      - Tokens also attend to global token summary (optional)

    Design intent:
      - Local window: focus on spikes / short-term morphology
      - Global tokens: preserve global context

    NOTE:
      This is a practical implementation for seq_len ~ 24/30/60; it is not an O(T log T) kernel,
      but it is correct and stable for your scale.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        window_size: int = 8,
        num_global_tokens: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.window_size = int(window_size)
        self.num_global_tokens = int(num_global_tokens)

        self.mha_local = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.mha_global = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])

        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout_rate)
        self.drop2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        """
        inputs: [B, T, D]
        """
        x = inputs
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        D = tf.shape(x)[2]

        # ---- 1) build global tokens (learnable summary tokens)
        # We create a small set of global tokens by taking last-k tokens then projecting,
        # which is stable and avoids having to store trainable tokens outside functional graph.
        # You can later switch to true trainable global tokens if desired.
        if self.num_global_tokens > 0:
            g = _gather_last_k(x, self.num_global_tokens)  # [B, G, D]
        else:
            g = None

        # ---- 2) local window attention per timestep (causal local)
        # outputs shape [B, T, D]
        local_out_list = []
        # Python loop is acceptable at seq_len <= ~60; your default is 24/30
        for t_idx in range(int(x.shape[1]) if x.shape[1] is not None else 0):
            local_out_list.append(_local_attention_one_step(self.mha_local, x, t_idx, self.window_size))
        if local_out_list:
            local_out = tf.concat(local_out_list, axis=1)
        else:
            # fallback for dynamic T (rare in your pipeline)
            # if T is dynamic, use while_loop
            def body(i, ta):
                out_i = _local_attention_one_step(self.mha_local, x, i, self.window_size)
                ta = ta.write(i, tf.squeeze(out_i, axis=1))
                return i + 1, ta

            ta = tf.TensorArray(dtype=x.dtype, size=T)
            i0 = tf.constant(0)
            _, ta = tf.while_loop(lambda i, _: i < T, body, [i0, ta])
            local_out = tf.transpose(ta.stack(), [1, 0, 2])  # [B,T,D]

        # ---- 3) global context mixing
        if g is not None:
            # (a) global tokens attend to full sequence: g' = Attn(g, x)
            g2 = self.mha_global(g, x)  # [B,G,D]

            # (b) tokens attend to global tokens: x' = Attn(x, g')
            global_to_x = self.mha_global(x, g2)  # [B,T,D]

            attn_out = local_out + global_to_x
        else:
            attn_out = local_out

        attn_out = self.drop1(attn_out, training=training)
        out1 = self.ln1(x + attn_out)

        ffn_out = self.ffn(out1)
        ffn_out = self.drop2(ffn_out, training=training)
        return self.ln2(out1 + ffn_out)
