import numpy as np

def make_sliding_windows(X: np.ndarray, y: np.ndarray, seq_len: int):
    """
    X: (T, F), y: (T,)
    Return:
      Xw: (N, seq_len, F)
      yw: (N,)
    where N = T - seq_len + 1, predicting y at time t using X[t-seq_len+1 : t+1].
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    T = len(X)
    if T < seq_len:
        raise ValueError(f"Not enough samples T={T} for seq_len={seq_len}")

    N = T - seq_len + 1
    Xw = np.zeros((N, seq_len, X.shape[1]), dtype=np.float32)
    yw = np.zeros((N,), dtype=np.float32)
    for i in range(N):
        Xw[i] = X[i:i+seq_len]
        yw[i] = y[i+seq_len-1]
    return Xw, yw
