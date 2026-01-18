import torch
import torch.nn as nn
from .registry import register
from src.train.torch_trainer import TorchTrainConfig, train_regressor, predict

class SimpleLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # last step
        return self.head(last)

@register("lstm")
def train_predict_lstm(X_train, y_train, X_valid, y_valid, X_test, model_cfg: dict, train_cfg: dict):
    n_features = X_train.shape[-1]
    model = SimpleLSTM(
        n_features=n_features,
        hidden=int(model_cfg.get("hidden", 64)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    tc = TorchTrainConfig(**train_cfg)
    model = train_regressor(model, X_train, y_train, X_valid, y_valid, tc)
    y_pred_test = predict(model, X_test)
    return model, y_pred_test
