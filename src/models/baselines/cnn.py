import torch
import torch.nn as nn
from .registry import register
from src.train.torch_trainer import TorchTrainConfig, train_regressor, predict

class SimpleCNN1D(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        # input: (B, T, F) -> transpose to (B, F, T)
        self.net = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (B, hidden, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)   # (B, F, T)
        x = self.net(x)
        return self.head(x)

@register("cnn")
def train_predict_cnn(X_train, y_train, X_valid, y_valid, X_test, model_cfg: dict, train_cfg: dict):
    n_features = X_train.shape[-1]
    model = SimpleCNN1D(
        n_features=n_features,
        hidden=int(model_cfg.get("hidden", 64)),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    tc = TorchTrainConfig(**train_cfg)
    model = train_regressor(model, X_train, y_train, X_valid, y_valid, tc)
    y_pred_test = predict(model, X_test)
    return model, y_pred_test
