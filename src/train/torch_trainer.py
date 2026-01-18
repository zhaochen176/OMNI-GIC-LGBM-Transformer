from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

@dataclass
class TorchTrainConfig:
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    seed: int = 42

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_regressor(model: nn.Module, X_train, y_train, X_valid, y_valid, cfg: TorchTrainConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = model.to(device)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    Xva = torch.tensor(X_valid, dtype=torch.float32)
    yva = torch.tensor(y_valid, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(Xva, yva), batch_size=cfg.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())
        v = float(np.mean(val_losses)) if val_losses else float("inf")

        if v < best_val:
            best_val = v
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict(model: nn.Module, X):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        y = model(Xt).detach().cpu().numpy().reshape(-1)
    return y
