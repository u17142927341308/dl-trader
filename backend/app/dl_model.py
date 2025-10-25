import os, json, math, pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, in_dim:int, hidden:int=64, layers:int=2, dropout:float=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out,_ = self.lstm(x)
        out = out[:, -1, :]        # letztes Zeitfenster
        out = self.head(out)
        return out.squeeze(-1)

def _build_windows(df: pd.DataFrame, feats, window:int=64):
    X = df[feats].values.astype(np.float32)
    y = df["y_ret"].values.astype(np.float32)
    xs, ys = [], []
    for i in range(window, len(df)):
        xs.append(X[i-window:i])
        ys.append(y[i])
    if not xs:
        return np.empty((0, window, len(feats)), np.float32), np.empty((0,), np.float32)
    return np.stack(xs), np.array(ys)

def _fit_scaler(X: np.ndarray):
    mu = X.mean(axis=(0,1)).tolist()
    sd = X.std(axis=(0,1)).tolist()
    sd = [s if s>1e-8 else 1.0 for s in sd]
    return {"mu": mu, "sd": sd}

def _apply_scaler(X: np.ndarray, scaler: dict):
    mu = np.array(scaler["mu"], dtype=np.float32)
    sd = np.array(scaler["sd"], dtype=np.float32)
    return (X - mu) / sd

def _save_scaler(p: pathlib.Path, scaler: dict):
    p.write_text(json.dumps(scaler), encoding="utf-8")

def _load_scaler(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8"))

def train_symbol(df: pd.DataFrame, feats, models_dir: pathlib.Path, symbol: str,
                 window:int=64, max_rows:int=2000, epochs:int=6, lr:float=1e-3, device:str=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    models_dir.mkdir(parents=True, exist_ok=True)
    mod_path = models_dir / f"{symbol}.pt"
    sc_path  = models_dir / f"{symbol}.norm.json"

    if len(df) > max_rows:
        df = df.iloc[-max_rows:].copy()

    X, y = _build_windows(df, feats, window=window)
    if len(X)==0:
        return None, None

    if sc_path.exists():
        scaler = _load_scaler(sc_path)
    else:
        scaler = _fit_scaler(X)
        _save_scaler(sc_path, scaler)
    X = _apply_scaler(X, scaler)

    # Train/Test Split: letztes 10% als Validierung
    n = len(X)
    n_val = max(50, int(0.1*n))
    Xtr, ytr = X[:n-n_val], y[:n-n_val]
    Xva, yva = X[n-n_val:], y[n-n_val:]

    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, device=device)
    Xva_t = torch.tensor(Xva, device=device)
    yva_t = torch.tensor(yva, device=device)

    model = LSTMRegressor(in_dim=X.shape[2]).to(device)
    if mod_path.exists():
        model.load_state_dict(torch.load(mod_path, map_location=device))

    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.MSELoss()

    bs = 64
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(Xtr_t), bs):
            idx = perm[i:i+bs]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            pred = model(xb)
            l = loss(pred, yb)
            opt.zero_grad()
            l.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_val = model(Xva_t)
        val_rmse = float(torch.sqrt(nn.functional.mse_loss(pred_val, yva_t)).cpu())

    torch.save(model.state_dict(), mod_path)
    return model, {"val_rmse": val_rmse, "scaler": scaler, "window": window, "device": device}

def predict_next(df: pd.DataFrame, feats, models_dir: pathlib.Path, symbol: str,
                 info: dict):
    device = info.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(in_dim=len(feats)).to(device)
    mod_path = models_dir / f"{symbol}.pt"
    sc_path  = models_dir / f"{symbol}.norm.json"
    if not (mod_path.exists() and sc_path.exists()):
        return None
    model.load_state_dict(torch.load(mod_path, map_location=device))
    scaler = _load_scaler(sc_path)
    window = info.get("window", 64)

    if len(df) < window+1:
        return None

    Xlast = df[feats].values.astype(np.float32)[-window:]
    Xlast = np.expand_dims(Xlast, 0)
    Xlast = _apply_scaler(Xlast, scaler)
    xt = torch.tensor(Xlast, device=device)
    model.eval()
    with torch.no_grad():
        pred = float(model(xt).cpu())
    return pred
