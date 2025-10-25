import os, json, time
from typing import Dict
import numpy as np
import pandas as pd
import torch
from torch import nn
from .data import download_prices, build_features

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
SYMS = [s.strip() for s in os.environ.get("SYMBOLS","SPY,QQQ").split(",") if s.strip()]
TRAIN_START = os.environ.get("TRAIN_START", "2010-01-01")

class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

def _train_one(sym: str) -> Dict:
    df_raw = download_prices(sym, TRAIN_START)
    feats_df, feats = build_features(df_raw)
    if feats_df is None or feats_df.empty or len(feats_df) < 400:
        return {"symbol": sym, "error": "not_enough_data", "n_rows": 0 if feats_df is None else int(len(feats_df))}
    X = feats_df[feats].values.astype(np.float32)
    y = feats_df["y_ret"].values.astype(np.float32).reshape(-1,1)
    n = len(X); n_tr = int(n * 0.8)
    Xtr, ytr = X[:n_tr], y[:n_tr]
    Xte, yte = X[n_tr:], y[n_tr:]

    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd[sd==0]=1.0
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd

    model = MLP(Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()

    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
    model.train()
    for _ in range(60):
        opt.zero_grad(); pred = model(Xt); loss = crit(pred, yt); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        Xv = torch.from_numpy(Xte); yv = torch.from_numpy(yte)
        pv = model(Xv); val_loss = float(crit(pv, yv).item())
        x_last = torch.from_numpy(((X[-1] - mu) / sd).reshape(1,-1))
        last_pred = float(model(x_last).item())

    out_dir = os.path.join(DATA_DIR, "models"); os.makedirs(out_dir, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd, "feats": feats, "trained_at": time.time(), "n_rows": n},
               os.path.join(out_dir, f"{sym}.pt"))
    meta = {"symbol": sym, "val_mse": val_loss, "last_pred": last_pred, "n_rows": int(n), "ts": time.time()}
    with open(os.path.join(out_dir, f"{sym}.json"), "w", encoding="utf-8") as f: json.dump(meta, f, indent=2)
    return meta

def train_daily() -> Dict:
    results, errors = [], []
    for s in SYMS:
        try:
            res = _train_one(s); results.append(res)
            if "error" in res: errors.append(f"{s}: {res['error']}")
        except Exception as e:
            errors.append(f"{s}: {type(e).__name__}: {e}")
    return {"models": results, "errors": errors}
