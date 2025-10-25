import os, csv, datetime as dt
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Dataset

SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS","SPY,QQQ").split(",")]
START   = os.getenv("START_DATE","2014-01-01")
TRAIN_S = os.getenv("TRAIN_START","2015-01-01")
TRAIN_E = os.getenv("TRAIN_END","2020-01-01")
DATA_DIR= os.getenv("DATA_DIR","/srv/data")

class WindowDS(Dataset):
    def __init__(self, X, y):
        self.X=torch.tensor(X, dtype=torch.float32)
        self.y=torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i].permute(1,0), self.y[i]

def train_epoch(model, dl, opt, device):
    model.train(); crit=torch.nn.HuberLoss(delta=1e-3)
    tot=n=0
    for xb,yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); pred = model(xb); loss = crit(pred, yb); loss.backward(); opt.step()
        tot += float(loss.item())*len(xb); n += len(xb)
    return tot/max(n,1)

def signal_from_pred(pred_ret, vol_fore, scale_k=1.2, cap=0.6, min_vol=1e-4):
    vol = np.maximum(vol_fore, min_vol)
    x = pred_ret/(scale_k*vol)
    w = np.clip(np.tanh(x), -cap, cap)
    side = "HOLD"
    if w >= 0.10: side="BUY"
    elif w <= -0.10: side="SELL"
    return float(w), side

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

def daily_train_and_signal():
    from app.data import download_prices, build_features
    from app.model import make_windows, TCNReg

    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z"
    rows=[]
    for sym in SYMBOLS:
        df_raw = download_prices(sym, START)
        df, feats = build_features(df_raw)

        idx = df.index
        if not (idx.min() <= pd.Timestamp(TRAIN_S) and idx.max() >= pd.Timestamp(TRAIN_E)):
            continue
        train_mask = (idx >= pd.Timestamp(TRAIN_S)) & (idx < pd.Timestamp(TRAIN_E))
        test_mask  = (idx >= pd.Timestamp(TRAIN_E))
        train_start = np.where(train_mask)[0][0]
        train_end   = np.where(train_mask)[0][-1]
        test_start  = np.where(test_mask)[0][0]

        X_train, y_train = make_windows(df, feats, "y_ret", win=64, start=train_start, end=train_end)
        X_test,  y_test  = make_windows(df, feats, "y_ret", win=64, start=test_start,  end=len(df)-1)
        if len(X_train)==0 or len(X_test)==0: 
            continue

        dl = DataLoader(WindowDS(X_train, y_train), batch_size=256, shuffle=True, drop_last=True)
        model = TCNReg(in_feats=len(feats)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
        for _ in range(8): _=train_epoch(model, dl, opt, device)

        xb = torch.tensor(X_test[-1:], dtype=torch.float32).permute(0,2,1).to(device)
        with torch.no_grad():
            pred = float(model(xb).cpu().numpy().ravel()[0])

        vol_all = df["ret1"].ewm(span=20).std().shift(1).bfill().values
        idx0 = test_start + 64
        vol_fore = float(vol_all[idx0 + len(X_test) - 1])

        w, side = signal_from_pred(pred, vol_fore, scale_k=1.2, cap=0.6)
        price = float(df["Close"].iloc[-1])
        rows.append([now, sym, side, w, price])

    sig_path = os.path.join(DATA_DIR, "signals.csv")
    write_header = not os.path.exists(sig_path)
    with open(sig_path, "a", newline="") as f:
        wr = csv.writer(f)
        if write_header:
            wr.writerow(["ts","symbol","side","weight","price"])
        for r in rows:
            wr.writerow(r)
    return rows
