import os, json, time, io, math, sys
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn

SYMBOLS = os.getenv("SYMBOLS","SPY,QQQ").split(",")
HISTORY_YEARS = int(os.getenv("HISTORY_YEARS","5"))
EPOCHS = int(os.getenv("EPOCHS","8"))
HORIZON = 1
DEVICE = "cpu"

DATA_DIR = os.path.join("docs","data")
os.makedirs(DATA_DIR, exist_ok=True)

def load_prices(sym:str):
    start = (datetime.now(timezone.utc) - timedelta(days=365*HISTORY_YEARS)).date().isoformat()
    df = yf.download(sym, start=start, auto_adjust=True, progress=False)
    if df is None or len(df)==0:
        return pd.DataFrame()
    df = df.reset_index().rename(columns={"Date":"ts"})
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize("UTC")
    use = df[["ts","Open","High","Low","Close","Volume"]].dropna().copy()
    return use

class LSTM(nn.Module):
    def __init__(self, in_dim=1, hid=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hid, 1)
    def forward(self, x):
        y,_ = self.lstm(x)
        return self.fc(y[:,-1,:])

def make_ds(close:pd.Series, win=60):
    ret = close.pct_change().fillna(0.0).values.astype(np.float32)
    X, y = [], []
    for i in range(win, len(ret)-HORIZON):
        X.append(ret[i-win:i].reshape(-1,1))
        y.append(ret[i+HORIZON]-ret[i+HORIZON-1])
    X = np.stack(X) if X else np.zeros((0,win,1), np.float32)
    y = np.array(y, np.float32).reshape(-1,1)
    return X, y

def train_symbol(sym:str):
    df = load_prices(sym)
    if df.empty: return None
    close = df["Close"]
    X, y = make_ds(close, win=60)
    if len(X)<100: return None

    model = LSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X, device=DEVICE)
    y_t = torch.tensor(y, device=DEVICE)

    model.train()
    for ep in range(EPOCHS):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()

    # Vorhersage nächster Schritt
    last_win = torch.tensor(X[-1:], device=DEVICE)
    model.eval()
    with torch.no_grad():
        next_ret = model(last_win).cpu().numpy().ravel()[0]
    # Gewicht in [-0.6,0.6]
    weight = float(np.clip(next_ret*20.0, -0.6, 0.6))
    side = "BUY" if weight>=0 else "SELL"
    price = float(close.iloc[-1])
    return {
        "symbol": sym,
        "weight": weight,
        "side": side,
        "price": price,
        "loss": float(loss.item()),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")
    }

def append_csv(path, header, rows, max_rows=500):
    if os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.DataFrame(rows, columns=header)
        out = pd.concat([old, new], axis=0)
        if len(out)>max_rows: out = out.iloc[-max_rows:]
    else:
        out = pd.DataFrame(rows, columns=header)
    out.to_csv(path, index=False)

def main():
    results = []
    per_symbol_hist = {}
    hist_path = os.path.join(DATA_DIR,"metrics.json")
    if os.path.exists(hist_path):
        with open(hist_path,"r",encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {"history": [], "per_symbol": {}}

    for sym in SYMBOLS:
        r = train_symbol(sym)
        if r is None: continue
        results.append(r)
        # pro Symbol Verlauf Gewicht
        per_symbol_hist.setdefault(sym, [])
        per_symbol_hist[sym].append({"ts": r["ts"], "weight": r["weight"]})

    # Signale anhängen
    if results:
        sig_rows = [[r["ts"], r["symbol"], r["side"], r["weight"], r["price"]] for r in results]
        append_csv(os.path.join(DATA_DIR,"signals.csv"),
                   ["ts","symbol","side","weight","price"], sig_rows, max_rows=1000)

        # Metrics aggregieren
        avg_loss = float(np.mean([r["loss"] for r in results]))
        metrics["history"].append({"ts": results[0]["ts"], "loss": avg_loss})
        for sym, hist in per_symbol_hist.items():
            metrics["per_symbol"].setdefault(sym, [])
            metrics["per_symbol"][sym].extend(hist)
            # kürzen
            if len(metrics["per_symbol"][sym])>1000:
                metrics["per_symbol"][sym] = metrics["per_symbol"][sym][-1000:]

        # 60-Tage Fenster für history (1x pro Tag / Stunde ok)
        if len(metrics["history"])>2000:
            metrics["history"]=metrics["history"][-2000:]

        with open(hist_path,"w",encoding="utf-8") as f:
            json.dump(metrics,f,ensure_ascii=False)

    # Return für Logs
    print(json.dumps({"written": len(results), "symbols": [r["symbol"] for r in results]}))

if __name__=="__main__":
    main()
