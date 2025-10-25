import os, sys, json, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import urllib.request

SYMBOLS = os.getenv("SYMBOLS","SPY,QQQ").split(",")
START = os.getenv("START_DATE","2015-01-01")
CAMPAIGN_DAYS = int(os.getenv("CAMPAIGN_DAYS","60"))
REPO_SLUG = os.getenv("REPO_SLUG","")

def download(symbol):
    for i in range(3):
        try:
            df = yf.download(symbol, start=START, auto_adjust=True, progress=False, threads=False)
            if df is not None and len(df)>250:
                df = df.rename(columns={c:c.title() for c in df.columns})
                return df[["Open","High","Low","Close","Volume"]].dropna().copy()
        except Exception as e:
            time.sleep(1.0*(i+1))
    return pd.DataFrame()

def build_features(df):
    if df is None or df.empty: return pd.DataFrame(), []
    px = df["Close"].copy()
    out = df.copy()
    out["ret1"]   = px.pct_change()
    out["vol20"]  = out["ret1"].ewm(span=20).std()
    out["ret5"]   = px.pct_change(5)
    out["ret20"]  = px.pct_change(20)
    out["ma20"]   = px.rolling(20).mean()
    out["price_z20"] = (px/out["ma20"] - 1)
    for c in ["ret5","ret20","price_z20"]:
        s = out[c]
        mu = s.rolling(252, min_periods=100).mean()
        sd = s.rolling(252, min_periods=100).std(ddof=0).replace(0, np.nan)
        out[c] = ((s - mu) / sd).fillna(0.0)
    feats = ["ret1","vol20","ret5","ret20","price_z20"]
    out = out[["Close"]+feats].dropna().copy()
    out["y"] = out["Close"].pct_change().shift(-1)
    out = out.dropna().copy()
    return out, feats

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.net(x)

def train_predict(df, feats):
    X = df[feats].values.astype(np.float32)
    y = df["y"].values.astype(np.float32).reshape(-1,1)
    n = len(df)
    if n<400: return 0.0
    # Train/Test split: last 60 as validation
    tr = n-60
    Xtr, ytr = X[:tr], y[:tr]
    Xte = X[tr:]
    # Standardize by train stats
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True); sd[sd==0]=1.0
    Xtr = (Xtr - mu)/sd
    Xte = (Xte - mu)/sd
    torch.manual_seed(7)
    model = MLP(Xtr.shape[1])
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr)
    model.train()
    for epoch in range(25):
        opt.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
    model.eval()
    x_last = torch.from_numpy(Xte[-1:].astype(np.float32))
    with torch.no_grad():
        pred = float(model(x_last).cpu().numpy()[0,0])
    return pred

def side_weight(pred, vol):
    # Gewicht kappen +/-0.6
    if vol<=1e-8: vol=1e-8
    s = np.tanh((pred/vol)*3.0)
    w = float(np.clip(s, -0.6, 0.6))
    side = "BUY" if w>0.02 else ("SELL" if w<-0.02 else "HOLD")
    return side, w

def try_fetch_prev(repo_slug):
    if not repo_slug: return {"rows":[]}
    url = f"https://raw.githubusercontent.com/{repo_slug}/gh-pages/data/signals.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return {"rows":[]}

def main():
    now = datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
    prev = try_fetch_prev(REPO_SLUG)
    rows = prev.get("rows", [])[-200:]
    for sym in SYMBOLS:
        df = download(sym)
        if df.empty: 
            continue
        feats_df, feats = build_features(df)
        if feats_df.empty: 
            continue
        pred = train_predict(feats_df, feats)
        vol_f = float(feats_df["ret1"].ewm(span=20).std().iloc[-1])
        side, w = side_weight(pred, vol_f)
        price = float(df["Close"].iloc[-1])
        rows.append({"ts":now, "symbol":sym, "side":side, "weight":w, "price":price})
    out = {"rows": rows[-200:]}
    os.makedirs("public/data", exist_ok=True)
    with open("public/data/signals.json","w",encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    # Statische Dateien kopieren
    os.makedirs("public", exist_ok=True)
    if os.path.exists("site/index.html"):
        with open("site/index.html","rb") as src, open("public/index.html","wb") as dst: dst.write(src.read())
    if os.path.exists("site/style.css"):
        with open("site/style.css","rb") as src, open("public/style.css","wb") as dst: dst.write(src.read())
    print(json.dumps({"written": len(out["rows"])}))

if __name__=="__main__":
    main()
