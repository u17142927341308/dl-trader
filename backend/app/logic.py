import os, csv, math, datetime as dt
import pandas as pd
import numpy as np

from .data import download_prices

DATA_DIR = os.environ.get("DATA_DIR","/app/data")
SYMBOLS = [s.strip() for s in os.environ.get("SYMBOLS","SPY,QQQ").split(",") if s.strip()]
START_DATE = os.environ.get("START_DATE","2015-01-01")

os.makedirs(DATA_DIR, exist_ok=True)

def build_features(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame(), []
    out = df.copy()
    px = out["Close"].copy()
    out["ret1"]   = px.pct_change()
    out["vol_20"] = out["ret1"].ewm(span=20).std()
    out["ret5"]   = px.pct_change(5)
    out["ret20"]  = px.pct_change(20)
    out["ma20"]   = px.rolling(20).mean()
    out["price_z_20"] = (px/out["ma20"] - 1)
    for col in ["ret5","ret20","price_z_20"]:
        s  = out[col]
        mu = s.rolling(252, min_periods=100).mean()
        sd = s.rolling(252, min_periods=100).std(ddof=0).replace(0, np.nan)
        out[col] = ((s - mu) / sd).fillna(0.0)
    feats = ["ret1","vol_20","ret5","ret20","price_z_20"]
    out = out[["Open","High","Low","Close","Volume"] + feats].dropna().copy()
    out["y_ret"] = out["Close"].pct_change().shift(-1)
    out = out.dropna()
    return out, feats

def _signal_from_row(row: pd.Series):
    # simple, smooth momentum/mean-reversion blend
    mom = float(row.get("ret20", 0.0))
    z   = float(row.get("price_z_20", 0.0))
    raw = 0.8*mom + 0.2*z
    w   = max(-0.6, min(0.6, raw*5.0))
    side = "BUY" if w > 0.05 else ("SELL" if w < -0.05 else "FLAT")
    return side, round(w, 2)

def run_daily():
    rows = []
    errors = []
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    for sym in SYMBOLS:
        try:
            df = download_prices(sym, START_DATE)
            if df is None or df.empty:
                errors.append(f"{sym}: no data")
                continue
            feats_df, feats = build_features(df)
            if feats_df.empty:
                errors.append(f"{sym}: no features")
                continue
            last = feats_df.iloc[-1]
            side, w = _signal_from_row(last)
            price = float(df["Close"].iloc[-1])
            rows.append([now, sym, side, w, price])
        except Exception as e:
            errors.append(f"{sym}: {type(e).__name__}: {e}")

    sig_path = os.path.join(DATA_DIR, "signals.csv")
    write_header = not os.path.exists(sig_path)
    if rows:
        with open(sig_path, "a", newline="") as f:
            wr = csv.writer(f)
            if write_header:
                wr.writerow(["ts","symbol","side","weight","price"])
            wr.writerows(rows)

    err_path = os.path.join(DATA_DIR, "last_errors.log")
    if errors:
        with open(err_path, "w", encoding="utf-8") as f:
            for line in errors:
                f.write(line + "\n")
    else:
        if os.path.exists(err_path):
            os.remove(err_path)

    return rows, errors
