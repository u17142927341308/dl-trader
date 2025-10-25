
import os, io, csv, json, pathlib, datetime as dt
import numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib

from app.data import download_prices, build_features

DATA_DIR = os.environ.get("DATA_DIR","/data")
MODELS_DIR = os.path.join(DATA_DIR,"models")
os.makedirs(MODELS_DIR, exist_ok=True)

SYMBOLS = [s.strip() for s in os.environ.get("SYMBOLS","SPY,QQQ").split(",") if s.strip()]
START_DATE = os.environ.get("START_DATE","2015-01-01")

def _model_path(sym:str)->str:
    return os.path.join(MODELS_DIR, f"{sym}.joblib")

def train_symbol(sym:str):
    df_raw = download_prices(sym, START_DATE)
    if df_raw is None or df_raw.empty:
        raise ValueError(f"no data for {sym}")
    feats_df, feats = build_features(df_raw)
    if feats_df is None or feats_df.empty:
        raise ValueError(f"no features for {sym}")
    X = feats_df[feats].astype(float).values
    y = feats_df["y_ret"].astype(float).values
    n = len(X)
    if n < 400:
        raise ValueError(f"not enough rows ({n}) for {sym}")
    split = int(n*0.8)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(64,32),
                             activation="relu",
                             solver="adam",
                             learning_rate_init=1e-3,
                             max_iter=400,
                             early_stopping=True,
                             n_iter_no_change=10,
                             validation_fraction=0.15,
                             random_state=42))
    ])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    mse = float(mean_squared_error(yte, pred))
    joblib.dump({"pipe": pipe, "feats": feats, "trained_at": dt.datetime.utcnow().isoformat()+"Z"}, _model_path(sym))
    return {"symbol": sym, "rows": n, "mse": mse}

def train_daily():
    results, errors = [], []
    for sym in SYMBOLS:
        try:
            res = train_symbol(sym); results.append(res)
        except Exception as e:
            errors.append(f"{sym}: {type(e).__name__}: {e}")
    return results, errors

def _predict_with_model(sym:str, feats_df:pd.DataFrame):
    p = _model_path(sym)
    if not os.path.exists(p): return None
    obj = joblib.load(p)
    feats = obj["feats"]
    if any(f not in feats_df.columns for f in feats): return None
    x = feats_df[feats].astype(float).iloc[[-1]].values
    pred = float(obj["pipe"].predict(x)[0])
    return pred

def _signal_from_pred(pred:float, vol:float):
    if vol is None or vol<=0: vol = 0.01
    import math
    w = math.tanh((pred/vol)*0.8) * 0.6
    side = "BUY" if w>=0 else "SELL"
    return abs(w), side if side=="BUY" else side

def run_daily_dl():
    rows, errors = [], []
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z"
    for sym in SYMBOLS:
        try:
            df_raw = download_prices(sym, START_DATE)
            if df_raw is None or df_raw.empty:
                raise ValueError("no data")
            feats_df, feats = build_features(df_raw)
            if feats_df is None or feats_df.empty:
                raise ValueError("no features")
            vol_all = feats_df["ret1"].ewm(span=20).std().shift(1).bfill().values
            vol_fore = float(vol_all[-1])
            pred = _predict_with_model(sym, feats_df)
            if pred is None:
                px = feats_df["price_z_20"].iloc[-1]
                pred = float(px)*0.01
            w_abs, side = _signal_from_pred(pred, vol_fore)
            price = float(df_raw["Close"].iloc[-1])
            rows.append([now, sym, side, (w_abs if side=="BUY" else -w_abs), price])
        except Exception as e:
            errors.append(f"{sym}: {type(e).__name__}: {e}")
    sig_path = os.path.join(DATA_DIR, "signals.csv")
    write_header = not os.path.exists(sig_path)
    if rows:
        with open(sig_path,"a",newline="") as f:
            wr = csv.writer(f)
            if write_header:
                wr.writerow(["ts","symbol","side","weight","price"])
            wr.writerows(rows)
    return rows, errors

def list_models():
    out = []
    for sym in SYMBOLS:
        p = _model_path(sym)
        if os.path.exists(p):
            info = joblib.load(p)
            out.append({"symbol": sym, "trained_at": info.get("trained_at")})
    return out
