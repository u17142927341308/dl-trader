import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_features(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame(), []
    px = df["Close"].copy()
    out = df.copy()
    out["ret1"]   = px.pct_change()
    out["vol_20"] = out["ret1"].ewm(span=20).std()
    out["ret5"]   = px.pct_change(5)
    out["ret20"]  = px.pct_change(20)
    out["ma20"]   = px.rolling(20).mean()
    out["price_z_20"] = (px/out["ma20"] - 1)
    for col in ["ret5","ret20","price_z_20"]:
        s = out[col]
        mu = s.rolling(252, min_periods=100).mean()
        sd = s.rolling(252, min_periods=100).std(ddof=0).replace(0, np.nan)
        out[col] = ((s - mu) / sd).fillna(0.0)
    feats = ["ret1","vol_20","ret5","ret20","price_z_20"]
    out = out[["Open","High","Low","Close","Volume"] + feats].dropna().copy()
    out["y_ret"] = out["Close"].pct_change().shift(-1)
    return out.dropna().copy(), feats

def train_predict_latest(df_feats: pd.DataFrame, feats):
    # Zeitbasierter Split: letzte 252 Börtage als Validationsfenster
    if len(df_feats) < 400:
        return None, np.nan
    X = df_feats[feats].values
    y = df_feats["y_ret"].values
    split = max(252, int(len(X)*0.2))
    Xtr, ytr = X[:-split], y[:-split]
    Xva, yva = X[-split:], y[-split:]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(64,32), activation="relu",
                             random_state=42, max_iter=500, early_stopping=True))
    ])
    pipe.fit(Xtr, ytr)
    pred_next = pipe.predict([X[-1]])[0]
    return pipe, float(pred_next)

def atr(df: pd.DataFrame, n=14):
    if df is None or df.empty: return np.nan
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high-low),
        (high-prev_close).abs(),
        (low-prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean().iloc[-1]

def make_signal(pred_ret, price, atr_value, thr=0.0015, atr_k=2.0):
    # Schwelle 0.15% – vermeidet Rauschen
    if np.isnan(pred_ret) or np.isnan(price) or np.isnan(atr_value):
        return "HOLD", 0.0, None, None
    if pred_ret > thr:
        side = "BUY"
        sl = price - atr_k*atr_value
        tp = price + 2*atr_k*atr_value
    elif pred_ret < -thr:
        side = "SELL"
        sl = price + atr_k*atr_value
        tp = price - 2*atr_k*atr_value
    else:
        side = "HOLD"
        sl = None
        tp = None
    # Gewicht proportional zur erwarteten Rendite / Volatilität, gedeckelt
    weight = float(np.clip(pred_ret / max(1e-6, atr_value/price), -1.0, 1.0))
    weight = round(weight, 2)
    return side, weight, (None if sl is None else round(float(sl),2)), (None if tp is None else round(float(tp),2))
