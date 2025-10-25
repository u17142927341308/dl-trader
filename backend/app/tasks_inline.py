import os, csv, json, pathlib, datetime as dt
import numpy as np
from app.data import download_prices, build_features
from app.dl_model import train_symbol, predict_next

DATA_DIR   = pathlib.Path(os.environ.get("DATA_DIR", "data"))
MODELS_DIR = pathlib.Path("models")
START_DATE = os.environ.get("START_DATE", "2015-01-01")
SYMBOLS    = [s.strip() for s in os.environ.get("SYMBOLS", "SPY,QQQ").split(",") if s.strip()]

def _ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

def _signal_from_pred(df, pred: float):
    vol = df["ret1"].ewm(span=20).std().iloc[-1]
    if vol is None or np.isnan(vol) or vol<=0: vol = 0.01
    z = pred / (vol + 1e-8)
    side = "BUY" if z>0.3 else ("SELL" if z<-0.3 else "FLAT")
    w = float(np.tanh(z/2))
    price = float(df["Close"].iloc[-1])
    if side == "BUY":
        sl = price * (1 - 2*vol)
    elif side == "SELL":
        sl = price * (1 + 2*vol)
    else:
        sl = None
    info = {"z": float(z), "vol": float(vol), "stop_loss": (None if sl is None else float(sl))}
    return side, w, price, info

def _update_state():
    state_path = DATA_DIR / "state.json"
    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00","Z")
    state = {"started_at": now, "runs": 0}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if "started_at" not in state:
        state["started_at"] = now
    state["runs"] = int(state.get("runs", 0)) + 1
    state["days_remaining"] = max(0, 60 - state["runs"])
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state

def run_daily():
    _ensure_dirs()
    rows, errors = [], []

    for sym in SYMBOLS:
        try:
            df_raw = download_prices(sym, START_DATE)
            feats_df, feats = build_features(df_raw)
            if feats_df is None or len(feats_df) < 100:
                raise ValueError("not enough data")

            model, meta = train_symbol(feats_df, feats, MODELS_DIR, sym, window=96, epochs=6)
            if model is None:
                raise ValueError("no training data")

            pred = predict_next(feats_df, feats, MODELS_DIR, sym, meta)
            if pred is None or np.isnan(pred):
                raise ValueError("no prediction")

            side, w, price, extra = _signal_from_pred(feats_df, pred)
            now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00","Z")
            rows.append([now, sym, side, w, price, extra.get("stop_loss")])
        except Exception as e:
            errors.append(f"{sym}: {type(e).__name__}: {e}")

    sig_path = DATA_DIR / "signals.csv"
    write_header = not sig_path.exists()
    if rows:
        with sig_path.open("a", newline="") as f:
            wr = csv.writer(f)
            if write_header:
                wr.writerow(["ts","symbol","side","weight","price","stop_loss"])
            wr.writerows(rows)

    state = _update_state()

    # Für Dashboard (GitHub Pages)
    docs = pathlib.Path("docs")
    docs.mkdir(exist_ok=True)
    (docs / "signals.json").write_text(
        json.dumps({"rows":[
            {"ts":r[0],"symbol":r[1],"side":r[2],"weight":r[3],"price":r[4],"stop_loss":r[5]}
            for r in rows
        ]}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    (docs / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    return rows, errors
