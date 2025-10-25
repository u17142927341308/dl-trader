import os, json, math, pathlib, datetime as dt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import download_prices

def _to_date(s):
    try:
        return dt.datetime.fromisoformat(str(s).replace('Z','+00:00')).date()
    except Exception:
        return None

def _load_signals(sig_csv: pathlib.Path, lookback_days=180):
    if not sig_csv.exists():
        return pd.DataFrame(columns=["ts","symbol","side","weight","price"])
    df = pd.read_csv(sig_csv)
    if "ts" in df.columns:
        df["date"] = df["ts"].apply(_to_date)
        cutoff = dt.date.today() - dt.timedelta(days=lookback_days)
        df = df[df["date"] >= cutoff]
    return df

def _next_trading_day(dates: pd.DatetimeIndex, day: dt.date):
    # erstes Datum > day
    idx = dates.get_indexer([pd.Timestamp(day)+pd.Timedelta(days=0)], method='bfill')
    return None if len(idx)==0 else (dates[idx[0]] if 0 <= idx[0] < len(dates) else None)

def make_daily_report(data_dir="data", docs_dir="docs"):
    data_dir = pathlib.Path(data_dir)
    docs_dir = pathlib.Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    sig_csv = data_dir / "signals.csv"
    sigs = _load_signals(sig_csv, lookback_days=180)
    if sigs.empty:
        # leere Platzhalter
        (docs_dir / "metrics.json").write_text(json.dumps({"note":"no signals yet"}, indent=2), encoding="utf-8")
        fig = plt.figure(figsize=(7,4))
        plt.title("No signals yet")
        plt.savefig(docs_dir / "backtest.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        return

    symbols = sorted(sigs["symbol"].unique())
    start_date = (sigs["date"].min() - dt.timedelta(days=5)).isoformat()

    # Kursdaten laden
    px = {}
    for sym in symbols + ["SPY"]:
        df = download_prices(sym, start_date)
        if df is None or df.empty: 
            continue
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df["ret1"] = df["Close"].pct_change()
        px[sym] = df

    # Equity-Kurve aus Signalen
    # Portfoliowert startet bei 1.0
    all_days = sorted(set(d.index.date for d in px.values() if d is not None))
    equity = pd.Series(index=pd.to_datetime(all_days), dtype=float).sort_index()
    equity.iloc[0] = 1.0
    spy = px.get("SPY")
    if spy is None or spy.empty:
        # Benchmark aus erstem Symbol approximieren
        anydf = next(iter(px.values()))
        spy = anydf[["Close"]].copy()
        spy["ret1"] = spy["Close"].pct_change()

    # Signal -> Ertrag am NÄCHSTEN Handelstag
    daily_pnl = pd.Series(0.0, index=equity.index)
    for _, r in sigs.iterrows():
        sym = r["symbol"]; side = str(r["side"]).upper(); w = float(r["weight"])
        df_sym = px.get(sym)
        if df_sym is None or df_sym.empty: 
            continue
        day = r.get("date")
        if not isinstance(day, dt.date): 
            continue
        nd = _next_trading_day(df_sym.index, day)
        if nd is None or nd.date() not in daily_pnl.index: 
            continue
        sign = 1.0 if side == "BUY" else (-1.0 if side == "SELL" else 0.0)
        # next-day return
        ret = df_sym.loc[nd, "ret1"]
        if pd.isna(ret):
            continue
        daily_pnl.loc[pd.Timestamp(nd.date())] += sign * abs(w) * ret

    # Kumulieren
    eq = (1.0 + daily_pnl.fillna(0.0)).cumprod()
    equity.update(eq)

    # Benchmark
    bmk = spy.reindex(equity.index).copy()
    bmk_eq = (1.0 + bmk["ret1"].fillna(0.0)).cumprod()

    # Kennzahlen
    def _max_dd(series):
        roll_max = series.cummax()
        dd = series/roll_max - 1.0
        return float(dd.min())

    def _sharpe(daily_ret):
        mu = daily_ret.mean()
        sd = daily_ret.std(ddof=0)
        if sd == 0 or pd.isna(sd): 
            return 0.0
        ann = (mu * 252.0) / sd
        return float(ann)

    daily_ret = equity.pct_change().fillna(0.0)
    metrics = {
        "period_days": int(len(equity)),
        "total_return_portfolio": float(equity.iloc[-1] - 1.0),
        "total_return_benchmark": float(bmk_eq.iloc[-1] - 1.0),
        "max_drawdown_portfolio": _max_dd(equity.fillna(method="ffill").fillna(1.0)),
        "sharpe_like": _sharpe(daily_ret),
        "as_of": dt.datetime.utcnow().isoformat() + "Z",
    }
    (docs_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Plot
    fig = plt.figure(figsize=(8,4))
    plt.plot(equity.index, equity.values, label="Portfolio")
    plt.plot(bmk_eq.index, bmk_eq.values, label="Benchmark (SPY)")
    plt.title("Equity Curve (last ~6 months)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(docs_dir / "backtest.png", dpi=150)
    plt.close(fig)
