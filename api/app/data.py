import os, io, csv, json, time, math, pathlib, datetime as dt
import pandas as pd
import numpy as np
import requests

from .config import DATA_DIR, START_DATE, POLY_KEY

def _normalize_ohlc(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    cols = {c.lower(): c for c in df.columns}
    # Standardisiere Spalten
    for want in ["open","high","low","close","volume"]:
        if want not in cols:
            # versuche Titelvarianten
            for cand in df.columns:
                if cand.lower().startswith(want):
                    cols[want]=cand
                    break
    out = pd.DataFrame({
        "Open":   df[cols.get("open","Open")].astype(float),
        "High":   df[cols.get("high","High")].astype(float),
        "Low":    df[cols.get("low","Low")].astype(float),
        "Close":  df[cols.get("close","Close")].astype(float),
        "Volume": df[cols.get("volume","Volume")].astype(float) if cols.get("volume","Volume") in df.columns else 0.0
    }, index=pd.to_datetime(df.index))
    out = out.sort_index()
    out.index.name = "Date"
    return out.dropna()

def _download_stooq(symbol: str, start: str) -> pd.DataFrame:
    # Stooq benötigt .us für US-Ticker
    s = f"{symbol.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={s}&i=d"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})
        df = df[df.index >= pd.Timestamp(start)]
        return _normalize_ohlc(df, symbol)
    except Exception:
        return pd.DataFrame()

def _download_yf(symbol: str, start: str) -> pd.DataFrame:
    # leichter, API-frei: yfinance JSON endpoint via Rapid route ist unzuverlässig im Container,
    # daher nutzen wir fallback nur, wenn es klappt.
    try:
        import yfinance as yf
        df = yf.download(symbol, start=start, auto_adjust=False, progress=False)
        if df is None or df.empty: return pd.DataFrame()
        df = df.rename(columns={"Adj Close":"Close"}) if "Adj Close" in df.columns else df
        return _normalize_ohlc(df, symbol)
    except Exception:
        return pd.DataFrame()

def _download_polygon(symbol: str, start: str) -> pd.DataFrame:
    if not POLY_KEY: return pd.DataFrame()
    try:
        end = pd.Timestamp.today().date().isoformat()
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={POLY_KEY}"
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        js = r.json()
        if not js or "results" not in js: return pd.DataFrame()
        rows = js["results"]
        df = pd.DataFrame(rows)
        if df.empty: return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.set_index("Date")
        df = df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
        return _normalize_ohlc(df[["Open","High","Low","Close","Volume"]], symbol)
    except Exception:
        return pd.DataFrame()

def download_prices(symbol: str, start: str):
    for fn in (_download_stooq, _download_yf, _download_polygon):
        df = fn(symbol, start)
        if df is not None and not df.empty:
            return df
    return pd.DataFrame()
