import os, io, time, math, datetime as dt
import pandas as pd
import numpy as np
import requests

POLY_KEY = os.environ.get("POLYGON_API_KEY","") or ""
START_DATE = os.environ.get("START_DATE","2015-01-01")

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
        df = df.set_index("Date")
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
            df = df.set_index("time")
    df = df.sort_index()
    cols_map = {c.lower():c for c in df.columns}
    # ensure standard names
    ren = {}
    for want in ["Open","High","Low","Close","Volume"]:
        # try common lowercase variants
        if want not in df.columns:
            low = want.lower()
            if low in [c.lower() for c in df.columns]:
                # map from first match
                for c in df.columns:
                    if c.lower()==low: ren[c]=want
    if ren:
        df = df.rename(columns=ren)
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[keep]
    # clean
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf,-np.inf], np.nan).dropna()
    # sometimes volume missing
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    return df

def _stooq_symbol(symbol: str) -> str:
    # US ETFs/Equities on stooq: lowercase + .us
    return f"{symbol.lower()}.us"

def _download_stooq(symbol: str, start: str) -> pd.DataFrame:
    try:
        sym = _stooq_symbol(symbol)
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        r = requests.get(url, timeout=15)
        if r.status_code != 200 or not r.text.strip():
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty: 
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
        df = df[df["Date"] >= pd.to_datetime(start)]
        df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})
        return _normalize(df)
    except Exception:
        return pd.DataFrame()

def _download_yf(symbol: str, start: str) -> pd.DataFrame:
    try:
        import yfinance as yf
        df = yf.download(symbol, start=start, auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(columns={"Date":"Date"})
        return _normalize(df)
    except Exception:
        return pd.DataFrame()

def _download_polygon(symbol: str, start: str) -> pd.DataFrame:
    if not POLY_KEY:
        return pd.DataFrame()
    try:
        today = pd.Timestamp.today(tz="UTC").date()
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
            f"{start}/{today}?adjusted=true&sort=asc&limit=50000&apiKey={POLY_KEY}"
        )
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        js = r.json()
        res = js.get("results", [])
        if not res:
            return pd.DataFrame()
        df = pd.DataFrame(res)
        df["Date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
        df = df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
        df = df[["Date","Open","High","Low","Close","Volume"]]
        return _normalize(df)
    except Exception:
        return pd.DataFrame()

def download_prices(symbol: str, start: str) -> pd.DataFrame:
    # robust multi-source: stooq -> polygon -> yfinance
    for fn in (_download_stooq, _download_polygon, _download_yf):
        df = fn(symbol, start)
        if df is not None and not df.empty:
            return df
    return pd.DataFrame()
