import os, csv, pathlib, traceback, datetime as dt
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel

from .logic import run_daily as _run_daily

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
SYMBOLS = [s.strip() for s in os.environ.get("SYMBOLS", "SPY,QQQ").split(",") if s.strip()]

os.makedirs(DATA_DIR, exist_ok=True)

class Health(BaseModel):
    status: str
    symbols: List[str]

app = FastAPI(title="dl-trader API", version="1.0")

@app.get("/health", response_model=Health)
def health():
    return Health(status="ok", symbols=SYMBOLS)

@app.post("/run_daily")
def run_daily_endpoint():
    try:
        rows, errs = _run_daily()
        return {"written": len(rows), "ok": len(errs)==0, "errors": errs}
    except Exception as e:
        tb = traceback.format_exc()
        msg = f"{type(e).__name__}: {e}"
        return {"written": 0, "ok": False, "errors": [msg], "traceback": tb}

@app.get("/signals")
def get_signals():
    p = os.path.join(DATA_DIR, "signals.csv")
    if not os.path.exists(p): 
        return {"rows": []}
    out: List[Dict[str, Any]] = []
    with open(p, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            out.append(r)
    return {"rows": out[-100:]}

@app.get("/debug/last_errors")
def last_errors():
    p = os.path.join(DATA_DIR, "last_errors.log")
    if not os.path.exists(p):
        return {"errors": []}
    return {"errors": pathlib.Path(p).read_text(encoding="utf-8").splitlines()}

try:
    from .dl import train_daily
    @app.post("/train_daily")
    def train_daily_endpoint():
        try:
            info = train_daily()
            ok = (len(info.get("errors", [])) == 0)
            return {"ok": ok, **info}
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            return {"ok": False, "errors": [msg]}
except Exception:
    pass
