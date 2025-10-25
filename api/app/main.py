import os, csv, pathlib, traceback
from fastapi import FastAPI
from pydantic import BaseModel
from .config import SYMBOLS, DATA_DIR
from .tasks_inline import run_daily
from .state import learning_progress, oos_ready, OOS_PATH, SIG_PATH

app = FastAPI(title="DL-Trader API v4.1")

class Health(BaseModel):
    status: str
    symbols: list[str]

@app.get("/health")
def health():
    return Health(status="ok", symbols=SYMBOLS)

@app.get("/status")
def status():
    done, total, remaining = learning_progress()
    return {
        "learning_days_done": done,
        "learning_days_total": total,
        "learning_days_remaining": remaining,
        "ready": oos_ready()
    }

@app.post("/run_daily")
def run_daily_endpoint():
    try:
        rows, errs = run_daily()
        return {"written": len(rows), "ok": len(errs)==0, "errors": errs}
    except Exception as e:
        tb = traceback.format_exc()
        msg = f"{type(e).__name__}: {e}"
        return {"written": 0, "ok": False, "errors": [msg], "traceback": tb}

@app.get("/oos")
def get_oos():
    p = OOS_PATH
    if not os.path.exists(p): return {"rows": []}
    out=[]
    with open(p) as f:
        rd = csv.DictReader(f)
        for r in rd: out.append(r)
    return {"rows": out[-200:]}

@app.get("/signals")
def get_signals():
    p = SIG_PATH
    if not os.path.exists(p): return {"rows": []}
    out=[]
    with open(p) as f:
        rd = csv.DictReader(f)
        for r in rd: out.append(r)
    return {"rows": out[-100:]}
