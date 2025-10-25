import os, json, time, datetime as dt
from pathlib import Path
from .dl import train_daily

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "models").mkdir(parents=True, exist_ok=True)

CAMPAIGN_DAYS = int(os.environ.get("CAMPAIGN_DAYS", "60"))
CAMPAIGN_FILE = DATA_DIR / "campaign.json"
PROGRESS_FILE = DATA_DIR / "progress.json"

def _utc_today():
    return dt.datetime.utcnow().date().isoformat()

def _load_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default

def _save_json(p: Path, obj):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(p)

def ensure_campaign():
    camp = _load_json(CAMPAIGN_FILE, {})
    if not camp.get("start_date"):
        camp = {"start_date": _utc_today(), "days_total": CAMPAIGN_DAYS}
        _save_json(CAMPAIGN_FILE, camp)
    return camp

def upsert_progress(entry):
    prog = _load_json(PROGRESS_FILE, [])
    today = _utc_today()
    if prog and prog[-1].get("date") == today:
        prog[-1] = entry
    else:
        prog.append(entry)
    _save_json(PROGRESS_FILE, prog)
    return prog

def train():
    camp = ensure_campaign()
    today = _utc_today()
    res = train_daily()
    entry = {
        "date": today,
        "ts": time.time(),
        "campaign": camp,
        "models": res.get("models", []),
        "errors": res.get("errors", []),
    }
    prog = upsert_progress(entry)
    start = dt.date.fromisoformat(camp["start_date"])
    elapsed = (dt.date.fromisoformat(today) - start).days + 1
    elapsed = max(1, min(elapsed, camp["days_total"]))
    pct = round(100.0 * elapsed / camp["days_total"], 2)
    return {
        "ok": len(res.get("errors", [])) == 0,
        "today": entry,
        "progress_percent": pct,
        "days_elapsed": elapsed,
        "days_total": camp["days_total"],
        "runs_saved": len(prog),
    }

if __name__ == "__main__":
    out = train()
    print(json.dumps(out, indent=2))
