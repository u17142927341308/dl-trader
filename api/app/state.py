import json, os
from datetime import datetime, timezone
from .config import DATA_DIR, LEARNING_DAYS

STATE_PATH = os.path.join(DATA_DIR, "state.json")
OOS_PATH   = os.path.join(DATA_DIR, "oos_predictions.csv")
SIG_PATH   = os.path.join(DATA_DIR, "signals.csv")

def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def load_state():
    if not os.path.exists(STATE_PATH):
        st = {"learning_started": _now_iso(), "oos_days": 0}
        save_state(st)
        return st
    return json.load(open(STATE_PATH,"r",encoding="utf-8"))

def save_state(st):
    json.dump(st, open(STATE_PATH,"w",encoding="utf-8"), indent=2)

def learning_progress():
    st = load_state()
    days = st.get("oos_days",0)
    return days, LEARNING_DAYS, max(0, LEARNING_DAYS - days)

def bump_oos_day_if_new(trading_date_str: str):
    st = load_state()
    # Einfache Logik: wenn trading_date größer als letzter gezählter Tag
    last_key = st.get("last_trading_date","")
    if trading_date_str and trading_date_str != last_key:
        st["oos_days"] = int(st.get("oos_days",0)) + 1
        st["last_trading_date"] = trading_date_str
        save_state(st)

def oos_ready():
    days, total, _ = learning_progress()
    return days >= total
