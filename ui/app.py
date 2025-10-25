
import os, datetime as dt
import pandas as pd
import requests
import streamlit as st

API = os.environ.get("API_BASE","http://api:8000")
CYCLE_START = os.environ.get("CYCLE_START")

st.set_page_config(page_title="DL Trader v4.1", layout="wide")
st.title("DL Trader v4.1")

@st.cache_data(ttl=30)
def _get_json(path):
    r = requests.get(f"{API}{path}", timeout=10)
    r.raise_for_status()
    return r.json()
def _post(path):
    r = requests.post(f"{API}{path}", timeout=120)
    r.raise_for_status()
    return r.json()

try:
    h = _get_json("/health")
    st.caption("Status: " + str(h.get("status","?")) + " · Symbole: " + ", ".join(h.get("symbols",[])))
except Exception as e:
    st.error("API nicht erreichbar: " + str(e))
    st.stop()

c1,c2,c3 = st.columns([1,1,1])
with c1:
    if st.button("Jetzt TRAINIEREN", use_container_width=True):
        with st.spinner("Training..."):
            out = _post("/train_daily")
        st.success(f"train_daily: {out.get('trained',0)} Modelle")
        st.cache_data.clear()
with c2:
    if st.button("Jetzt RUN (DL)", use_container_width=True):
        with st.spinner("run_daily_dl..."):
            out = _post("/run_daily_dl")
        st.success("run_daily_dl: written=" + str(out.get("written")))
        st.cache_data.clear()
with c3:
    if st.button("Neu laden", use_container_width=True):
        st.cache_data.clear()

mods = _get_json("/models").get("models",[])
if mods:
    st.caption("Modelle: " + ", ".join([f"{m['symbol']}@{m['trained_at']}" for m in mods]))

data = _get_json("/signals")
rows = data.get("rows", [])
df = pd.DataFrame(rows)
if df.empty:
    st.info("Noch keine Signale.")
    st.stop()

df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
df = df.dropna(subset=["ts"]).sort_values("ts")
df["date"] = df["ts"].dt.date
today_utc = dt.datetime.utcnow().date()

st.subheader("60-Tage-Lernphase")
if not CYCLE_START:
    CYCLE_START = str(df["date"].min())
start_date = pd.to_datetime(CYCLE_START).date()
u = [d for d in df["date"].unique().tolist() if d >= start_date]
days = len(u)
ratio = min(days/60.0, 1.0)
st.progress(ratio, text=f"Fortschritt: {days}/60 Tage seit {start_date}")
st.caption("Nach 60 Tagen werden fixe Entry/Exit+S/L Regeln gezogen.")

st.subheader("Heutige Signale")
df_today = df[df["date"] == today_utc].copy()
if df_today.empty:
    st.write("Heute noch keine neuen Signale.")
else:
    for col in ("weight","price"):
        df_today[col] = pd.to_numeric(df_today[col], errors="coerce")
    short = df_today[["ts","symbol","side","weight","price"]].copy()
    short["ts"] = short["ts"].dt.strftime("%H:%M:%S UTC")
    st.dataframe(short.reset_index(drop=True), use_container_width=True, height=220)

st.subheader("Verlauf (letzte 50)")
tail = df.tail(50).copy()
for col in ("weight","price"):
    tail[col] = pd.to_numeric(tail[col], errors="coerce")
view = tail[["ts","symbol","side","weight","price"]].copy()
view["ts"] = view["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
st.dataframe(view.reset_index(drop=True), use_container_width=True, height=320)
