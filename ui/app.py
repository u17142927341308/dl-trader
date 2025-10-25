import os, json, requests
from datetime import datetime, timezone, date
import streamlit as st
import pandas as pd

API_CANDIDATES = [
    os.getenv("API_URL", "").rstrip("/"),
    "http://api:8000",
    "http://localhost:8000",
]
API_CANDIDATES = [u for u in API_CANDIDATES if u]

def api_get(path):
    for base in API_CANDIDATES:
        try:
            r = requests.get(f"{base}{path}", timeout=10)
            if r.ok:
                return r.json()
        except Exception:
            pass
    return None

def api_post(path):
    for base in API_CANDIDATES:
        try:
            r = requests.post(f"{base}{path}", timeout=30)
            if r.ok:
                return r.json()
        except Exception:
            pass
    return None

def parse_ts(ts):
    # Eingänge wie "2025-10-24T22:43:34Z"
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None

st.set_page_config(page_title="DL Trader", layout="wide")
st.title("DL Trader – Live Signale & Kampagnenfortschritt")

# Kopfzeile: Health + Button
health = api_get("/health") or {}
symbols = health.get("symbols", [])

cols = st.columns([1,1,2])
with cols[0]:
    st.markdown("**API Status:** " + ("✅ OK" if health else "⚠️ nicht erreichbar"))
with cols[1]:
    st.markdown("**Symbole:** " + (", ".join(symbols) if symbols else "—"))
with cols[2]:
    if st.button("Tageslauf jetzt ausführen"):
        res = api_post("/run_daily")
        if res:
            st.success(f"Run ok: written={res.get('written')} errors={len(res.get('errors',[]))}")
        else:
            st.error("Run fehlgeschlagen: API nicht erreichbar")

# Signale laden
data = api_get("/signals") or {"rows": []}
rows = data.get("rows", [])

# Kampagnen-Fortschritt (60 Tage)
def campaign_progress(rows):
    if not rows:
        return 0, 60, 0.0, None
    tss = [parse_ts(r.get("ts","")) for r in rows]
    tss = [t for t in tss if t is not None]
    if not tss:
        return 0, 60, 0.0, None
    start = min(tss).date()
    today = date.today()
    elapsed = max(0, (today - start).days + 1)  # inkl. heutigem Tag
    total = 60
    remain = max(0, total - elapsed)
    pct = min(1.0, elapsed / total)
    return elapsed, remain, pct, start

elapsed, remain, pct, start_date = campaign_progress(rows)

st.subheader("60-Tage Kampagne")
pcol1, pcol2, pcol3 = st.columns([3,1,1])
with pcol1:
    st.progress(pct, text=f"{elapsed} / 60 Tage")
with pcol2:
    st.metric("Vergangene Tage", f"{elapsed}")
with pcol3:
    st.metric("Rest", f"{remain}")
if start_date:
    st.caption(f"Kampagnenstart: {start_date.isoformat()} (automatisch aus ersten Signalen)")

# Kompakte Kacheln pro Symbol (letztes Signal)
st.subheader("Aktuelle Signale")
if not rows:
    st.info("Noch keine Signale vorhanden.")
else:
    # letztes Signal je Symbol
    latest = {}
    for r in rows:
        sym = r.get("symbol","")
        ts = parse_ts(r.get("ts","")) or datetime(1970,1,1,tzinfo=timezone.utc)
        if sym and (sym not in latest or ts > latest[sym]["_ts"]):
            rr = dict(r)
            rr["_ts"] = ts
            latest[sym] = rr

    cols = st.columns(max(1, min(4, len(latest))))
    i = 0
    for sym in sorted(latest):
        slot = cols[i % len(cols)]
        with slot:
            r = latest[sym]
            side = str(r.get("side","")).upper()
            w = float(r.get("weight", 0.0))
            price = float(r.get("price", 0.0)) if r.get("price") not in (None,"") else None
            ts_h = r.get("ts","")
            side_badge = "🟢 BUY" if side == "BUY" else ("🔴 SELL" if side == "SELL" else "⚪ HOLD")
            st.markdown(f"### {sym}")
            st.markdown(f"**{side_badge}**  \nGewichtung: **{w:.2f}**")
            if price is not None and price > 0:
                st.caption(f"Preis: {price:.2f}")
            st.caption(f"Zeit: {ts_h}")
        i += 1

# Kleine, begrenzte Tabelle mit den letzten 12 Signalen (optional)
with st.expander("Letzte 12 Signale"):
    if rows:
        tail = rows[-12:]
        # hübsch formatieren
        for r in tail:
            if "weight" in r:
                try:
                    r["weight"] = float(r["weight"])
                except Exception:
                    pass
        df = pd.DataFrame(tail)[["ts","symbol","side","weight","price"]]
        st.dataframe(df, use_container_width=True)
    else:
        st.write("—")

st.caption("Hinweis: Der Tageslauf wird (per Scheduler/CI) einmal täglich ausgeführt. Der 60-Tage-Fortschritt zeigt die laufende Kampagne.")
