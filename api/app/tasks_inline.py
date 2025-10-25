import os, csv, pathlib, traceback
from datetime import datetime, timezone
import pandas as pd

from .config import DATA_DIR, START_DATE, SYMBOLS
from .data import download_prices
from .training import build_features, train_predict_latest, atr, make_signal
from .state import OOS_PATH, SIG_PATH, bump_oos_day_if_new, oos_ready

def run_daily():
    rows = []
    errors = []
    pathlib.Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    for sym in SYMBOLS:
        try:
            df_raw = download_prices(sym, START_DATE)
            if df_raw is None or df_raw.empty:
                raise ValueError("no data")
            df_feat, feats = build_features(df_raw)
            model, pred_next = train_predict_latest(df_feat, feats)
            px  = float(df_raw["Close"].iloc[-1])
            day = str(df_raw.index[-1].date())
            atr_val = atr(df_raw)

            # OOS-Log schreiben
            write_header = not os.path.exists(OOS_PATH)
            with open(OOS_PATH,"a",newline="") as f:
                wr = csv.writer(f)
                if write_header:
                    wr.writerow(["ts","trading_day","symbol","pred_ret","price","atr14"])
                wr.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             day, sym, round(float(pred_next),6), round(px,2),
                             ("" if pd.isna(atr_val) else round(float(atr_val),4))])

            # Fortschritt hochzählen (einmal pro neuem Handelstag)
            bump_oos_day_if_new(day)

            # Nach Lernphase echte Signale schreiben
            if oos_ready():
                side, weight, sl, tp = make_signal(pred_next, px, atr_val)
                if side != "HOLD":
                    write_sig_header = not os.path.exists(SIG_PATH)
                    with open(SIG_PATH,"a",newline="") as f:
                        wr = csv.writer(f)
                        if write_sig_header:
                            wr.writerow(["ts","symbol","side","weight","price","stop_loss","take_profit"])
                        wr.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                     sym, side, weight, round(px,2), sl, tp])
                    rows.append([sym, side, weight, round(px,2), sl, tp])
        except Exception as e:
            errors.append(f"{sym}: {type(e).__name__}: {e}")

    # last_errors.log pflegen
    err_path = os.path.join(DATA_DIR,"last_errors.log")
    if errors:
        with open(err_path,"w") as f:
            for line in errors: f.write(line+"\n")
    else:
        if os.path.exists(err_path): os.remove(err_path)
    return rows, errors
