import os
from pathlib import Path
DATA_DIR   = Path(os.getenv("DATA_DIR","/srv/data")); DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH    = str(DATA_DIR / "trader.sqlite")
SYMBOLS    = [s.strip() for s in os.getenv("SYMBOLS","SPY,QQQ,IWM,TLT,GLD,EEM").split(",")]
START_DATE = os.getenv("START_DATE","2014-01-01")
TRAIN_START= os.getenv("TRAIN_START","2015-01-01")
TRAIN_END  = os.getenv("TRAIN_END","2020-01-01")
