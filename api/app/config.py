import os

DATA_DIR = os.getenv("DATA_DIR", "/data")
os.makedirs(DATA_DIR, exist_ok=True)

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS","SPY,QQQ").split(",") if s.strip()]
START_DATE = os.getenv("START_DATE","2015-01-01")

LEARNING_DAYS = int(os.getenv("LEARNING_DAYS","60"))
MARKET_TZ = os.getenv("MARKET_TZ","Europe/Berlin")
RUN_AT = os.getenv("RUN_AT","18:05")

POLY_KEY = os.getenv("POLYGON_API_KEY","").strip()
