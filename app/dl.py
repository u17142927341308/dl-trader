import os, json, time
from pathlib import Path

def train_daily():
    data_dir = Path(os.environ.get("DATA_DIR", "/app/data"))
    models_dir = data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "placeholder.json"
    payload = {"ts": time.time(), "note": "stub model until training is wired"}
    model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "models": [{"symbol": "SPY", "path": str(model_path)}],
        "errors": []
    }
