from pathlib import Path
import sys, json, datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.tasks_inline import run_daily  # nutzt eure bestehende Logik

Path("site").mkdir(parents=True, exist_ok=True)
rows, errs = run_daily()

payload = {
    "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    "rows": [
        {"ts": r[0], "symbol": r[1], "side": r[2], "weight": r[3], "price": r[4]}
        for r in rows
    ],
    "errors": errs,
}

Path("site/signals.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>DL-Trader — Daily Signals</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
  body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
  h1 {{ font-size: 22px; margin-bottom: 8px; }}
  .ts {{ color:#666; font-size: 13px; margin-bottom: 16px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill,minmax(240px,1fr)); gap: 12px; }}
  .card {{ border:1px solid #e5e7eb; border-radius:14px; padding:14px; }}
  .sym {{ font-weight:600; font-size:16px; }}
  .side.buy {{ color:#0a7f2e; font-weight:600; }}
  .side.sell {{ color:#b91c1c; font-weight:600; }}
  .muted {{ color:#6b7280; font-size:13px; }}
</style>
</head>
<body>
<h1>Daily Signals</h1>
<div class="ts">Generated at: {payload["generated_at"]}</div>
<div class="grid">
"""
for r in payload["rows"]:
    side_class = "buy" if str(r["side"]).upper() == "BUY" else "sell"
    html += f'''
  <div class="card">
    <div class="sym">{r["symbol"]}</div>
    <div class="side {side_class}">{r["side"]}</div>
    <div class="muted">Weight: {float(r["weight"]):.3f}</div>
    <div class="muted">Price: {float(r["price"]):.2f}</div>
  </div>
'''
html += """
</div>
</body>
</html>
"""
Path("site/index.html").write_text(html, encoding="utf-8")
print("OK: site/index.html and site/signals.json generated")
