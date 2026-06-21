// gold-bot dashboard — fetches pre-generated JSON artifacts and renders them.
// Pure static front-end: no execution, no secrets. Every panel degrades
// gracefully to an empty state when its artifact is missing or stale.
//
// Phase: this is the dashboard SHELL. It already wires status, signal, strategy,
// metrics, equity and trades from data/*.json when present. Richer views
// (plateau heatmap, per-fold WFO, sortable/exportable trade log) land in Phase 8.

const DATA = "data";
const STALE_AFTER_MS = 1000 * 60 * 90; // a 1h-signal older than 90m is stale

async function loadJSON(name) {
  try {
    const res = await fetch(`${DATA}/${name}?t=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

function fmtMoney(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return `$${Number(x).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}
function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString();
}
function setText(id, txt) {
  const el = document.getElementById(id);
  if (el) el.textContent = txt;
}

function renderStatus(status) {
  if (!status) return;
  const synthetic = status.data_source === "synthetic";
  setText("search-state", (status.search_state || "—") + (synthetic ? " · ⚠️ SYNTHETIC DEMO" : ""));
  setText("dd-headroom", fmtMoney(status.account_headroom_to_trailing_dd));
  setText("daily-state", status.daily_loss_state || "—");
  setText("data-asof", fmtTime(status.data_as_of));
  setText("last-run", `Last run: ${fmtTime(status.generated_at)}`);
}

function renderSignal(sig) {
  const body = document.getElementById("signal-body");
  if (!body || !sig || !sig.signal) return;

  const stale = sig.generated_at && (Date.now() - new Date(sig.generated_at).getTime() > STALE_AFTER_MS);
  const side = sig.signal;
  body.classList.remove("empty");
  body.innerHTML = `
    <div class="signal-big signal-${side}">${side}${stale ? ' <span class="stale-flag">STALE</span>' : ""}</div>
    <dl class="kv">
      <dt>Instrument</dt><dd>${sig.instrument || "—"} (${sig.timeframe || "—"})</dd>
      <dt>Entry</dt><dd>${sig.entry ?? "—"}</dd>
      <dt>Stop loss</dt><dd>${sig.stop_loss ?? "—"}</dd>
      <dt>Take profit</dt><dd>${sig.take_profit ?? "—"}</dd>
      <dt>Size</dt><dd>${sig.position_size_contracts ?? "—"} contracts</dd>
      <dt>Risk</dt><dd>${fmtMoney(sig.risk_dollars)}</dd>
      <dt>DD headroom</dt><dd>${fmtMoney(sig.account_headroom_to_trailing_dd)}</dd>
      <dt>Strategy</dt><dd>${sig.strategy_id || "—"}</dd>
      <dt>Valid until</dt><dd>${fmtTime(sig.valid_until)}</dd>
    </dl>
    <p class="hint">${sig.confidence_notes || ""}</p>`;
}

function paramsStr(params) {
  return params ? Object.entries(params).map(([k, v]) => `${k}=${v}`).join(", ") : "—";
}

function renderStrategy(strat, best) {
  const body = document.getElementById("strategy-body");
  if (!body) return;

  if (strat && strat.strategy_id) {
    body.classList.remove("empty");
    body.innerHTML = `
      <div class="signal-big signal-LONG">ACTIVE</div>
      <dl class="kv">
        <dt>ID</dt><dd>${strat.strategy_id}</dd>
        <dt>Family</dt><dd>${strat.family || "—"}</dd>
        <dt>Params</dt><dd>${paramsStr(strat.params)}</dd>
        <dt>Holdout Sharpe</dt><dd>${strat.holdout?.sharpe ?? "—"}</dd>
        <dt>Holdout PF</dt><dd>${strat.holdout?.profit_factor ?? "—"}</dd>
      </dl>`;
    return;
  }

  // No accepted strategy — show the best candidate found and why it was rejected.
  if (best && best.strategy_id) {
    body.classList.remove("empty");
    const fails = (best.gate_failures || []).join(", ") || "—";
    body.innerHTML = `
      <div class="signal-big signal-FLAT">NOT ACCEPTED</div>
      <p class="hint">Best candidate found this cycle (shown for transparency; not live):</p>
      <dl class="kv">
        <dt>ID</dt><dd>${best.strategy_id}</dd>
        <dt>Family</dt><dd>${best.family}</dd>
        <dt>Params</dt><dd>${paramsStr(best.params)}</dd>
        <dt>OOS Sharpe</dt><dd>${best.oos_sharpe ?? "—"}</dd>
        <dt>OOS Profit factor</dt><dd>${best.oos_profit_factor ?? "—"}</dd>
        <dt># Trades</dt><dd>${best.n_trades ?? "—"}</dd>
        <dt>Gate failures</dt><dd>${fails}</dd>
      </dl>`;
  }
}

function renderMetrics(metrics) {
  const body = document.getElementById("metrics-body");
  if (!body || !metrics || !metrics.rows) return;
  body.classList.remove("empty");
  const rows = metrics.rows.map(
    (r) => `<tr><td>${r.name}</td><td>${r.oos ?? "—"}</td><td>${r.holdout ?? "—"}</td></tr>`
  ).join("");
  body.innerHTML = `<table><thead><tr><th>Metric</th><th>OOS</th><th>Holdout</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderEquity(eq) {
  const empty = document.getElementById("equity-empty");
  const canvas = document.getElementById("equity-chart");
  if (!eq || !eq.equity || !canvas || typeof Chart === "undefined") return;
  if (empty) empty.style.display = "none";
  new Chart(canvas, {
    type: "line",
    data: {
      labels: eq.timestamps || eq.equity.map((_, i) => i),
      datasets: [
        { label: "Equity ($)", data: eq.equity, borderColor: "#e3b341", borderWidth: 1.5, pointRadius: 0 },
        { label: "Trailing-DD floor", data: eq.trailing_dd_floor || [], borderColor: "#f85149", borderWidth: 1, borderDash: [4, 4], pointRadius: 0 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#e6edf3" } } },
      scales: { x: { ticks: { color: "#8b949e" } }, y: { ticks: { color: "#8b949e" } } },
    },
  });
}

function renderTrades(trades) {
  const body = document.getElementById("trades-body");
  if (!body || !trades || !trades.rows || !trades.rows.length) return;
  body.classList.remove("empty");
  const cols = Object.keys(trades.rows[0]);
  const head = cols.map((c) => `<th>${c}</th>`).join("");
  const rows = trades.rows.slice(-200).map(
    (r) => `<tr>${cols.map((c) => `<td>${r[c]}</td>`).join("")}</tr>`
  ).join("");
  body.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>`;
}

function renderWFO(wfo) {
  const body = document.getElementById("wfo-body");
  if (!body || !wfo) return;
  if (wfo.total_trials === undefined && !wfo.folds) return;
  body.classList.remove("empty");
  body.innerHTML = `
    <dl class="kv">
      <dt>Total trials</dt><dd>${wfo.total_trials ?? "—"}</dd>
      <dt>Folds</dt><dd>${wfo.folds?.length ?? "—"}</dd>
      <dt>Deflated Sharpe</dt><dd>${wfo.deflated_sharpe ?? "—"}</dd>
      <dt>WF efficiency</dt><dd>${wfo.walk_forward_efficiency ?? "—"}</dd>
    </dl>
    <p class="hint">Trial count is shown for honesty about multiple testing.</p>`;
}

async function main() {
  const [status, signals, strategy, best, metrics, equity, trades, wfo] = await Promise.all([
    loadJSON("status.json"),
    loadJSON("signals.json"),
    loadJSON("current_strategy.json"),
    loadJSON("best_candidate.json"),
    loadJSON("metrics.json"),
    loadJSON("equity_curve.json"),
    loadJSON("trades.json"),
    loadJSON("walkforward.json"),
  ]);
  renderStatus(status);
  renderSignal(signals);
  renderStrategy(strategy, best);
  renderMetrics(metrics);
  renderEquity(equity);
  renderTrades(trades);
  renderWFO(wfo);
}

document.addEventListener("DOMContentLoaded", main);
