# gold-bot — Gold Futures Strategy Discovery & Signal Dashboard

A research-and-signal system for trading CME COMEX **Gold (GC)** and **Micro
Gold (MGC)** futures on a **Tradovate 50,000 USD funded account**. It searches
for trading strategies with anti-overfitting discipline, validates them
walk-forward under the funded account's *trailing drawdown* and *daily loss*
rules, and publishes signals + analytics to a static dashboard.

> ⚠️ **Risk disclaimers**
> - Trading futures is leveraged and can lose more than deposited.
> - Backtests and walk-forward results do **not** guarantee future performance.
> - This system **generates and displays signals only — it does not place orders.**
> - Validate on a Tradovate paper/sim account for a meaningful period before
>   risking a funded account.
> - The strategy search is constrained to respect the funded account's trailing
>   drawdown and daily loss limit. **Never disable those checks to chase returns.**

---

## Architecture: compute layer vs. static Pages

GitHub Pages serves **static files only** — it cannot run Python, place broker
orders, or hold secrets. So the system is split in two:

```
        ┌──────────────────────────────────────────┐
        │  COMPUTE LAYER  (GitHub Actions, Python)   │
        │  data → backtest → search → walk-forward   │
        │  → gating → signal generation              │
        │            writes JSON ↓                   │
        └──────────────────────────────────────────┘
                          │  commits
                          ▼
        ┌──────────────────────────────────────────┐
        │  docs/data/*.json   (published artifacts)  │
        └──────────────────────────────────────────┘
                          │  fetch()
                          ▼
        ┌──────────────────────────────────────────┐
        │  PRESENTATION LAYER (GitHub Pages, static) │
        │  vanilla JS + CDN charts; displays only    │
        └──────────────────────────────────────────┘
```

All heavy lifting runs in Actions. The browser only fetches pre-generated JSON
and renders it. **No live order execution from the browser. Signals are
displayed, not auto-fired.** True live auto-execution would require a separate
always-on process (e.g. Oracle Cloud Free VPS) talking to the Tradovate API;
the signal module is designed so that can be bolted on later, but broker
credentials never go near the public site.

## Funded-account constraints (inviolable)

Encoded in `config/settings.py` (`AccountRules`), overridable via `.env`.
Defaults are placeholders for a typical 50K plan — **confirm with your provider**:

| Parameter            | Default   | Meaning                                                        |
|----------------------|-----------|----------------------------------------------------------------|
| Account size         | 50,000 USD| Starting balance                                               |
| Trailing max DD      | 2,000 USD | EOD trailing; ratchets up on new highs, never down; breach=dead|
| Daily loss limit     | 1,000 USD | Hit it → trading stops for the day                             |
| Profit target (eval) | 3,000 USD | Evaluation pass threshold                                      |
| Max contracts        | 10 MGC-eq | Funded-plan position cap                                       |

Instruments: **GC** (100 oz, 0.10 tick = $10) and **MGC** (10 oz, 0.10 tick =
$1). 10 MGC = 1 GC. MGC is the default for granular sizing.

> Any strategy that would have breached the trailing drawdown or daily loss
> limit at any point out-of-sample is **rejected**, regardless of total return.

---

## Build status & roadmap

The project is built in phases (see the original brief, §11). Each phase is
typed, tested, and committed before moving on.

- [x] **Phase 1 — Scaffold, config, data adapter + caching, indicators + no-look-ahead tests**
- [x] **Phase 2 — Strategy ABC + families (ema_cross, rsi_bollinger, donchian_breakout, macd_trend) + registry**
- [x] **Phase 3 — Event-driven engine (trailing-DD, daily-loss, costs) + fast runner; reconciled**
- [ ] Phase 4 — Metrics suite (Deflated Sharpe, Monte Carlo) with tests
- [ ] Phase 5 — Walk-forward + optimizer + gating + orchestrator
- [ ] Phase 6 — Risk manager wired into backtest + signal paths
- [ ] Phase 7 — Signal generator + JSON exporters + schema
- [~] Phase 8 — Dashboard front-end (shell live on Pages; rich views pending)
- [~] Phase 9 — GitHub Actions + Pages deployment (Pages wired; compute workflows pending)
- [ ] Phase 10 — Docs, final test pass, "extend to live Tradovate" note

## What Phase 1 delivers

```
gold-bot/
├── config/
│   ├── settings.py        # pydantic-settings: account rules, instruments, costs
│   └── search_space.yaml  # 4 strategy families + parameter grids
├── src/gold_bot/
│   ├── data/
│   │   ├── adapter.py          # DataAdapter ABC + normalize_ohlcv() contract
│   │   ├── yfinance_adapter.py # free default source (GC=F), cache-backed
│   │   └── cache.py            # parquet cache + hash integrity checks
│   ├── features/
│   │   └── indicators.py       # SMA/EMA/RSI/ATR/Bollinger/MACD/Donchian, all causal
│   ├── strategies/
│   │   ├── base.py             # Strategy ABC -> target-position {-1,0,1}, causal
│   │   ├── ema_cross.py        # trend
│   │   ├── rsi_bollinger.py    # mean reversion
│   │   ├── donchian_breakout.py# breakout
│   │   ├── macd_trend.py       # trend + regime filter
│   │   └── registry.py         # auto-register families + search-space expansion
│   ├── risk/
│   │   └── prop_rules.py       # TrailingDrawdown + DailyLossLimit (path-dependent)
│   └── backtest/
│       ├── event_engine.py     # bar-by-bar verifier: prop rules + costs + stops
│       └── vectorbt_runner.py  # fast vectorised pass for the search (reconciled)
├── docs/                       # gold-bot dashboard (GitHub Pages root)
└── tests/                      # 53 tests: no-look-ahead (indicators + signals),
                                # cache integrity, config, registry/grids, prop-rule
                                # edge cases, engine breaches, event/fast reconcile
```

**Design highlights**

- **No look-ahead, provably.** Every indicator is tested for *prefix-stability*:
  the value at bar `t` computed on `data[0:t]` must equal the full-series value
  restricted to `[0:t]`. If appending future bars changes the past, the test
  fails. See `tests/test_indicators.py`.
- **Pluggable data.** Strategy code only ever sees a `DataAdapter`. Swapping
  yfinance for a paid intraday source (Databento/CME/Tradovate) means
  implementing one interface — no strategy changes.
- **Honest caching.** The parquet cache is a pure speed optimisation with hash
  integrity; deleting `.cache/` never changes results.

## Setup

```bash
cd gold-bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit account numbers to match your provider

# Quality gate
ruff check .
pytest
```

## Why a "keep searching until it's profitable" loop is dangerous (and how we tame it)

Naively retrying strategies until one looks profitable **is data-snooping** and
will blow up a funded account. The search (Phase 5) is therefore disciplined:

1. A **holdout** test block is touched exactly once, on the single finalist.
2. Strategies are scored on **out-of-sample walk-forward** performance, never
   in-sample.
3. Every trial is logged so the **Deflated Sharpe Ratio** can penalise the fact
   that we tried many variants.
4. An **acceptance gate** requires OOS Sharpe/profit-factor thresholds, no
   trailing-DD/daily-loss breach, a parameter *plateau* (not a lonely spike),
   walk-forward efficiency above a floor, and Monte-Carlo bust-probability ≤ 5%.
5. **If nothing passes, nothing is shipped.** The dashboard says "searching /
   none found" rather than lowering the bar. That honesty is a feature.
