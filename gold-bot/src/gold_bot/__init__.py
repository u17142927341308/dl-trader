"""gold-bot: Gold futures (GC/MGC) strategy discovery & signal system.

See README.md for architecture. The package is organised into layers:
    data/      - pluggable price-data adapters + caching
    features/  - indicators (no look-ahead, unit-tested)
    strategies/- strategy families (added in phase 2)
    risk/      - prop-firm-aware risk management (phase 3/6)
    backtest/  - vectorbt fast pass + event-driven verifier (phase 3)
    search/    - walk-forward optimisation + acceptance gating (phase 5)
    signals/   - apply the chosen strategy to latest bars (phase 7)
    reporting/ - write docs/data/*.json for the dashboard (phase 7)
"""

__version__ = "0.1.0"
