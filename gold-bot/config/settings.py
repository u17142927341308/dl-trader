"""Central configuration for gold-bot.

Everything that a strategy / backtester / risk manager needs to know about the
funded account, the instruments, and the trading costs lives here as typed,
validated settings. Values can be overridden via environment variables (prefix
``GOLDBOT_``) or a local ``.env`` file.

The funded-account rules in :class:`AccountRules` are treated by the rest of the
system as *inviolable* risk constraints, not suggestions. A backtest that would
have breached the trailing drawdown or daily loss limit is rejected outright.

NOTE: the defaults below are sensible placeholders for a "typical" 50K funded
plan. Confirm the exact numbers with your prop firm and override them via .env.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class InstrumentSpec(BaseModel):
    """Contract specification for a single futures instrument.

    Gold is quoted in USD per troy ounce. A "tick" is the minimum price
    increment; ``tick_value`` is what one tick is worth for one contract.
    """

    symbol: str
    yahoo_symbol: str  # what the yfinance adapter requests as a price proxy
    point_value: float  # USD per 1.00 move in price, per contract
    tick_size: float = 0.10
    tick_value: float = Field(..., description="USD per tick per contract")

    @property
    def ticks_per_point(self) -> float:
        return 1.0 / self.tick_size


# GC: full-size, 100 oz -> $10 per 0.10 tick -> $100 per 1.00 point.
GC = InstrumentSpec(
    symbol="GC", yahoo_symbol="GC=F", point_value=100.0, tick_size=0.10, tick_value=10.0
)
# MGC: micro, 10 oz -> $1 per 0.10 tick -> $10 per 1.00 point. 10 MGC == 1 GC.
MGC = InstrumentSpec(
    symbol="MGC", yahoo_symbol="GC=F", point_value=10.0, tick_size=0.10, tick_value=1.0
)

INSTRUMENTS: dict[str, InstrumentSpec] = {"GC": GC, "MGC": MGC}


class AccountRules(BaseModel):
    """Tradovate 50K funded-account risk rules. INVIOLABLE in the backtester."""

    account_size: float = 50_000.0
    # End-of-day trailing max drawdown: ratchets up with new equity highs,
    # never moves down. Breaching it = account dead.
    trailing_drawdown: float = 2_500.0
    # Hitting this intraday loss stops trading for the rest of the day.
    daily_loss_limit: float = 1_250.0
    # Evaluation profit target.
    profit_target: float = 3_000.0
    # Max contracts (in MGC-equivalents) the funded plan permits.
    max_contracts: int = 10


class CostModel(BaseModel):
    """Realistic trading frictions. No frictionless fills allowed."""

    # Commission per contract per side (round-turn = 2x this). Tradovate +
    # exchange + NFA fees for micro gold are ~$0.5-1.5/side depending on plan.
    commission_per_side: float = 0.74
    # Slippage modelled in ticks per side. >= 1 tick is mandatory.
    slippage_ticks: float = 1.0


class Settings(BaseSettings):
    """Top-level settings object, populated from env / .env."""

    model_config = SettingsConfigDict(
        env_prefix="GOLDBOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Default instrument to trade/size in.
    default_instrument: str = "MGC"

    # Flat overrides for the most-tweaked account numbers (env-friendly scalars).
    account_size: float = 50_000.0
    trailing_drawdown: float = 2_500.0
    daily_loss_limit: float = 1_250.0
    profit_target: float = 3_000.0
    max_contracts: int = 10

    commission_per_side: float = 0.74
    slippage_ticks: float = 1.0

    # Data history start (longest reliable GC history is what we want).
    history_start: str = "2007-01-01"
    data_cache_dir: str = ".cache"

    # Reproducibility.
    random_seed: int = 7

    @property
    def account_rules(self) -> AccountRules:
        return AccountRules(
            account_size=self.account_size,
            trailing_drawdown=self.trailing_drawdown,
            daily_loss_limit=self.daily_loss_limit,
            profit_target=self.profit_target,
            max_contracts=self.max_contracts,
        )

    @property
    def cost_model(self) -> CostModel:
        return CostModel(
            commission_per_side=self.commission_per_side,
            slippage_ticks=self.slippage_ticks,
        )

    @property
    def instrument(self) -> InstrumentSpec:
        return INSTRUMENTS[self.default_instrument]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor so the whole app shares one Settings instance."""
    return Settings()
