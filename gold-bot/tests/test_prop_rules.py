"""Unit tests for the path-dependent prop-firm rule simulators."""

from __future__ import annotations

from gold_bot.risk.prop_rules import DailyLossLimit, TrailingDrawdown


def test_initial_floor() -> None:
    tdd = TrailingDrawdown(start_equity=50_000, amount=2_000)
    assert tdd.floor == 48_000
    assert not tdd.check(48_500)
    assert tdd.headroom(48_500) == 500


def test_eod_ratchet_raises_floor_only_on_close() -> None:
    tdd = TrailingDrawdown(50_000, 2_000, ratchet_mode="eod")
    # Intraday spike to 51,000 that is NOT held to the close must NOT move the
    # floor in EOD mode.
    tdd.update_intraday(51_000)
    assert tdd.floor == 48_000
    # Day closes at 50,500 -> peak ratchets, floor rises to 48,500.
    tdd.on_day_close(50_500)
    assert tdd.floor == 48_500
    # A lower subsequent close never lowers the floor.
    tdd.on_day_close(50_100)
    assert tdd.floor == 48_500


def test_intraday_ratchet_mode() -> None:
    tdd = TrailingDrawdown(50_000, 2_000, ratchet_mode="intraday")
    tdd.check(51_000)  # check() calls update_intraday
    assert tdd.floor == 49_000


def test_gap_down_through_floor_is_fatal() -> None:
    tdd = TrailingDrawdown(50_000, 2_000)
    assert not tdd.check(48_400)
    # Gap straight through the floor.
    assert tdd.check(47_500)
    assert tdd.dead
    assert tdd.breach_equity == 47_500
    # Death latches even if equity recovers.
    assert tdd.check(60_000)


def test_breach_exactly_at_floor() -> None:
    tdd = TrailingDrawdown(50_000, 2_000)
    assert tdd.check(48_000)  # touching the floor kills it


def test_floor_after_ratchet_then_drawdown() -> None:
    tdd = TrailingDrawdown(50_000, 2_000)
    tdd.on_day_close(53_000)  # peak 53k, floor 51k
    assert tdd.floor == 51_000
    assert not tdd.check(51_500)
    assert tdd.check(50_900)  # below the ratcheted floor -> dead


def test_daily_loss_limit() -> None:
    dll = DailyLossLimit(1_000)
    assert dll.loss(50_000, 49_300) == 700
    assert not dll.breached(50_000, 49_300)
    assert dll.breached(50_000, 49_000)
    assert dll.breached(50_000, 48_500)
