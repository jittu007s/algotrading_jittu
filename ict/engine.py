"""Live/paper trading engine for the ICT strategy.

Run from algo-trading/:  python -m ict.engine

Paper mode (config.yaml mode: paper) logs signals and simulated fills to
the journal and Telegram, placing no orders. Live mode additionally
requires DRY_RUN=false in ../.env - two independent switches must agree
before real money moves.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, time as dtime, timedelta

from . import config as cfg_mod
from .alerts import TelegramAlerter
from .bias import BiasEngine
from .datafeed import PollingFeed
from .journal import Journal
from .models import Bias, Candle, LiquidityLevel, PaperTrade, Setup, SwingKind
from .order_manager import OrderManager
from .risk import DayRiskManager, size_position
from .structure import (SetupScanner, entry_level, entry_triggered,
                        latest_swing_stop, setup_invalidated)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("ict.engine")

INTERVAL_SECONDS = {"FIVE_MINUTE": 300, "THREE_MINUTE": 180, "ONE_MINUTE": 60}


def _static_levels(feed: PollingFeed, now: datetime, skip_first_min: int) -> list[LiquidityLevel]:
    """Previous-day high/low + today's opening-range extremes."""
    levels: list[LiquidityLevel] = []
    daily = feed.fetch("ONE_DAY", now - timedelta(days=10), now)
    prev = [c for c in daily if c.timestamp.date() < now.date()]
    if prev:
        levels.append(LiquidityLevel(prev[-1].high, SwingKind.HIGH, "pdh"))
        levels.append(LiquidityLevel(prev[-1].low, SwingKind.LOW, "pdl"))
    session_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    ors = feed.fetch("FIVE_MINUTE", session_open,
                     session_open + timedelta(minutes=skip_first_min))
    if ors:
        levels.append(LiquidityLevel(max(c.high for c in ors), SwingKind.HIGH, "or_high"))
        levels.append(LiquidityLevel(min(c.low for c in ors), SwingKind.LOW, "or_low"))
    return levels


class Engine:
    def __init__(self):
        self.cfg = cfg_mod.load()
        self.paper = self.cfg.mode != "live"
        self.feed = PollingFeed()
        self.bias_engine = BiasEngine(self.feed, self.cfg.bias_daily_tf,
                                      self.cfg.bias_intraday_tf,
                                      self.cfg.structure.swing_k_bias)
        self.scanner = SetupScanner(
            swing_k=self.cfg.structure.swing_k_setup,
            min_pierce=self.cfg.structure.sweep_min_pierce,
            needs_rejection=self.cfg.structure.sweep_needs_rejection,
            body_mult=self.cfg.structure.displacement_body_mult,
            avg_body_period=self.cfg.structure.avg_body_period,
            validity=self.cfg.structure.setup_validity_candles)
        self.orders = OrderManager(self.feed.client, paper=self.paper,
                                   strike_mode=self.cfg.options.strike,
                                   delta_assumed=self.cfg.options.delta_assumed)
        self.risk = DayRiskManager(self.cfg.capital,
                                   self.cfg.risk.max_daily_loss_pct,
                                   self.cfg.risk.equity_drawdown_stop_pct)
        self.journal = Journal(self.cfg.journal_db)
        self.alerts = TelegramAlerter(self.cfg.telegram_enabled)

        self.pending_setup: Setup | None = None
        self.pending_expiry_index: int = 0
        self.trade: PaperTrade | None = None
        self.trade_ctx: dict = {}
        self.lot_size = 75  # refreshed from instrument master at entry

    # ------------------------------------------------------------------
    def run(self) -> None:
        interval_s = INTERVAL_SECONDS[self.cfg.setup_tf]
        last_seen: datetime | None = None
        logger.info("ICT engine started (mode=%s)", self.cfg.mode)
        self.alerts.send(f"ICT engine started ({self.cfg.mode})")

        while True:
            try:
                now = datetime.now()
                self.risk.new_session(now.date())
                bias = self.bias_engine.refresh(now)

                if last_seen is None or last_seen.date() != now.date():
                    self.scanner.set_static_levels(
                        _static_levels(self.feed, now, self.cfg.session.skip_first_minutes))

                candles = self.feed.fetch(self.cfg.setup_tf, now - timedelta(days=4), now)
                for candle in self.feed.closed_only(candles, interval_s, now):
                    if last_seen and candle.timestamp <= last_seen:
                        continue
                    last_seen = candle.timestamp
                    live = candle.timestamp + timedelta(seconds=interval_s) > \
                        now.replace(microsecond=0) - timedelta(seconds=interval_s * 2)
                    self.on_candle(candle, bias, act=live)

                next_close = (int(now.timestamp() // interval_s) + 1) * interval_s
                time.sleep(max(next_close + 8 - now.timestamp(), 5))
            except KeyboardInterrupt:
                logger.info("stopping")
                break
            except Exception:
                logger.exception("engine loop error; retrying next candle")
                time.sleep(30)

    # ------------------------------------------------------------------
    def on_candle(self, candle: Candle, bias: Bias, act: bool = True) -> None:
        t = candle.timestamp.time()

        if self.trade is not None:
            self._manage_trade(candle, act)

        session_start = dtime(9, 15 + self.cfg.session.skip_first_minutes) \
            if 15 + self.cfg.session.skip_first_minutes < 60 else dtime(9, 45)
        in_entry_window = session_start <= t < self.cfg.session.no_entry_after

        setup = self.scanner.on_candle(candle, bias)
        if setup is not None and self.trade is None and in_entry_window \
                and self.risk.can_trade():
            self.pending_setup = setup
            self.pending_expiry_index = self.cfg.structure.setup_validity_candles
            self.journal.log_signal("setup", {
                "bias": bias.value, "fvg": [setup.fvg.low, setup.fvg.high],
                "stop_spot": setup.stop_spot, "sweep": setup.sweep.level.label})
            self.alerts.send(f"SETUP {bias.value}: FVG {setup.fvg.low:.1f}-"
                             f"{setup.fvg.high:.1f}, stop {setup.stop_spot:.1f}")

        if self.pending_setup is not None and self.trade is None:
            self._try_enter(candle, act, in_entry_window)

        if self.trade is not None and t >= self.cfg.session.square_off:
            self._exit(candle, "square_off", act)

    # ------------------------------------------------------------------
    def _try_enter(self, candle: Candle, act: bool, in_window: bool) -> None:
        setup = self.pending_setup
        assert setup is not None
        self.pending_expiry_index -= 1
        if setup_invalidated(setup, candle) or self.pending_expiry_index <= 0 \
                or not in_window or not self.risk.can_trade():
            if self.pending_expiry_index <= 0 or setup_invalidated(setup, candle):
                self.pending_setup = None
            return
        if not entry_triggered(setup, candle, self.cfg.structure.entry_point):
            return

        entry_spot = entry_level(setup, self.cfg.structure.entry_point)
        sl_dist = abs(entry_spot - setup.stop_spot)
        direction = setup.bias.value

        option = self.orders.select_option(entry_spot, direction)
        self.lot_size = option["quantity_per_lot"]
        lots = size_position(self.cfg.capital, self.cfg.risk.risk_per_trade_pct,
                             sl_dist, self.cfg.options.delta_assumed, self.lot_size)
        if lots < 1:
            logger.info("setup skipped: 1 lot exceeds risk budget (SL dist %.1f)", sl_dist)
            self.journal.log_signal("skip_oversize", {"sl_dist": sl_dist})
            self.pending_setup = None
            return

        tag = f"ict{candle.timestamp:%H%M}"
        if act:
            self.orders.buy(option, lots, tag)
        levels = {"sweep": setup.sweep.level.label, "sweep_extreme": setup.sweep.extreme,
                  "mss_swing": setup.mss.broken_swing.price,
                  "fvg": [setup.fvg.low, setup.fvg.high], "lots": lots}
        self.trade = PaperTrade(direction=setup.bias, entry_time=candle.timestamp,
                                entry_spot=entry_spot, stop_spot=setup.stop_spot,
                                levels=levels)
        self.trade_ctx = {
            "option": option, "lots": lots,
            "risk_dist": sl_dist,
            "journal_id": self.journal.open_trade(
                self.cfg.mode, direction, option["symbol"], lots,
                candle.timestamp, entry_spot, None, setup.stop_spot, levels),
        }
        self.alerts.send(f"ENTRY {direction} {option['symbol']} x{lots} lots "
                         f"@spot {entry_spot:.1f}, stop {setup.stop_spot:.1f}")
        self.pending_setup = None

    def _manage_trade(self, candle: Candle, act: bool) -> None:
        trade = self.trade
        assert trade is not None
        long = trade.direction == Bias.BULLISH
        risk = self.trade_ctx["risk_dist"]

        hit_stop = candle.low <= trade.stop_spot if long else candle.high >= trade.stop_spot
        if hit_stop:
            self._exit(candle, "stop", act, price=trade.stop_spot)
            return

        favorable = (candle.high - trade.entry_spot) if long else (trade.entry_spot - candle.low)
        if not trade.partial_done and favorable >= self.cfg.management.partial_at_r * risk:
            trade.partial_done = True
            lots = self.trade_ctx["lots"]
            if lots >= 2 and act:
                half = lots // 2
                self.orders.sell_market(self.trade_ctx["option"], half, "partial_1R")
                self.trade_ctx["lots"] = lots - half
            # pay yourself: stop to entry either way
            trade.stop_spot = trade.entry_spot
            self.journal.log_signal("partial_1R", {"new_stop": trade.stop_spot})

        # Trail behind the most recent confirmed 5m swing - but only once
        # the trade has banked +1R (unless configured immediate). Trailing
        # from entry uses the entry pullback itself as the "swing" and
        # scratches winners within a couple of candles.
        may_trail = trade.partial_done or self.cfg.management.trail_start == "immediate"
        if may_trail:
            lvl = latest_swing_stop(self.scanner.candles[-40:], self.scanner.swing_k,
                                    long, self.cfg.management.swing_trail_buffer)
            if lvl is not None:
                if long and lvl < candle.close:
                    trade.stop_spot = max(trade.stop_spot, lvl)
                elif not long and lvl > candle.close:
                    trade.stop_spot = min(trade.stop_spot, lvl)

    def _exit(self, candle: Candle, reason: str, act: bool, price: float | None = None) -> None:
        trade = self.trade
        assert trade is not None
        exit_spot = price if price is not None else candle.close
        if act:
            self.orders.sell_market(self.trade_ctx["option"], self.trade_ctx["lots"], reason)
        long = trade.direction == Bias.BULLISH
        move = (exit_spot - trade.entry_spot) if long else (trade.entry_spot - exit_spot)
        pnl = move * self.cfg.options.delta_assumed * self.lot_size * self.trade_ctx["lots"]
        self.risk.record_trade(pnl)
        self.journal.close_trade(self.trade_ctx["journal_id"], candle.timestamp,
                                 exit_spot, None, reason, pnl)
        self.alerts.send(f"EXIT {reason} @spot {exit_spot:.1f} pnl~₹{pnl:.0f} "
                         f"(day ₹{self.risk.realized:.0f})")
        self.trade = None
        self.trade_ctx = {}


if __name__ == "__main__":
    Engine().run()
