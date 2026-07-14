"""Broker-agnostic core logic for the SMA-crossover option strategy.

Kept independent of the Angel One SmartAPI client so it can be unit tested
and backtested (see backtest.py) without any network/broker access.

Rules implemented, symmetrically for both directions (see README.md):

  The moving average is an SMMA (smoothed moving average, a.k.a. RMA /
  Wilder's MA) - the same line as TradingView's "SMMA 20 close" - not a
  simple mean.

  LONG (buy ATM CE):
    0. Fresh-cross filter: at least one candle must have CLOSED BELOW the
       SMMA since the last long setup, so only genuine cross-up setups
       arm - not every pair of closes in an ongoing uptrend.
    1. Two consecutive *closed* candles close above the SMMA -> setup armed.
    2. Entry: a later candle's high crosses above the high of the 2nd
       (more recent) of those two candles.
    3. Stop loss: low of the candle immediately before the entry candle.
    4. Target level: entry + risk_reward * (entry - stop_loss). Reaching
       it does NOT close the trade - it switches to TRAILING mode: the
       stop jumps to lock at least +1R (entry + risk) and from then on
       ratchets up along the SMMA each closed candle, never loosening.
       The trade exits when price trades back to the trailed stop.
    5. Early exit (only while the target has not yet been reached): price
       touches the SMMA twice while retracing down from the highest point
       made after entry.

  SHORT (buy ATM PE): the exact mirror — a fresh cross-down, two closes
  below the SMMA, entry on a break below the 2nd candle's low, stop loss
  at the previous candle's high; on reaching the 1:2 level the stop locks
  -1R below entry and trails the SMMA downward.
"""

from collections import deque
from datetime import datetime
from enum import Enum, auto
from typing import Optional


class Candle:
    __slots__ = ("timestamp", "open", "high", "low", "close")

    def __init__(self, timestamp: datetime, open: float, high: float, low: float, close: float):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close


class Signal(Enum):
    NONE = auto()
    ENTER_LONG_CE = auto()
    ENTER_SHORT_PE = auto()
    EXIT = auto()


class ExitReason(Enum):
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TARGET_HIT = "target_hit"
    SMA_DOUBLE_TOUCH = "sma_double_touch"
    TIME_EXIT = "time_exit"
    FORCED_EOD = "forced_eod"


class StrategyEvent:
    def __init__(self, signal: Signal, price: Optional[float], reason: Optional[ExitReason] = None,
                 stop_loss: Optional[float] = None, target: Optional[float] = None,
                 note: Optional[str] = None):
        self.signal = signal
        self.price = price
        self.reason = reason
        self.stop_loss = stop_loss
        self.target = target
        self.note = note

    def __repr__(self):
        return (f"StrategyEvent(signal={self.signal}, price={self.price}, "
                f"reason={self.reason}, stop_loss={self.stop_loss}, target={self.target}, "
                f"note={self.note})")


class SmaCrossOptionStrategy:
    def __init__(self, sma_period: int = 20, risk_reward: float = 2.0):
        self.sma_period = sma_period
        self.risk_reward = risk_reward

        history_len = max(200, sma_period + 5)
        self._candles = deque(maxlen=history_len)
        self._sma_history = deque(maxlen=history_len)
        self._seed_closes = []   # first `sma_period` closes seed the SMMA
        self._smma = None

        # Fresh-cross tracking: a long setup needs a close below the SMMA
        # since the last long arm (mirror for shorts), so we only trade
        # genuine crossovers, not every candle pair inside a trend.
        self._seen_close_below = False
        self._seen_close_above = False

        self.state = "IDLE"       # IDLE -> ARMED -> IN_POSITION
        self.direction = None      # "LONG" or "SHORT" while ARMED / IN_POSITION
        self.trigger_level = None  # high (long) / low (short) of the 2nd setup candle

        self.entry_price = None
        self.stop_loss = None
        self.target = None
        self.trailing = False   # True once the 1:2 level was reached
        self._risk = None       # per-trade risk (|entry - initial SL|)
        self.extreme_since_entry = None  # highest high (long) / lowest low (short)
        self.sma_touch_count = 0
        self._currently_touching_sma = False

    # ------------------------------------------------------------------
    def _update_smma(self, close):
        """SMMA / RMA: seeded with the simple mean of the first `period`
        closes, then smma = (prev * (period - 1) + close) / period."""
        if self._smma is None:
            self._seed_closes.append(close)
            if len(self._seed_closes) >= self.sma_period:
                self._smma = sum(self._seed_closes) / self.sma_period
        else:
            self._smma = (self._smma * (self.sma_period - 1) + close) / self.sma_period
        return self._smma

    # ------------------------------------------------------------------
    def on_closed_candle(self, candle: Candle) -> StrategyEvent:
        sma = self._update_smma(candle.close)
        self._candles.append(candle)
        self._sma_history.append(sma)

        # Track which side of the SMMA closes land on, in every state, so a
        # cross that happens mid-trade still validates the next setup.
        if sma is not None:
            if candle.close < sma:
                self._seen_close_below = True
            elif candle.close > sma:
                self._seen_close_above = True

        if self.state == "IN_POSITION":
            return self._process_in_position(candle, sma)
        if self.state == "ARMED":
            return self._process_armed(candle, sma)
        return self._process_idle(candle, sma)

    def _process_idle(self, candle, sma):
        if sma is None or len(self._candles) < 2:
            return StrategyEvent(Signal.NONE, candle.close)

        prev_candle = self._candles[-2]
        prev_sma = self._sma_history[-2]
        if prev_sma is None:
            return StrategyEvent(Signal.NONE, candle.close)

        if (prev_candle.close > prev_sma and candle.close > sma
                and self._seen_close_below):
            self.state = "ARMED"
            self.direction = "LONG"
            self.trigger_level = candle.high  # high of the 2nd (latest) candle
            self._seen_close_below = False    # consume the cross
        elif (prev_candle.close < prev_sma and candle.close < sma
                and self._seen_close_above):
            self.state = "ARMED"
            self.direction = "SHORT"
            self.trigger_level = candle.low   # low of the 2nd (latest) candle
            self._seen_close_above = False

        return StrategyEvent(Signal.NONE, candle.close)

    def _process_armed(self, candle, sma):
        # Setup invalidated if price closes back across the SMA before triggering.
        crossed_back = (
            sma is not None
            and ((self.direction == "LONG" and candle.close < sma)
                 or (self.direction == "SHORT" and candle.close > sma))
        )
        if crossed_back:
            self._reset()
            # The invalidating candle may itself start an opposite setup.
            return self._process_idle(candle, sma)

        triggered = (
            (self.direction == "LONG" and candle.high >= self.trigger_level)
            or (self.direction == "SHORT" and candle.low <= self.trigger_level)
        )
        if not triggered:
            return StrategyEvent(Signal.NONE, candle.close)

        entry_price = self.trigger_level
        prev_candle = self._candles[-2]  # candle immediately before the entry candle

        if self.direction == "LONG":
            sl = prev_candle.low
            risk = entry_price - sl
            if risk <= 0:
                risk = entry_price * 0.005  # degenerate fallback, shouldn't normally happen
                sl = entry_price - risk
            target = entry_price + self.risk_reward * risk
            signal = Signal.ENTER_LONG_CE
            self.extreme_since_entry = candle.high
        else:
            sl = prev_candle.high
            risk = sl - entry_price
            if risk <= 0:
                risk = entry_price * 0.005
                sl = entry_price + risk
            target = entry_price - self.risk_reward * risk
            signal = Signal.ENTER_SHORT_PE
            self.extreme_since_entry = candle.low

        self.state = "IN_POSITION"
        self.entry_price = entry_price
        self.stop_loss = sl
        self.target = target
        self.trailing = False
        self._risk = risk
        self.sma_touch_count = 0
        self._currently_touching_sma = False

        return StrategyEvent(signal, entry_price, stop_loss=sl, target=target)

    def _process_in_position(self, candle, sma):
        long = self.direction == "LONG"

        # 1. Stop check against the stop as it stood BEFORE this candle.
        hit_sl = candle.low <= self.stop_loss if long else candle.high >= self.stop_loss
        if hit_sl:
            exit_price = self.stop_loss
            reason = ExitReason.TRAILING_STOP if self.trailing else ExitReason.STOP_LOSS
            self._reset()
            return StrategyEvent(Signal.EXIT, exit_price, reason=reason)

        made_new_extreme = (candle.high > self.extreme_since_entry) if long \
            else (candle.low < self.extreme_since_entry)
        self.extreme_since_entry = max(self.extreme_since_entry, candle.high) if long \
            else min(self.extreme_since_entry, candle.low)

        # 2. Reaching the 1:2 level switches to trailing instead of exiting:
        #    lock at least +/-1R, then ratchet the stop along the SMMA.
        if not self.trailing:
            hit_target = candle.high >= self.target if long else candle.low <= self.target
            if hit_target:
                self.trailing = True
                if long:
                    lock = self.entry_price + self._risk
                    self.stop_loss = max(self.stop_loss, lock, sma if sma is not None else lock)
                else:
                    lock = self.entry_price - self._risk
                    self.stop_loss = min(self.stop_loss, lock, sma if sma is not None else lock)
                self.target = None
                return StrategyEvent(Signal.NONE, candle.close,
                                     stop_loss=self.stop_loss, note="trailing_activated")
        else:
            if sma is not None:
                self.stop_loss = max(self.stop_loss, sma) if long else min(self.stop_loss, sma)

        # 3. Double-SMMA-touch early exit applies only before the trade
        #    earns trailing mode (afterwards the trailed stop handles exits).
        if not self.trailing:
            touching = sma is not None and candle.low <= sma <= candle.high
            if not made_new_extreme and touching and not self._currently_touching_sma:
                self.sma_touch_count += 1
                self._currently_touching_sma = True
                if self.sma_touch_count >= 2:
                    exit_price = candle.close
                    self._reset()
                    return StrategyEvent(Signal.EXIT, exit_price, reason=ExitReason.SMA_DOUBLE_TOUCH)
            elif not touching:
                self._currently_touching_sma = False

        return StrategyEvent(Signal.NONE, candle.close)

    def force_exit(self, price: Optional[float], reason: ExitReason = ExitReason.FORCED_EOD) -> StrategyEvent:
        """Call at end-of-day (or on shutdown) to flatten any open position."""
        self._reset()
        return StrategyEvent(Signal.EXIT, price, reason=reason)

    def _reset(self):
        self.state = "IDLE"
        self.direction = None
        self.trigger_level = None
        self.entry_price = None
        self.stop_loss = None
        self.target = None
        self.trailing = False
        self._risk = None
        self.extreme_since_entry = None
        self.sma_touch_count = 0
        self._currently_touching_sma = False


class OpeningRangeBreakout:
    """Opening Range Breakout (ORB) - one of the oldest and most widely
    documented professional intraday strategies (Crabel 1990; validated on
    modern data by Zarattini & Aziz 2023), adapted to this bot's framework:

      - Opening range (OR) = the high/low of the first `or_minutes` of the
        session (default 15 min = the first five 3-minute candles).
      - LONG (buy ATM CE): a candle CLOSES above the OR high. Stop = OR low.
      - SHORT (buy ATM PE): a candle CLOSES below the OR low. Stop = OR high.
      - Risk guard: if the OR is wider than `max_risk_points`, the trade is
        skipped - a stop further than that is not worth an ATM option.
      - Management is identical to the SMMA strategy: at risk_reward x risk
        the stop locks +/-1R and trails the SMMA; exits on the trailed stop.
      - Discipline: at most ONE long and ONE short attempt per day - a core
        part of why ORB survives costs. No re-entries after a stop-out.

    Exposes the same interface/state fields as SmaCrossOptionStrategy so
    bot.py and the backtesters can drive either interchangeably.
    """

    def __init__(self, sma_period: int = 20, risk_reward: float = 2.3,
                 or_minutes: int = 3, max_risk_points: float = 75.0):
        self.sma_period = sma_period
        self.risk_reward = risk_reward
        self.or_minutes = or_minutes
        self.max_risk_points = max_risk_points

        self._seed_closes = []
        self._smma = None

        self._session = None
        self._or_high = None
        self._or_low = None
        self._traded_long = False
        self._traded_short = False

        self.state = "IDLE"
        self.direction = None
        self.entry_price = None
        self.stop_loss = None
        self.target = None
        self.trailing = False
        self._risk = None
        self.extreme_since_entry = None

    def _update_smma(self, close):
        if self._smma is None:
            self._seed_closes.append(close)
            if len(self._seed_closes) >= self.sma_period:
                self._smma = sum(self._seed_closes) / self.sma_period
        else:
            self._smma = (self._smma * (self.sma_period - 1) + close) / self.sma_period
        return self._smma

    def on_closed_candle(self, candle: Candle) -> StrategyEvent:
        sma = self._update_smma(candle.close)

        day = candle.timestamp.date()
        if day != self._session:
            self._session = day
            self._or_high = candle.high
            self._or_low = candle.low
            self._traded_long = False
            self._traded_short = False
            self._or_announced = False
            if self.state == "IN_POSITION":  # shouldn't happen (square-off), but be safe
                self._reset()
            return StrategyEvent(Signal.NONE, candle.close)

        session_open = candle.timestamp.replace(hour=9, minute=15, second=0, microsecond=0)
        in_or_window = (candle.timestamp - session_open).total_seconds() / 60 < self.or_minutes
        if in_or_window:
            self._or_high = max(self._or_high, candle.high)
            self._or_low = min(self._or_low, candle.low)
            return StrategyEvent(Signal.NONE, candle.close)

        # one-time diagnostic once the range is final, so zero-trade days
        # are explainable (range too wide vs no breakout)
        note = None
        if not getattr(self, "_or_announced", True):
            self._or_announced = True
            rng = self._or_high - self._or_low
            note = f"OR {self._or_low:.1f}-{self._or_high:.1f} ({rng:.1f} pts)"
            if rng > self.max_risk_points:
                note += f" > cap {self.max_risk_points:g} - breakouts will be SKIPPED"

        if self.state == "IN_POSITION":
            return self._manage(candle, sma)

        # --- entries: close through the opening range, one shot per side ---
        or_range = self._or_high - self._or_low
        if candle.close > self._or_high and not self._traded_long:
            self._traded_long = True
            if or_range <= self.max_risk_points:
                return self._enter("LONG", candle.close, self._or_low + or_range /2, candle, note=note)
            note = (note + " | " if note else "") + \
                f"LONG breakout at {candle.close:.1f} skipped (range {or_range:.1f} > cap)"
        elif candle.close < self._or_low and not self._traded_short:
            self._traded_short = True
            if or_range <= self.max_risk_points:
                return self._enter("SHORT", candle.close, self._or_high  - or_range /2, candle, note=note)
            note = (note + " | " if note else "") + \
                f"SHORT breakout at {candle.close:.1f} skipped (range {or_range:.1f} > cap)"

        return StrategyEvent(Signal.NONE, candle.close, note=note)

    def _enter(self, direction, entry_price, sl, candle, note=None):
        risk = abs(entry_price - sl)
        if risk <= 0:
            return StrategyEvent(Signal.NONE, candle.close)
        self.state = "IN_POSITION"
        self.direction = direction
        self.entry_price = entry_price
        self.stop_loss = sl
        self._risk = risk
        self.trailing = False
        if direction == "LONG":
            self.target = entry_price + self.risk_reward * risk
            self.extreme_since_entry = candle.high
            return StrategyEvent(Signal.ENTER_LONG_CE, entry_price, stop_loss=sl,
                                 target=self.target, note=note)
        self.target = entry_price - self.risk_reward * risk
        self.extreme_since_entry = candle.low
        return StrategyEvent(Signal.ENTER_SHORT_PE, entry_price, stop_loss=sl,
                             target=self.target, note=note)

    def _manage(self, candle, sma):
        long = self.direction == "LONG"

        hit_sl = candle.low <= self.stop_loss if long else candle.high >= self.stop_loss
        if hit_sl:
            exit_price = self.stop_loss
            reason = ExitReason.TRAILING_STOP if self.trailing else ExitReason.STOP_LOSS
            self._reset()
            return StrategyEvent(Signal.EXIT, exit_price, reason=reason)

        self.extreme_since_entry = max(self.extreme_since_entry, candle.high) if long \
            else min(self.extreme_since_entry, candle.low)

        if not self.trailing:
            hit_target = candle.high >= self.target if long else candle.low <= self.target
            if hit_target:
                self.trailing = True
                if long:
                    lock = self.entry_price + self._risk
                    self.stop_loss = max(self.stop_loss, lock, sma if sma is not None else lock)
                else:
                    lock = self.entry_price - self._risk
                    self.stop_loss = min(self.stop_loss, lock, sma if sma is not None else lock)
                self.target = None
                return StrategyEvent(Signal.NONE, candle.close,
                                     stop_loss=self.stop_loss, note="trailing_activated")
        elif sma is not None:
            self.stop_loss = max(self.stop_loss, sma) if long else min(self.stop_loss, sma)

        return StrategyEvent(Signal.NONE, candle.close)

    def force_exit(self, price, reason: ExitReason = ExitReason.FORCED_EOD) -> StrategyEvent:
        self._reset()
        return StrategyEvent(Signal.EXIT, price, reason=reason)

    def _reset(self):
        self.state = "IDLE"
        self.direction = None
        self.entry_price = None
        self.stop_loss = None
        self.target = None
        self.trailing = False
        self._risk = None
        self.extreme_since_entry = None
