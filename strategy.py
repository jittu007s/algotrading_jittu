"""Broker-agnostic core logic for the SMA-crossover option strategy.

Kept independent of the Angel One SmartAPI client so it can be unit tested
and backtested (see backtest.py) without any network/broker access.

Rules implemented (see README.md for the full spec and the assumptions
made to turn the English description into precise logic):

  1. Two consecutive *closed* candles close above the SMA -> setup "armed".
  2. Entry trigger: a later candle's high crosses above the high of the
     2nd (more recent) of those two candles -> BUY 1 lot ATM CE.
  3. Stop loss: most recent confirmed swing low (fractal low) before entry.
  4. Target: 1:2 risk/reward measured on the underlying, off the entry
     price and stop loss.
  5. Early exit: price touches the SMA twice while retracing from the
     highest point made after entry ("touch back SMA twice while falling
     from top").
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
    EXIT = auto()


class ExitReason(Enum):
    STOP_LOSS = "stop_loss"
    TARGET = "target"
    SMA_DOUBLE_TOUCH = "sma_double_touch"
    FORCED_EOD = "forced_eod"


class StrategyEvent:
    def __init__(self, signal: Signal, price: Optional[float], reason: Optional[ExitReason] = None,
                 stop_loss: Optional[float] = None, target: Optional[float] = None):
        self.signal = signal
        self.price = price
        self.reason = reason
        self.stop_loss = stop_loss
        self.target = target

    def __repr__(self):
        return (f"StrategyEvent(signal={self.signal}, price={self.price}, "
                f"reason={self.reason}, stop_loss={self.stop_loss}, target={self.target})")


class SmaCrossOptionStrategy:
    def __init__(self, sma_period: int = 20, swing_fractal: int = 2, swing_lookback: int = 30,
                 risk_reward: float = 2.0):
        self.sma_period = sma_period
        self.swing_fractal = swing_fractal
        self.swing_lookback = swing_lookback
        self.risk_reward = risk_reward

        history_len = max(200, swing_lookback + swing_fractal * 2 + 5)
        self._candles = deque(maxlen=history_len)
        self._sma_history = deque(maxlen=history_len)
        self._closes = deque(maxlen=sma_period)

        self.state = "IDLE"  # IDLE -> ARMED -> IN_POSITION
        self.trigger_high = None
        self._pending_sl = None

        self.entry_price = None
        self.stop_loss = None
        self.target = None
        self.peak_since_entry = None
        self.sma_touch_count = 0
        self._currently_touching_sma = False

    # ------------------------------------------------------------------
    def _sma(self):
        if len(self._closes) < self.sma_period:
            return None
        return sum(self._closes) / self.sma_period

    def _find_swing_low(self):
        """Most recent confirmed fractal low: a candle whose low is the
        lowest among `swing_fractal` candles on either side of it."""
        candles = list(self._candles)[-self.swing_lookback:]
        n = self.swing_fractal
        for i in range(len(candles) - n - 1, n - 1, -1):
            window = candles[i - n:i + n + 1]
            if candles[i].low == min(c.low for c in window):
                return candles[i].low
        return min((c.low for c in candles), default=None)

    # ------------------------------------------------------------------
    def on_closed_candle(self, candle: Candle) -> StrategyEvent:
        self._closes.append(candle.close)
        sma = self._sma()
        self._candles.append(candle)
        self._sma_history.append(sma)

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

        if prev_candle.close > prev_sma and candle.close > sma:
            self.state = "ARMED"
            self.trigger_high = candle.high  # high of the 2nd (latest) candle
            self._pending_sl = self._find_swing_low()

        return StrategyEvent(Signal.NONE, candle.close)

    def _process_armed(self, candle, sma):
        # Setup invalidated if price closes back below the SMA before triggering.
        if sma is not None and candle.close < sma:
            self.state = "IDLE"
            self.trigger_high = None
            self._pending_sl = None
            return StrategyEvent(Signal.NONE, candle.close)

        if candle.high >= self.trigger_high:
            entry_price = self.trigger_high
            sl = self._pending_sl if self._pending_sl is not None else candle.low
            risk = entry_price - sl
            if risk <= 0:
                risk = entry_price * 0.01  # degenerate fallback, shouldn't normally happen
                sl = entry_price - risk
            target = entry_price + self.risk_reward * risk

            self.state = "IN_POSITION"
            self.entry_price = entry_price
            self.stop_loss = sl
            self.target = target
            self.peak_since_entry = candle.high
            self.sma_touch_count = 0
            self._currently_touching_sma = False

            return StrategyEvent(Signal.ENTER_LONG_CE, entry_price, stop_loss=sl, target=target)

        return StrategyEvent(Signal.NONE, candle.close)

    def _process_in_position(self, candle, sma):
        if candle.low <= self.stop_loss:
            exit_price = self.stop_loss
            self._reset()
            return StrategyEvent(Signal.EXIT, exit_price, reason=ExitReason.STOP_LOSS)

        if candle.high >= self.target:
            exit_price = self.target
            self._reset()
            return StrategyEvent(Signal.EXIT, exit_price, reason=ExitReason.TARGET)

        made_new_peak = candle.high > self.peak_since_entry
        self.peak_since_entry = max(self.peak_since_entry, candle.high)

        touching = sma is not None and candle.low <= sma <= candle.high
        if not made_new_peak and touching and not self._currently_touching_sma:
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
        self.trigger_high = None
        self._pending_sl = None
        self.entry_price = None
        self.stop_loss = None
        self.target = None
        self.peak_since_entry = None
        self.sma_touch_count = 0
        self._currently_touching_sma = False
