# Broker-agnostic core logic for the SMA-crossover option strategy.
#
# Kept independent of the Angel One SmartAPI client so it can be unit tested
# and backtested (see backtest.py) without any network/broker access.
#
# Rules implemented, symmetrically for both directions (see README.md):
#
#   The moving average is an SMMA (smoothed moving average, a.k.a. RMA /
#   Wilder's MA) - the same line as TradingView's "SMMA 20 close" - not a
#   simple mean.
#
#   LONG (buy ATM CE):
#     0. Fresh-cross filter: at least one candle must have CLOSED BELOW the
#        SMMA since the last long setup, so only genuine cross-up setups
#        arm - not every pair of closes in an ongoing uptrend.
#     1. Two consecutive *closed* candles close above the SMMA -> setup armed.
#     2. Entry: a later candle's high crosses above the high of the 2nd
#        (more recent) of those two candles.
#     3. Stop loss: low of the candle immediately before the entry candle.
#     4. Target level: entry + risk_reward * (entry - stop_loss). Reaching
#        it does NOT close the trade - it switches to TRAILING mode: the
#        stop jumps to lock at least +1R (entry + risk) and from then on
#        ratchets up along the SMMA each closed candle, never loosening.
#        The trade exits when price trades back to the trailed stop.
#     5. Early exit (only while the target has not yet been reached): price
#        touches the SMMA twice while retracing down from the highest point
#        made after entry.
#
#   SHORT (buy ATM PE): the exact mirror — a fresh cross-down, two closes
#   below the SMMA, entry on a break below the 2nd candle's low, stop loss
#   at the previous candle's high; on reaching the 1:2 level the stop locks
#   -1R below entry and trails the SMMA downward.

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
    BE_STOP = "be_stop"
    TARGET_HIT = "target_hit"
    TARGET_3R = "target_3r"
    TIMEOUT_EXIT = "timeout_prev_extreme"
    TRAILING_STOP = "trailing_stop"
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
        self._seed_closes = []    # first `sma_period` closes seed the SMMA
        self._smma = None

        # Fresh-cross tracking: a long setup needs a close below the SMMA
        # since the last long arm (mirror for shorts), so we only trade
        # genuine crossovers, not every candle pair inside a trend.
        self._seen_close_below = False
        self._seen_close_above = False

        self.state = "IDLE"           # IDLE -> ARMED -> IN_POSITION
        self.direction = None         # "LONG" or "SHORT" while ARMED / IN_POSITION
        self.trigger_level = None  # high (long) / low (short) of the 2nd setup candle

        self.entry_price = None
        self.stop_loss = None
        self.target = None
        self.trailing = False    # True once the 1:2 level was reached
        self._risk = None         # per-trade risk (|entry - initial SL|)
        self.extreme_since_entry = None   # highest high (long) / lowest low (short)
        self.sma_touch_count = 0
        self._currently_touching_sma = False

    # -------- SMA computation -----------------
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

    # -------- Main logic -----------------
    def on_closed_candle(self, candle: Candle) -> StrategyEvent:
        sma = self._update_smma(candle.close)
        self._candles.append(candle)
        self._sma_history.append(sma)

        # Track which side of the SMA closes land on, in every state, so a
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

    # -------- IDLE state: waiting for setup arm ------
    def _process_idle(self, candle: Candle, sma: Optional[float]) -> StrategyEvent:
        """In IDLE, scan for a fresh setup to arm."""
        if sma is None or len(self._candles) < 2:
            return StrategyEvent(Signal.NONE, None)

        long_setup = (self._seen_close_below and
                      candle.close > sma and self._candles[-2].close > sma)
        short_setup = (self._seen_close_above and
                       candle.close < sma and self._candles[-2].close < sma)

        if long_setup:
            self.state = "ARMED"
            self.direction = "LONG"
            self.trigger_level = self._candles[-2].high
            self._seen_close_below = False
            return StrategyEvent(Signal.NONE, None, note="long arm")
        elif short_setup:
            self.state = "ARMED"
            self.direction = "SHORT"
            self.trigger_level = self._candles[-2].low
            self._seen_close_above = False
            return StrategyEvent(Signal.NONE, None, note="short arm")

        return StrategyEvent(Signal.NONE, None)

    # -------- ARMED state: waiting for entry trigger ------
    def _process_armed(self, candle: Candle, sma: Optional[float]) -> StrategyEvent:
        """In ARMED, wait for the entry trigger (break of the 2nd candle's extreme)."""
        if self.direction == "LONG":
            if candle.high >= self.trigger_level:
                # Entry triggered
                self.entry_price = self.trigger_level
                self.stop_loss = self._candles[-2].low
                self._risk = self.entry_price - self.stop_loss
                self.target = self.entry_price + self.risk_reward * self._risk
                self.extreme_since_entry = candle.high
                self.sma_touch_count = 0
                self._currently_touching_sma = False
                self.state = "IN_POSITION"
                self.trailing = False
                return StrategyEvent(Signal.ENTER_LONG_CE, self.entry_price,
                                     stop_loss=self.stop_loss, target=self.target,
                                     note="long entry")
        elif self.direction == "SHORT":
            if candle.low <= self.trigger_level:
                # Entry triggered
                self.entry_price = self.trigger_level
                self.stop_loss = self._candles[-2].high
                self._risk = self.stop_loss - self.entry_price
                self.target = self.entry_price - self.risk_reward * self._risk
                self.extreme_since_entry = candle.low
                self.sma_touch_count = 0
                self._currently_touching_sma = False
                self.state = "IN_POSITION"
                self.trailing = False
                return StrategyEvent(Signal.ENTER_SHORT_PE, self.entry_price,
                                     stop_loss=self.stop_loss, target=self.target,
                                     note="short entry")

        return StrategyEvent(Signal.NONE, None)

    # -------- IN_POSITION state: manage the trade ------
    def _process_in_position(self, candle: Candle, sma: Optional[float]) -> StrategyEvent:
        """In IN_POSITION, manage stop/target/exit logic."""
        if self.direction == "LONG":
            # Check stop loss
            if candle.low <= self.stop_loss:
                self._reset()
                return StrategyEvent(Signal.EXIT, self.stop_loss,
                                     reason=ExitReason.STOP_LOSS, note="long SL hit")

            # Check target or trailing stop
            if not self.trailing:
                # FIX: Added None check for self.target before comparison
                if self.target is not None and candle.high >= self.target:
                    self.trailing = True
                    self.stop_loss = self.entry_price + self._risk  # lock +1R
                    return StrategyEvent(Signal.NONE, None, note="long target hit, entering trail")
            else:
                # Trailing mode
                if sma is not None and (candle.close < sma or candle.low < sma):
                    self._currently_touching_sma = True
                    self.sma_touch_count += 1
                    if self.sma_touch_count >= 2:
                        self._reset()
                        return StrategyEvent(Signal.EXIT, candle.close,
                                             reason=ExitReason.SMA_DOUBLE_TOUCH, note="long 2x SMA touch in trail")
                else:
                    self._currently_touching_sma = False

                # Trail the stop up along the SMA
                if sma is not None:
                    new_stop = max(sma - self._risk * 0.5, self.stop_loss)  # gentle trail
                    self.stop_loss = max(self.stop_loss, new_stop)

                # Check trailing stop
                if candle.low <= self.stop_loss:
                    self._reset()
                    return StrategyEvent(Signal.EXIT, self.stop_loss,
                                         reason=ExitReason.TRAILING_STOP, note="long trailing SL hit")

            self.extreme_since_entry = max(self.extreme_since_entry, candle.high)

        elif self.direction == "SHORT":
            # Check stop loss
            if candle.high >= self.stop_loss:
                self._reset()
                return StrategyEvent(Signal.EXIT, self.stop_loss,
                                     reason=ExitReason.STOP_LOSS, note="short SL hit")

            # Check target or trailing stop
            if not self.trailing:
                # FIX: Added None check for self.target before comparison
                if self.target is not None and candle.low <= self.target:
                    self.trailing = True
                    self.stop_loss = self.entry_price - self._risk  # lock -1R
                    return StrategyEvent(Signal.NONE, None, note="short target hit, entering trail")
            else:
                # Trailing mode
                if sma is not None and (candle.close > sma or candle.high > sma):
                    self._currently_touching_sma = True
                    self.sma_touch_count += 1
                    if self.sma_touch_count >= 2:
                        self._reset()
                        return StrategyEvent(Signal.EXIT, candle.close,
                                             reason=ExitReason.SMA_DOUBLE_TOUCH, note="short 2x SMA touch in trail")
                else:
                    self._currently_touching_sma = False

                # Trail the stop down along the SMA
                if sma is not None:
                    new_stop = min(sma + self._risk * 0.5, self.stop_loss)  # gentle trail
                    self.stop_loss = min(self.stop_loss, new_stop)

                # Check trailing stop
                if candle.high >= self.stop_loss:
                    self._reset()
                    return StrategyEvent(Signal.EXIT, self.stop_loss,
                                         reason=ExitReason.TRAILING_STOP, note="short trailing SL hit")

            self.extreme_since_entry = min(self.extreme_since_entry, candle.low)

        return StrategyEvent(Signal.NONE, None)

    def force_exit(self, price: Optional[float], reason: ExitReason = ExitReason.FORCED_EOD) -> StrategyEvent:
        """Force an immediate exit, typically at end-of-day."""
        if self.state != "IN_POSITION":
            self._reset()
            return StrategyEvent(Signal.NONE, None)

        exit_price = price if price is not None else self.entry_price
        self._reset()
        return StrategyEvent(Signal.EXIT, exit_price, reason=reason, note=f"forced exit ({reason.value})")

    def _reset(self):
        """Return to IDLE state and clear position state."""
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
