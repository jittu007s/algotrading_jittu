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
    """Opening Range Breakout with the full user-specified trade handling.

    ENTRIES
      - Opening range (OR) = high/low of the first `or_minutes` of the day.
      - LONG (buy CE): a candle CLOSES above the OR high; SHORT (buy PE):
        a close below the OR low. Stop = opposite side of the range; the
        trade is skipped if the range exceeds `max_risk_points`.
      - After a LONG stop-out, the long trigger level becomes the DAY's
        high so far (mirror for shorts) - re-entry must beat it.
      - Once BOTH sides have stopped out, any further entry requires a
        touch of the (updated) OR high/low, then a retracement of at
        least `retrace_points`, then a retest of the same level. The
        retest entry's stop is the retracement extreme.

    MANAGEMENT (rr_shift, per spec)
      - At `risk_reward` x risk (2R): SL shifts to the ENTRY price, the
        target extends to `extended_target_r` x risk (3R), and a
        `timeout_minutes` clock starts. Target -> exit at 3R; SL -> exit
        at entry; clock expiry -> exit at the previous candle's high
        (long) / low (short), falling back to the close.
      - Independent guard: in profit but 2R still missing after
        `be_after_minutes` -> SL shifts to entry.
    """

    def __init__(self, sma_period: int = 20, risk_reward: float = 2.0,
                 or_minutes: int = 15, max_risk_points: float = 60.0,
                 extended_target_r: float = 3.0, timeout_minutes: int = 15,
                 be_after_minutes: int = 30, retrace_points: float = 15.0,
                 stop_mode: str = "opposite"):
        # sma_period retained for constructor compatibility (unused)
        self.risk_reward = risk_reward
        self.or_minutes = or_minutes
        self.max_risk_points = max_risk_points
        self.extended_target_r = extended_target_r
        self.timeout_minutes = timeout_minutes
        self.be_after_minutes = be_after_minutes
        self.retrace_points = retrace_points
        self.stop_mode = stop_mode

        self._session = None
        self._prev = None
        self._reset_session()
        self._reset_position()

    # -- state resets -----------------------------------------------------
    def _reset_session(self):
        self._or_high = None
        self._or_low = None
        self._day_high = None
        self._day_low = None
        self._or_announced = False
        self._long_done = False
        self._short_done = False
        self._long_stopped = False
        self._short_stopped = False
        self._machine = None          # None | wait_touch | wait_retrace | wait_retest
        self._machine_side = None
        self._retrace_extreme = None

    def _reset_position(self):
        self.state = "IDLE"
        self.direction = None
        self.entry_price = None
        self.stop_loss = None
        self.target = None
        self._risk = None
        self._shifted = False
        self._shift_t = None
        self._entry_t = None
        self._from_retest = False
        self.trailing = False   # kept for interface compatibility

    # -- main --------------------------------------------------------------
    def on_closed_candle(self, candle: Candle) -> StrategyEvent:
        from datetime import timedelta

        day = candle.timestamp.date()
        if day != self._session:
            self._session = day
            self._reset_session()
            if self.state == "IN_POSITION":
                self._reset_position()
            self._or_high = candle.high
            self._or_low = candle.low
            self._day_high = candle.high
            self._day_low = candle.low
            self._prev = candle
            return StrategyEvent(Signal.NONE, candle.close)

        self._day_high = max(self._day_high, candle.high)
        self._day_low = min(self._day_low, candle.low)

        session_open = candle.timestamp.replace(hour=9, minute=15, second=0, microsecond=0)
        if (candle.timestamp - session_open).total_seconds() / 60 < self.or_minutes:
            self._or_high = max(self._or_high, candle.high)
            self._or_low = min(self._or_low, candle.low)
            self._prev = candle
            return StrategyEvent(Signal.NONE, candle.close)

        note = None
        if not self._or_announced:
            self._or_announced = True
            rng = self._or_high - self._or_low
            note = f"OR {self._or_low:.1f}-{self._or_high:.1f} ({rng:.1f} pts)"
            if rng > self.max_risk_points:
                note += f" > cap {self.max_risk_points:g} - breakouts will be SKIPPED"

        if self.state == "IN_POSITION":
            ev = self._manage(candle)
            if ev.signal == Signal.EXIT:
                shift_note = self._after_exit(ev.reason)
                if shift_note:
                    ev.note = (ev.note + " | " if ev.note else "") + shift_note
            self._prev = candle
            return ev

        ev = self._entries(candle, note)
        self._prev = candle
        return ev

    # -- entries -----------------------------------------------------------
    def _entries(self, c: Candle, note):
        if self._long_done and self._short_done:
            return StrategyEvent(Signal.NONE, c.close, note=note)

        if self._machine is not None:
            return self._machine_step(c, note)

        rng = self._or_high - self._or_low
        mid = self._or_low + rng / 2
        if c.close > self._or_high and not self._long_done:
            sl = mid if self.stop_mode == "mid_range" else self._or_low
            return self._enter("LONG", c.close, sl, c, note)
        if c.close < self._or_low and not self._short_done:
            sl = mid if self.stop_mode == "mid_range" else self._or_high
            return self._enter("SHORT", c.close, sl, c, note)
        return StrategyEvent(Signal.NONE, c.close, note=note)

    def _machine_step(self, c: Candle, note):
        """Both sides stopped: touch -> retrace -> retest state machine."""
        add = None
        if self._machine == "wait_touch":
            if c.high >= self._or_high and not self._long_done:
                self._machine = "wait_retrace"
                self._machine_side = "LONG"
                self._retrace_extreme = c.low
                add = f"retest: touched high {self._or_high:.1f}, waiting for retracement"
            elif c.low <= self._or_low and not self._short_done:
                self._machine = "wait_retrace"
                self._machine_side = "SHORT"
                self._retrace_extreme = c.high
                add = f"retest: touched low {self._or_low:.1f}, waiting for retracement"
        elif self._machine == "wait_retrace":
            long = self._machine_side == "LONG"
            level = self._or_high if long else self._or_low
            self._retrace_extreme = min(self._retrace_extreme, c.low) if long \
                else max(self._retrace_extreme, c.high)
            retraced = (level - c.low >= self.retrace_points) if long \
                else (c.high - level >= self.retrace_points)
            if retraced:
                self._machine = "wait_retest"
                add = f"retest: retraced {self.retrace_points:g}+ pts, waiting for retest of {level:.1f}"
        elif self._machine == "wait_retest":
            long = self._machine_side == "LONG"
            level = self._or_high if long else self._or_low
            self._retrace_extreme = min(self._retrace_extreme, c.low) if long \
                else max(self._retrace_extreme, c.high)
            hit = c.high >= level if long else c.low <= level
            if hit:
                sl = self._retrace_extreme
                risk = abs(level - sl)
                self._machine = "wait_touch"   # re-arm for a future cycle
                self._machine_side = None
                if 0 < risk <= self.max_risk_points:
                    return self._enter("LONG" if long else "SHORT", level, sl, c, note,
                                       from_retest=True)
                add = f"retest entry skipped: risk {risk:.1f} outside cap"
        if add:
            note = (note + " | " if note else "") + add
        return StrategyEvent(Signal.NONE, c.close, note=note)

    def _enter(self, direction, entry_price, sl, candle, note=None, from_retest=False):
        risk = abs(entry_price - sl)
        if risk <= 0:
            return StrategyEvent(Signal.NONE, candle.close, note=note)
        if risk > self.max_risk_points:
            note = (note + " | " if note else "") + \
                f"{direction} entry at {entry_price:.1f} skipped: stop {sl:.1f} = " \
                f"{risk:.1f} pts risk > cap {self.max_risk_points:g}"
            return StrategyEvent(Signal.NONE, candle.close, note=note)
        self.state = "IN_POSITION"
        self.direction = direction
        self.entry_price = entry_price
        self.stop_loss = sl
        self._risk = risk
        self._shifted = False
        self._shift_t = None
        self._entry_t = candle.timestamp
        self._from_retest = from_retest
        if direction == "LONG":
            self.target = entry_price + self.risk_reward * risk
            sig = Signal.ENTER_LONG_CE
        else:
            self.target = entry_price - self.risk_reward * risk
            sig = Signal.ENTER_SHORT_PE
        if from_retest:
            note = (note + " | " if note else "") + "retest entry"
        return StrategyEvent(sig, entry_price, stop_loss=sl, target=self.target, note=note)

    # -- management (rr_shift + 30-min BE guard) ----------------------------
    def _manage(self, c: Candle) -> StrategyEvent:
        from datetime import timedelta
        long = self.direction == "LONG"

        hit = c.low <= self.stop_loss if long else c.high >= self.stop_loss
        if hit:
            at_entry = (self.stop_loss >= self.entry_price) if long \
                else (self.stop_loss <= self.entry_price)
            reason = ExitReason.BE_STOP if at_entry else ExitReason.STOP_LOSS
            price = self.stop_loss
            self._capture_exit_context()
            self._reset_position()
            return StrategyEvent(Signal.EXIT, price, reason=reason)

        note = None
        if not self._shifted:
            reached = (c.high >= self.entry_price + self.risk_reward * self._risk) if long \
                else (c.low <= self.entry_price - self.risk_reward * self._risk)
            if reached:
                self._shifted = True
                self._shift_t = c.timestamp
                self.stop_loss = self.entry_price
                self.target = (self.entry_price + self.extended_target_r * self._risk) if long \
                    else (self.entry_price - self.extended_target_r * self._risk)
                note = (f"{self.risk_reward:g}R reached: SL->entry {self.stop_loss:.1f}, "
                        f"target {self.target:.1f}, {self.timeout_minutes}-min clock on")
            elif c.timestamp - self._entry_t >= timedelta(minutes=self.be_after_minutes):
                in_profit = c.close > self.entry_price if long else c.close < self.entry_price
                be_set = (self.stop_loss >= self.entry_price) if long \
                    else (self.stop_loss <= self.entry_price)
                if in_profit and not be_set:
                    self.stop_loss = self.entry_price
                    note = (f"{self.be_after_minutes} min in profit without "
                            f"{self.risk_reward:g}R: SL->entry {self.stop_loss:.1f}")

        if self._shifted:
            hit_tgt = c.high >= self.target if long else c.low <= self.target
            if hit_tgt:
                price = self.target
                self._capture_exit_context()
                self._reset_position()
                return StrategyEvent(Signal.EXIT, price, reason=ExitReason.TARGET_3R)
            if c.timestamp >= self._shift_t + timedelta(minutes=self.timeout_minutes):
                lvl = (self._prev.high if long else self._prev.low) if self._prev else c.close
                touched = c.high >= lvl if long else c.low <= lvl
                price = lvl if touched else c.close
                self._capture_exit_context()
                self._reset_position()
                return StrategyEvent(Signal.EXIT, price, reason=ExitReason.TIMEOUT_EXIT)

        return StrategyEvent(Signal.NONE, c.close, note=note)

    def _capture_exit_context(self):
        self._exited_direction = self.direction
        self._exited_from_retest = getattr(self, "_from_retest", False)

    def _after_exit(self, reason) -> str:
        """Book-keeping after our own exit: level shifts and side flags.
        Any stop (initial or breakeven) re-arms the side with the day's
        extreme as its new level; a win retires the side (or, for a
        retest-cycle win, the whole day)."""
        note = ""
        stopped = reason in (ExitReason.STOP_LOSS, ExitReason.BE_STOP)
        direction = self._exited_direction
        from_retest = self._exited_from_retest
        if stopped:
            if direction == "LONG":
                self._long_stopped = True
                self._or_high = self._day_high
                note = f"level shift: long trigger -> day high {self._or_high:.1f}"
            else:
                self._short_stopped = True
                self._or_low = self._day_low
                note = f"level shift: short trigger -> day low {self._or_low:.1f}"
            if self._long_stopped and self._short_stopped and self._machine is None:
                self._machine = "wait_touch"
                note += " | both sides stopped: retest mode ON"
        else:
            if from_retest:
                self._long_done = True
                self._short_done = True
                note = "retest trade won - done for the day"
            elif direction == "LONG":
                self._long_done = True
            else:
                self._short_done = True
        return note

    def force_exit(self, price, reason: ExitReason = ExitReason.FORCED_EOD) -> StrategyEvent:
        self._reset_position()
        return StrategyEvent(Signal.EXIT, price, reason=reason)
