"""Regime-Adaptive confidence-scored strategy.

Design rationale (full write-up in STRATEGY_BLUEPRINT.md): instead of
stacking indicators, one instrument is chosen per orthogonal axis -

  trend      -> EMA(50) side + 15-minute EMA(20) slope (multi-timeframe)
  strength   -> ADX(14): trades only in trending regimes, stands aside in chop
  momentum   -> RSI(14) band + direction (Stoch/CCI/MACD would be redundant)
  volatility -> ATR(14) vs its own average: dead markets and news spikes both
                blocked; the stop distance adapts to ATR
  structure  -> pullback-to-EMA20 then resumption breakout (trade trigger)

A confidence score (0-100) gates entries; risk is managed per-day with a
max-trades cap, a consecutive-loss stop and a daily loss limit in R.

Data honesty: Nifty INDEX candles carry no volume and SmartAPI gives no
order-flow/IV-chain feed, so volume, IV-rank, and dealer-positioning ideas
are deliberately absent rather than faked.

Interface-compatible with SmaCrossOptionStrategy (bot.py / backtesters can
drive any of the three strategies interchangeably).
"""

from collections import deque
from datetime import time as dtime

from strategy import Candle, ExitReason, Signal, StrategyEvent


class _EMA:
    def __init__(self, period):
        self.k = 2 / (period + 1)
        self.value = None

    def update(self, x):
        self.value = x if self.value is None else x * self.k + self.value * (1 - self.k)
        return self.value


class _Wilder:
    """Wilder smoothing used by RSI/ATR/ADX."""
    def __init__(self, period):
        self.period = period
        self.value = None
        self._seed = []

    def update(self, x):
        if self.value is None:
            self._seed.append(x)
            if len(self._seed) >= self.period:
                self.value = sum(self._seed) / self.period
        else:
            self.value = (self.value * (self.period - 1) + x) / self.period
        return self.value


class RegimeAdaptiveStrategy:
    # score weights (sum 100) - see blueprint for the reasoning per block
    W_TREND_ADX = 10
    W_TREND_EMA50 = 10
    W_TREND_MTF = 10
    W_MOM_BAND = 15
    W_MOM_RISING = 10
    W_VOL_ALIVE = 10
    W_VOL_SANE = 10
    W_STRUCT_PULLBACK = 10
    W_STRUCT_BREAK = 15

    def __init__(self, score_threshold=70, adx_min=20.0, atr_period=14, rsi_period=14,
                 atr_stop_mult=1.2, breakeven_r=1.0, trail_r=1.5,
                 max_trades_per_day=3, max_consec_losses=2, daily_loss_limit_r=2.0,
                 time_exit_bars=25, time_exit_min_r=0.5,
                 no_entry_before=dtime(9, 30), no_entry_after=dtime(15, 0)):
        self.score_threshold = score_threshold
        self.adx_min = adx_min
        self.atr_stop_mult = atr_stop_mult
        self.breakeven_r = breakeven_r
        self.trail_r = trail_r
        self.max_trades_per_day = max_trades_per_day
        self.max_consec_losses = max_consec_losses
        self.daily_loss_limit_r = daily_loss_limit_r
        self.time_exit_bars = time_exit_bars
        self.time_exit_min_r = time_exit_min_r
        self.no_entry_before = no_entry_before
        self.no_entry_after = no_entry_after

        # indicators (3-minute)
        self.ema20 = _EMA(20)
        self.ema50 = _EMA(50)
        self._rsi_gain = _Wilder(rsi_period)
        self._rsi_loss = _Wilder(rsi_period)
        self.rsi = None
        self._rsi_prev = None
        self._atr = _Wilder(atr_period)
        self._plus_dm = _Wilder(atr_period)
        self._minus_dm = _Wilder(atr_period)
        self._adx = _Wilder(atr_period)
        self.adx = None
        self._atr_window = deque(maxlen=50)   # ATR history for "alive/sane" bands
        self._recent = deque(maxlen=7)        # (low, high, ema20) for pullback check
        self._prev_candle = None
        self._prev_close = None

        # 15-minute aggregation for the higher-timeframe trend
        self._m15_bucket = None
        self._m15_close = None
        self.ema20_15m = _EMA(20)
        self._m15_history = deque(maxlen=5)

        # day risk state
        self._session = None
        self._trades_today = 0
        self._consec_losses = 0
        self._day_r = 0.0

        # position state (same fields the other strategies expose)
        self.state = "IDLE"
        self.direction = None
        self.entry_price = None
        self.stop_loss = None
        self.target = None       # informational: the +2R reference level
        self.trailing = False
        self._risk = None
        self._breakeven_done = False
        self._bars_in_trade = 0
        self.last_score = None   # for logging/inspection

    # ------------------------------------------------------------------
    def _update_indicators(self, c):
        prev = self._prev_candle
        self.ema20.update(c.close)
        self.ema50.update(c.close)

        if self._prev_close is not None:
            chg = c.close - self._prev_close
            avg_g = self._rsi_gain.update(max(chg, 0.0))
            avg_l = self._rsi_loss.update(max(-chg, 0.0))
            if avg_g is not None and avg_l is not None:
                self._rsi_prev = self.rsi
                self.rsi = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1 + avg_g / avg_l)

        if prev is not None:
            tr = max(c.high - c.low, abs(c.high - prev.close), abs(c.low - prev.close))
            atr = self._atr.update(tr)
            if atr is not None:
                self._atr_window.append(atr)
            up, dn = c.high - prev.high, prev.low - c.low
            pdm = up if (up > dn and up > 0) else 0.0
            mdm = dn if (dn > up and dn > 0) else 0.0
            spdm, smdm = self._plus_dm.update(pdm), self._minus_dm.update(mdm)
            if atr and spdm is not None and smdm is not None and atr > 0:
                pdi, mdi = 100 * spdm / atr, 100 * smdm / atr
                dx = 100 * abs(pdi - mdi) / (pdi + mdi) if (pdi + mdi) > 0 else 0.0
                self.adx = self._adx.update(dx)

        # 15-minute close series (buckets aligned to 9:15)
        minutes = (c.timestamp.hour - 9) * 60 + c.timestamp.minute - 15
        bucket = minutes // 15
        if self._m15_bucket is not None and bucket != self._m15_bucket and self._m15_close is not None:
            v = self.ema20_15m.update(self._m15_close)
            if v is not None:
                self._m15_history.append(v)
        self._m15_bucket = bucket
        self._m15_close = c.close

        if self.ema20.value is not None:
            self._recent.append((c.low, c.high, self.ema20.value))
        self._prev_candle = c
        self._prev_close = c.close

    def _mtf_trend(self):
        """+1 rising 15m EMA20, -1 falling, 0 unknown/flat."""
        h = self._m15_history
        if len(h) < 3:
            return 0
        if h[-1] > h[-3]:
            return 1
        if h[-1] < h[-3]:
            return -1
        return 0

    def _atr_state(self):
        """(alive, sane): ATR not dead vs its 50-bar mean, and not a spike."""
        if self._atr.value is None or len(self._atr_window) < 20:
            return False, False
        mean = sum(self._atr_window) / len(self._atr_window)
        a = self._atr.value
        return a >= 0.8 * mean, a <= 2.5 * mean

    def _score(self, c, direction):
        long = direction == "LONG"
        s = 0
        if self.adx is not None and self.adx >= self.adx_min:
            s += self.W_TREND_ADX
        if self.ema50.value is not None and ((c.close > self.ema50.value) if long else (c.close < self.ema50.value)):
            s += self.W_TREND_EMA50
        if self._mtf_trend() == (1 if long else -1):
            s += self.W_TREND_MTF
        if self.rsi is not None and ((50 <= self.rsi <= 72) if long else (28 <= self.rsi <= 50)):
            s += self.W_MOM_BAND
        if self.rsi is not None and self._rsi_prev is not None and \
                ((self.rsi > self._rsi_prev) if long else (self.rsi < self._rsi_prev)):
            s += self.W_MOM_RISING
        alive, sane = self._atr_state()
        if alive:
            s += self.W_VOL_ALIVE
        if sane:
            s += self.W_VOL_SANE
        # structure: pulled back to EMA20 recently, then breaks the previous bar
        recent = list(self._recent)[:-1]
        if long:
            if any(lo <= e for lo, _hi, e in recent[-6:]):
                s += self.W_STRUCT_PULLBACK
            if self._prev_ref is not None and c.close > self._prev_ref.high and c.close > self.ema20.value:
                s += self.W_STRUCT_BREAK
        else:
            if any(hi >= e for _lo, hi, e in recent[-6:]):
                s += self.W_STRUCT_PULLBACK
            if self._prev_ref is not None and c.close < self._prev_ref.low and c.close < self.ema20.value:
                s += self.W_STRUCT_BREAK
        return s

    # ------------------------------------------------------------------
    def on_closed_candle(self, candle: Candle) -> StrategyEvent:
        # keep a reference to the candle BEFORE this one for the breakout check
        self._prev_ref = self._prev_candle
        self._update_indicators(candle)

        day = candle.timestamp.date()
        if day != self._session:
            self._session = day
            self._trades_today = 0
            self._consec_losses = 0
            self._day_r = 0.0
            if self.state == "IN_POSITION":
                self._reset_position()

        if self.state == "IN_POSITION":
            return self._manage(candle)

        # ---- entry gates -------------------------------------------------
        t = candle.timestamp.time()
        if not (self.no_entry_before <= t < self.no_entry_after):
            return StrategyEvent(Signal.NONE, candle.close)
        if (self._trades_today >= self.max_trades_per_day
                or self._consec_losses >= self.max_consec_losses
                or self._day_r <= -self.daily_loss_limit_r):
            return StrategyEvent(Signal.NONE, candle.close)
        if self.adx is None or self.adx < self.adx_min:     # ranging regime: stand aside
            return StrategyEvent(Signal.NONE, candle.close)
        if self.ema20.value is None or self.ema50.value is None or self._atr.value is None:
            return StrategyEvent(Signal.NONE, candle.close)

        direction = "LONG" if candle.close > self.ema50.value else "SHORT"

        # hard trigger: the structural breakout must be present - the score
        # alone (trend+momentum+vol) must never fire an entry mid-bar-nowhere
        if self._prev_ref is None:
            return StrategyEvent(Signal.NONE, candle.close)
        if direction == "LONG":
            has_break = candle.close > self._prev_ref.high and candle.close > self.ema20.value
        else:
            has_break = candle.close < self._prev_ref.low and candle.close < self.ema20.value
        if not has_break:
            return StrategyEvent(Signal.NONE, candle.close)

        score = self._score(candle, direction)
        self.last_score = score
        if score < self.score_threshold:
            return StrategyEvent(Signal.NONE, candle.close)

        entry = candle.close
        risk = self.atr_stop_mult * self._atr.value
        if direction == "LONG":
            sl = entry - risk
            self.target = entry + 2 * risk
            sig = Signal.ENTER_LONG_CE
        else:
            sl = entry + risk
            self.target = entry - 2 * risk
            sig = Signal.ENTER_SHORT_PE

        self.state = "IN_POSITION"
        self.direction = direction
        self.entry_price = entry
        self.stop_loss = sl
        self._risk = risk
        self.trailing = False
        self._breakeven_done = False
        self._bars_in_trade = 0
        self._trades_today += 1
        return StrategyEvent(sig, entry, stop_loss=sl, target=self.target,
                             note=f"score={score}")

    def _manage(self, c):
        long = self.direction == "LONG"
        self._bars_in_trade += 1

        hit_sl = c.low <= self.stop_loss if long else c.high >= self.stop_loss
        if hit_sl:
            return self._close(self.stop_loss,
                               ExitReason.TRAILING_STOP if self.trailing else ExitReason.STOP_LOSS)

        move = (c.high - self.entry_price) if long else (self.entry_price - c.low)
        r = move / self._risk

        if not self._breakeven_done and r >= self.breakeven_r:
            self.stop_loss = max(self.stop_loss, self.entry_price) if long \
                else min(self.stop_loss, self.entry_price)
            self._breakeven_done = True
        if not self.trailing and r >= self.trail_r:
            self.trailing = True
        if self.trailing and self.ema20.value is not None:
            self.stop_loss = max(self.stop_loss, self.ema20.value) if long \
                else min(self.stop_loss, self.ema20.value)

        # theta guard: a trade that goes nowhere for `time_exit_bars` candles
        # is bleeding option premium - cut it.
        unreal = ((c.close - self.entry_price) if long else (self.entry_price - c.close)) / self._risk
        if not self.trailing and self._bars_in_trade >= self.time_exit_bars and unreal < self.time_exit_min_r:
            return self._close(c.close, ExitReason.TIME_EXIT)

        return StrategyEvent(Signal.NONE, c.close)

    def _close(self, price, reason):
        realized = ((price - self.entry_price) if self.direction == "LONG"
                    else (self.entry_price - price)) / self._risk
        self._day_r += realized
        self._consec_losses = self._consec_losses + 1 if realized < 0 else 0
        self._reset_position()
        return StrategyEvent(Signal.EXIT, price, reason=reason)

    def force_exit(self, price, reason: ExitReason = ExitReason.FORCED_EOD) -> StrategyEvent:
        if self.state == "IN_POSITION" and price is not None and self._risk:
            realized = ((price - self.entry_price) if self.direction == "LONG"
                        else (self.entry_price - price)) / self._risk
            self._day_r += realized
        self._reset_position()
        return StrategyEvent(Signal.EXIT, price, reason=reason)

    def _reset_position(self):
        self.state = "IDLE"
        self.direction = None
        self.entry_price = None
        self.stop_loss = None
        self.target = None
        self.trailing = False
        self._risk = None
        self._breakeven_done = False
        self._bars_in_trade = 0
