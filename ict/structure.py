"""Pure setup-detection logic: swings, liquidity sweeps, market structure
shifts (MSS), and fair value gaps (FVG).

Everything here is deterministic and broker-free so it can be unit tested
against synthetic candles (tests/test_structure.py). "ICT" has no canonical
specification - the definitions implemented are the ones in the project
brief, stated precisely:

  swing (fractal): candle whose high (low) is the strict maximum (minimum)
      of the k candles on each side. Confirmed only k candles later.
  liquidity sweep: a candle pierces >= `min_pierce` points beyond a
      liquidity level (prior 5m swing, previous-day high/low, opening-range
      extreme) and, if `needs_rejection`, closes back on the level's side.
  MSS: after a sweep, a candle whose body >= `body_mult` x the average body
      of the last `avg_body_period` candles CLOSES beyond the most recent
      opposing 5m swing in the trade direction.
  FVG: in the displacement leg, candle triple (i-2, i-1, i) where
      low(i) > high(i-2) (bullish) leaves the imbalance
      [high(i-2), low(i)]; mirror for bearish.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from typing import Deque, List, Optional

from .models import (FVG, MSS, Bias, Candle, LiquidityLevel, Setup, Sweep,
                     Swing, SwingKind)

logger = logging.getLogger(__name__)


def find_swings(candles: List[Candle], k: int) -> List[Swing]:
    """All confirmed fractal swings in a series (strict extremes over k
    candles each side)."""
    swings: List[Swing] = []
    for i in range(k, len(candles) - k):
        window = candles[i - k:i + k + 1]
        c = candles[i]
        if c.high == max(w.high for w in window) and \
                sum(1 for w in window if w.high == c.high) == 1:
            swings.append(Swing(SwingKind.HIGH, c.high, c.timestamp, i))
        if c.low == min(w.low for w in window) and \
                sum(1 for w in window if w.low == c.low) == 1:
            swings.append(Swing(SwingKind.LOW, c.low, c.timestamp, i))
    return swings


def detect_bias(candles: List[Candle], k: int) -> Bias:
    """Bias from the last two confirmed swing highs and lows:
    HH + HL -> bullish, LH + LL -> bearish, anything else neutral."""
    swings = find_swings(candles, k)
    highs = [s for s in swings if s.kind == SwingKind.HIGH][-2:]
    lows = [s for s in swings if s.kind == SwingKind.LOW][-2:]
    if len(highs) < 2 or len(lows) < 2:
        return Bias.NEUTRAL
    hh = highs[1].price > highs[0].price
    hl = lows[1].price > lows[0].price
    lh = highs[1].price < highs[0].price
    ll = lows[1].price < lows[0].price
    if hh and hl:
        return Bias.BULLISH
    if lh and ll:
        return Bias.BEARISH
    return Bias.NEUTRAL


def combine_bias(daily: Bias, intraday: Bias) -> Bias:
    """Daily and 1H must not conflict; intraday leads when daily is neutral."""
    if daily == Bias.NEUTRAL:
        return intraday
    if intraday == Bias.NEUTRAL or intraday == daily:
        return daily
    return Bias.NEUTRAL


class SetupScanner:
    """Incremental 5-minute scanner: feed closed candles, receive a Setup
    when a sweep -> MSS -> FVG sequence completes in the bias direction.

    State machine per bias direction:
      WAIT_SWEEP -> (liquidity taken + rejection) -> WAIT_MSS
      WAIT_MSS   -> (displacement close through opposing swing) -> scan FVG
      FVG found  -> Setup emitted; caller handles entry on FVG retrace.
    A stage expires `validity` candles after the sweep.
    """

    def __init__(self, swing_k: int = 2, min_pierce: float = 3.0,
                 needs_rejection: bool = True, body_mult: float = 1.5,
                 avg_body_period: int = 20, validity: int = 12,
                 max_history: int = 400):
        self.swing_k = swing_k
        self.min_pierce = min_pierce
        self.needs_rejection = needs_rejection
        self.body_mult = body_mult
        self.avg_body_period = avg_body_period
        self.validity = validity

        self.candles: List[Candle] = []
        self.max_history = max_history
        self.static_levels: List[LiquidityLevel] = []   # PDH/PDL/OR extremes
        self._bodies: Deque[float] = deque(maxlen=avg_body_period)

        self._pending_sweep: Optional[Sweep] = None
        self._sweep_index: Optional[int] = None
        self.stats: Counter = Counter()   # funnel: sweeps / mss_no_fvg / setups / expired

    # -- context ---------------------------------------------------------
    def set_static_levels(self, levels: List[LiquidityLevel]) -> None:
        self.static_levels = list(levels)

    def _swings(self) -> List[Swing]:
        return find_swings(self.candles, self.swing_k)

    def _avg_body(self) -> Optional[float]:
        if len(self._bodies) < self.avg_body_period:
            return None
        return sum(self._bodies) / len(self._bodies)

    def _liquidity(self, bias: Bias) -> List[LiquidityLevel]:
        """Levels whose stops the sweep should take: below price for longs
        (sell-side), above for shorts (buy-side)."""
        want = SwingKind.LOW if bias == Bias.BULLISH else SwingKind.HIGH
        levels = [lv for lv in self.static_levels if lv.kind == want]
        for s in self._swings()[-12:]:
            if s.kind == want:
                levels.append(LiquidityLevel(s.price, want, "swing_" + want.value))
        return levels

    # -- main ------------------------------------------------------------
    def on_candle(self, candle: Candle, bias: Bias) -> Optional[Setup]:
        self.candles.append(candle)
        if len(self.candles) > self.max_history:
            del self.candles[:len(self.candles) - self.max_history]
        self._bodies.append(candle.body)
        i = len(self.candles) - 1

        if bias == Bias.NEUTRAL:
            self._pending_sweep = None
            return None

        # expire a stale sweep
        if self._pending_sweep is not None and self._sweep_index is not None \
                and i - self._sweep_index > self.validity:
            logger.debug("sweep expired without MSS")
            self.stats["sweep_expired_no_mss"] += 1
            self._pending_sweep = None

        if self._pending_sweep is None:
            self._detect_sweep(candle, i, bias)
            return None

        return self._detect_mss_and_fvg(candle, i, bias)

    # -- stages ----------------------------------------------------------
    def _detect_sweep(self, c: Candle, i: int, bias: Bias) -> None:
        for level in self._liquidity(bias):
            if bias == Bias.BULLISH:
                pierced = c.low <= level.price - self.min_pierce
                rejected = (not self.needs_rejection) or c.close > level.price
            else:
                pierced = c.high >= level.price + self.min_pierce
                rejected = (not self.needs_rejection) or c.close < level.price
            if pierced and rejected:
                extreme = c.low if bias == Bias.BULLISH else c.high
                self._pending_sweep = Sweep(level, extreme, c.timestamp, i)
                self._sweep_index = i
                self.stats["sweeps"] += 1
                logger.info("sweep of %s @ %.2f (extreme %.2f)", level.label, level.price, extreme)
                return

    def _detect_mss_and_fvg(self, c: Candle, i: int, bias: Bias) -> Optional[Setup]:
        avg = self._avg_body()
        if avg is None or avg <= 0:
            return None
        if c.body < self.body_mult * avg:
            return None

        # opposing swing to break: most recent swing high (long) / low (short)
        # confirmed BEFORE the sweep candle
        want = SwingKind.HIGH if bias == Bias.BULLISH else SwingKind.LOW
        candidates = [s for s in self._swings()
                      if s.kind == want and s.index <= (self._sweep_index or 0)]
        if not candidates:
            return None
        swing = candidates[-1]

        broke = c.close > swing.price if bias == Bias.BULLISH else c.close < swing.price
        if not broke:
            return None

        fvg = self._find_fvg(i, bias)
        if fvg is None:
            logger.info("MSS without FVG - no entry basis, discarding sweep")
            self.stats["mss_without_fvg"] += 1
            self._pending_sweep = None
            return None

        sweep = self._pending_sweep
        assert sweep is not None
        stop = sweep.extreme - self.min_pierce if bias == Bias.BULLISH \
            else sweep.extreme + self.min_pierce
        setup = Setup(bias=bias, sweep=sweep, mss=MSS(swing, i, c.timestamp),
                      fvg=fvg, stop_spot=round(stop, 2), created_index=i)
        self.stats["setups"] += 1
        logger.info("setup complete: %s FVG %.2f-%.2f stop %.2f",
                    bias.value, fvg.low, fvg.high, setup.stop_spot)
        self._pending_sweep = None
        return setup

    def _find_fvg(self, i: int, bias: Bias) -> Optional[FVG]:
        """Newest FVG formed in the displacement leg (since the sweep)."""
        start = max((self._sweep_index or 0), 2)
        for j in range(i, start - 1, -1):
            c1, c3 = self.candles[j - 2], self.candles[j]
            if bias == Bias.BULLISH and c3.low > c1.high:
                return FVG(low=c1.high, high=c3.low, created_index=j, timestamp=c3.timestamp)
            if bias == Bias.BEARISH and c3.high < c1.low:
                return FVG(low=c3.high, high=c1.low, created_index=j, timestamp=c3.timestamp)
        return None


def entry_level(setup: Setup, mode: str = "midpoint") -> float:
    """Where the resting entry sits: 'midpoint' of the FVG (the brief's
    default) or its near 'edge' (first touch of the gap - fills far more
    setups at slightly worse prices)."""
    if mode == "edge":
        return setup.fvg.high if setup.bias == Bias.BULLISH else setup.fvg.low
    return setup.fvg.midpoint


def entry_triggered(setup: Setup, candle: Candle, mode: str = "midpoint") -> bool:
    """Entry: first pullback into the FVG entry level."""
    level = entry_level(setup, mode)
    return candle.low <= level <= candle.high


def setup_invalidated(setup: Setup, candle: Candle) -> bool:
    """Setup dies if price trades back beyond the sweep-anchored stop
    before the entry fills."""
    if setup.bias == Bias.BULLISH:
        return candle.close < setup.stop_spot
    return candle.close > setup.stop_spot
