"""Unit tests for swing/sweep/MSS/FVG detection on synthetic candles.

Run from algo-trading/:  python -m unittest discover ict/tests -v
"""

import unittest
from datetime import datetime, timedelta

from ict.models import Bias, Candle, LiquidityLevel, SwingKind
from ict.structure import (SetupScanner, combine_bias, detect_bias,
                           entry_triggered, find_swings, setup_invalidated)

T0 = datetime(2026, 7, 14, 9, 15)


def c(i: int, o: float, h: float, l: float, cl: float, step_min: int = 5) -> Candle:
    return Candle(T0 + timedelta(minutes=step_min * i), o, h, l, cl)


def flat(i: int, px: float, wobble: float = 2.0) -> Candle:
    return c(i, px, px + wobble, px - wobble, px + 0.5)


class TestSwings(unittest.TestCase):
    def test_finds_isolated_high_and_low(self):
        candles = [flat(0, 100), flat(1, 100), c(2, 100, 120, 99, 101),
                   flat(3, 100), flat(4, 100),
                   c(5, 100, 101, 80, 100), flat(6, 100), flat(7, 100)]
        swings = find_swings(candles, k=2)
        highs = [s for s in swings if s.kind == SwingKind.HIGH]
        lows = [s for s in swings if s.kind == SwingKind.LOW]
        self.assertEqual([s.price for s in highs], [120])
        self.assertEqual([s.price for s in lows], [80])
        self.assertEqual(highs[0].index, 2)
        self.assertEqual(lows[0].index, 5)

    def test_needs_k_candles_each_side(self):
        candles = [flat(0, 100), c(1, 100, 130, 99, 100)]  # high at the edge
        self.assertEqual(find_swings(candles, k=2), [])


class TestBias(unittest.TestCase):
    def _trend(self, up: bool):
        candles = []
        px = 100.0
        for i in range(40):
            leg = (i // 5) % 2 == 0
            d = (1.6 if leg else -0.7) if up else (-1.6 if leg else 0.7)
            px += d * 3
            candles.append(c(i, px, px + 4, px - 4, px + (1 if up else -1)))
        return candles

    def test_uptrend_is_bullish(self):
        self.assertEqual(detect_bias(self._trend(True), k=2), Bias.BULLISH)

    def test_downtrend_is_bearish(self):
        self.assertEqual(detect_bias(self._trend(False), k=2), Bias.BEARISH)

    def test_combine_conflict_is_neutral(self):
        self.assertEqual(combine_bias(Bias.BULLISH, Bias.BEARISH), Bias.NEUTRAL)
        self.assertEqual(combine_bias(Bias.NEUTRAL, Bias.BEARISH), Bias.BEARISH)
        self.assertEqual(combine_bias(Bias.BULLISH, Bias.NEUTRAL), Bias.BULLISH)


def bullish_sequence(scanner: SetupScanner):
    """Feed a canonical bullish sweep->MSS->FVG sequence; returns the setup
    and the index it fired at. Layout:
      - 24 quiet candles around 100 (seeds avg body, forms swing low 94 and
        swing high ~103)
      - sweep candle: wick to 89 (through swing low 94 by >3), close 101
      - displacement candle: big body closing over the last swing high,
        leaving an FVG
    """
    setup = None
    # quiet base with a defined swing low at 94 (idx 10) and swing high 103 (idx 16)
    for i in range(24):
        if i == 10:
            cd = c(i, 100, 101, 94, 100)      # swing low 94
        elif i == 16:
            cd = c(i, 100, 103, 99, 100.5)    # swing high 103
        else:
            cd = flat(i, 100, 1.5)
        setup = scanner.on_candle(cd, Bias.BULLISH) or setup
    # sweep: pierce 94 by >=3 and reject back above
    setup = scanner.on_candle(c(24, 100, 100.5, 89, 100.8), Bias.BULLISH) or setup
    # small pause candle (also candle1 of the future FVG)
    setup = scanner.on_candle(c(25, 100.8, 101.5, 100.2, 101), Bias.BULLISH) or setup
    # displacement: body 101 -> 110 (huge vs avg), closes over swing high 103,
    # candle low 104 > candle-1(idx 24...) wait: FVG triple is (24,25,26):
    # low(26)=104 > high(24)=100.5 -> bullish FVG [100.5, 104] ... uses (25,26,27)? scanner picks newest
    setup = scanner.on_candle(c(26, 104, 110, 104, 109.5), Bias.BULLISH) or setup
    return setup


class TestSweepMssFvg(unittest.TestCase):
    def setUp(self):
        self.scanner = SetupScanner(swing_k=2, min_pierce=3.0, body_mult=1.5,
                                    avg_body_period=10, validity=12)

    def test_full_bullish_sequence(self):
        setup = bullish_sequence(self.scanner)
        self.assertIsNotNone(setup, "sweep->MSS->FVG should produce a setup")
        self.assertEqual(setup.bias, Bias.BULLISH)
        self.assertAlmostEqual(setup.sweep.extreme, 89.0)
        self.assertAlmostEqual(setup.stop_spot, 86.0)          # extreme - pierce
        self.assertGreater(setup.fvg.high, setup.fvg.low)
        self.assertGreaterEqual(setup.fvg.low, 100.0)          # gap over candle1 high

    def test_no_setup_without_sweep(self):
        setup = None
        for i in range(30):
            setup = self.scanner.on_candle(flat(i, 100, 1.5), Bias.BULLISH) or setup
        self.assertIsNone(setup)

    def test_sweep_without_rejection_ignored(self):
        for i in range(24):
            cd = c(i, 100, 101, 94, 100) if i == 10 else flat(i, 100, 1.5)
            self.scanner.on_candle(cd, Bias.BULLISH)
        # pierces the 94 swing low but CLOSES below it -> no rejection
        self.scanner.on_candle(c(24, 100, 100.2, 89, 90.5), Bias.BULLISH)
        self.assertIsNone(self.scanner._pending_sweep)

    def test_sweep_expires(self):
        for i in range(24):
            cd = c(i, 100, 101, 94, 100) if i == 10 else flat(i, 100, 1.5)
            self.scanner.on_candle(cd, Bias.BULLISH)
        self.scanner.on_candle(c(24, 100, 100.5, 89, 100.8), Bias.BULLISH)
        self.assertIsNotNone(self.scanner._pending_sweep)
        for i in range(25, 25 + 14):   # > validity candles of drift
            self.scanner.on_candle(flat(i, 100.5, 1.0), Bias.BULLISH)
        self.assertIsNone(self.scanner._pending_sweep)

    def test_entry_and_invalidation_helpers(self):
        setup = bullish_sequence(self.scanner)
        mid = setup.fvg.midpoint
        self.assertTrue(entry_triggered(setup, c(30, mid + 2, mid + 3, mid - 1, mid + 1)))
        self.assertFalse(entry_triggered(setup, c(31, mid + 5, mid + 6, mid + 4, mid + 5)))
        self.assertTrue(setup_invalidated(setup, c(32, 90, 91, 84, 85)))
        self.assertFalse(setup_invalidated(setup, c(33, 100, 101, 99, 100)))

    def test_static_level_sweep_pdl(self):
        scanner = SetupScanner(swing_k=2, min_pierce=3.0, body_mult=1.5,
                               avg_body_period=10, validity=12)
        scanner.set_static_levels([LiquidityLevel(95.0, SwingKind.LOW, "pdl")])
        for i in range(20):
            scanner.on_candle(flat(i, 100, 1.5), Bias.BULLISH)
        scanner.on_candle(c(20, 100, 100.5, 91.5, 100.4), Bias.BULLISH)
        self.assertIsNotNone(scanner._pending_sweep)
        self.assertEqual(scanner._pending_sweep.level.label, "pdl")


if __name__ == "__main__":
    unittest.main()
