"""Unit tests for the SMC detection primitives (FVG, Order Block, equal
levels). Run:  python -m unittest discover -s ict/tests -t ."""

import unittest
from datetime import datetime, timedelta

from ict.models import Candle, SwingKind
from ict.smc import find_equal_levels, find_fvgs, find_order_blocks

T0 = datetime(2026, 7, 14, 9, 15)


def c(i, o, h, l, cl):
    return Candle(T0 + timedelta(minutes=3 * i), o, h, l, cl)


def flat(i, px, w=1.5):
    return c(i, px, px + w, px - w, px + 0.3)


class TestFVG(unittest.TestCase):
    def test_bullish_gap_detected_and_sized(self):
        # candle0 high 101, candle2 low 104 -> bullish gap [101, 104]
        candles = [c(0, 100, 101, 99, 100), c(1, 101, 104, 100, 103),
                   c(2, 104, 108, 104, 107)]
        gaps = find_fvgs(candles, min_size=1.0, require_unmitigated=False)
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0].low, 101)
        self.assertAlmostEqual(gaps[0].high, 104)

    def test_min_size_filter(self):
        candles = [c(0, 100, 101, 99, 100), c(1, 101, 102, 100.5, 101.5),
                   c(2, 101.5, 103, 101.2, 102)]   # gap [101, 101.2] = 0.2
        self.assertEqual(find_fvgs(candles, min_size=1.0, require_unmitigated=False), [])

    def test_unmitigated_filter_drops_filled_gap(self):
        # bullish gap [101,104] at idx2, then idx3 trades back through midpoint 102.5
        candles = [c(0, 100, 101, 99, 100), c(1, 101, 104, 100, 103),
                   c(2, 104, 108, 104, 107), c(3, 107, 108, 102, 103)]
        self.assertEqual(find_fvgs(candles, require_unmitigated=True), [])
        self.assertEqual(len(find_fvgs(candles, require_unmitigated=False)), 1)

    def test_ranked_largest_first(self):
        candles = [c(0, 100, 101, 99, 100), c(1, 101, 104, 100, 103), c(2, 104, 108, 104, 107),
                   flat(3, 108), flat(4, 108),
                   c(5, 108, 110, 107, 109), c(6, 110, 130, 109, 128), c(7, 130, 140, 125, 138)]
        gaps = find_fvgs(candles, require_unmitigated=False)
        self.assertGreaterEqual(len(gaps), 2)
        self.assertGreaterEqual(gaps[0].high - gaps[0].low, gaps[1].high - gaps[1].low)


class TestOrderBlock(unittest.TestCase):
    def test_bullish_ob_is_last_bearish_before_impulse(self):
        candles = []
        # base with a swing high ~103 to break
        for i in range(8):
            candles.append(flat(i, 100, 1.2) if i != 3 else c(3, 100, 103, 99, 100.5))
        # a clean bearish candle (the OB), then a big bullish displacement breaking 103
        candles.append(c(8, 100, 100.5, 97, 98))      # bearish OB at idx 8
        candles.append(c(9, 98, 112, 97.5, 111))      # displacement up, closes > 103
        obs = find_order_blocks(candles, swing_k=2, displacement_mult=1.5,
                                avg_period=8, only_untested=False)
        self.assertTrue(obs)
        ob = next(b for b in obs if b.index == 8)
        self.assertEqual(ob.kind, SwingKind.LOW)      # bullish/demand OB
        self.assertAlmostEqual(ob.low, 97)
        self.assertAlmostEqual(ob.high, 100.5)

    def test_untested_filter(self):
        candles = []
        for i in range(8):
            candles.append(flat(i, 100, 1.2) if i != 3 else c(3, 100, 103, 99, 100.5))
        candles.append(c(8, 100, 100.5, 97, 98))
        candles.append(c(9, 98, 112, 97.5, 111))
        candles.append(c(10, 111, 113, 98, 99))       # trades back into the OB midpoint ~98.75
        untested = find_order_blocks(candles, swing_k=2, avg_period=8, only_untested=True)
        self.assertFalse(any(b.index == 8 for b in untested))

    def test_no_ob_without_displacement(self):
        candles = [flat(i, 100, 1.0) for i in range(12)]   # no impulse
        self.assertEqual(find_order_blocks(candles, avg_period=8), [])


class TestEqualLevels(unittest.TestCase):
    def test_equal_highs_clustered(self):
        # two isolated swing highs at ~120 within tolerance
        candles = [flat(0, 100), flat(1, 100), c(2, 100, 120, 99, 101),
                   flat(3, 100), flat(4, 100), c(5, 100, 121, 99, 101),
                   flat(6, 100), flat(7, 100)]
        levels = find_equal_levels(candles, swing_k=2, tolerance=3.0, min_count=2)
        highs = [l for l in levels if l.kind == SwingKind.HIGH]
        self.assertEqual(len(highs), 1)
        self.assertEqual(highs[0].count, 2)
        self.assertAlmostEqual(highs[0].price, 120.5)

    def test_far_apart_highs_not_equal(self):
        candles = [flat(0, 100), flat(1, 100), c(2, 100, 120, 99, 101),
                   flat(3, 100), flat(4, 100), c(5, 100, 140, 99, 101),
                   flat(6, 100), flat(7, 100)]
        highs = [l for l in find_equal_levels(candles, tolerance=3.0)
                 if l.kind == SwingKind.HIGH]
        self.assertEqual(highs, [])


if __name__ == "__main__":
    unittest.main()
