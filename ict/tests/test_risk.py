import unittest
from datetime import date

from ict.risk import DayRiskManager, size_position


class TestSizing(unittest.TestCase):
    def test_basic_sizing(self):
        # capital 2L, 0.75% risk = 1500 budget; 30-pt spot SL * 0.5 delta
        # * 75 qty = 1125/lot -> 1 lot
        self.assertEqual(size_position(200000, 0.75, 30, 0.5, 75), 1)

    def test_skip_when_one_lot_too_big(self):
        # 80-pt SL * 0.5 * 75 = 3000/lot > 1500 budget -> 0 lots (skip)
        self.assertEqual(size_position(200000, 0.75, 80, 0.5, 75), 0)

    def test_multiple_lots(self):
        # 10-pt SL * 0.5 * 75 = 375/lot -> 4 lots within 1500
        self.assertEqual(size_position(200000, 0.75, 10, 0.5, 75), 4)

    def test_degenerate_inputs(self):
        self.assertEqual(size_position(200000, 0.75, 0, 0.5, 75), 0)
        self.assertEqual(size_position(200000, 0.75, 30, 0, 75), 0)


class TestDayRisk(unittest.TestCase):
    def setUp(self):
        self.rm = DayRiskManager(capital=200000, max_daily_loss_pct=1.25,
                                 equity_drawdown_stop_pct=1.0)
        self.rm.new_session(date(2026, 7, 14))

    def test_daily_loss_halts(self):
        self.rm.record_trade(-1500)
        self.assertTrue(self.rm.can_trade())
        self.rm.record_trade(-1100)   # total -2600 > 2500 limit
        self.assertFalse(self.rm.can_trade())
        self.assertIn("daily loss", self.rm.halt_reason)

    def test_equity_giveback_halts(self):
        self.rm.record_trade(+3000)   # peak 3000
        self.rm.record_trade(-2100)   # gave back 2100 >= 2000 (1% of 2L)
        self.assertFalse(self.rm.can_trade())
        self.assertIn("giveback", self.rm.halt_reason)

    def test_new_session_resets(self):
        self.rm.record_trade(-3000)
        self.assertFalse(self.rm.can_trade())
        self.rm.new_session(date(2026, 7, 15))
        self.assertTrue(self.rm.can_trade())


if __name__ == "__main__":
    unittest.main()
