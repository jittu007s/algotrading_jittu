"""Position sizing and account-level risk guards."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

logger = logging.getLogger(__name__)


def size_position(capital: float, risk_pct: float, spot_sl_distance: float,
                  delta: float, lot_size: int) -> int:
    """Lots to buy so that (premium move if spot hits SL) <= risk budget.

    premium risk / unit ~= spot_sl_distance * delta. Returns 0 if even one
    lot exceeds the budget - the trade must then be skipped, never taken
    oversized.
    """
    if spot_sl_distance <= 0 or delta <= 0 or lot_size <= 0:
        return 0
    budget = capital * risk_pct / 100.0
    per_lot_risk = spot_sl_distance * delta * lot_size
    return max(int(budget // per_lot_risk), 0)


@dataclass
class DayRiskManager:
    """Tracks realized P&L, equity peak, and enforces the hard stops:
    max daily loss, and a trailing giveback from the intraday equity peak.
    All amounts are in rupees.
    """
    capital: float
    max_daily_loss_pct: float
    equity_drawdown_stop_pct: float

    _day: date | None = None
    realized: float = 0.0
    equity_peak: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    _trades: int = field(default=0)

    def new_session(self, day: date) -> None:
        if day != self._day:
            self._day = day
            self.realized = 0.0
            self.equity_peak = 0.0
            self.halted = False
            self.halt_reason = ""
            self._trades = 0

    def record_trade(self, pnl_rupees: float) -> None:
        self.realized += pnl_rupees
        self._trades += 1
        self.equity_peak = max(self.equity_peak, self.realized)
        self._check()

    def _check(self) -> None:
        max_loss = self.capital * self.max_daily_loss_pct / 100.0
        if self.realized <= -max_loss:
            self.halted = True
            self.halt_reason = (f"daily loss limit hit "
                                f"({self.realized:.0f} <= -{max_loss:.0f})")
        giveback = self.capital * self.equity_drawdown_stop_pct / 100.0
        if self.equity_peak > 0 and self.equity_peak - self.realized >= giveback:
            self.halted = True
            self.halt_reason = (f"equity giveback stop "
                                f"(peak {self.equity_peak:.0f} -> {self.realized:.0f})")
        if self.halted:
            logger.warning("TRADING HALTED for the day: %s", self.halt_reason)

    def can_trade(self) -> bool:
        return not self.halted
