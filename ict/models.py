"""Typed domain objects shared across the ICT modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def bullish(self) -> bool:
        return self.close > self.open


class Bias(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SwingKind(Enum):
    HIGH = "high"
    LOW = "low"


@dataclass(frozen=True)
class Swing:
    kind: SwingKind
    price: float
    timestamp: datetime
    index: int              # index into the candle series it was found on


@dataclass(frozen=True)
class LiquidityLevel:
    """A level resting stops accumulate around: prior swing high/low,
    previous day high/low, or the opening-range extreme."""
    price: float
    kind: SwingKind         # HIGH = buy-side liquidity, LOW = sell-side
    label: str              # e.g. "swing_low", "pdl", "or_low"


@dataclass(frozen=True)
class Sweep:
    level: LiquidityLevel
    extreme: float          # the stop-hunt wick extreme (SL anchor)
    timestamp: datetime
    index: int


@dataclass(frozen=True)
class MSS:
    """Market structure shift: displacement close through a swing point."""
    broken_swing: Swing
    displacement_index: int
    timestamp: datetime


@dataclass(frozen=True)
class FVG:
    """3-candle imbalance left by the displacement move."""
    low: float
    high: float
    created_index: int
    timestamp: datetime

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2


@dataclass
class Setup:
    """A completed sweep -> MSS -> FVG sequence awaiting entry."""
    bias: Bias
    sweep: Sweep
    mss: MSS
    fvg: FVG
    stop_spot: float                 # spot SL: just beyond the swept extreme
    created_index: int
    entered: bool = False
    invalidated: bool = False


@dataclass
class PaperTrade:
    direction: Bias
    entry_time: datetime
    entry_spot: float
    stop_spot: float
    partial_done: bool = False
    exit_time: Optional[datetime] = None
    exit_spot: Optional[float] = None
    exit_reason: Optional[str] = None
    levels: dict = field(default_factory=dict)   # journalled setup levels
