"""Bias engine: multi-timeframe directional bias for the session.

Daily and 1H swing structure vote; conflicts mean NEUTRAL and no trades.
Recomputed at session start and refreshed hourly as new 1H candles close.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List

from .models import Bias, Candle
from .structure import combine_bias, detect_bias

logger = logging.getLogger(__name__)


class BiasEngine:
    def __init__(self, feed, daily_tf: str, intraday_tf: str, swing_k: int):
        self.feed = feed
        self.daily_tf = daily_tf
        self.intraday_tf = intraday_tf
        self.swing_k = swing_k
        self._bias = Bias.NEUTRAL
        self._last_refresh: datetime | None = None

    @property
    def bias(self) -> Bias:
        return self._bias

    def refresh(self, now: datetime) -> Bias:
        if self._last_refresh and now - self._last_refresh < timedelta(hours=1):
            return self._bias
        daily: List[Candle] = self.feed.fetch(
            self.daily_tf, now - timedelta(days=90), now)
        hourly: List[Candle] = self.feed.fetch(
            self.intraday_tf, now - timedelta(days=15), now)
        d = detect_bias(daily, self.swing_k)
        h = detect_bias(hourly, self.swing_k)
        self._bias = combine_bias(d, h)
        self._last_refresh = now
        logger.info("bias refresh: daily=%s 1h=%s -> %s", d.value, h.value, self._bias.value)
        return self._bias
