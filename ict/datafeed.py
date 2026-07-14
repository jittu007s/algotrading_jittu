"""Market data access for the ICT system.

v1 uses the proven polling approach (one getCandleData request per closed
candle, long back-off on AB1021 rate limits) via the existing
AngelBrokingClient in ../angel_api.py.

A SmartAPI SmartWebSocketV2 feed would cut latency, but a websocket layer
that has never been run against the live endpoint is a liability in a
trading system - reconnect/resubscribe behaviour must be verified against
the real feed. The seam is `CandleFeed`: swap `PollingFeed` for a
websocket implementation without touching the engine. (When building it:
handle on_close -> exponential-backoff reconnect, resubscribe on open,
heartbeat every 30s, and drop ticks older than the last aggregated candle.)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Protocol

sys.path.insert(0, str(Path(__file__).parent.parent))  # for angel_api/config

import config as bot_config  # the existing .env-driven broker config
from angel_api import AngelBrokingClient

from .models import Candle

logger = logging.getLogger(__name__)


class CandleFeed(Protocol):
    def fetch(self, interval: str, from_dt: datetime, to_dt: datetime) -> List[Candle]: ...


class PollingFeed:
    """Historical/latest candles for the Nifty spot index via REST."""

    def __init__(self, client: Optional[AngelBrokingClient] = None):
        if client is None:
            client = AngelBrokingClient(
                bot_config.API_KEY, bot_config.CLIENT_CODE,
                bot_config.PASSWORD, bot_config.TOTP_SECRET)
            client.login()
        self.client = client

    def fetch(self, interval: str, from_dt: datetime, to_dt: datetime) -> List[Candle]:
        raw = self.client.get_candles(
            exchange=bot_config.UNDERLYING_EXCHANGE,
            symboltoken=bot_config.UNDERLYING_TOKEN,
            interval=interval,
            from_dt=from_dt,
            to_dt=to_dt,
        )
        candles = [
            Candle(timestamp=datetime.fromisoformat(r[0]).replace(tzinfo=None),
                   open=r[1], high=r[2], low=r[3], close=r[4])
            for r in raw
        ]
        candles.sort(key=lambda c: c.timestamp)
        return candles

    def closed_only(self, candles: List[Candle], interval_seconds: int,
                    now: Optional[datetime] = None) -> List[Candle]:
        now = now or datetime.now()
        return [c for c in candles
                if c.timestamp + timedelta(seconds=interval_seconds) <= now]
