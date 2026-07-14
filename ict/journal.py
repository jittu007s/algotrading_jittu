"""SQLite trade journal.

Screenshots are not producible from a headless bot; instead every trade
row stores the full setup levels (sweep level/extreme, MSS swing, FVG
bounds, stop) as JSON so the chart can be reconstructed after the fact.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,
    direction TEXT NOT NULL,
    symbol TEXT,
    lots INTEGER,
    entry_time TEXT, entry_spot REAL, entry_premium REAL,
    stop_spot REAL,
    exit_time TEXT, exit_spot REAL, exit_premium REAL,
    exit_reason TEXT,
    pnl_rupees REAL,
    levels_json TEXT
);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail_json TEXT
);
"""


class Journal:
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self.db_path = Path(db_path)

    def log_signal(self, kind: str, detail: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO signals (ts, kind, detail_json) VALUES (?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), kind,
             json.dumps(detail, default=str)))
        self._conn.commit()

    def open_trade(self, mode: str, direction: str, symbol: Optional[str],
                   lots: int, entry_time: datetime, entry_spot: float,
                   entry_premium: Optional[float], stop_spot: float,
                   levels: dict[str, Any]) -> int:
        cur = self._conn.execute(
            "INSERT INTO trades (mode, direction, symbol, lots, entry_time,"
            " entry_spot, entry_premium, stop_spot, levels_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (mode, direction, symbol, lots, entry_time.isoformat(timespec="seconds"),
             entry_spot, entry_premium, stop_spot, json.dumps(levels, default=str)))
        self._conn.commit()
        return int(cur.lastrowid)

    def close_trade(self, trade_id: int, exit_time: datetime, exit_spot: float,
                    exit_premium: Optional[float], exit_reason: str,
                    pnl_rupees: Optional[float]) -> None:
        self._conn.execute(
            "UPDATE trades SET exit_time=?, exit_spot=?, exit_premium=?,"
            " exit_reason=?, pnl_rupees=? WHERE id=?",
            (exit_time.isoformat(timespec="seconds"), exit_spot, exit_premium,
             exit_reason, pnl_rupees, trade_id))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
