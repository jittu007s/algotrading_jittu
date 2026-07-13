"""Resolve the ATM Nifty option trading symbol/token from Angel One's
public instrument (scrip) master, and cache it locally for a day.
"""

import json
import logging
import time
from datetime import date, datetime
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

SCRIP_MASTER_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
CACHE_FILE = Path(__file__).parent / ".scrip_master_cache.json"
CACHE_TTL_SECONDS = 24 * 60 * 60


def load_scrip_master(force_refresh: bool = False):
    if not force_refresh and CACHE_FILE.exists():
        age = time.time() - CACHE_FILE.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            with open(CACHE_FILE) as f:
                return json.load(f)

    logger.info("Downloading Angel One instrument (scrip) master...")
    resp = requests.get(SCRIP_MASTER_URL, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f)
    return data


def _parse_expiry(raw: str) -> date:
    # Angel One stores option expiry as e.g. "28MAR2024".
    return datetime.strptime(raw, "%d%b%Y").date()


def find_atm_option(scrip_master, spot_price: float, option_type: str = "CE",
                     underlying: str = "NIFTY", strike_step: int = 50) -> dict:
    """Return {symbol, token, expiry, strike, lotsize} for the ATM option
    of the nearest upcoming (weekly/monthly) expiry.

    NOTE: field names/formats in Angel One's scrip master have changed
    across API revisions in the past - re-verify `instrumenttype`,
    `strike` scaling and `expiry` format against the live file if this
    stops matching anything.
    """
    atm_strike = round(spot_price / strike_step) * strike_step
    today = date.today()

    candidates = []
    for item in scrip_master:
        if item.get("instrumenttype") != "OPTIDX":
            continue
        if item.get("name") != underlying:
            continue
        symbol = item.get("symbol", "")
        if not symbol.endswith(option_type):
            continue
        try:
            expiry = _parse_expiry(item["expiry"])
            strike = float(item["strike"]) / 100  # Angel One stores strike * 100
        except (KeyError, ValueError):
            continue
        if expiry < today or strike != atm_strike:
            continue
        candidates.append((expiry, item, strike))

    if not candidates:
        raise LookupError(f"No {option_type} option found for {underlying} ATM strike {atm_strike}")

    candidates.sort(key=lambda c: c[0])
    expiry, item, strike = candidates[0]
    return {
        "symbol": item["symbol"],
        "token": item["token"],
        "expiry": expiry,
        "strike": strike,
        "lotsize": int(item["lotsize"]) if item.get("lotsize") else None,
    }
