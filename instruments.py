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


def find_offset_option(scrip_master, spot_price: float, option_type: str = "CE",
                       offset: int = 0, underlying: str = "NIFTY",
                       strike_step: int = 50, option_exchange: str = None,
                       as_of: date = None) -> dict:
    """Return {symbol, token, expiry, strike, lotsize} for the option `offset`
    strikes away from ATM, of the nearest expiry on/after `as_of`.

    `offset` is signed and measured towards out-of-the-money:
      offset > 0  -> OTM (CE: strikes ABOVE spot; PE: strikes BELOW spot)
      offset < 0  -> ITM (CE: strikes BELOW spot; PE: strikes ABOVE spot)
      offset == 0 -> ATM
    Because a CE is OTM above spot and a PE is OTM below it, the offset is
    added for a CE and subtracted for a PE.

    `option_exchange` (e.g. "NFO" for NIFTY/BANKNIFTY, "BFO" for SENSEX)
    disambiguates instruments that share a `name` across exchanges. `as_of`
    lets a backtest pick the expiry that was current on a past date.

    NOTE: field names/formats in Angel One's scrip master have changed
    across API revisions in the past - re-verify `instrumenttype`,
    `strike` scaling and `expiry` format against the live file if this
    stops matching anything.
    """
    atm_strike = round(spot_price / strike_step) * strike_step
    signed = offset if option_type == "CE" else -offset
    target_strike = atm_strike + signed * strike_step
    floor_date = as_of or date.today()

    candidates = []
    for item in scrip_master:
        if item.get("instrumenttype") != "OPTIDX":
            continue
        if item.get("name") != underlying:
            continue
        if option_exchange and item.get("exch_seg") != option_exchange:
            continue
        symbol = item.get("symbol", "")
        if not symbol.endswith(option_type):
            continue
        try:
            expiry = _parse_expiry(item["expiry"])
            strike = float(item["strike"]) / 100  # Angel One stores strike * 100
        except (KeyError, ValueError):
            continue
        if expiry < floor_date or strike != target_strike:
            continue
        candidates.append((expiry, item, strike))

    if not candidates:
        raise LookupError(
            f"No {option_type} {underlying} option at strike {target_strike} "
            f"(ATM {atm_strike}, offset {offset}) on/after {floor_date}")

    candidates.sort(key=lambda c: c[0])
    expiry, item, strike = candidates[0]
    return {
        "symbol": item["symbol"],
        "token": item["token"],
        "expiry": expiry,
        "strike": strike,
        "lotsize": int(item["lotsize"]) if item.get("lotsize") else None,
    }


def find_atm_option(scrip_master, spot_price: float, option_type: str = "CE",
                     underlying: str = "NIFTY", strike_step: int = 50,
                     option_exchange: str = None) -> dict:
    """ATM option of the nearest upcoming expiry (offset 0)."""
    return find_offset_option(scrip_master, spot_price, option_type=option_type,
                              offset=0, underlying=underlying,
                              strike_step=strike_step, option_exchange=option_exchange)
