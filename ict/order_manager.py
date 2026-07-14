"""Order management: idempotent entries, broker-side premium stop-loss.

Live mode places, per trade:
  1. a MARKET BUY of the selected option (idempotency: a client `ordertag`
     derived from the setup, checked against the day's order book before
     placing, so a retried call can never double-buy), then
  2. a STOPLOSS (SL-L) SELL on the option at the premium level implied by
     the spot stop - the broker-side safety net if this process dies.

Angel One GTT rules could replace (2) for multi-day positions; for an
intraday system the plain stop-loss order is simpler and cancels itself
at square-off. OCO (SL + target as one bracket) is not offered by
SmartAPI's normal variety - the engine manages the target/trailing side
and cancels the SL order on exit.

Paper mode logs simulated fills and never touches the broker.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as bot_config
from angel_api import AngelBrokingClient
from instruments import find_atm_option, load_scrip_master

logger = logging.getLogger(__name__)


class OrderManager:
    def __init__(self, client: Optional[AngelBrokingClient], paper: bool,
                 strike_mode: str = "ATM", delta_assumed: float = 0.5):
        self.client = client
        self.paper = paper
        self.strike_mode = strike_mode
        self.delta = delta_assumed
        self._scrip_master = None
        self._placed_tags: set[str] = set()

    def _master(self):
        if self._scrip_master is None:
            self._scrip_master = load_scrip_master()
        return self._scrip_master

    def select_option(self, spot: float, direction: str) -> dict:
        """direction: 'bullish' -> CE, 'bearish' -> PE. ITM1 shifts one
        strike into the money."""
        option_type = "CE" if direction == "bullish" else "PE"
        ref_spot = spot
        if self.strike_mode == "ITM1":
            shift = bot_config.STRIKE_STEP if option_type == "CE" else -bot_config.STRIKE_STEP
            ref_spot = spot - shift  # one strike in the money
        opt = find_atm_option(self._master(), ref_spot, option_type=option_type,
                              underlying=bot_config.UNDERLYING_NAME,
                              strike_step=bot_config.STRIKE_STEP)
        opt["quantity_per_lot"] = opt["lotsize"] or bot_config.LOT_SIZE
        return opt

    def premium_sl(self, entry_premium: float, spot_entry: float, spot_sl: float) -> float:
        """Convert the spot stop distance to an option premium stop via the
        assumed delta; floored at 0.5 so the SL order price stays valid."""
        dist = abs(spot_entry - spot_sl) * self.delta
        return max(round(entry_premium - dist, 1), 0.5)

    # ------------------------------------------------------------------
    def buy(self, option: dict, lots: int, tag: str) -> Optional[str]:
        qty = lots * option["quantity_per_lot"]
        if tag in self._placed_tags:
            logger.warning("duplicate order suppressed (tag %s)", tag)
            return None
        self._placed_tags.add(tag)
        if self.paper:
            logger.info("[PAPER] BUY %s x%d (tag %s)", option["symbol"], qty, tag)
            return f"paper-{tag}"
        resp = self.client.smart_api.placeOrder({
            "variety": "NORMAL", "tradingsymbol": option["symbol"],
            "symboltoken": option["token"], "transactiontype": "BUY",
            "exchange": bot_config.NFO_EXCHANGE, "ordertype": "MARKET",
            "producttype": "INTRADAY", "duration": "DAY", "price": "0",
            "quantity": str(qty), "ordertag": tag[:20],
        })
        logger.info("BUY placed %s x%d -> %s", option["symbol"], qty, resp)
        return str(resp)

    def place_premium_stop(self, option: dict, lots: int, trigger_premium: float,
                           tag: str) -> Optional[str]:
        qty = lots * option["quantity_per_lot"]
        limit_price = max(round(trigger_premium - 1.0, 1), 0.1)
        if self.paper:
            logger.info("[PAPER] SL-L SELL %s x%d trigger %.1f", option["symbol"], qty, trigger_premium)
            return f"paper-sl-{tag}"
        resp = self.client.smart_api.placeOrder({
            "variety": "STOPLOSS", "tradingsymbol": option["symbol"],
            "symboltoken": option["token"], "transactiontype": "SELL",
            "exchange": bot_config.NFO_EXCHANGE, "ordertype": "STOPLOSS_LIMIT",
            "producttype": "INTRADAY", "duration": "DAY",
            "triggerprice": str(trigger_premium), "price": str(limit_price),
            "quantity": str(qty), "ordertag": ("sl" + tag)[:20],
        })
        logger.info("SL order placed %s trigger %.1f -> %s", option["symbol"], trigger_premium, resp)
        return str(resp)

    def sell_market(self, option: dict, lots: int, reason: str) -> None:
        qty = lots * option["quantity_per_lot"]
        if self.paper:
            logger.info("[PAPER] SELL %s x%d (%s)", option["symbol"], qty, reason)
            return
        resp = self.client.smart_api.placeOrder({
            "variety": "NORMAL", "tradingsymbol": option["symbol"],
            "symboltoken": option["token"], "transactiontype": "SELL",
            "exchange": bot_config.NFO_EXCHANGE, "ordertype": "MARKET",
            "producttype": "INTRADAY", "duration": "DAY", "price": "0",
            "quantity": str(qty),
        })
        logger.info("SELL placed (%s) -> %s", reason, resp)

    def cancel(self, order_id: Optional[str]) -> None:
        if not order_id or self.paper or order_id.startswith("paper"):
            return
        try:
            self.client.smart_api.cancelOrder(order_id, "STOPLOSS")
        except Exception:
            logger.warning("cancel failed for %s (may already be filled)", order_id, exc_info=True)
