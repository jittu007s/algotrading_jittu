"""Thin wrapper around the official Angel One (Angel Broking) SmartAPI
Python SDK (`smartapi-python`): login, historical candles, order placement.
"""

import logging
import time
from datetime import datetime

import pyotp
from SmartApi import SmartConnect

logger = logging.getLogger(__name__)


class AngelBrokingClient:
    def __init__(self, api_key: str, client_code: str, password: str, totp_secret: str):
        if not all([api_key, client_code, password, totp_secret]):
            raise ValueError("Missing Angel One credentials - check your .env file")

        self.client_code = client_code
        self.smart_api = SmartConnect(api_key=api_key)
        self._totp_secret = totp_secret
        self._password = password
        self.feed_token = None

    def login(self):
        totp = pyotp.TOTP(self._totp_secret).now()
        session = self.smart_api.generateSession(self.client_code, self._password, totp)
        if not session or not session.get("status"):
            raise RuntimeError(f"Angel One login failed: {session}")
        self.feed_token = self.smart_api.getfeedToken()
        logger.info("Logged in to Angel One SmartAPI as %s", self.client_code)
        return session

    def logout(self):
        try:
            self.smart_api.terminateSession(self.client_code)
        except Exception:
            logger.warning("Logout call failed (non-fatal)", exc_info=True)

    def get_candles(self, exchange: str, symboltoken: str, interval: str,
                     from_dt: datetime, to_dt: datetime, retries: int = 3):
        """Returns a list of [timestamp, open, high, low, close, volume]."""
        params = {
            "exchange": exchange,
            "symboltoken": symboltoken,
            "interval": interval,
            "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
            "todate": to_dt.strftime("%Y-%m-%d %H:%M"),
        }
        last_err = None
        for attempt in range(retries):
            try:
                resp = self.smart_api.getCandleData(params)
                if resp and resp.get("status"):
                    return resp.get("data") or []
                last_err = resp
            except Exception as exc:  # network / rate-limit hiccups
                last_err = exc
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"getCandleData failed after {retries} attempts: {last_err}")

    def place_market_order(self, exchange: str, tradingsymbol: str, symboltoken: str,
                            transaction_type: str, quantity: int,
                            producttype: str = "INTRADAY", variety: str = "NORMAL"):
        """transaction_type: 'BUY' or 'SELL'."""
        order_params = {
            "variety": variety,
            "tradingsymbol": tradingsymbol,
            "symboltoken": symboltoken,
            "transactiontype": transaction_type,
            "exchange": exchange,
            "ordertype": "MARKET",
            "producttype": producttype,
            "duration": "DAY",
            "price": "0",
            "quantity": str(quantity),
        }
        resp = self.smart_api.placeOrder(order_params)
        logger.info("Order placed: %s -> response: %s", order_params, resp)
        return resp
