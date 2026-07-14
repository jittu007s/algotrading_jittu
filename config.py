import os

from dotenv import load_dotenv

load_dotenv()

# --- Angel One (SmartAPI) credentials --------------------------------------
API_KEY = os.getenv("ANGEL_API_KEY")
CLIENT_CODE = os.getenv("ANGEL_CLIENT_CODE")
PASSWORD = os.getenv("ANGEL_MPIN")
TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET")

# --- Underlying used for candle/SMA/swing-low signal generation ------------
UNDERLYING_NAME = "NIFTY"
UNDERLYING_EXCHANGE = "NSE"
UNDERLYING_TOKEN = "99926000"  # Nifty 50 index token on Angel One's instrument master
NFO_EXCHANGE = "NFO"
STRIKE_STEP = 50

# Verify the current NSE Nifty options lot size before trading - it is
# revised periodically by the exchange (75 as of 2025). find_atm_option()
# will use the lot size from the live instrument master when available and
# fall back to this constant only if that field is missing.
LOT_SIZE = 65

# --- Strategy parameters -----------------------------------------------
# Angel One candle interval enum: ONE_MINUTE, THREE_MINUTE, FIVE_MINUTE,
# TEN_MINUTE, FIFTEEN_MINUTE, THIRTY_MINUTE, ONE_HOUR, ONE_DAY
CANDLE_INTERVAL = "THREE_MINUTE"
SMA_PERIOD = 18   # period of the SMMA (smoothed MA, TradingView "SMMA 20 close")
RISK_REWARD = 2.5

# Which strategy the live bot runs: "SMMA_CROSS" (the original rules) or
# "ORB" (Opening Range Breakout). backtest_today.py always compares both.
STRATEGY = "SMMA_CROSS"
OR_MINUTES = 3            # ORB: opening range = first N minutes of the session
ORB_MAX_RISK_POINTS = 80   # ORB: skip the trade if the range (= risk) is wider
ORB_CANDLE_INTERVAL = "THREE_MINUTE"  # ORB runs on 5-min candles (others use CANDLE_INTERVAL)

# --- Execution / safety --------------------------------------------------
PRODUCT_TYPE = "INTRADAY"
ORDER_VARIETY = "NORMAL"

# When true (default), signals are logged but no real orders are sent.
# Flip to false only after you've validated behaviour end-to-end.
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# No NEW positions at or after this time (IST) - a trade opened minutes
# before square-off never gets room to work and just burns brokerage.
NO_ENTRY_AFTER_HOUR_MINUTE = (15, 0)

# Force-flatten any open position by this time (IST) regardless of strategy
# state, so an intraday option position never gets carried overnight.
SQUARE_OFF_HOUR_MINUTE = (15, 20)

# How long after a candle boundary to wait before fetching it (gives Angel
# One's servers time to finalize the bar). One data request per candle.
FETCH_DELAY_SECONDS = 8

# Cool-down after an AB1021 "Too many requests" response. Angel One's
# historical endpoint has a strict quota; back off generously.
RATE_LIMIT_COOLDOWN_SECONDS = 60
