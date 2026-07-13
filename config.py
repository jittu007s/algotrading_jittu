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
LOT_SIZE = 75

# --- Strategy parameters -----------------------------------------------
# Angel One candle interval enum: ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE,
# THIRTY_MINUTE, ONE_HOUR, ONE_DAY, ...
CANDLE_INTERVAL = "FIFTEEN_MINUTE"
SMA_PERIOD = 20
SWING_FRACTAL = 2       # candles on each side used to confirm a swing low
SWING_LOOKBACK = 30     # how far back to search for the swing low
RISK_REWARD = 2.0

# --- Execution / safety --------------------------------------------------
PRODUCT_TYPE = "INTRADAY"
ORDER_VARIETY = "NORMAL"

# When true (default), signals are logged but no real orders are sent.
# Flip to false only after you've validated behaviour end-to-end.
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# Force-flatten any open position by this time (IST) regardless of strategy
# state, so an intraday option position never gets carried overnight.
SQUARE_OFF_HOUR_MINUTE = (15, 20)

POLL_SECONDS = 15
