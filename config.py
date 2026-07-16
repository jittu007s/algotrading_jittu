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

# Where the SMMA_CROSS strategy reads its candles:
#   "INDEX_THEN_OPTION" - two-stage: the crossover fires on the INDEX chart
#              to pick a direction, then the bot goes to the strike
#              STRIKE_OFFSET strikes OTM/ITM and only BUYs once the SAME
#              crossover ALSO confirms on that option's OWN candles. Exit
#              when the premium hits OPTION_TARGET_PREMIUM_PCT (+100%) or the
#              trailed stop (2nd-last candle low; first stop = 3rd-last low).
#   "OPTION" - run the crossover directly on the weekly ATM option's OWN
#              candles and BUY on a cross-up (ATM CE and PE watched in
#              parallel, long-only per option).
#   "INDEX"  - the original behaviour: crossover on the index chart, mapping
#              a long setup -> ATM CE and a short setup -> ATM PE.
SMMA_SOURCE = "INDEX_THEN_OPTION"

# --- Multi-index / two-stage (index signal -> offset option -> option confirm) ---
# Which index the bot trades. Its underlying candles drive stage 1.
INDEX = "NIFTY"            # "NIFTY" | "BANKNIFTY" | "SENSEX"
# Strikes away from ATM for the option leg, signed towards OTM:
#   +N -> N strikes OTM,  -N -> N strikes ITM,  0 -> ATM.
STRIKE_OFFSET = 2
# Candles to wait for the option-chart crossover to confirm after the index
# signal; if it does not confirm within this many candles the setup is dropped.
OPTION_CONFIRM_VALIDITY = 10
# Book the whole option position when the premium gains this fraction
# (1.0 = +100%, i.e. the premium doubles).
OPTION_TARGET_PREMIUM_PCT = 1.0

# Per-index instrument map. VERIFY tokens / strike steps / exchanges against
# the live scrip master before trading - Angel One revises lot sizes and
# occasionally the index tokens. Index candles are read from `under_exchange`
# / `under_token`; options are resolved on `option_exchange` (NFO for NSE
# indices, BFO for the BSE SENSEX).
INDEXES = {
    "NIFTY":     {"name": "NIFTY",     "under_exchange": "NSE", "under_token": "99926000",
                  "option_exchange": "NFO", "strike_step": 50},
    "BANKNIFTY": {"name": "BANKNIFTY", "under_exchange": "NSE", "under_token": "99926009",
                  "option_exchange": "NFO", "strike_step": 100},
    "SENSEX":    {"name": "SENSEX",    "under_exchange": "BSE", "under_token": "99919000",
                  "option_exchange": "BFO", "strike_step": 100},
}


def index_config(name: str = None) -> dict:
    """Instrument map for the active (or named) index."""
    return INDEXES[name or INDEX]

# Which strategy the live bot runs: "SMMA_CROSS" (the original rules) or
# "ORB" (Opening Range Breakout). backtest_today.py always compares both.
STRATEGY = "PULLBACK"   # "PULLBACK" | "FVG_RETEST" | "ORB" | "SMMA_CROSS" | "REGIME"
OR_MINUTES = 15             # ORB: opening range = first N minutes of the session
ORB_MAX_RISK_POINTS = 80   # ORB: skip the trade if the range (= risk) is wider
ORB_EXTENDED_TARGET_R = 3.0   # after 2R the target extends to this R multiple
ORB_TIMEOUT_MINUTES = 15      # after the 2R shift: exit at prev candle high/low if neither hits
ORB_BE_AFTER_MINUTES = 60     # in profit but no 2R for this long -> SL to entry
ORB_RETRACE_POINTS = 15       # retest mode: minimum retracement before the retest entry
ORB_RETEST_STOP_LOOKBACK = 10 # retest stop = extreme of the last N candles (recent swing)
ORB_STOP_MODE = "mid_range"   # initial-entry stop: "mid_range" (user experiment) | "opposite"
ORB_CANDLE_INTERVAL = "FIFTEEN_MINUTE"  # candle timeframe ORB runs on

# --- PULLBACK strategy (mark 9:15-9:30 high/low, cross -> return -> confirm) ---
PB_OR_MINUTES = 15            # marking window in minutes (15 = until 09:30)
PB_RISK_REWARD = 2.0          # initial 1:2
PB_MAX_RISK_POINTS = 60       # stop clamp
PB_NUM_LOTS = 1               # >1 enables the 50%-at-2R scale-out with 4R runner
PB_PULLBACK_VALIDITY = 20     # candles a cross stays valid awaiting return+confirm
PB_TARGET_CAP_R = 10          # trail the stop until price reaches this R multiple, then exit
PB_CANDLE_INTERVAL = "THREE_MINUTE"

# --- FVG_RETEST strategy (sell into bearish FVG retest / buy into bullish) ---
FVG_MIN_SIZE = 5.0            # ignore gaps smaller than this many points
FVG_BUFFER = 2.0             # stop sits this far beyond the gap's far edge
FVG_RISK_REWARD = 2.0        # 1:2 before trailing kicks in
FVG_MAX_RISK_POINTS = 60     # stop clamp
FVG_TARGET_CAP_R = 10        # trail winners to this R multiple, then exit
FVG_MAX_AGE = 60             # candles a gap stays tradeable after forming
FVG_CANDLE_INTERVAL = "THREE_MINUTE"

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
