"""Live trading loop.

Polls closed candles for the Nifty 50 index, feeds them into
SmaCrossOptionStrategy, and executes ATM option entries/exits (CE for long
signals, PE for short signals) on Angel One via SmartAPI when a signal fires.

Run with DRY_RUN=true (the default - see config.py / .env) to log signals
without sending real orders while you validate behaviour against the live
market. Only set DRY_RUN=false once you've done that, and start with a
single lot.
"""

import logging
import time
from datetime import datetime, timedelta

import config
from angel_api import AngelBrokingClient
from instruments import find_atm_option, load_scrip_master
from strategy import Candle, Signal, SmaCrossOptionStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bot")

INTERVAL_SECONDS = {
    "ONE_MINUTE": 60,
    "THREE_MINUTE": 180,
    "FIVE_MINUTE": 300,
    "TEN_MINUTE": 600,
    "FIFTEEN_MINUTE": 900,
    "THIRTY_MINUTE": 1800,
    "ONE_HOUR": 3600,
}


def seconds_until_next_candle(now):
    """Sleep until just after the next candle boundary closes, so we make
    ONE data request per candle instead of hammering the rate-limited
    historical API every few seconds."""
    interval_s = INTERVAL_SECONDS[config.CANDLE_INTERVAL]
    epoch = now.timestamp()
    next_close = (int(epoch // interval_s) + 1) * interval_s
    return max(next_close + config.FETCH_DELAY_SECONDS - epoch, 1.0)


def fetch_new_candles(client, last_seen_ts):
    now = datetime.now()
    from_dt = now - timedelta(days=4)  # covers weekends/holidays for SMA warm-up
    raw = client.get_candles(
        exchange=config.UNDERLYING_EXCHANGE,
        symboltoken=config.UNDERLYING_TOKEN,
        interval=config.CANDLE_INTERVAL,
        from_dt=from_dt,
        to_dt=now,
    )
    interval_s = INTERVAL_SECONDS[config.CANDLE_INTERVAL]
    candles = []
    for row in raw:
        ts = datetime.fromisoformat(row[0])
        naive_ts = ts.replace(tzinfo=None)
        if last_seen_ts and naive_ts <= last_seen_ts:
            continue
        # Skip the still-forming candle: only act on candles whose window
        # has fully elapsed, otherwise signals fire on incomplete data.
        if naive_ts + timedelta(seconds=interval_s) > now:
            continue
        candles.append(Candle(timestamp=naive_ts, open=row[1], high=row[2], low=row[3], close=row[4]))
    candles.sort(key=lambda c: c.timestamp)
    return candles


def is_past_square_off(now):
    h, m = config.SQUARE_OFF_HOUR_MINUTE
    return (now.hour, now.minute) >= (h, m)


def enter_position(client, scrip_master, spot_price, event, option_type):
    option = find_atm_option(
        scrip_master, spot_price, option_type=option_type,
        underlying=config.UNDERLYING_NAME, strike_step=config.STRIKE_STEP,
    )
    qty = option["lotsize"] or config.LOT_SIZE
    logger.info(
        "ENTRY signal (%s) @ spot=%.2f -> BUY %s qty=%s | SL(underlying)=%.2f target(underlying)=%.2f",
        option_type, spot_price, option["symbol"], qty, event.stop_loss, event.target,
    )

    if not config.DRY_RUN:
        client.place_market_order(
            exchange=config.NFO_EXCHANGE,
            tradingsymbol=option["symbol"],
            symboltoken=option["token"],
            transaction_type="BUY",
            quantity=qty,
            producttype=config.PRODUCT_TYPE,
        )

    option["quantity"] = qty
    return option


def exit_position(client, held_option, event):
    reason = event.reason.value if event.reason else "unknown"
    logger.info("EXIT signal (%s) -> SELL %s qty=%s", reason, held_option["symbol"], held_option["quantity"])

    if not config.DRY_RUN:
        client.place_market_order(
            exchange=config.NFO_EXCHANGE,
            tradingsymbol=held_option["symbol"],
            symboltoken=held_option["token"],
            transaction_type="SELL",
            quantity=held_option["quantity"],
            producttype=config.PRODUCT_TYPE,
        )


def main():
    client = AngelBrokingClient(config.API_KEY, config.CLIENT_CODE, config.PASSWORD, config.TOTP_SECRET)
    client.login()

    scrip_master = load_scrip_master()

    strategy = SmaCrossOptionStrategy(
        sma_period=config.SMA_PERIOD,
        risk_reward=config.RISK_REWARD,
    )

    held_option = None  # dict with symbol/token/quantity while a position is open
    last_seen_ts = None

    logger.info("Bot started. DRY_RUN=%s, interval=%s, sma=%s", config.DRY_RUN, config.CANDLE_INTERVAL, config.SMA_PERIOD)

    while True:
        try:
            now = datetime.now()

            if held_option and is_past_square_off(now):
                event = strategy.force_exit(price=None)
                exit_position(client, held_option, event)
                held_option = None

            for candle in fetch_new_candles(client, last_seen_ts):
                last_seen_ts = candle.timestamp
                event = strategy.on_closed_candle(candle)

                if event.signal in (Signal.ENTER_LONG_CE, Signal.ENTER_SHORT_PE) and not held_option:
                    option_type = "CE" if event.signal == Signal.ENTER_LONG_CE else "PE"
                    held_option = enter_position(client, scrip_master, candle.close, event, option_type)
                elif event.signal == Signal.EXIT and held_option:
                    exit_position(client, held_option, event)
                    held_option = None

            time.sleep(seconds_until_next_candle(datetime.now()))

        except Exception as exc:
            if "rate" in str(exc).lower() or "AB1021" in str(exc) or "Too many requests" in str(exc):
                logger.warning("Rate limited by Angel One; cooling down %ss", config.RATE_LIMIT_COOLDOWN_SECONDS)
                time.sleep(config.RATE_LIMIT_COOLDOWN_SECONDS)
            else:
                logger.exception("Error in main loop; retrying next candle")
                time.sleep(seconds_until_next_candle(datetime.now()))


if __name__ == "__main__":
    main()
