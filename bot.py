"""Live trading loop.

Polls closed candles for the Nifty 50 index, feeds them into
SmaCrossOptionStrategy, and executes ATM CE entries/exits on Angel One via
SmartAPI when a signal fires.

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


def fetch_new_candles(client, last_seen_ts):
    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=2)
    raw = client.get_candles(
        exchange=config.UNDERLYING_EXCHANGE,
        symboltoken=config.UNDERLYING_TOKEN,
        interval=config.CANDLE_INTERVAL,
        from_dt=from_dt,
        to_dt=to_dt,
    )
    candles = []
    for row in raw:
        ts = datetime.fromisoformat(row[0])
        if last_seen_ts and ts <= last_seen_ts:
            continue
        candles.append(Candle(timestamp=ts, open=row[1], high=row[2], low=row[3], close=row[4]))
    candles.sort(key=lambda c: c.timestamp)
    return candles


def is_past_square_off(now):
    h, m = config.SQUARE_OFF_HOUR_MINUTE
    return (now.hour, now.minute) >= (h, m)


def enter_position(client, scrip_master, spot_price, event):
    option = find_atm_option(
        scrip_master, spot_price, option_type="CE",
        underlying=config.UNDERLYING_NAME, strike_step=config.STRIKE_STEP,
    )
    qty = option["lotsize"] or config.LOT_SIZE
    logger.info(
        "ENTRY signal @ spot=%.2f -> BUY %s qty=%s | SL(underlying)=%.2f target(underlying)=%.2f",
        spot_price, option["symbol"], qty, event.stop_loss, event.target,
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
        swing_fractal=config.SWING_FRACTAL,
        swing_lookback=config.SWING_LOOKBACK,
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

                if event.signal == Signal.ENTER_LONG_CE and not held_option:
                    held_option = enter_position(client, scrip_master, candle.close, event)
                elif event.signal == Signal.EXIT and held_option:
                    exit_position(client, held_option, event)
                    held_option = None

        except Exception:
            logger.exception("Error in main loop; will retry after poll interval")

        time.sleep(config.POLL_SECONDS)


if __name__ == "__main__":
    main()
