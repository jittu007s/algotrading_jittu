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
from instruments import find_atm_option, find_offset_option, load_scrip_master
from quant_strategy import RegimeAdaptiveStrategy
from strategy import (Candle, FVGRetestStrategy, OpeningRangeBreakout,
                      OptionPremiumStrategy, PullbackConfirmStrategy, Signal,
                      SmaCrossOptionStrategy)


def build_strategy():
    if config.STRATEGY == "FVG_RETEST":
        return FVGRetestStrategy(
            min_size=config.FVG_MIN_SIZE, buffer=config.FVG_BUFFER,
            risk_reward=config.FVG_RISK_REWARD, max_risk_points=config.FVG_MAX_RISK_POINTS,
            target_cap_r=config.FVG_TARGET_CAP_R, fvg_max_age=config.FVG_MAX_AGE)
    if config.STRATEGY == "PULLBACK":
        return PullbackConfirmStrategy(
            or_minutes=config.PB_OR_MINUTES, risk_reward=config.PB_RISK_REWARD,
            max_risk_points=config.PB_MAX_RISK_POINTS, num_lots=config.PB_NUM_LOTS,
            pullback_validity=config.PB_PULLBACK_VALIDITY,
            target_cap_r=config.PB_TARGET_CAP_R)
    if config.STRATEGY == "ORB":
        return OpeningRangeBreakout(
            risk_reward=config.RISK_REWARD,
            or_minutes=config.OR_MINUTES, max_risk_points=config.ORB_MAX_RISK_POINTS,
            extended_target_r=config.ORB_EXTENDED_TARGET_R,
            timeout_minutes=config.ORB_TIMEOUT_MINUTES,
            be_after_minutes=config.ORB_BE_AFTER_MINUTES,
            retrace_points=config.ORB_RETRACE_POINTS,
            stop_mode=config.ORB_STOP_MODE,
            retest_stop_lookback=config.ORB_RETEST_STOP_LOOKBACK)
    if config.STRATEGY == "REGIME":
        return RegimeAdaptiveStrategy()
    return SmaCrossOptionStrategy(sma_period=config.SMA_PERIOD, risk_reward=config.RISK_REWARD)


def active_interval():
    """Each strategy runs on its own candle timeframe."""
    if config.STRATEGY == "FVG_RETEST":
        return config.FVG_CANDLE_INTERVAL
    if config.STRATEGY == "PULLBACK":
        return config.PB_CANDLE_INTERVAL
    if config.STRATEGY == "ORB":
        return config.ORB_CANDLE_INTERVAL
    return config.CANDLE_INTERVAL

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
    interval_s = INTERVAL_SECONDS[active_interval()]
    epoch = now.timestamp()
    next_close = (int(epoch // interval_s) + 1) * interval_s
    return max(next_close + config.FETCH_DELAY_SECONDS - epoch, 1.0)


def fetch_new_candles(client, last_seen_ts):
    now = datetime.now()
    from_dt = now - timedelta(days=4)  # covers weekends/holidays for SMA warm-up
    raw = client.get_candles(
        exchange=config.UNDERLYING_EXCHANGE,
        symboltoken=config.UNDERLYING_TOKEN,
        interval=active_interval(),
        from_dt=from_dt,
        to_dt=now,
    )
    interval_s = INTERVAL_SECONDS[active_interval()]
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


def is_past_entry_cutoff(now):
    h, m = config.NO_ENTRY_AFTER_HOUR_MINUTE
    return (now.hour, now.minute) >= (h, m)


def enter_position(client, scrip_master, candle, event, option_type):
    option = find_atm_option(
        scrip_master, candle.close, option_type=option_type,
        underlying=config.UNDERLYING_NAME, strike_step=config.STRIKE_STEP,
    )
    qty = option["lotsize"] or config.LOT_SIZE
    logger.info(
        "ENTRY signal (%s) candle=%s entry=%.2f spot=%.2f -> BUY %s qty=%s | SL=%.2f target=%.2f",
        option_type, candle.timestamp, event.price, candle.close,
        option["symbol"], qty, event.stop_loss, event.target,
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
    price = "n/a" if event.price is None else f"{event.price:.2f}"
    logger.info("EXIT signal (%s) exit(underlying)=%s -> SELL %s qty=%s",
                reason, price, held_option["symbol"], held_option["quantity"])

    if not config.DRY_RUN:
        client.place_market_order(
            exchange=config.NFO_EXCHANGE,
            tradingsymbol=held_option["symbol"],
            symboltoken=held_option["token"],
            transaction_type="SELL",
            quantity=held_option["quantity"],
            producttype=config.PRODUCT_TYPE,
        )


# ---------------------------------------------------------------------------
# OPTION-chart mode: run the SMMA crossover directly on the weekly-expiry ATM
# option's OWN candles and BUY the option on a cross-up. Both the ATM CE and
# ATM PE are watched in parallel as independent "lanes" (long-only per
# option). The ATM strike is re-resolved from live spot whenever a lane is
# flat, so a fresh setup always trades the current ATM; a held lane keeps its
# option until the strategy exits it.
# ---------------------------------------------------------------------------

def build_option_strategy():
    return SmaCrossOptionStrategy(sma_period=config.SMA_PERIOD,
                                  risk_reward=config.RISK_REWARD, long_only=True)


class OptionLane:
    """One option (CE or PE) tracked on its own candles with its own SMMA."""

    def __init__(self, option_type):
        self.option_type = option_type   # "CE" or "PE"
        self.strategy = build_option_strategy()
        self.token = None
        self.symbol = None
        self.lotsize = None
        self.last_seen_ts = None
        self.held = None   # dict{symbol,token,quantity} while a position is open

    def repoint(self, option):
        """Switch this lane to a new ATM strike and start its SMMA fresh."""
        self.strategy = build_option_strategy()
        self.token = option["token"]
        self.symbol = option["symbol"]
        self.lotsize = option["lotsize"]
        self.last_seen_ts = None


def latest_spot(client, interval):
    now = datetime.now()
    raw = client.get_candles(
        exchange=config.UNDERLYING_EXCHANGE, symboltoken=config.UNDERLYING_TOKEN,
        interval=interval, from_dt=now - timedelta(days=4), to_dt=now)
    return raw[-1][4] if raw else None


def fetch_candles_since(client, token, exchange, interval, last_seen_ts):
    """Closed candles for `token` newer than last_seen_ts (skips the
    still-forming bar)."""
    now = datetime.now()
    raw = client.get_candles(exchange=exchange, symboltoken=token, interval=interval,
                             from_dt=now - timedelta(days=4), to_dt=now)
    interval_s = INTERVAL_SECONDS[interval]
    out = []
    for row in raw:
        ts = datetime.fromisoformat(row[0]).replace(tzinfo=None)
        if last_seen_ts and ts <= last_seen_ts:
            continue
        if ts + timedelta(seconds=interval_s) > now:
            continue
        out.append(Candle(timestamp=ts, open=row[1], high=row[2], low=row[3], close=row[4]))
    out.sort(key=lambda c: c.timestamp)
    return out


def warm_lane(client, lane, interval):
    """Feed the new option's recent history into its SMMA (state only, no
    orders) so a cross-up can fire without waiting for a fresh warm-up."""
    candles = fetch_candles_since(client, lane.token, config.NFO_EXCHANGE, interval, None)
    for c in candles:
        lane.strategy.on_closed_candle(c)
        lane.last_seen_ts = c.timestamp
    if lane.strategy.state == "IN_POSITION":
        lane.strategy.force_exit(price=None)  # a warm-up "position" was never real
    logger.info("[%s] warmed %s on %d candles (token=%s)",
                lane.option_type, lane.symbol, len(candles), lane.token)


def buy_option(client, lane, event):
    qty = lane.lotsize or config.LOT_SIZE
    logger.info("[%s] ENTRY %s option_price=%.2f qty=%s | SL=%.2f target=%.2f",
                lane.option_type, lane.symbol, event.price, qty,
                event.stop_loss, event.target)
    if not config.DRY_RUN:
        client.place_market_order(
            exchange=config.NFO_EXCHANGE, tradingsymbol=lane.symbol,
            symboltoken=lane.token, transaction_type="BUY", quantity=qty,
            producttype=config.PRODUCT_TYPE)
    lane.held = {"symbol": lane.symbol, "token": lane.token, "quantity": qty}


def sell_option(client, lane, event):
    reason = event.reason.value if event.reason else "unknown"
    price = "n/a" if event.price is None else f"{event.price:.2f}"
    held = lane.held or {"symbol": lane.symbol, "token": lane.token,
                         "quantity": lane.lotsize or config.LOT_SIZE}
    logger.info("[%s] EXIT (%s) %s option_price=%s qty=%s",
                lane.option_type, reason, held["symbol"], price, held["quantity"])
    if not config.DRY_RUN:
        client.place_market_order(
            exchange=config.NFO_EXCHANGE, tradingsymbol=held["symbol"],
            symboltoken=held["token"], transaction_type="SELL",
            quantity=held["quantity"], producttype=config.PRODUCT_TYPE)


def run_option_chart_loop(client, scrip_master):
    interval = config.CANDLE_INTERVAL
    interval_s = INTERVAL_SECONDS[interval]
    lanes = {"CE": OptionLane("CE"), "PE": OptionLane("PE")}
    start_time = datetime.now()

    logger.info("Bot started (OPTION-chart SMMA). DRY_RUN=%s interval=%s sma=%s",
                config.DRY_RUN, interval, config.SMA_PERIOD)

    while True:
        try:
            now = datetime.now()
            spot = latest_spot(client, interval)

            for otype, lane in lanes.items():
                if lane.held and is_past_square_off(now):
                    event = lane.strategy.force_exit(price=None)
                    sell_option(client, lane, event)
                    lane.held = None
                    continue

                # Re-resolve the ATM strike only while the lane is flat, so a
                # held option is never swapped out from under an open trade.
                if lane.held is None and spot is not None:
                    option = find_atm_option(
                        scrip_master, spot, option_type=otype,
                        underlying=config.UNDERLYING_NAME, strike_step=config.STRIKE_STEP)
                    if option["token"] != lane.token:
                        lane.repoint(option)
                        warm_lane(client, lane, interval)

                if lane.token is None:
                    continue

                for candle in fetch_candles_since(client, lane.token,
                                                  config.NFO_EXCHANGE, interval,
                                                  lane.last_seen_ts):
                    lane.last_seen_ts = candle.timestamp
                    is_live = candle.timestamp + timedelta(seconds=interval_s) > start_time
                    event = lane.strategy.on_closed_candle(candle)
                    if not is_live:
                        continue   # pre-start candles warm state only

                    if event.note:
                        logger.info("[%s %s] note: %s", otype, lane.symbol, event.note)

                    if event.signal == Signal.ENTER_LONG_CE and lane.held is None:
                        if is_past_entry_cutoff(datetime.now()):
                            lane.strategy.force_exit(price=None)
                            logger.info("[%s] entry suppressed - past %02d:%02d cutoff",
                                        otype, *config.NO_ENTRY_AFTER_HOUR_MINUTE)
                        else:
                            buy_option(client, lane, event)
                    elif event.signal == Signal.EXIT and lane.held is not None:
                        sell_option(client, lane, event)
                        lane.held = None

            time.sleep(seconds_until_next_candle(datetime.now()))

        except Exception as exc:
            if "rate" in str(exc).lower() or "AB1021" in str(exc) or "Too many requests" in str(exc):
                logger.warning("Rate limited by Angel One; cooling down %ss",
                               config.RATE_LIMIT_COOLDOWN_SECONDS)
                time.sleep(config.RATE_LIMIT_COOLDOWN_SECONDS)
            else:
                logger.exception("Error in option-chart loop; retrying next candle")
                time.sleep(seconds_until_next_candle(datetime.now()))


# ---------------------------------------------------------------------------
# TWO-STAGE mode (INDEX_THEN_OPTION): the SMMA crossover fires on the INDEX to
# pick a direction; the bot then goes to the strike STRIKE_OFFSET strikes
# OTM/ITM and only BUYs once the SAME crossover ALSO confirms on that option's
# OWN candles. The option leg exits at +100% premium or its trailed stop
# (initial = 3rd-last candle low, then trailed to the 2nd-last candle low).
# ---------------------------------------------------------------------------

def build_index_strategy():
    # Index stage: fire the direction on the 2nd consecutive close across the
    # SMMA (signal only - never holds an index position).
    return SmaCrossOptionStrategy(sma_period=config.SMA_PERIOD,
                                  risk_reward=config.RISK_REWARD, signal_only=True)


def build_option_leg_strategy():
    if getattr(config, "OPTION_LEG_MODE", "premium_ladder") == "premium_ladder":
        return OptionPremiumStrategy(
            sma_period=config.SMA_PERIOD, swing_k=config.OPTION_SWING_K,
            swing_lookback=config.OPTION_SWING_LOOKBACK,
            fallback_risk_pct=config.OPTION_FALLBACK_RISK_PCT,
            ladder_start_pct=config.OPTION_LADDER_START_PCT,
            ladder_step_pct=config.OPTION_LADDER_STEP_PCT,
            ladder_lock_offset_pct=config.OPTION_LADDER_LOCK_OFFSET_PCT)
    return SmaCrossOptionStrategy(
        sma_period=config.SMA_PERIOD, risk_reward=config.RISK_REWARD, long_only=True,
        target_mode="premium_pct", target_premium_pct=config.OPTION_TARGET_PREMIUM_PCT,
        trail_mode="prev2_extreme")


def resolve_leg_option(scrip_master, spot, option_type, ic):
    return find_offset_option(
        scrip_master, spot, option_type=option_type, offset=config.STRIKE_OFFSET,
        underlying=ic["name"], strike_step=ic["strike_step"],
        option_exchange=ic["option_exchange"])


def run_index_option_confirm_loop(client, scrip_master):
    ic = config.index_config()
    interval = config.CANDLE_INTERVAL
    interval_s = INTERVAL_SECONDS[interval]
    index_strategy = build_index_strategy()

    index_last_ts = None
    pending = None   # dict{option, otype, strat, last_ts, age} awaiting option confirm
    held = None      # dict{symbol, token, quantity, strat, last_ts, otype}
    start_time = datetime.now()

    logger.info("Bot started (INDEX_THEN_OPTION). index=%s offset=%+d DRY_RUN=%s "
                "interval=%s sma=%s target=+%.0f%%",
                config.INDEX, config.STRIKE_OFFSET, config.DRY_RUN, interval,
                config.SMA_PERIOD, config.OPTION_TARGET_PREMIUM_PCT * 100)

    def option_candles(token, last_ts):
        return fetch_candles_since(client, token, ic["option_exchange"], interval, last_ts)

    while True:
        try:
            now = datetime.now()

            # -- square-off any open option position --------------------------
            if held and is_past_square_off(now):
                event = held["strat"].force_exit(price=None)
                sell_held(client, held, event)
                held = None

            # -- STAGE 2a: manage an open option position ---------------------
            if held:
                for candle in option_candles(held["token"], held["last_ts"]):
                    held["last_ts"] = candle.timestamp
                    event = held["strat"].on_closed_candle(candle)
                    if event.note:
                        logger.info("[%s %s] %s", held["otype"], held["symbol"], event.note)
                    if event.signal == Signal.EXIT:
                        sell_held(client, held, event)
                        held = None
                        break

            # -- STAGE 2b: wait for the option chart to confirm ---------------
            if pending and not held:
                for candle in option_candles(pending["option"]["token"], pending["last_ts"]):
                    pending["last_ts"] = candle.timestamp
                    pending["age"] += 1
                    event = pending["strat"].on_closed_candle(candle)
                    if event.signal == Signal.ENTER_LONG_CE:
                        if is_past_entry_cutoff(datetime.now()):
                            pending["strat"].force_exit(price=None)
                            logger.info("[%s] confirm suppressed - past cutoff", pending["otype"])
                            pending = None
                        else:
                            held = buy_leg(client, pending, event)
                            pending = None
                        break
                    if pending["age"] > config.OPTION_CONFIRM_VALIDITY:
                        logger.info("[%s] option confirm expired (%d candles), dropping setup",
                                    pending["otype"], pending["age"])
                        pending = None
                        break

            # -- STAGE 1: index crossover picks a direction -------------------
            for candle in fetch_candles_since(client, ic["under_token"],
                                              ic["under_exchange"], interval, index_last_ts):
                index_last_ts = candle.timestamp
                is_live = candle.timestamp + timedelta(seconds=interval_s) > start_time
                event = index_strategy.on_closed_candle(candle)
                if not is_live:
                    continue
                if event.signal in (Signal.ENTER_LONG_CE, Signal.ENTER_SHORT_PE) \
                        and held is None:
                    if is_past_entry_cutoff(datetime.now()):
                        index_strategy.force_exit(price=None)
                        continue
                    otype = "CE" if event.signal == Signal.ENTER_LONG_CE else "PE"
                    try:
                        option = resolve_leg_option(scrip_master, candle.close, otype, ic)
                    except LookupError as exc:
                        logger.warning("Could not resolve %s option: %s", otype, exc)
                        continue
                    strat = build_option_leg_strategy()
                    warm_from_history(client, strat, option["token"], ic["option_exchange"], interval)
                    pending = {"option": option, "otype": otype, "strat": strat,
                               "last_ts": None, "age": 0}
                    logger.info("[STAGE1] index %s @%.2f -> watch %s (strike %s) for confirm",
                                otype, candle.close, option["symbol"], option["strike"])

            time.sleep(seconds_until_next_candle(datetime.now()))

        except Exception as exc:
            if "rate" in str(exc).lower() or "AB1021" in str(exc) or "Too many requests" in str(exc):
                logger.warning("Rate limited by Angel One; cooling down %ss",
                               config.RATE_LIMIT_COOLDOWN_SECONDS)
                time.sleep(config.RATE_LIMIT_COOLDOWN_SECONDS)
            else:
                logger.exception("Error in index-option loop; retrying next candle")
                time.sleep(seconds_until_next_candle(datetime.now()))


def warm_from_history(client, strat, token, exchange, interval):
    """Feed an option's recent history into a strategy (state only) so its
    SMMA is ready to detect a crossover immediately."""
    for c in fetch_candles_since(client, token, exchange, interval, None):
        strat.on_closed_candle(c)
    if strat.state == "IN_POSITION":
        strat.force_exit(price=None)


def buy_leg(client, pending, event):
    option = pending["option"]
    qty = option["lotsize"] or config.LOT_SIZE
    target_txt = "ladder" if event.target is None else f"{event.target:.2f}"
    logger.info("[STAGE2] CONFIRM %s option_price=%.2f qty=%s | SL=%.2f target=%s",
                option["symbol"], event.price, qty, event.stop_loss, target_txt)
    if not config.DRY_RUN:
        client.place_market_order(
            exchange=config.index_config()["option_exchange"], tradingsymbol=option["symbol"],
            symboltoken=option["token"], transaction_type="BUY", quantity=qty,
            producttype=config.PRODUCT_TYPE)
    return {"symbol": option["symbol"], "token": option["token"], "quantity": qty,
            "strat": pending["strat"], "last_ts": pending["last_ts"], "otype": pending["otype"]}


def sell_held(client, held, event):
    reason = event.reason.value if event.reason else "unknown"
    price = "n/a" if event.price is None else f"{event.price:.2f}"
    logger.info("[EXIT] (%s) %s option_price=%s qty=%s",
                reason, held["symbol"], price, held["quantity"])
    if not config.DRY_RUN:
        client.place_market_order(
            exchange=config.index_config()["option_exchange"], tradingsymbol=held["symbol"],
            symboltoken=held["token"], transaction_type="SELL", quantity=held["quantity"],
            producttype=config.PRODUCT_TYPE)


def main():
    client = AngelBrokingClient(config.API_KEY, config.CLIENT_CODE, config.PASSWORD, config.TOTP_SECRET)
    client.login()

    scrip_master = load_scrip_master()

    if config.STRATEGY == "SMMA_CROSS":
        mode = getattr(config, "SMMA_SOURCE", "INDEX")
        if mode == "INDEX_THEN_OPTION":
            run_index_option_confirm_loop(client, scrip_master)
            return
        if mode == "OPTION":
            run_option_chart_loop(client, scrip_master)
            return

    strategy = build_strategy()

    held_option = None  # dict with symbol/token/quantity while a position is open
    last_seen_ts = None

    # Candles that closed before the bot started are HISTORY: they warm up
    # the strategy state (SMA, setups) but must NEVER place orders. Only
    # candles that close after this moment are tradeable.
    start_time = datetime.now()
    interval_s = INTERVAL_SECONDS[active_interval()]
    warmup_done = False

    logger.info("Bot started. DRY_RUN=%s, strategy=%s, interval=%s, sma=%s",
                config.DRY_RUN, config.STRATEGY, active_interval(), config.SMA_PERIOD)

    while True:
        try:
            now = datetime.now()

            if held_option and is_past_square_off(now):
                event = strategy.force_exit(price=None)
                exit_position(client, held_option, event)
                held_option = None

            for candle in fetch_new_candles(client, last_seen_ts):
                last_seen_ts = candle.timestamp

                is_live = candle.timestamp + timedelta(seconds=interval_s) > start_time
                if not is_live:
                    strategy.on_closed_candle(candle)  # state only - no orders on history
                    continue

                if not warmup_done:
                    warmup_done = True
                    if strategy.state == "IN_POSITION":
                        # A "position" opened during warm-up was never real -
                        # discard it instead of selling something we don't hold.
                        strategy.force_exit(price=None)
                        logger.info("Discarded phantom warm-up position; starting flat")
                    logger.info("Warm-up complete; live from %s", candle.timestamp)

                event = strategy.on_closed_candle(candle)

                if event.note:
                    logger.info("strategy note: %s", event.note)

                if event.signal in (Signal.ENTER_LONG_CE, Signal.ENTER_SHORT_PE) and not held_option:
                    if is_past_entry_cutoff(datetime.now()):
                        # Too close to square-off for the trade to work -
                        # discard the strategy's position state, take no order.
                        strategy.force_exit(price=None)
                        logger.info("Entry signal suppressed - past %02d:%02d no-entry cutoff",
                                    *config.NO_ENTRY_AFTER_HOUR_MINUTE)
                    else:
                        option_type = "CE" if event.signal == Signal.ENTER_LONG_CE else "PE"
                        held_option = enter_position(client, scrip_master, candle, event, option_type)
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
