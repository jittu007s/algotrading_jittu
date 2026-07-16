"""Backtest the TWO-STAGE strategy (INDEX_THEN_OPTION) on the option chart.

For each of the last N sessions it:
  1. runs the SMMA crossover on the INDEX to generate direction signals,
  2. for each signal, resolves the strike STRIKE_OFFSET strikes OTM/ITM,
  3. fetches THAT option's own candles and runs the option-leg SMMA
     crossover to decide whether to actually BUY (the confirmation),
  4. manages the option position: exit at +OPTION_TARGET_PREMIUM_PCT, else
     trail the stop (initial = 3rd-last candle low, then 2nd-last candle low).

Usage (same .env credentials as bot.py; reads market data only):
    python backtest_option_chart.py                 # config.INDEX, config.STRIKE_OFFSET, 5 days
    python backtest_option_chart.py NIFTY 2 5       # NIFTY, +2 OTM, last 5 sessions
    python backtest_option_chart.py BANKNIFTY -1 5  # BANKNIFTY, 1 strike ITM
    python backtest_option_chart.py SENSEX 2 3      # SENSEX, +2 OTM, last 3 sessions

IMPORTANT data caveat: Angel One's scrip master only lists LIVE (unexpired)
instruments, so an offset strike from a past session whose weekly expiry has
already passed cannot be resolved and that signal is reported as SKIPPED
(reason "No ... option ..."). Option-chart backtesting is therefore reliable
only for sessions within the currently-live expiries; older days will show
skips. The index-signal side is always shown so you can see what WOULD have
been taken.
"""

import sys
import time
from datetime import datetime, time as dtime, timedelta

import config
from angel_api import AngelBrokingClient
from instruments import find_offset_option, load_scrip_master
from strategy import (Candle, ExitReason, OptionPremiumStrategy, Signal,
                      SmaCrossOptionStrategy)

NO_ENTRY_AFTER = dtime(*config.NO_ENTRY_AFTER_HOUR_MINUTE)
SQUARE_OFF = dtime(*config.SQUARE_OFF_HOUR_MINUTE)
INTERVAL = config.CANDLE_INTERVAL

# Angel One's historical (getCandleData) endpoint is strictly rate limited.
# The backtester fires one call per unique option strike, so we pace them to
# stay under the limit instead of tripping it and getting banned mid-run.
FETCH_MIN_INTERVAL = getattr(config, "BACKTEST_FETCH_MIN_INTERVAL", 1.5)
_last_fetch = [0.0]


def _throttle():
    gap = time.time() - _last_fetch[0]
    if gap < FETCH_MIN_INTERVAL:
        time.sleep(FETCH_MIN_INTERVAL - gap)
    _last_fetch[0] = time.time()


def fetch_candles(client, exchange, token, from_dt, to_dt):
    _throttle()
    raw = client.get_candles(exchange=exchange, symboltoken=token, interval=INTERVAL,
                             from_dt=from_dt, to_dt=to_dt)
    out = [Candle(timestamp=datetime.fromisoformat(r[0]).replace(tzinfo=None),
                  open=r[1], high=r[2], low=r[3], close=r[4]) for r in raw]
    out.sort(key=lambda c: c.timestamp)
    return out


def build_index_strategy():
    return SmaCrossOptionStrategy(sma_period=config.SMA_PERIOD,
                                  risk_reward=config.RISK_REWARD, signal_only=True)


def build_option_leg(target_pct):
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
        target_mode="premium_pct", target_premium_pct=target_pct, trail_mode="prev2_extreme")


def simulate_option_leg(ocandles, signal_time, validity, target_pct):
    """Warm the option-leg SMMA on candles up to the index-signal time, then
    look for the option-chart confirmation within `validity` candles and,
    if it confirms, manage the trade to exit. Returns a result dict."""
    strat = build_option_leg(target_pct)
    post = []
    for c in ocandles:
        if c.timestamp <= signal_time:
            strat.on_closed_candle(c)
        else:
            post.append(c)
    if strat.state == "IN_POSITION":
        strat.force_exit(price=None)   # a warm-up "position" was never real

    entry = None
    entry_time = sl = target = None
    age = 0
    for c in post:
        if c.timestamp.time() >= SQUARE_OFF:
            if entry is not None:
                strat.force_exit(price=None)
                return dict(confirmed=True, entry=entry, entry_time=entry_time,
                            exit=c.open, exit_time=c.timestamp, reason=ExitReason.FORCED_EOD.value,
                            sl=sl, target=target)
            return dict(confirmed=False, reason="eod_before_confirm")

        ev = strat.on_closed_candle(c)
        if entry is None:
            age += 1
            if ev.signal == Signal.ENTER_LONG_CE:
                entry, entry_time, sl, target = ev.price, c.timestamp, ev.stop_loss, ev.target
            elif age > validity:
                return dict(confirmed=False, reason="no_confirm")
        elif ev.signal == Signal.EXIT:
            return dict(confirmed=True, entry=entry, entry_time=entry_time,
                        exit=ev.price, exit_time=c.timestamp,
                        reason=ev.reason.value if ev.reason else "?", sl=sl, target=target)

    if entry is not None:
        last = post[-1]
        return dict(confirmed=True, entry=entry, entry_time=entry_time, exit=last.close,
                    exit_time=last.timestamp, reason="open_at_data_end", sl=sl, target=target)
    return dict(confirmed=False, reason="no_confirm")


def replay_day(index_day, warmup, scrip, ic, offset, day, option_cache, client, validity, target_pct):
    idx = build_index_strategy()
    for c in warmup:
        idx.on_closed_candle(c)
    if idx.state != "IDLE":
        idx.force_exit(price=None)

    trades, skips = [], []
    busy_until = None
    for c in index_day:
        ev = idx.on_closed_candle(c)
        if busy_until and c.timestamp <= busy_until:
            continue
        if ev.signal not in (Signal.ENTER_LONG_CE, Signal.ENTER_SHORT_PE):
            continue
        if c.timestamp.time() >= NO_ENTRY_AFTER:
            idx.force_exit(price=None)
            continue

        otype = "CE" if ev.signal == Signal.ENTER_LONG_CE else "PE"
        try:
            option = find_offset_option(scrip, c.close, option_type=otype, offset=offset,
                                        underlying=ic["name"], strike_step=ic["strike_step"],
                                        option_exchange=ic["option_exchange"], as_of=day)
        except LookupError as exc:
            skips.append((c.timestamp, otype, str(exc)))
            continue

        key = (option["token"], day)
        if key not in option_cache:
            frm = datetime.combine(day, dtime(9, 0))
            to = datetime.combine(day, dtime(15, 35))
            cooldown = getattr(config, "RATE_LIMIT_COOLDOWN_SECONDS", 30)
            last_exc = None
            for attempt in range(3):
                try:
                    # Only a genuine empty response (status ok, no data - e.g. an
                    # expired strike) is cached; a rate-limit/network error is
                    # transient and must NOT poison the cache, or every later
                    # signal on this strike would wrongly skip as "no candles".
                    option_cache[key] = fetch_candles(client, ic["option_exchange"],
                                                      option["token"], frm, to)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < 2:
                        time.sleep(cooldown)
            if last_exc is not None:
                skips.append((c.timestamp, otype, f"fetch failed: {last_exc}"))
                continue
        ocandles = option_cache[key]
        if not ocandles:
            skips.append((c.timestamp, otype, "no option candles"))
            continue

        res = simulate_option_leg(ocandles, c.timestamp, validity, target_pct)
        if res["confirmed"]:
            move = res["exit"] - res["entry"]
            res.update(otype=otype, strike=option["strike"], symbol=option["symbol"],
                       index_time=c.timestamp, points=move, pct=move / res["entry"] * 100.0)
            trades.append(res)
            busy_until = res["exit_time"]
        else:
            skips.append((c.timestamp, otype, res["reason"]))
    return trades, skips


def main():
    index_name = config.INDEX
    offset = config.STRIKE_OFFSET
    n_days = 5
    positional = [a for a in sys.argv[1:]]
    if positional and positional[0].upper() in config.INDEXES:
        index_name = positional.pop(0).upper()
    if positional:
        offset = int(positional.pop(0))
    if positional:
        n_days = int(positional.pop(0))

    ic = config.index_config(index_name)
    target_pct = config.OPTION_TARGET_PREMIUM_PCT
    validity = config.OPTION_CONFIRM_VALIDITY
    calendar_days = max(21, int(n_days * 1.7) + 7)

    client = AngelBrokingClient(config.API_KEY, config.CLIENT_CODE, config.PASSWORD, config.TOTP_SECRET)
    client.login()
    scrip = load_scrip_master()

    now = datetime.now()
    index_candles = fetch_candles(client, ic["under_exchange"], ic["under_token"],
                                  now - timedelta(days=calendar_days), now)

    all_dates = sorted({c.timestamp.date() for c in index_candles})
    target_dates = all_dates[-n_days:]
    print(f"{index_name} offset {offset:+d}  target +{target_pct*100:.0f}%  "
          f"confirm-window {validity} candles  ({INTERVAL})")
    print(f"Index data: {len(index_candles)} candles over {len(all_dates)} sessions "
          f"({all_dates[0]} .. {all_dates[-1]}); testing {len(target_dates)} session(s)\n")

    option_cache = {}
    g_trades = g_wins = 0
    g_points = g_pct = 0.0
    total_skips = 0
    for day in target_dates:
        warmup = [c for c in index_candles if c.timestamp.date() < day]
        day_candles = [c for c in index_candles if c.timestamp.date() == day]
        if not day_candles:
            continue
        trades, skips = replay_day(day_candles, warmup, scrip, ic, offset, day,
                                   option_cache, client, validity, target_pct)
        day_pts = sum(t["points"] for t in trades)
        day_pct = sum(t["pct"] for t in trades)
        wins = sum(1 for t in trades if t["points"] > 0)
        g_trades += len(trades); g_wins += wins
        g_points += day_pts; g_pct += day_pct
        total_skips += len(skips)
        print(f"  {day}: {len(trades)} trade(s), premium net {day_pts:+.2f} pts "
              f"({day_pct:+.1f}% on entry), {len(skips)} skipped")
        for t in trades:
            tgt = "ladder" if t["target"] is None else f"{t['target']:.2f}"
            print(f"      {t['otype']} {t['strike']:.0f} idx@{t['index_time']:%H:%M} "
                  f"buy {t['entry_time']:%H:%M} entry={t['entry']:.2f} sl={t['sl']:.2f} "
                  f"target={tgt} -> exit {t['exit_time']:%H:%M} {t['exit']:.2f} "
                  f"({t['reason']}) {t['points']:+.2f} ({t['pct']:+.1f}%)")
        for ts, otype, reason in skips:
            print(f"      · skip {ts:%H:%M} {otype}: {reason}")

    wr = (g_wins / g_trades) if g_trades else 0.0
    print(f"\nTOTAL: {g_trades} confirmed trade(s), win rate {wr:.0%}, "
          f"premium net {g_points:+.2f} pts ({g_pct:+.1f}% summed on entry), "
          f"{total_skips} signal(s) skipped")
    print("\n(Premium points/percent are on the OPTION price. Real fills differ by")
    print(" spread + brokerage; skips are mostly expired strikes not in the live master.)")
    client.logout()


if __name__ == "__main__":
    main()
