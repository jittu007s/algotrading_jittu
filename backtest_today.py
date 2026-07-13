"""Replay today's (or any day's) 3-minute Nifty candles through the exact
live strategy and print what the bot WOULD have done - entries, exits,
and total points - without placing any orders.

Usage (uses the same .env credentials as bot.py):
    python backtest_today.py               # today
    python backtest_today.py 2026-07-10    # a specific past date

Also writes the day's candles to nifty_<date>_3m.csv so the run can be
re-analysed later without another API call.

Reads market data only - never places orders, regardless of DRY_RUN.
"""

import csv
import sys
from datetime import datetime, time as dtime

import config
from angel_api import AngelBrokingClient
from strategy import Candle, ExitReason, Signal, SmaCrossOptionStrategy

NO_ENTRY_AFTER = dtime(*config.NO_ENTRY_AFTER_HOUR_MINUTE)
SQUARE_OFF = dtime(*config.SQUARE_OFF_HOUR_MINUTE)


def fetch_day(day: datetime.date):
    """Fetch the target day PLUS several prior days for SMMA warm-up, so
    the moving average is continuous across sessions (like TradingView and
    the live bot) instead of being seeded from the day's first candles.

    Returns (warmup_candles, day_candles)."""
    from datetime import timedelta

    client = AngelBrokingClient(config.API_KEY, config.CLIENT_CODE, config.PASSWORD, config.TOTP_SECRET)
    client.login()
    raw = client.get_candles(
        exchange=config.UNDERLYING_EXCHANGE,
        symboltoken=config.UNDERLYING_TOKEN,
        interval=config.CANDLE_INTERVAL,
        from_dt=datetime.combine(day - timedelta(days=6), dtime(9, 15)),
        to_dt=datetime.combine(day, dtime(15, 30)),
    )
    client.logout()
    candles = [
        Candle(
            timestamp=datetime.fromisoformat(r[0]).replace(tzinfo=None),
            open=r[1], high=r[2], low=r[3], close=r[4],
        )
        for r in raw
    ]
    candles.sort(key=lambda c: c.timestamp)
    warmup = [c for c in candles if c.timestamp.date() < day]
    day_candles = [c for c in candles if c.timestamp.date() == day]
    return warmup, day_candles


def save_csv(candles, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close"])
        for c in candles:
            w.writerow([c.timestamp.isoformat(), c.open, c.high, c.low, c.close])


def replay(candles, warmup=()):
    """Run the strategy over the day with the live bot's session rules:
    warm up silently on prior days' candles, then trade the target day
    with no new entries at/after NO_ENTRY_AFTER and square-off at
    SQUARE_OFF."""
    strategy = SmaCrossOptionStrategy(sma_period=config.SMA_PERIOD, risk_reward=config.RISK_REWARD)
    for c in warmup:
        strategy.on_closed_candle(c)
    if strategy.state != "IDLE":
        strategy.force_exit(price=None)  # discard any phantom warm-up position

    trades, open_trade = [], None

    for candle in candles:
        t = candle.timestamp.time()

        if open_trade and t >= SQUARE_OFF:
            strategy.force_exit(price=candle.close)
            open_trade.update(exit_time=candle.timestamp, exit_price=candle.close,
                              reason=ExitReason.FORCED_EOD.value)
            trades.append(open_trade)
            open_trade = None
            continue

        event = strategy.on_closed_candle(candle)

        if event.signal in (Signal.ENTER_LONG_CE, Signal.ENTER_SHORT_PE):
            if t >= NO_ENTRY_AFTER:
                strategy.force_exit(price=None)  # suppressed, same as live bot
                continue
            open_trade = {
                "side": "LONG (CE)" if event.signal == Signal.ENTER_LONG_CE else "SHORT (PE)",
                "entry_time": candle.timestamp, "entry": event.price,
                "sl": event.stop_loss, "target": event.target, "trailed": False,
            }
        elif event.note == "trailing_activated" and open_trade:
            open_trade["trailed"] = True
        elif event.signal == Signal.EXIT and open_trade:
            open_trade.update(exit_time=candle.timestamp, exit_price=event.price,
                              reason=event.reason.value if event.reason else "?")
            trades.append(open_trade)
            open_trade = None

    for tr in trades:
        move = tr["exit_price"] - tr["entry"]
        tr["points"] = move if tr["side"].startswith("LONG") else -move
    return trades


def report(day, trades):
    print(f"\n=== Strategy replay for {day} "
          f"(SMMA {config.SMA_PERIOD}, {config.CANDLE_INTERVAL}, RR 1:{config.RISK_REWARD:g}, "
          f"trailing, no entries after {NO_ENTRY_AFTER}) ===\n")
    if not trades:
        print("No trades were signalled.")
        return
    for tr in trades:
        print(f"  {tr['side']:<10} {tr['entry_time']:%H:%M} -> {tr['exit_time']:%H:%M}  "
              f"entry={tr['entry']:.2f} sl={tr['sl']:.2f} exit={tr['exit_price']:.2f} "
              f"({tr['reason']}{', trailed' if tr['trailed'] else ''})  "
              f"points={tr['points']:+.2f}")
    wins = [t for t in trades if t["points"] > 0]
    total = sum(t["points"] for t in trades)
    print(f"\n  Trades: {len(trades)}  Wins: {len(wins)}  Losses: {len(trades)-len(wins)}"
          f"  Win rate: {len(wins)/len(trades):.0%}")
    print(f"  NET: {total:+.2f} points on the underlying")
    print("  (option P&L differs: ATM delta ~0.5 at entry, plus time decay)")


if __name__ == "__main__":
    day = datetime.strptime(sys.argv[1], "%Y-%m-%d").date() if len(sys.argv) > 1 else datetime.now().date()
    warmup, candles = fetch_day(day)
    if not candles:
        print(f"No candle data returned for {day} (holiday/weekend?)")
        sys.exit(1)
    csv_path = f"nifty_{day:%Y%m%d}_3m.csv"
    save_csv(warmup + candles, csv_path)
    print(f"Fetched {len(candles)} candles for {day} (+{len(warmup)} warm-up) -> saved to {csv_path}")
    trades = replay(candles, warmup=warmup)
    report(day, trades)
