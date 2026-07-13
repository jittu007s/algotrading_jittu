"""Backtest the last N trading sessions (default 5) and compare all three
strategies on identical data:

  SMMA_CROSS - the original SMMA-crossover rules (live default)
  ORB        - Opening Range Breakout (15-min range, one shot per side)
  REGIME     - Regime-Adaptive confidence-scored strategy (see
               quant_strategy.py / STRATEGY_BLUEPRINT.md)

Usage (same .env credentials as bot.py; reads market data only, never
places orders):
    python backtest_today.py                  # last 5 sessions, all strategies
    python backtest_today.py 22               # last ~1 month of sessions
    python backtest_today.py 22 ORB           # 1 month, ORB only
    python backtest_today.py 2026-07-10       # one specific date
    python backtest_today.py 2026-07-10 ORB   # one date, one strategy

Each session is replayed with the live bot's rules: SMMA/indicators warm
up on all candles BEFORE that session, entries blocked at/after 15:00,
square-off at 15:20. Candles are cached to nifty_3m_history.csv.
"""

import csv
import sys
from datetime import datetime, time as dtime, timedelta

import config
from angel_api import AngelBrokingClient
from quant_strategy import RegimeAdaptiveStrategy
from strategy import Candle, ExitReason, Signal, SmaCrossOptionStrategy, OpeningRangeBreakout

NO_ENTRY_AFTER = dtime(*config.NO_ENTRY_AFTER_HOUR_MINUTE)
SQUARE_OFF = dtime(*config.SQUARE_OFF_HOUR_MINUTE)

# name -> (factory, candle interval it runs on)
STRATEGIES = {
    "SMMA_CROSS": (lambda: SmaCrossOptionStrategy(
        sma_period=config.SMA_PERIOD, risk_reward=config.RISK_REWARD),
        config.CANDLE_INTERVAL),
    "ORB": (lambda: OpeningRangeBreakout(
        sma_period=config.SMA_PERIOD, risk_reward=config.RISK_REWARD,
        or_minutes=config.OR_MINUTES, max_risk_points=config.ORB_MAX_RISK_POINTS),
        config.ORB_CANDLE_INTERVAL),
    "REGIME": (lambda: RegimeAdaptiveStrategy(), config.CANDLE_INTERVAL),
}

_INTERVAL_LABEL = {"ONE_MINUTE": "1m", "THREE_MINUTE": "3m", "FIVE_MINUTE": "5m",
                   "TEN_MINUTE": "10m", "FIFTEEN_MINUTE": "15m"}


def fetch_history(client, interval, calendar_days=21):
    now = datetime.now()
    raw = client.get_candles(
        exchange=config.UNDERLYING_EXCHANGE,
        symboltoken=config.UNDERLYING_TOKEN,
        interval=interval,
        from_dt=now - timedelta(days=calendar_days),
        to_dt=now,
    )
    candles = [
        Candle(timestamp=datetime.fromisoformat(r[0]).replace(tzinfo=None),
               open=r[1], high=r[2], low=r[3], close=r[4])
        for r in raw
    ]
    candles.sort(key=lambda c: c.timestamp)
    return candles


def save_csv(candles, interval):
    path = f"nifty_{_INTERVAL_LABEL.get(interval, interval)}_history.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close"])
        for c in candles:
            w.writerow([c.timestamp.isoformat(), c.open, c.high, c.low, c.close])


def replay_day(day_candles, warmup, factory):
    strategy = factory()
    for c in warmup:
        strategy.on_closed_candle(c)
    if strategy.state != "IDLE":
        strategy.force_exit(price=None)

    trades, open_trade, diag = [], None, []
    for candle in day_candles:
        t = candle.timestamp.time()

        if open_trade and t >= SQUARE_OFF:
            strategy.force_exit(price=candle.close)
            open_trade.update(exit_time=candle.timestamp, exit_price=candle.close,
                              reason=ExitReason.FORCED_EOD.value)
            trades.append(open_trade)
            open_trade = None
            continue

        event = strategy.on_closed_candle(candle)

        if event.note and event.signal not in (Signal.ENTER_LONG_CE, Signal.ENTER_SHORT_PE) \
                and ("OR " in event.note or "skipped" in event.note):
            diag.append(f"{candle.timestamp:%H:%M} {event.note}")

        if event.signal in (Signal.ENTER_LONG_CE, Signal.ENTER_SHORT_PE):
            if t >= NO_ENTRY_AFTER:
                strategy.force_exit(price=None)
                continue
            open_trade = {
                "side": "LONG (CE)" if event.signal == Signal.ENTER_LONG_CE else "SHORT (PE)",
                "entry_time": candle.timestamp, "entry": event.price,
                "sl": event.stop_loss, "note": event.note or "",
            }
        elif event.signal == Signal.EXIT and open_trade:
            open_trade.update(exit_time=candle.timestamp, exit_price=event.price,
                              reason=event.reason.value if event.reason else "?")
            trades.append(open_trade)
            open_trade = None

    # a position still open at data end (e.g. running the script mid-session)
    if open_trade:
        last = day_candles[-1]
        open_trade.update(exit_time=last.timestamp, exit_price=last.close, reason="open_at_data_end")
        trades.append(open_trade)

    for tr in trades:
        move = tr["exit_price"] - tr["entry"]
        tr["points"] = move if tr["side"].startswith("LONG") else -move
    return trades, diag


def summarize(trades):
    wins = [t for t in trades if t["points"] > 0]
    total = sum(t["points"] for t in trades)
    return len(trades), len(wins), total


def main():
    target_dates = None
    n_days = 5
    selected = dict(STRATEGIES)
    for arg in sys.argv[1:]:
        if arg.upper() in STRATEGIES:
            selected = {arg.upper(): STRATEGIES[arg.upper()]}
        elif "-" in arg:
            target_dates = [datetime.strptime(arg, "%Y-%m-%d").date()]
        else:
            n_days = int(arg)

    # enough calendar days to cover the requested sessions plus warm-up
    calendar_days = max(21, int(n_days * 1.7) + 7)

    client = AngelBrokingClient(config.API_KEY, config.CLIENT_CODE, config.PASSWORD, config.TOTP_SECRET)
    client.login()
    data = {}
    for interval in {iv for _f, iv in selected.values()}:
        data[interval] = fetch_history(client, interval, calendar_days=calendar_days)
        save_csv(data[interval], interval)
    client.logout()

    base = data[next(iter(selected.values()))[1]]
    all_dates = sorted({c.timestamp.date() for c in base})
    if target_dates is None:
        target_dates = all_dates[-n_days:]
    print(f"Data: {len(base)} base candles across {len(all_dates)} sessions "
          f"({all_dates[0]} .. {all_dates[-1]}); testing {len(target_dates)} session(s)\n")

    grand = {}
    for name, (factory, interval) in selected.items():
        candles = data[interval]
        print(f"================ {name} ({_INTERVAL_LABEL.get(interval, interval)} candles) ================")
        g_trades, g_wins, g_total = 0, 0, 0.0
        for day in target_dates:
            warmup = [c for c in candles if c.timestamp.date() < day]
            day_candles = [c for c in candles if c.timestamp.date() == day]
            if not day_candles:
                continue
            trades, diag = replay_day(day_candles, warmup, factory)
            n, w, tot = summarize(trades)
            g_trades += n; g_wins += w; g_total += tot
            print(f"  {day}: {n} trades, net {tot:+8.2f} pts")
            for d in diag:
                print(f"      · {d}")
            for tr in trades:
                print(f"      {tr['side']:<10} {tr['entry_time']:%H:%M}->{tr['exit_time']:%H:%M} "
                      f"entry={tr['entry']:.2f} sl={tr['sl']:.2f} exit={tr['exit_price']:.2f} "
                      f"({tr['reason']}) {tr['points']:+.2f}"
                      + (f"  [{tr['note']}]" if tr.get("note") else ""))
        wr = (g_wins / g_trades) if g_trades else 0.0
        print(f"  TOTAL: {g_trades} trades, win rate {wr:.0%}, NET {g_total:+.2f} points\n")
        grand[name] = (g_trades, wr, g_total)

    print("================ COMPARISON ================")
    for name, (n, wr, tot) in sorted(grand.items(), key=lambda kv: -kv[1][2]):
        print(f"  {name:<12} trades={n:<3} win rate={wr:>4.0%}  net={tot:+9.2f} pts")
    print("\n(Points are on the Nifty index. Option P&L differs: ATM delta ~0.5,")
    print(" theta decay, and ~2-4 points of spread+charges per round trip.)")


if __name__ == "__main__":
    main()
