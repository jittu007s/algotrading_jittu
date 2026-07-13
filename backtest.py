"""Offline backtest of SmaCrossOptionStrategy against a CSV of historical
underlying candles (e.g. exported via AngelBrokingClient.get_candles, or
any other Nifty 15m/5m OHLC source).

CSV columns expected: timestamp,open,high,low,close

IMPORTANT: this reports P&L in underlying (Nifty) points, as an
approximation of the strategy's edge - it does NOT model option premium
behaviour (time decay, IV changes, delta). Real option P&L will differ;
use this only to sanity-check the entry/exit rules before going live.

Usage:
    python backtest.py path/to/nifty_15m.csv
"""

import csv
import sys
from datetime import datetime

from strategy import Candle, Signal, SmaCrossOptionStrategy


def load_candles(path):
    candles = []
    with open(path) as f:
        for row in csv.DictReader(f):
            candles.append(Candle(
                timestamp=datetime.fromisoformat(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            ))
    candles.sort(key=lambda c: c.timestamp)
    return candles


def run_backtest(candles, **strategy_kwargs):
    strategy = SmaCrossOptionStrategy(**strategy_kwargs)
    trades = []
    open_trade = None

    for candle in candles:
        event = strategy.on_closed_candle(candle)
        if event.signal in (Signal.ENTER_LONG_CE, Signal.ENTER_SHORT_PE):
            open_trade = {
                "side": "LONG" if event.signal == Signal.ENTER_LONG_CE else "SHORT",
                "entry_time": candle.timestamp,
                "entry_price": event.price,
                "stop_loss": event.stop_loss,
                "target": event.target,
            }
        elif event.signal == Signal.EXIT and open_trade:
            open_trade["exit_time"] = candle.timestamp
            open_trade["exit_price"] = event.price
            open_trade["reason"] = event.reason.value if event.reason else None
            move = event.price - open_trade["entry_price"]
            open_trade["points"] = move if open_trade["side"] == "LONG" else -move
            trades.append(open_trade)
            open_trade = None

    return trades


def summarize(trades):
    if not trades:
        print("No trades generated.")
        return

    wins = [t for t in trades if t["points"] > 0]
    losses = [t for t in trades if t["points"] <= 0]
    total_points = sum(t["points"] for t in trades)

    print(f"Trades: {len(trades)}  Wins: {len(wins)}  Losses: {len(losses)}  "
          f"Win rate: {len(wins) / len(trades):.1%}")
    print(f"Total points (underlying): {total_points:.2f}  "
          f"Avg points/trade: {total_points / len(trades):.2f}\n")

    for t in trades:
        print(f"  {t['side']:<5} {t['entry_time']} -> {t.get('exit_time')} | "
              f"entry={t['entry_price']:.2f} exit={t.get('exit_price'):.2f} "
              f"reason={t.get('reason')} points={t['points']:.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python backtest.py <candles.csv>")
        sys.exit(1)

    all_candles = load_candles(sys.argv[1])
    all_trades = run_backtest(all_candles)
    summarize(all_trades)
