"""Backtest the ICT sweep->MSS->FVG strategy over recent sessions.

Run from algo-trading/:  python -m ict.backtest_ict [N_SESSIONS]

Uses 5-minute candles for setups (1-minute refinement can be added at the
cost of much heavier data pulls), daily+1H for bias, and the same session
rules as the engine. P&L is reported in spot points and in approximate
premium rupees via the configured delta.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, time as dtime, timedelta

from . import config as cfg_mod
from .datafeed import PollingFeed
from .models import Bias, Candle, LiquidityLevel, SwingKind
from .risk import DayRiskManager, size_position
from .structure import (SetupScanner, combine_bias, detect_bias, entry_level,
                        entry_triggered, setup_invalidated, find_swings)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("ict.backtest")


def day_levels(daily: list[Candle], day, m5: list[Candle], skip_min: int):
    prev = [c for c in daily if c.timestamp.date() < day]
    levels = []
    if prev:
        levels.append(LiquidityLevel(prev[-1].high, SwingKind.HIGH, "pdh"))
        levels.append(LiquidityLevel(prev[-1].low, SwingKind.LOW, "pdl"))
    ors = [c for c in m5 if c.timestamp.date() == day
           and c.timestamp.time() < dtime(9, 15 + skip_min)]
    if ors:
        levels.append(LiquidityLevel(max(c.high for c in ors), SwingKind.HIGH, "or_high"))
        levels.append(LiquidityLevel(min(c.low for c in ors), SwingKind.LOW, "or_low"))
    return levels


def run(n_sessions: int = 10) -> None:
    cfg = cfg_mod.load()
    feed = PollingFeed()
    now = datetime.now()

    m5 = feed.fetch("FIVE_MINUTE", now - timedelta(days=int(n_sessions * 1.7) + 10), now)
    h1 = feed.fetch("ONE_HOUR", now - timedelta(days=60), now)
    daily = feed.fetch("ONE_DAY", now - timedelta(days=180), now)

    dates = sorted({c.timestamp.date() for c in m5})[-n_sessions:]
    all_trades = []
    funnel = {"sweeps": 0, "sweep_expired_no_mss": 0, "mss_without_fvg": 0,
              "setups": 0, "entered": 0, "expired_unentered": 0,
              "invalidated": 0, "skipped_oversize": 0, "neutral_days": 0}
    oversize_dists = []

    for day in dates:
        day_m5 = [c for c in m5 if c.timestamp.date() == day]
        warm_m5 = [c for c in m5 if c.timestamp.date() < day][-300:]
        if not day_m5:
            continue

        d_bias = detect_bias([c for c in daily if c.timestamp.date() < day],
                             cfg.structure.swing_k_bias)
        h_bias = detect_bias([c for c in h1 if c.timestamp.date() < day][-120:],
                             cfg.structure.swing_k_bias)
        bias = combine_bias(d_bias, h_bias)

        scanner = SetupScanner(
            swing_k=cfg.structure.swing_k_setup,
            min_pierce=cfg.structure.sweep_min_pierce,
            needs_rejection=cfg.structure.sweep_needs_rejection,
            body_mult=cfg.structure.displacement_body_mult,
            avg_body_period=cfg.structure.avg_body_period,
            validity=cfg.structure.setup_validity_candles)
        scanner.set_static_levels(day_levels(daily, day, m5, cfg.session.skip_first_minutes))
        for c in warm_m5:
            scanner.on_candle(c, Bias.NEUTRAL)   # history for swings/avg body only

        risk = DayRiskManager(cfg.capital, cfg.risk.max_daily_loss_pct,
                              cfg.risk.equity_drawdown_stop_pct)
        risk.new_session(day)

        pending = None
        pending_ttl = 0
        trade = None
        day_trades = []

        session_start = dtime(9, 15 + cfg.session.skip_first_minutes)

        for c in day_m5:
            t = c.timestamp.time()

            # ---- manage open trade -----------------------------------
            if trade is not None:
                long = trade["dir"] == Bias.BULLISH
                hit = c.low <= trade["stop"] if long else c.high >= trade["stop"]
                if hit or t >= cfg.session.square_off:
                    exit_spot = trade["stop"] if hit else c.close
                    move = (exit_spot - trade["entry"]) if long else (trade["entry"] - exit_spot)
                    pnl = move * cfg.options.delta_assumed * 75 * trade["lots"]
                    risk.record_trade(pnl)
                    trade.update(exit=exit_spot, exit_t=c.timestamp,
                                 reason="stop" if hit else "square_off",
                                 points=move, pnl=pnl)
                    day_trades.append(trade)
                    trade = None
                else:
                    fav = (c.high - trade["entry"]) if long else (trade["entry"] - c.low)
                    if not trade["partial"] and fav >= trade["risk"]:
                        trade["partial"] = True
                        trade["stop"] = trade["entry"]
                    swings = find_swings(scanner.candles[-40:], scanner.swing_k)
                    if long:
                        lows = [s.price for s in swings if s.kind == SwingKind.LOW]
                        if lows:
                            trade["stop"] = max(trade["stop"], lows[-1])
                    else:
                        highs = [s.price for s in swings if s.kind == SwingKind.HIGH]
                        if highs:
                            trade["stop"] = min(trade["stop"], highs[-1])

            # ---- scan / enter ----------------------------------------
            setup = scanner.on_candle(c, bias)
            in_window = session_start <= t < cfg.session.no_entry_after
            if setup and trade is None and in_window and risk.can_trade():
                pending, pending_ttl = setup, cfg.structure.setup_validity_candles

            if pending and trade is None:
                pending_ttl -= 1
                if setup_invalidated(pending, c):
                    funnel["invalidated"] += 1
                    pending = None
                elif pending_ttl <= 0:
                    funnel["expired_unentered"] += 1
                    pending = None
                elif in_window and risk.can_trade() and \
                        entry_triggered(pending, c, cfg.structure.entry_point):
                    entry = entry_level(pending, cfg.structure.entry_point)
                    dist = abs(entry - pending.stop_spot)
                    lots = size_position(cfg.capital, cfg.risk.risk_per_trade_pct,
                                         dist, cfg.options.delta_assumed, 75)
                    if lots >= 1:
                        funnel["entered"] += 1
                        trade = {"dir": pending.bias, "entry": entry,
                                 "stop": pending.stop_spot, "risk": dist,
                                 "lots": lots, "partial": False,
                                 "entry_t": c.timestamp}
                    else:
                        funnel["skipped_oversize"] += 1
                        oversize_dists.append(dist)
                    pending = None

        if bias == Bias.NEUTRAL:
            funnel["neutral_days"] += 1
        for k in ("sweeps", "sweep_expired_no_mss", "mss_without_fvg", "setups"):
            funnel[k] += scanner.stats.get(k, 0)
        print(f"{day} bias={bias.value:<8} trades={len(day_trades)}"
              f"  pnl=₹{sum(tr['pnl'] for tr in day_trades):+9.0f}"
              + ("  [HALTED: " + risk.halt_reason + "]" if risk.halted else ""))
        for tr in day_trades:
            print(f"    {tr['dir'].value:<8} {tr['entry_t']:%H:%M}->{tr['exit_t']:%H:%M} "
                  f"entry={tr['entry']:.1f} stop_out={tr['exit']:.1f} ({tr['reason']}) "
                  f"{tr['points']:+.1f} pts x{tr['lots']} lots = ₹{tr['pnl']:+.0f}")
        all_trades += day_trades

    wins = [t for t in all_trades if t["pnl"] > 0]
    total = sum(t["pnl"] for t in all_trades)
    print(f"\nTOTAL: {len(all_trades)} trades, "
          f"win rate {(len(wins)/len(all_trades)) if all_trades else 0:.0%}, "
          f"net ~₹{total:+.0f} (delta-approximated premium, before charges)")
    print("\nFUNNEL (why setups did or didn't become trades):")
    print(f"  neutral-bias days (no trading): {funnel['neutral_days']}")
    print(f"  sweeps detected:            {funnel['sweeps']}")
    print(f"  ├─ expired without MSS:     {funnel['sweep_expired_no_mss']}")
    print(f"  ├─ MSS but no FVG:          {funnel['mss_without_fvg']}")
    print(f"  └─ full setups formed:      {funnel['setups']}")
    print(f"     ├─ entered:              {funnel['entered']}")
    print(f"     ├─ expired unentered:    {funnel['expired_unentered']} (no pullback to '{cfg.structure.entry_point}')")
    print(f"     ├─ invalidated pre-fill: {funnel['invalidated']}")
    print(f"     └─ skipped oversize:     {funnel['skipped_oversize']}"
          + (f" (SL distances {min(oversize_dists):.0f}-{max(oversize_dists):.0f} pts;"
             f" budget allows ~{cfg.capital*cfg.risk.risk_per_trade_pct/100/(cfg.options.delta_assumed*75):.0f} pts)"
             if oversize_dists else ""))


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
