"""Kronos foundation-model options strategy + backtest for weekly index options.

Uses Kronos (https://github.com/shiyu-coder/Kronos) - an open-source foundation
model for financial candlesticks - via `KronosTokenizer`, `Kronos` and
`KronosPredictor`. Kronos forecasts the INDEX's next candles; the sign and size
of the predicted move choose a side (CE when it predicts up, PE when down) and
the weekly option leg (STRIKE_OFFSET strikes OTM/ITM) executes and manages the
trade. Works for NIFTY, BANKNIFTY and SENSEX weekly options through the shared
`config.INDEXES` map and `instruments.find_offset_option`.

Pieces
------
  KronosForecaster    - loads the tokenizer + model + predictor (lazily) and
                        turns a window of index candles into a `Forecast`. The
                        predictor is dependency-injectable so the signal logic
                        and backtest are unit-testable without the heavy model.
  KronosSignalStrategy- turns a Forecast into a CE / PE / no-trade decision.
  KronosOptionBacktest- walk-forward backtest over the last N sessions per
                        index/offset, reusing the tested option leg (2-close
                        confirm + percentage-ladder exit) from
                        backtest_option_chart.

Install (heavy; ideally a CUDA GPU):
    pip install -r requirements-kronos.txt
    # then clone Kronos so `from model import ...` resolves, or pip-install it.
See README / requirements-kronos.txt.

NOTE on volume: NIFTY/SENSEX index candles carry no volume, so the volume and
amount columns Kronos expects are filled with zeros. Predictions still work but
are driven by price alone.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta
from typing import List

import config
from instruments import find_offset_option
from strategy import Candle, ExitReason, OptionPremiumStrategy, Signal
# Reuse the tested option-leg simulation, throttled fetch and builders.
from backtest_option_chart import (build_option_leg, fetch_candles,
                                    simulate_option_leg)

INTERVAL = config.CANDLE_INTERVAL
NO_ENTRY_AFTER = dtime(*config.NO_ENTRY_AFTER_HOUR_MINUTE)
SQUARE_OFF = dtime(*config.SQUARE_OFF_HOUR_MINUTE)

_INTERVAL_SECONDS = {
    "ONE_MINUTE": 60, "THREE_MINUTE": 180, "FIVE_MINUTE": 300, "TEN_MINUTE": 600,
    "FIFTEEN_MINUTE": 900, "THIRTY_MINUTE": 1800, "ONE_HOUR": 3600,
}


# ---------------------------------------------------------------------------
# Forecast container - decouples the signal logic from pandas / the model.
# ---------------------------------------------------------------------------
@dataclass
class Forecast:
    cur_close: float
    open: List[float] = field(default_factory=list)
    high: List[float] = field(default_factory=list)
    low: List[float] = field(default_factory=list)
    close: List[float] = field(default_factory=list)

    @property
    def terminal_close(self) -> float:
        return self.close[-1] if self.close else self.cur_close

    @property
    def move_pct(self) -> float:
        """Predicted % move from the current close to the horizon-end close."""
        if not self.close or self.cur_close == 0:
            return 0.0
        return (self.terminal_close - self.cur_close) / self.cur_close * 100.0

    @property
    def max_up_pct(self) -> float:
        if not self.high or self.cur_close == 0:
            return 0.0
        return (max(self.high) - self.cur_close) / self.cur_close * 100.0

    @property
    def max_down_pct(self) -> float:
        if not self.low or self.cur_close == 0:
            return 0.0
        return (self.cur_close - min(self.low)) / self.cur_close * 100.0


def _load_kronos():
    """Import Kronos from whatever layout is installed, with a clear error."""
    for mod in ("model", "kronos", "Kronos"):
        try:
            m = __import__(mod, fromlist=["Kronos", "KronosTokenizer", "KronosPredictor"])
            return m.Kronos, m.KronosTokenizer, m.KronosPredictor
        except Exception:
            continue
    raise ImportError(
        "Kronos is not importable. Install it, e.g.:\n"
        "  git clone https://github.com/shiyu-coder/Kronos\n"
        "  pip install -r Kronos/requirements.txt  (torch, huggingface_hub, ...)\n"
        "and make its `model` package importable (add the clone to PYTHONPATH),\n"
        "or `pip install kronos-forecasting` if a packaged build is available.")


class KronosForecaster:
    """Lazily loads Kronos and turns index candles into a `Forecast`.

    `predictor` may be injected (any object with a `.predict(...)` returning a
    DataFrame that has open/high/low/close columns) - used for testing and to
    let callers share one loaded model across indices.
    """

    def __init__(self, model_name: str = None, tokenizer_name: str = None,
                 device: str = None, max_context: int = None,
                 T: float = None, top_p: float = None, sample_count: int = None,
                 predictor=None, interval: str = INTERVAL):
        self.model_name = model_name or config.KRONOS_MODEL
        self.tokenizer_name = tokenizer_name or config.KRONOS_TOKENIZER
        self.device = device if device is not None else config.KRONOS_DEVICE
        self.max_context = max_context or config.KRONOS_MAX_CONTEXT
        self.T = config.KRONOS_T if T is None else T
        self.top_p = config.KRONOS_TOP_P if top_p is None else top_p
        self.sample_count = config.KRONOS_SAMPLE_COUNT if sample_count is None else sample_count
        self.interval = interval
        self._predictor = predictor

    def _ensure_predictor(self):
        if self._predictor is not None:
            return
        Kronos, KronosTokenizer, KronosPredictor = _load_kronos()
        device = self.device
        if device is None:
            try:
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
        model = Kronos.from_pretrained(self.model_name)
        self._predictor = KronosPredictor(model, tokenizer, device=device,
                                          max_context=self.max_context)

    def forecast(self, candles: List[Candle], horizon: int) -> Forecast:
        """Predict the next `horizon` candles from `candles` (chronological)."""
        self._ensure_predictor()
        import pandas as pd

        step = timedelta(seconds=_INTERVAL_SECONDS.get(self.interval, 180))
        x_df = pd.DataFrame({
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            # index candles have no volume; Kronos still needs the columns.
            "volume": [float(getattr(c, "volume", 0.0) or 0.0) for c in candles],
            "amount": [0.0 for _ in candles],
        })
        x_timestamp = pd.Series([pd.Timestamp(c.timestamp) for c in candles])
        last = candles[-1].timestamp
        y_timestamp = pd.Series([pd.Timestamp(last + step * (i + 1)) for i in range(horizon)])

        pred = self._predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=horizon, T=self.T, top_p=self.top_p,
            sample_count=self.sample_count, verbose=False)

        return Forecast(
            cur_close=candles[-1].close,
            open=[float(v) for v in pred["open"].tolist()],
            high=[float(v) for v in pred["high"].tolist()],
            low=[float(v) for v in pred["low"].tolist()],
            close=[float(v) for v in pred["close"].tolist()])


class KronosSignalStrategy:
    """Turns a Kronos index forecast into a CE / PE / no-trade decision."""

    def __init__(self, forecaster: KronosForecaster, lookback: int = None,
                 horizon: int = None, threshold_pct: float = None):
        self.forecaster = forecaster
        self.lookback = lookback or config.KRONOS_LOOKBACK
        self.horizon = horizon or config.KRONOS_HORIZON
        self.threshold_pct = config.KRONOS_THRESHOLD_PCT if threshold_pct is None else threshold_pct

    def decide(self, candles: List[Candle]):
        """Return (Signal, Forecast) for the context ending at candles[-1].
        Signal is ENTER_LONG_CE / ENTER_SHORT_PE / NONE."""
        if len(candles) < 2:
            return Signal.NONE, None
        window = candles[-self.lookback:]
        fc = self.forecaster.forecast(window, self.horizon)
        if fc.move_pct >= self.threshold_pct:
            return Signal.ENTER_LONG_CE, fc
        if fc.move_pct <= -self.threshold_pct:
            return Signal.ENTER_SHORT_PE, fc
        return Signal.NONE, fc


def simulate_option_direct(ocandles, signal_time, build_leg):
    """Enter the option directly at the first candle after `signal_time` (no
    2-close confirmation) and manage it with the option leg's ladder. Requires
    an OptionPremiumStrategy (it uses force_enter_long / swing-low stop)."""
    strat = build_leg()
    if not isinstance(strat, OptionPremiumStrategy):
        raise TypeError("direct entry needs OPTION_LEG_MODE='premium_ladder'")
    post = []
    for c in ocandles:
        if c.timestamp <= signal_time:
            strat.on_closed_candle(c)
        else:
            post.append(c)
    if strat.state == "IN_POSITION":
        strat.force_exit(price=None)
    if not post:
        return dict(confirmed=False, reason="no_candles_after_signal")

    first = post[0]
    if first.timestamp.time() >= SQUARE_OFF:
        return dict(confirmed=False, reason="eod_before_entry")
    ev = strat.force_enter_long(first)
    entry, sl, entry_time = ev.price, ev.stop_loss, first.timestamp

    for c in post[1:]:
        if c.timestamp.time() >= SQUARE_OFF:
            strat.force_exit(price=None)
            return dict(confirmed=True, entry=entry, entry_time=entry_time, exit=c.open,
                        exit_time=c.timestamp, reason=ExitReason.FORCED_EOD.value, sl=sl, target=None)
        ev = strat.on_closed_candle(c)
        if ev.signal == Signal.EXIT:
            return dict(confirmed=True, entry=entry, entry_time=entry_time, exit=ev.price,
                        exit_time=c.timestamp, reason=ev.reason.value if ev.reason else "?",
                        sl=sl, target=None)
    last = post[-1]
    return dict(confirmed=True, entry=entry, entry_time=entry_time, exit=last.close,
                exit_time=last.timestamp, reason="open_at_data_end", sl=sl, target=None)


class KronosOptionBacktest:
    """Walk-forward backtest of the Kronos strategy on weekly index options."""

    def __init__(self, index_name: str, offset: int = None, strategy: KronosSignalStrategy = None,
                 require_confirm: bool = None, decision_every: int = None,
                 validity: int = None, target_pct: float = None):
        self.index_name = index_name
        self.ic = config.index_config(index_name)
        self.offset = config.STRIKE_OFFSET if offset is None else offset
        self.strategy = strategy
        self.require_confirm = (config.KRONOS_REQUIRE_OPTION_CONFIRM
                                if require_confirm is None else require_confirm)
        self.decision_every = decision_every or config.KRONOS_DECISION_EVERY
        self.validity = validity or config.OPTION_CONFIRM_VALIDITY
        self.target_pct = config.OPTION_TARGET_PREMIUM_PCT if target_pct is None else target_pct

    # -- option data (throttled fetch + retry, no poisoning) --------------
    def _option_candles(self, client, token, day, cache, skips, when, otype):
        key = (token, day)
        if key in cache:
            return cache[key]
        frm = datetime.combine(day, dtime(9, 0))
        to = datetime.combine(day, dtime(15, 35))
        cooldown = getattr(config, "RATE_LIMIT_COOLDOWN_SECONDS", 30)
        last_exc = None
        for attempt in range(3):
            try:
                cache[key] = fetch_candles(client, self.ic["option_exchange"], token, frm, to)
                return cache[key]
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(cooldown)
        skips.append((when, otype, f"fetch failed: {last_exc}"))
        return None

    def replay_day(self, index_day, warmup, scrip, day, client, cache):
        trades, skips = [], []
        busy_until = None
        history = list(warmup)   # rolling context (prior sessions + intraday)
        for i, c in enumerate(index_day):
            history.append(c)
            if busy_until and c.timestamp <= busy_until:
                continue
            if i % self.decision_every != 0:
                continue
            if c.timestamp.time() >= NO_ENTRY_AFTER:
                continue

            signal, fc = self.strategy.decide(history)
            if signal not in (Signal.ENTER_LONG_CE, Signal.ENTER_SHORT_PE):
                continue
            otype = "CE" if signal == Signal.ENTER_LONG_CE else "PE"
            try:
                option = find_offset_option(
                    scrip, c.close, option_type=otype, offset=self.offset,
                    underlying=self.ic["name"], strike_step=self.ic["strike_step"],
                    option_exchange=self.ic["option_exchange"], as_of=day)
            except LookupError as exc:
                skips.append((c.timestamp, otype, str(exc)))
                continue

            ocandles = self._option_candles(client, option["token"], day, cache,
                                            skips, c.timestamp, otype)
            if ocandles is None:
                continue
            if not ocandles:
                skips.append((c.timestamp, otype, "no option candles"))
                continue

            if self.require_confirm:
                res = simulate_option_leg(ocandles, c.timestamp, self.validity, self.target_pct)
            else:
                res = simulate_option_direct(ocandles, c.timestamp,
                                             lambda: build_option_leg(self.target_pct))
            if res.get("confirmed"):
                move = res["exit"] - res["entry"]
                res.update(otype=otype, strike=option["strike"], symbol=option["symbol"],
                           index_time=c.timestamp, points=move, pct=move / res["entry"] * 100.0,
                           pred_move=(fc.move_pct if fc else 0.0))
                trades.append(res)
                busy_until = res["exit_time"]
            else:
                skips.append((c.timestamp, otype, res.get("reason", "no_confirm")))
        return trades, skips


def _fetch_index(client, ic, calendar_days):
    now = datetime.now()
    return fetch_candles(client, ic["under_exchange"], ic["under_token"],
                         now - timedelta(days=calendar_days), now)


def run_backtest(client, forecaster: KronosForecaster, index_name: str,
                 offset: int, n_days: int, scrip):
    ic = config.index_config(index_name)
    strat = KronosSignalStrategy(forecaster)
    bt = KronosOptionBacktest(index_name, offset=offset, strategy=strat)

    calendar_days = max(21, int(n_days * 1.7) + 7)
    index_candles = _fetch_index(client, ic, calendar_days)
    all_dates = sorted({c.timestamp.date() for c in index_candles})
    target_dates = all_dates[-n_days:]

    print(f"KRONOS {index_name} offset {offset:+d}  horizon {strat.horizon} "
          f"threshold {strat.threshold_pct}%  confirm={bt.require_confirm}  ({INTERVAL})")
    print(f"Index data: {len(index_candles)} candles over {len(all_dates)} sessions "
          f"({all_dates[0]} .. {all_dates[-1]}); testing {len(target_dates)} session(s)\n")

    cache = {}
    g_trades = g_wins = 0
    g_points = g_pct = 0.0
    total_skips = 0
    for day in target_dates:
        warmup = [c for c in index_candles if c.timestamp.date() < day]
        day_candles = [c for c in index_candles if c.timestamp.date() == day]
        if not day_candles:
            continue
        trades, skips = bt.replay_day(day_candles, warmup, scrip, day, client, cache)
        day_pts = sum(t["points"] for t in trades)
        day_pct = sum(t["pct"] for t in trades)
        wins = sum(1 for t in trades if t["points"] > 0)
        g_trades += len(trades); g_wins += wins
        g_points += day_pts; g_pct += day_pct
        total_skips += len(skips)
        print(f"  {day}: {len(trades)} trade(s), premium net {day_pts:+.2f} pts "
              f"({day_pct:+.1f}% on entry), {len(skips)} skipped")
        for t in trades:
            print(f"      {t['otype']} {t['strike']:.0f} pred{t['pred_move']:+.2f}% "
                  f"idx@{t['index_time']:%H:%M} buy {t['entry_time']:%H:%M} "
                  f"entry={t['entry']:.2f} sl={t['sl']:.2f} -> exit {t['exit_time']:%H:%M} "
                  f"{t['exit']:.2f} ({t['reason']}) {t['points']:+.2f} ({t['pct']:+.1f}%)")
        for ts, otype, reason in skips:
            print(f"      · skip {ts:%H:%M} {otype}: {reason}")

    wr = (g_wins / g_trades) if g_trades else 0.0
    print(f"\nTOTAL: {g_trades} trade(s), win rate {wr:.0%}, premium net {g_points:+.2f} pts "
          f"({g_pct:+.1f}% summed on entry), {total_skips} skipped")


def main():
    import argparse
    from angel_api import AngelBrokingClient
    from instruments import load_scrip_master

    ap = argparse.ArgumentParser(description="Kronos weekly-option backtest")
    ap.add_argument("index", nargs="?", default=config.INDEX,
                    help="NIFTY | BANKNIFTY | SENSEX (default config.INDEX)")
    ap.add_argument("offset", nargs="?", type=int, default=config.STRIKE_OFFSET,
                    help="signed strike offset (+OTM / -ITM)")
    ap.add_argument("days", nargs="?", type=int, default=5, help="sessions to test")
    args = ap.parse_args()

    index_name = args.index.upper()
    if index_name not in config.INDEXES:
        raise SystemExit(f"Unknown index {index_name}; choose from {list(config.INDEXES)}")

    client = AngelBrokingClient(config.API_KEY, config.CLIENT_CODE, config.PASSWORD, config.TOTP_SECRET)
    client.login()
    scrip = load_scrip_master()

    forecaster = KronosForecaster()   # loads the real model lazily on first use
    run_backtest(client, forecaster, index_name, args.offset, args.days, scrip)
    client.logout()


if __name__ == "__main__":
    main()
