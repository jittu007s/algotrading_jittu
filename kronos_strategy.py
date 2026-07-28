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

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta
from typing import List

# Import torch BEFORE pandas / numpy / the broker SDK. On Windows those pull in
# their own MKL/OpenMP DLLs, and if they load first torch's native init can die
# with "[WinError 1114] ... c10.dll". Importing it first is harmless elsewhere,
# and staying quiet here keeps non-Kronos users unaffected (the real, actionable
# error is raised later by _load_kronos()).
try:
    import torch as _torch  # noqa: F401
except Exception:
    _torch = None

import config
from instruments import find_offset_option, is_expiry_day
from sizing import compute_size
from strategy import Candle, ExitReason, OptionPremiumStrategy, Signal
# Reuse the tested option-leg simulation, throttled fetch and builders.
from backtest_option_chart import (build_option_leg, fetch_candles,
                                    simulate_option_leg)

INTERVAL = config.CANDLE_INTERVAL
NO_ENTRY_AFTER = dtime(*config.NO_ENTRY_AFTER_HOUR_MINUTE)
SQUARE_OFF = dtime(*config.SQUARE_OFF_HOUR_MINUTE)
EXPIRY_SWITCH = dtime(*getattr(config, "EXPIRY_SWITCH_HOUR_MINUTE", (14, 45)))

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


def _kronos_root_candidates():
    """Directories that might BE the Kronos repo root (i.e. contain `model/`).

    Checked in priority order. Auto-discovery matters because Kronos's `model`
    package uses absolute imports (`from model.x import ...`), which only
    resolve when the repo ROOT is on sys.path - and because a tracked
    config.py can be overwritten by a git pull, so we must not depend on it.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    for src in (getattr(config, "KRONOS_REPO_PATH", None), os.getenv("KRONOS_REPO_PATH")):
        if src:
            out += [src, os.path.join(src, "Kronos")]
    # common spots: beside this file, in the cwd, one level up, or in $HOME
    for base in (here, os.getcwd(), os.path.dirname(here), os.path.expanduser("~")):
        out += [os.path.join(base, "Kronos"), os.path.join(base, "kronos"), base]
    return out


def _find_kronos_root():
    """First candidate directory that actually contains a `model` package."""
    seen = set()
    for d in _kronos_root_candidates():
        if not d or d in seen:
            continue
        seen.add(d)
        try:
            if os.path.isdir(os.path.join(d, "model")):
                return d
        except OSError:
            continue
    return None


def _load_kronos():
    """Import Kronos, auto-discovering a clone of github.com/shiyu-coder/Kronos.

    Looks for the repo ROOT (the folder containing the `model` package) via
    config.KRONOS_REPO_PATH, the KRONOS_REPO_PATH env var, then beside this
    file / the cwd / $HOME, and puts it on sys.path so Kronos's absolute
    imports resolve.

    Returns (Kronos, KronosTokenizer, KronosPredictor). Only a module that
    actually exposes all three classes is accepted - so a bare `Kronos`
    namespace folder is never returned by mistake - and if nothing works the
    real underlying import errors (e.g. a missing torch/einops) are surfaced."""
    root = _find_kronos_root()
    if root and root not in sys.path:
        sys.path.insert(0, root)

    # Import torch FIRST. On Windows, loading pandas/numpy/SmartApi (which pull
    # their own MKL/OpenMP DLLs) before torch can break torch's native init
    # with "[WinError 1114] ... c10.dll". Importing torch up front usually
    # avoids that, and if torch itself is broken we report it clearly rather
    # than burying it under seven failed module guesses.
    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise ImportError(
            f"PyTorch failed to load ({type(exc).__name__}: {exc}).\n"
            "This is a torch installation problem, not a Kronos one. On Windows "
            "'[WinError 1114] ... c10.dll' usually means:\n"
            "  * the Microsoft Visual C++ Redistributable is missing - install "
            "the x64 vc_redist from Microsoft, or\n"
            "  * a CPU/CUDA build mismatch - reinstall a clean CPU build:\n"
            "      pip uninstall -y torch\n"
            "      pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  * a DLL clash with another native package (numpy/MKL). Try a fresh "
            "venv, or set KMP_DUPLICATE_LIB_OK=TRUE.\n"
            "Verify with:  python -c \"import torch; print(torch.__version__)\"") from exc

    names = ("Kronos", "KronosTokenizer", "KronosPredictor")
    # `model` / `model.kronos` are the real homes; the rest are fallbacks for
    # other clone layouts. Bare `Kronos`/`kronos` are tried last and only used
    # if they truly carry the classes (attribute check below).
    candidates = ("model", "model.kronos", "Kronos.model", "Kronos.model.kronos",
                  "kronos.model", "kronos", "Kronos")
    errors = []
    for modname in candidates:
        try:
            m = __import__(modname, fromlist=list(names))
        except Exception as exc:   # ModuleNotFoundError, torch import errors, ...
            errors.append(f"{modname}: {type(exc).__name__}: {exc}")
            continue
        for target in (m, getattr(m, "model", None)):
            if target is not None and all(hasattr(target, n) for n in names):
                return target.Kronos, target.KronosTokenizer, target.KronosPredictor
        errors.append(f"{modname}: imported but missing {names}")

    tried = "\n  ".join(errors)
    found = f"auto-detected Kronos root: {root}" if root else (
        "NO Kronos repo root found (no directory containing a `model` package "
        "in config.KRONOS_REPO_PATH, $KRONOS_REPO_PATH, beside kronos_strategy.py, "
        "the cwd, or $HOME)")
    raise ImportError(
        "Kronos could not be loaded.\n"
        f"{found}\n"
        "Fix by EITHER cloning next to this script:\n"
        "    git clone https://github.com/shiyu-coder/Kronos\n"
        "  (a `Kronos/model/` folder beside kronos_strategy.py is picked up "
        "automatically)\n"
        "OR pointing at an existing clone WITHOUT editing tracked files:\n"
        "    set KRONOS_REPO_PATH=C:\\path\\to\\Kronos          (Windows cmd)\n"
        "    export KRONOS_REPO_PATH=/path/to/Kronos           (bash/git-bash)\n"
        "That path must be the folder CONTAINING the `model` package.\n"
        "Also ensure deps are installed: pip install -r requirements-kronos.txt\n"
        f"Import attempts:\n  {tried}")


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


def simulate_option_direct(ocandles, signal_time, build_leg, carry_forward=False):
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
    # A carried (next-expiry) position is never squared off, so the intraday
    # square-off / no-entry gates do not apply to it.
    if not carry_forward:
        if first.timestamp.time() >= SQUARE_OFF:
            return dict(confirmed=False, reason="eod_before_entry")
        # match the live bot: no NEW position at/after the no-entry cutoff
        if first.timestamp.time() >= NO_ENTRY_AFTER:
            return dict(confirmed=False, reason="entry_past_cutoff")
    ev = strat.force_enter_long(first)
    entry, sl, entry_time = ev.price, ev.stop_loss, first.timestamp

    for c in post[1:]:
        if not carry_forward and c.timestamp.time() >= SQUARE_OFF:
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
        self.loss_cooldown = getattr(config, "KRONOS_LOSS_COOLDOWN_CANDLES", 0)
        self.max_losses_per_day = getattr(config, "KRONOS_MAX_LOSSES_PER_DAY", 0)

    # -- option data (throttled fetch + retry, no poisoning) --------------
    def _option_candles(self, client, token, day, cache, skips, when, otype,
                        span_days=0):
        """Option candles for `day`; `span_days` extends the window forward so
        a carried-forward position can be tracked into later sessions."""
        key = (token, day, span_days)
        if key in cache:
            return cache[key]
        frm = datetime.combine(day, dtime(9, 0))
        to = datetime.combine(day + timedelta(days=span_days), dtime(15, 35))
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
        cooldown_until = None   # set after a loss, to stop same-move churn
        losses = 0
        history = list(warmup)   # rolling context (prior sessions + intraday)
        for i, c in enumerate(index_day):
            history.append(c)
            if busy_until and c.timestamp <= busy_until:
                continue
            if self.max_losses_per_day and losses >= self.max_losses_per_day:
                continue
            if cooldown_until and c.timestamp < cooldown_until:
                continue
            if i % self.decision_every != 0:
                continue
            # Expiry-day rule: past the switch time, don't touch the contract
            # expiring today - use the next expiry and carry it forward. Such a
            # trade is exempt from the intraday cutoff (it is never squared off).
            carry = (c.timestamp.time() >= EXPIRY_SWITCH
                     and is_expiry_day(scrip, underlying=self.ic["name"],
                                       option_exchange=self.ic["option_exchange"],
                                       as_of=day))
            cutoff_applies = not (carry and getattr(
                config, "CARRY_FORWARD_IGNORES_ENTRY_CUTOFF", True))
            if cutoff_applies and c.timestamp.time() >= NO_ENTRY_AFTER:
                continue

            signal, fc = self.strategy.decide(history)
            if signal not in (Signal.ENTER_LONG_CE, Signal.ENTER_SHORT_PE):
                continue
            otype = "CE" if signal == Signal.ENTER_LONG_CE else "PE"
            try:
                option = find_offset_option(
                    scrip, c.close, option_type=otype, offset=self.offset,
                    underlying=self.ic["name"], strike_step=self.ic["strike_step"],
                    option_exchange=self.ic["option_exchange"], as_of=day,
                    skip_same_day=carry)
            except LookupError as exc:
                skips.append((c.timestamp, otype, str(exc)))
                continue

            # a carried position lives past today, so pull later sessions too
            span = getattr(config, "CARRY_FORWARD_TRACK_DAYS", 5) if carry else 0
            ocandles = self._option_candles(client, option["token"], day, cache,
                                            skips, c.timestamp, otype, span_days=span)
            if ocandles is None:
                continue
            if not ocandles:
                skips.append((c.timestamp, otype, "no option candles"))
                continue

            if self.require_confirm:
                res = simulate_option_leg(ocandles, c.timestamp, self.validity,
                                          self.target_pct, carry_forward=carry)
            else:
                res = simulate_option_direct(ocandles, c.timestamp,
                                             lambda: build_option_leg(self.target_pct),
                                             carry_forward=carry)
            if res.get("confirmed"):
                move = res["exit"] - res["entry"]
                # size the trade the same way the live bot would
                sz = compute_size(getattr(config, "CAPITAL_FALLBACK", None), res["entry"],
                                  option["lotsize"] or config.LOT_SIZE,
                                  utilisation_pct=config.FUND_UTILISATION_PCT,
                                  min_lots=config.MIN_LOTS)
                res.update(otype=otype, strike=option["strike"], symbol=option["symbol"],
                           index_time=c.timestamp, points=move, pct=move / res["entry"] * 100.0,
                           pred_move=(fc.move_pct if fc else 0.0),
                           lots=sz.lots, qty=sz.quantity, cost=sz.cost,
                           rupees=move * sz.quantity if sz.ok else 0.0, sized_ok=sz.ok,
                           carry=carry, expiry=option.get("expiry"))
                trades.append(res)
                busy_until = res["exit_time"]
                if move < 0:
                    losses += 1
                    if self.loss_cooldown:
                        step = timedelta(seconds=_INTERVAL_SECONDS.get(INTERVAL, 180))
                        cooldown_until = res["exit_time"] + step * self.loss_cooldown
            else:
                skips.append((c.timestamp, otype, res.get("reason", "no_confirm")))
        return trades, skips


def diagnose_signal(client, forecaster, index_name, n_days):
    """Measure Kronos's raw directional accuracy on the INDEX - no options, no
    stops, no theta. This isolates signal quality from execution: if the hit
    rate is ~50%, no amount of stop/ladder tuning will make the strategy work.

    For each decision point we compare the predicted move against the index's
    ACTUAL move over the same horizon, and bucket by predicted magnitude."""
    ic = config.index_config(index_name)
    strat = KronosSignalStrategy(forecaster)
    horizon, thr = strat.horizon, strat.threshold_pct
    every = config.KRONOS_DECISION_EVERY
    calendar_days = max(21, int(n_days * 1.7) + 7)
    candles = _fetch_index(client, ic, calendar_days)
    all_dates = sorted({c.timestamp.date() for c in candles})
    target = all_dates[-n_days:]

    print(f"KRONOS SIGNAL DIAGNOSTIC - {index_name} ({INTERVAL})")
    print(f"horizon={horizon} candles  threshold={thr}%  sessions={len(target)}")
    print("Measures the model alone: predicted direction vs the index's actual "
          "move over the same horizon.\n")

    recs = []
    for day in target:
        warm = [c for c in candles if c.timestamp.date() < day]
        day_c = [c for c in candles if c.timestamp.date() == day]
        hist = list(warm)
        for i, c in enumerate(day_c):
            hist.append(c)
            if i % every != 0:
                continue
            fut = day_c[i + 1:i + 1 + horizon]
            if len(fut) < horizon:
                break
            window = hist[-strat.lookback:]
            if len(window) < min(strat.lookback, 20):
                continue      # too little context for a meaningful forecast
            fc = forecaster.forecast(window, horizon)
            actual = (fut[-1].close - c.close) / c.close * 100.0
            recs.append((fc.move_pct, actual))

    if not recs:
        print("No decision points with a full horizon of forward data.")
        return

    def report(label, rows):
        if not rows:
            print(f"  {label:<22} (none)")
            return
        hits = sum(1 for p, a in rows if (p > 0) == (a > 0))
        mean_a = sum(a for _p, a in rows) / len(rows)
        # mean actual move in the predicted direction (the tradeable edge)
        edge = sum(a if p > 0 else -a for p, a in rows) / len(rows)
        print(f"  {label:<22} n={len(rows):<5} hit={hits/len(rows):>5.1%}  "
              f"mean actual={mean_a:+.3f}%  edge={edge:+.4f}%")

    print("Directional accuracy (50% = coin flip, no edge):")
    report("all forecasts", recs)
    fired = [r for r in recs if abs(r[0]) >= thr]
    report(f"|pred| >= {thr}% (traded)", fired)
    for lo, hi in ((thr, 0.25), (0.25, 0.4), (0.4, 99)):
        report(f"|pred| {lo}-{hi}%", [r for r in recs if lo <= abs(r[0]) < hi])
    ups = [r for r in fired if r[0] > 0]
    dns = [r for r in fired if r[0] < 0]
    print("\nBias check (a healthy model should fire both ways):")
    report("predicted UP (CE)", ups)
    report("predicted DOWN (PE)", dns)
    print(f"\n  UP/DOWN split: {len(ups)} / {len(dns)}")
    print("\nHow to read this: 'edge' is the mean index move in the predicted")
    print("direction. It must comfortably exceed option spread + theta to be")
    print("tradeable. An edge near 0 (or hit rate near 50%) means the signal")
    print("itself has no value at this horizon/threshold - tuning stops cannot")
    print("fix that; change the horizon/threshold/model or drop the strategy.")


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
    g_points = g_pct = g_rupees = 0.0
    total_skips = 0
    for day in target_dates:
        warmup = [c for c in index_candles if c.timestamp.date() < day]
        day_candles = [c for c in index_candles if c.timestamp.date() == day]
        if not day_candles:
            continue
        trades, skips = bt.replay_day(day_candles, warmup, scrip, day, client, cache)
        day_pts = sum(t["points"] for t in trades)
        day_pct = sum(t["pct"] for t in trades)
        day_rs = sum(t.get("rupees", 0.0) for t in trades)
        wins = sum(1 for t in trades if t["points"] > 0)
        g_trades += len(trades); g_wins += wins
        g_points += day_pts; g_pct += day_pct; g_rupees += day_rs
        total_skips += len(skips)
        print(f"  {day}: {len(trades)} trade(s), premium net {day_pts:+.2f} pts "
              f"({day_pct:+.1f}% on entry), P&L {day_rs:+,.0f}, {len(skips)} skipped")
        for t in trades:
            print(f"      {t['otype']} {t['strike']:.0f} pred{t['pred_move']:+.2f}% "
                  f"idx@{t['index_time']:%H:%M} buy {t['entry_time']:%H:%M} "
                  f"entry={t['entry']:.2f} sl={t['sl']:.2f} -> exit {t['exit_time']:%H:%M} "
                  f"{t['exit']:.2f} ({t['reason']}) {t['points']:+.2f} ({t['pct']:+.1f}%) "
                  f"x{t.get('qty', 0)} ({t.get('lots', 0)} lots) = {t.get('rupees', 0.0):+,.0f}"
                  + (f"  [CARRY -> expiry {t.get('expiry')}]" if t.get("carry") else ""))
        for ts, otype, reason in skips:
            print(f"      · skip {ts:%H:%M} {otype}: {reason}")

    wr = (g_wins / g_trades) if g_trades else 0.0
    cap = getattr(config, "CAPITAL_FALLBACK", 0.0) or 0.0
    print(f"\nTOTAL: {g_trades} trade(s), win rate {wr:.0%}, premium net {g_points:+.2f} pts "
          f"({g_pct:+.1f}% summed on entry), {total_skips} skipped")
    print(f"P&L: {g_rupees:+,.0f}" + (f"  on capital {cap:,.0f} "
          f"({g_rupees / cap * 100:+.1f}%)" if cap else "")
          + f"   [sizing: {config.FUND_UTILISATION_PCT:.0f}% of funds, "
            f"min {config.MIN_LOTS} lots, ATM offset {offset:+d}]")


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
    ap.add_argument("--diagnose", action="store_true",
                    help="measure the model's raw directional accuracy on the "
                         "index (no options/stops/theta) instead of backtesting")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override KRONOS_THRESHOLD_PCT for this run")
    ap.add_argument("--horizon", type=int, default=None,
                    help="override KRONOS_HORIZON for this run")
    args = ap.parse_args()

    index_name = args.index.upper()
    if index_name not in config.INDEXES:
        raise SystemExit(f"Unknown index {index_name}; choose from {list(config.INDEXES)}")
    if args.threshold is not None:
        config.KRONOS_THRESHOLD_PCT = args.threshold
    if args.horizon is not None:
        config.KRONOS_HORIZON = args.horizon

    client = AngelBrokingClient(config.API_KEY, config.CLIENT_CODE, config.PASSWORD, config.TOTP_SECRET)
    client.login()

    forecaster = KronosForecaster()   # loads the real model lazily on first use
    if args.diagnose:
        diagnose_signal(client, forecaster, index_name, args.days)
    else:
        scrip = load_scrip_master()
        run_backtest(client, forecaster, index_name, args.offset, args.days, scrip)
    client.logout()


if __name__ == "__main__":
    main()
