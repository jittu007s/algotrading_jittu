# ICT / Smart Money Concepts system — Nifty options via Angel One

An intraday sweep → MSS → FVG strategy with multi-timeframe bias,
capital-percentage risk sizing, a SQLite journal, and paper/live modes.

> **Honesty note.** "ICT/SMC" has no canonical, testable specification —
> every coded version is one interpretation. The definitions implemented
> here are the ones in the project brief, stated precisely in
> `structure.py`'s docstring. Whether they carry edge is exactly what the
> paper mode and backtester exist to find out. Nothing here is validated
> yet; do not run live until it is.

## Strategy pipeline

1. **Bias engine** (`bias.py`): swing structure (HH/HL vs LH/LL) on Daily
   and 1H Nifty. Conflict = NEUTRAL = no trades. Bullish → CE only,
   bearish → PE only.
2. **Setup scanner** (`structure.py`, 5-min):
   - **Liquidity sweep**: wick ≥ 3 pts beyond a prior 5m swing, previous
     day high/low, or opening-range extreme, closing back inside.
   - **MSS**: within 12 candles, a displacement candle (body ≥ 1.5× avg)
     closes through the most recent opposing swing.
   - **FVG**: the 3-candle imbalance left by the displacement leg.
3. **Entry** (`engine.py`): first pullback into the FVG midpoint. Option:
   ATM (or ITM1 in config). Stop: beyond the sweep extreme. A broker-side
   premium SL-L order backs the software stop in live mode.
4. **Risk** (`risk.py`): 0.75%/trade sizing (skip if 1 lot exceeds it),
   1.25% daily hard stop, 1% equity-giveback stop, partial at +1R (lots
   permitting; stop to entry either way), trail behind 5m swings, no
   entries 09:15–09:30 or after 14:45, square-off 15:10.

## Layout

```
ict/
  config.yaml        all knobs (capital, risk %, structure params, session times)
  config.py          typed loader
  models.py          Candle/Swing/Sweep/MSS/FVG/Setup dataclasses
  structure.py       pure detection logic  <- unit tested
  bias.py            multi-timeframe bias engine
  risk.py            sizing + daily risk guards  <- unit tested
  journal.py         SQLite journal (setup levels stored as JSON per trade)
  datafeed.py        polling feed via ../angel_api.py (websocket seam documented)
  order_manager.py   idempotent orders, premium SL, paper mode
  engine.py          the live/paper loop        python -m ict.engine
  backtest_ict.py    session replays            python -m ict.backtest_ict 10
  tests/             synthetic-candle unit tests
```

## Running (from `algo-trading/`)

```bash
pip install -r requirements.txt          # adds pyyaml
python -m unittest discover -s ict/tests -t .   # 18 tests
python -m ict.backtest_ict 10            # backtest last 10 sessions
python -m ict.engine                     # paper mode (default)
```

Live mode needs BOTH `mode: live` in `config.yaml` **and** `DRY_RUN=false`
in `../.env`. Keep it in paper mode until the backtest and several weeks
of paper signals look sane.

## Deliberate deviations from the brief (and why)

- **Polling, not websocket**: an untested websocket layer is a liability
  in a trading system. The polling feed is proven code; `datafeed.py`
  documents the seam and the reconnect rules for a future ws upgrade.
- **No OCO**: SmartAPI's normal order variety has no true OCO bracket.
  Live mode places the broker-side premium SL order; the engine manages
  target/trailing and cancels the SL on its own exits.
- **1-min entry refinement** not implemented: entry fills at the FVG
  midpoint on the 5-min candle that touches it. Adding 1-min data
  multiplies API load for marginal backtest fidelity; do it only if paper
  results justify it.
- **Screenshots**: a headless bot can't screenshot charts; the journal
  stores every setup's levels as JSON so the chart can be redrawn.
- **Premium P&L is delta-approximated** (0.5 ATM) in paper/backtest modes.
  Real option fills differ (gamma, theta, spread); live mode uses real
  fills but the backtest numbers are approximations by construction.
