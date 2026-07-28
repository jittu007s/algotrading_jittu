# Kronos foundation-model options strategy

`kronos_strategy.py` uses [Kronos](https://github.com/shiyu-coder/Kronos) — an
open-source foundation model for financial candlesticks — to forecast the
**index** and trade **weekly options** on NIFTY, BANKNIFTY and SENSEX.

## How it works

1. **Forecast (KronosForecaster).** A rolling window of `KRONOS_LOOKBACK` index
   candles is fed to `KronosTokenizer` + `Kronos` via `KronosPredictor.predict`,
   which returns the next `KRONOS_HORIZON` candles.
2. **Signal (KronosSignalStrategy).** The predicted move from the current close
   to the horizon-end close decides a side: `>= KRONOS_THRESHOLD_PCT` → **CE**
   (bullish), `<= -threshold` → **PE** (bearish), otherwise no trade.
3. **Option leg.** The weekly option `STRIKE_OFFSET` strikes OTM/ITM is resolved
   with `find_offset_option` (per-index exchange/strike-step from
   `config.INDEXES`). Execution reuses the tested option leg:
   - `KRONOS_REQUIRE_OPTION_CONFIRM = True` → wait for the option premium's own
     2-close cross-up, then the percentage-ladder exit (30%→lock 10%, 50%→30% …).
   - `False` → enter directly at the first candle after the signal, same ladder
     exit, swing-low initial stop.

## Install (heavy; GPU recommended)

```bash
pip install -r requirements-kronos.txt
git clone https://github.com/shiyu-coder/Kronos
```

Then point the bot at the clone. **Easiest (no PYTHONPATH needed):** set the
path in `config.py` — it's the folder that contains the `model` package:

```python
# config.py
KRONOS_REPO_PATH = r"C:\Users\KUMARJ\Kronos"      # Windows
# KRONOS_REPO_PATH = "/home/you/Kronos"           # Linux/mac
```

or set the `KRONOS_REPO_PATH` environment variable, or add the clone to
`PYTHONPATH` yourself. Model/tokenizer weights download from Hugging Face on
first use (`config.KRONOS_MODEL`, `config.KRONOS_TOKENIZER`).

> The `ImportError: Kronos is not importable` message means step 2/3 above is
> missing — install the deps and set `KRONOS_REPO_PATH`.

## Backtest

```bash
python kronos_strategy.py NIFTY 2 5        # NIFTY, +2 OTM, last 5 sessions
python kronos_strategy.py BANKNIFTY 0 5    # BANKNIFTY, ATM
python kronos_strategy.py SENSEX -1 3      # SENSEX, 1 strike ITM, 3 sessions
```

Reads market data only (never places orders). Reports each trade in option
premium points and %, plus the Kronos predicted move that triggered it.

## Notes / caveats

- **No volume on the index.** NIFTY/SENSEX candles carry no volume, so the
  volume/amount columns Kronos expects are filled with zeros — predictions are
  price-only.
- **Expired strikes.** Like the SMMA option backtest, sessions whose weekly
  expiry has passed can't resolve their option in the live scrip master and are
  reported as skipped.
- **Config knobs** live in `config.py` under the `KRONOS_*` block: model ids,
  device, lookback/horizon, threshold, decision cadence, sampling (`T`,
  `top_p`, `sample_count`) and the confirm-vs-direct switch.
- The predictor is dependency-injectable (`KronosForecaster(predictor=...)`),
  so the signal logic and backtest are unit-tested without the heavy model.
```
