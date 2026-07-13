# Regime-Adaptive Nifty Options Strategy — Blueprint

This is the design document for `quant_strategy.py` (`STRATEGY = "REGIME"`).
It answers, honestly, the brief "design the best institutional-grade
strategy" — including what that phrase can and cannot mean for a retail
bot trading 1 lot of Nifty options over a candle API.

## 0. Honesty constraints (read first)

**There is no fetchable "world's best trader's strategy."** Genuinely
elite performance (Renaissance, Jane Street, Optiver) rests on
infrastructure, private data, and speed — not an indicator recipe. Anyone
claiming otherwise is selling something.

**What your data can and cannot support.** The bot sees 3-minute OHLC of
the Nifty *index* via SmartAPI. Therefore:

| Requested concept | Verdict |
|---|---|
| Volume, OBV, CMF, MFI, VWAP, volume profile, delta volume | **Impossible honestly** — index candles carry no volume. Faking it would be worse than omitting it. (Could be added later using the Nifty *futures* token, which does have volume.) |
| Order flow, footprint, dealer gamma, max pain, PCR, OI walls | **Not available** on this feed; needs option-chain and depth data plus meaningful investment in cleaning it. Omitted rather than pretended. |
| IV, IV rank, expected move, VIX regime | Partially possible later via India VIX token; not in v1. |
| ICT / SMC (order blocks, FVG, liquidity sweeps) | Discretionary pattern languages without agreed programmable definitions; any coded version is one blogger's interpretation. Excluded to avoid untestable rules — the *pullback-to-value then breakout* logic captures the same "buy the retrace in a trend" essence measurably. |
| Trend, momentum, volatility, structure, regime, MTF, risk rules | **Fully implementable** — this is what v1 is built from. |

**Overfitting is the main enemy.** Every additional indicator and
threshold is a knob that can be tuned to look great on 5 days and lose
money on the 6th. This design deliberately uses ONE instrument per
orthogonal information axis and keeps most parameters fixed.

## 1. Indicator selection — complements, redundancies, rejects

- **Trend**: EMA(50) side + slope of EMA(20) on 15-minute closes (MTF
  confirmation). *Rejected as redundant*: SMA/SMMA/HMA/Supertrend/Ichimoku —
  all are smoothed price; adding them re-counts the same vote.
- **Trend strength / regime**: ADX(14). Below 20 → ranging → **no trades**
  (long options bleed theta in chop; standing aside IS the edge). *Rejected*:
  Choppiness Index (duplicates ADX).
- **Momentum**: RSI(14) with a band (55–72 long side implemented as 50–72
  with a rising requirement; mirror for shorts). Bands prevent both
  buying dead momentum and buying blow-off tops. *Rejected as redundant*:
  Stochastic, CCI, ROC, TSI, MACD (all first-derivatives of price; RSI is
  the most studied and bounded).
- **Volatility**: ATR(14) against its own 50-bar mean. Two gates: `alive`
  (ATR ≥ 0.8×mean — dead tape kills option longs) and `sane` (ATR ≤
  2.5×mean — a news spike is untradeable noise; this is the programmable
  proxy for a news filter). ATR also sizes the stop. *Rejected*: Bollinger
  width, HV — same information as ATR.
- **Structure (the trigger)**: price pulled back to EMA(20) within the last
  6 bars, then closes above the previous bar's high and above EMA(20)
  (mirror for shorts). This is the programmable core of "pullback +
  continuation", the concept common to Wyckoff re-accumulation, SMC
  "return to order block", and classic trend following.

## 2. Regime logic

| Regime | Detection | Action |
|---|---|---|
| Trending | ADX ≥ 20 and MTF EMA slope agrees | Pullback-continuation model (this strategy) |
| Ranging | ADX < 20 | **Stand aside** — no long-option strategy survives chop after theta and spread |
| Dead vol | ATR < 0.8× its mean | Blocked by the `alive` gate |
| Spike vol | ATR > 2.5× its mean | Blocked by the `sane` gate |
| Alternative model | ORB (separate strategy in `strategy.py`) is the breakout-day model; run it side by side in the backtester and pick per your findings | |

## 3. Confidence score (0–100, trade at ≥ 70)

| Component | Points | Why it exists |
|---|---|---|
| ADX ≥ 20 | 10 | filters chop, the #1 killer of the old SMMA system |
| Price on trend side of EMA(50) | 10 | never fight the local trend |
| 15-min EMA(20) slope agrees | 10 | MTF agreement kills counter-trend scalps |
| RSI in band (50–72 / 28–50) | 15 | momentum present but not exhausted |
| RSI rising/falling with trade | 10 | momentum *now*, not stale |
| ATR alive | 10 | movement exists to pay for theta + spread |
| ATR sane | 10 | not a news bar |
| Pullback to EMA(20) in last 6 bars | 10 | entry at value, not at extension |
| Breakout of previous bar (hard trigger) | 15 | price confirms; **mandatory** regardless of score |

Score < 70 → no trade. The breakout is a hard gate so trend+momentum
alone can never fire mid-nowhere.

## 4. Exits

1. **Initial stop**: entry ∓ 1.2×ATR(14) — adapts to the tape instead of
   the previous candle's arbitrary low.
2. **Break-even**: at +1R the stop moves to entry ("never let a 1R winner
   lose").
3. **Trailing**: at +1.5R the stop trails EMA(20), ratcheting only.
4. **Time exit**: open ≥ 25 bars (~75 min) with < +0.5R unrealized →
   exit at close. This is the *theta guard*: an option that goes nowhere
   for an hour is losing money even at an unchanged index level.
5. **Session exits**: no entries before 09:30 (opening auction noise/gap
   filter) or after 15:00; hard square-off 15:20 (before broker RMS).

Partial profit booking is **not implemented**: at 1 lot there is nothing
to partial. It becomes meaningful at 2+ lots (book 1 lot at +1R).

## 5. Risk management (all enforced in code, per day)

- Max **3 trades** per day.
- Stop after **2 consecutive losses**.
- Daily loss limit **2R** — hit it and the bot stands down for the day.
- One position at a time; fixed 1 lot. Kelly sizing is *deliberately
  rejected*: Kelly needs a stable, known edge; on an unproven system it
  maximises the speed of ruin. Fixed minimum size until ≥ 100 live/dry
  trades exist.

## 6. Options selection guidance

- **Weekly ATM** for this bot (delta ≈ 0.5, most liquid, tightest spread).
- Consider **ITM (delta 0.6–0.7)** if theta bleed dominates results —
  more intrinsic, less decay, at more capital per lot.
- Avoid expiry-day afternoon longs (gamma is cheap but theta is brutal).
- Liquidity: Nifty weekly ATM typically has 1–2 point spreads; if the
  quoted spread exceeds ~3–4 points, skip — the strategy's average edge
  per trade is smaller than a bad fill.

## 7. Backtesting metrics & optimization discipline

`backtest_today.py` replays the last N sessions (default 5) for all three
strategies on identical data, reporting per-day trades, win rate, and net
points. Judge with: expectancy (points/trade), win rate vs payoff, max
intraday drawdown, and trade count (an edge on 3 trades is noise).

**May be tuned** (one at a time, on data not used for the tuning):
`score_threshold` (70), `adx_min` (20), `atr_stop_mult` (1.2),
`time_exit_bars` (25). **Fixed by design** (do not tune): indicator
periods (14/20/50 are conventions — tuning them is curve-fitting),
risk caps, session times, the one-instrument-per-axis structure.

5 sessions ≈ 10–20 trades — enough to compare *behaviour* (does it stand
aside in chop? does it ride trends?), **not** enough to prove an edge.
Treat 5-day results as a smoke test; accumulate ≥ 100 trades before
believing any number.

## 8. Why it should beat the named baselines (and the honest caveat)

- **Pure SMMA / EMA cross / Supertrend**: identical single-axis trend
  logic; all lose the same way — chopped in ranges. The ADX regime gate
  removes precisely those trades (your 13-Jul replay: 16 of 23 exits were
  chop stop-outs).
- **VWAP strategies**: not honestly computable on a volume-less index feed.
- **RSI-only / mean reversion**: buying options against trend fights both
  direction and theta; long premium needs the fat-tailed *with-trend* move.
- **ICT/SMC**: untestable as commonly stated; the testable kernel
  (pullback to value, then displacement) is exactly this strategy's trigger.
- **Trend following (raw)**: kept — but with momentum/vol gates and a theta
  guard that raw trend following lacks when expressed through options.

**Caveat**: "should statistically outperform" means *better expectancy per
trade through fewer bad trades*, not guaranteed profit. No 3-minute
indicator strategy has a large raw edge; survival comes from the risk
layer. Expect losing days and losing weeks.

## 9. Common failure modes

1. **Gap opens**: indicators carry yesterday's levels; 09:30 entry gate +
   ATR-sane filter mitigate, not eliminate.
2. **Whipsaw around ADX 20**: regime flips bar to bar; the score threshold
   usually keeps marginal setups out, but expect some.
3. **Theta on slow trend days**: index drifts up 30 points in 4 hours —
   direction right, option flat. Time exit cuts these at small cost.
4. **Event days** (budget, RBI, elections): no calendar feed exists in the
   bot — **do not run it on known event days**; that judgment stays human.
5. **Expiry-day gamma games**: ATM behaviour changes near expiry;
   consider standing aside after 13:00 on expiry day.
6. **Five-day overconfidence**: the backtest window is a smoke test, not
   a track record.
