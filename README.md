# Nifty SMA-Cross Option Strategy — Angel One (SmartAPI) Algo Bot

A standalone Python algo-trading bot for Angel One, separate from the
Taskify web app in the rest of this repo. It implements the strategy
described below using Angel One's official **SmartAPI**.

> **Not financial advice.** This is example/educational code. Options
> trading carries substantial risk of loss. Backtest and paper/dry-run
> test thoroughly before risking real capital, and only after you
> understand exactly what the code does.

## 1. Strategy specification

Plain-English rule → precise implementation (`strategy.py`):

The strategy runs on **3-minute candles** (`CANDLE_INTERVAL` in
`config.py`) and trades **both directions** symmetrically:

| Rule | LONG (buys ATM **CE**) | SHORT (buys ATM **PE**) |
|---|---|---|
| Setup | Two consecutive closed candles with `close > SMA(period)` → **ARMED**. Invalidated if price closes back below the SMA before triggering. | Two consecutive closed candles with `close < SMA(period)` → **ARMED**. Invalidated if price closes back above the SMA. |
| Entry trigger | A later candle's `high` crosses the **high of the 2nd setup candle**. | A later candle's `low` crosses the **low of the 2nd setup candle**. |
| Stop loss | **Low of the candle immediately before the entry candle.** | **High of the candle immediately before the entry candle.** |
| Target | `entry + 2 × (entry − SL)` (1:2 RR on the underlying, configurable via `RISK_REWARD`). | `entry − 2 × (SL − entry)`. |
| Early exit | Price touches the SMA twice while retracing **down** from the highest point made after entry. | Price touches the SMA twice while retracing **up** from the lowest point made after entry. |

Each distinct SMA touch is edge-detected — one continuous touch spanning
several candles counts once, and the 2nd touch exits immediately.

Notes / assumptions worth double-checking against your own intent before
going live:
- SL/target/SMA-touch are evaluated on the **underlying index**, not the
  option premium (which doesn't move linearly with the index) — this is
  the standard way to run "trade the index, execute via options"
  strategies. The option position is simply squared off when the
  underlying condition fires.
- Both legs only ever **buy** options (CE for longs, PE for shorts), so
  the maximum loss per trade is capped at the premium paid.
- Only one open position at a time — no pyramiding/re-entry while already
  in a trade.
- Any open position is force-flattened by `SQUARE_OFF_HOUR_MINUTE`
  (default 15:20 IST) so it never carries overnight.

## 2. Files

- `strategy.py` — pure, broker-agnostic strategy state machine (unit
  testable, no network calls).
- `angel_api.py` — thin wrapper over the official `smartapi-python` SDK
  (login, historical candles, market orders).
- `instruments.py` — downloads/caches Angel One's instrument master and
  resolves the ATM CE trading symbol/token for the nearest expiry.
- `bot.py` — the live polling loop that wires the above together.
- `backtest.py` — replays a CSV of historical Nifty candles through the
  same `strategy.py` to sanity-check the rules before going live.
- `config.py` / `.env.example` — strategy parameters and credentials.

## 3. Setup

```bash
cd algo-trading
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then fill in your real credentials
```

Backtest first (no credentials needed, just a CSV of `timestamp,open,high,low,close`):

```bash
python backtest.py path/to/nifty_15m.csv
```

Then dry-run live (default `DRY_RUN=true` — logs signals, sends no orders):

```bash
python bot.py
```

Only once you're satisfied with dry-run behaviour, set `DRY_RUN=false` in
`.env` and start with the default 1 lot.

## 4. Getting Angel One API access & enabling algo/API trading

Angel One doesn't have a feature to "upload" a script into their
platform — SmartAPI is a REST/WebSocket API. **Your code runs on your own
machine/server** and places orders into your Angel One account over that
API. Steps:

1. **Have an active Angel One trading + demat account** (client code +
   MPIN you use to log in).
2. **Create a developer app** at the SmartAPI portal — https://smartapi.angelbroking.com
   - Log in with your Angel One client code.
   - Go to "My Apps" → "Create New App".
   - Pick the app type you need — SmartAPI issues separate keys for
     **Trading APIs**, **Market Feeds APIs**, and **Historical Data
     APIs**. This bot needs a **Trading APIs** key (it covers order
     placement + historical candles).
   - Save the generated **API Key**.
3. **Enable TOTP-based login** in the Angel One app: Profile/Settings →
   enable 2FA via an authenticator app. When you scan the QR code, Angel
   also shows the underlying base32 secret — save that secret as
   `ANGEL_TOTP_SECRET`. The bot uses `pyotp` to generate the 6-digit code
   programmatically for each login instead of typing it manually.
4. Fill `ANGEL_API_KEY`, `ANGEL_CLIENT_CODE`, `ANGEL_MPIN`,
   `ANGEL_TOTP_SECRET` into `.env`.
5. **Test the login** with `DRY_RUN=true` first — `bot.py` will log in,
   pull candles, and print signals without placing any orders.
6. **Regulatory / compliance check before going live**: SEBI's algo
   trading framework for retail investors (rolled out through 2025)
   requires algo strategies used over broker APIs to be registered/tagged
   with the exchange through the broker, with order-level algo IDs. Check
   Angel One's current SmartAPI/algo-trading policy and any registration
   step required on your account **before** enabling `DRY_RUN=false` —
   requirements have been evolving and Angel One's own dashboard/support
   is the authoritative source, not this README.
7. Be mindful of SmartAPI's request rate limits (order placement and
   quote/candle endpoints are all rate-limited); `bot.py` polls on a
   configurable interval (`POLL_SECONDS`, default 15s) rather than
   hammering the API.

## 5. Before running this with real money — hardening suggestions

This is a compact reference implementation, not a production trading
system. Consider before going live with size:
- Place a genuine broker-side stop-loss order on the option leg as a
  safety net (in case your process/connection dies mid-trade) in
  addition to the software-managed exit logic here.
- Move from polling `getCandleData` to Angel One's WebSocket feed for
  lower-latency signal detection.
- Add reconnect/retry handling around the WebSocket/session (SmartAPI
  sessions expire and need periodic refresh).
- Add persistent trade logging/alerting (e.g. to a DB or Slack/Telegram)
  so you have an audit trail and can be notified of entries/exits.
- Confirm the current Nifty lot size and strike interval (`LOT_SIZE`,
  `STRIKE_STEP` in `config.py`) — both are revised periodically by NSE.
