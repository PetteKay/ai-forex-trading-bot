# Control the bot from Claude, Codex or ChatGPT (MCP)

`mcp_server.py` exposes the bot as an MCP server. Your AI app (Claude Code, Claude Desktop, OpenAI Codex, ChatGPT or any MCP client) does the analysis with its own model. **No NVIDIA/OpenAI/Anthropic API key is needed for this mode.** Every order still goes through the bot's risk governor and guards.

```
You ── "Analyse EURUSD and trade it if it qualifies" ──► Claude / Codex / ChatGPT
                                                             │  MCP tool calls
                                                             ▼
                                          mcp_server.py  (on the PC running MT5)
                     analysis, news, ML gate, lessons ──┤
         preview → risk governor, guards, sizing → order ──┴──► MetaTrader 5
```

## Before you start

- **A Windows PC with MetaTrader 5 open and logged in.** The `MetaTrader5` Python package is Windows-only, so the server runs on that PC.
- **The dependencies:** `py -m pip install -r requirements.txt`. This includes `mcp>=2.3,<3`.
- **A demo account first.**

## Safety settings (`config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `MCP_TRADING_MODE` | `"dry_run"` | `"off"`: trading tools refuse. `"dry_run"`: `place_trade` only reports what it would do. `"live"`: sends orders. |
| `MCP_HTTP_ALLOW_TRADING` | `False` | Over HTTP (ChatGPT), trading tools refuse unless this is `True`. Read-only tools always work. |
| `SIZING_MODE` / `RISK_PER_TRADE_PCT` | `"fixed"` / `0.5` | Same sizing as the bot. `"risk_pct"` sizes every trade to a % of equity. |
| `MCP_LESSONS` | `"client"` | Who writes lessons for trades closed over MCP. `"client"`: your AI app, via `get_trades_to_review` → `save_lesson` (no key needed). `"llm"`: the bot's own reflection agent (needs `LLM_API_KEY`). |

Every `place_trade`, live or dry run, is checked against:
- the **risk governor** (daily loss cap, kill switch, drawdown halt, max positions and orders per day). It reads `observability/governor_state_<account>.json`, so limits hit by the bot also block the MCP server, and the other way round. Each MT5 account has its own file.
  - Closed trades are booked before every check, including ones closed by `close_position`, by their stop loss / take profit, or while the server was off. So a losing close counts toward the daily loss cap.
- the **economic-calendar blackout**
- the **duplicate** and **correlation** guards
- the **spread filter** (if enabled)
- **position sizing**: it refuses rather than rounding a lot size up past the broker minimum
- **what the bot has learned**, the same rules `bot.py` applies:
  - the **setup blacklist** (N losses in a row on a symbol in the same regime)
  - the **ML gate** (`META_LABEL_MIN_PROB`) and **Kelly** sizing from its win probability
  - **loss-streak** and **volatility** size cuts

  `preview_trade` shows the ML probability and every size cut it made.

## MCP trades learn like the bot's own trades
When a trade opened over MCP closes (by `close_position`, its stop loss / take profit, or while the server was off), the server books it the same way `bot.py` does:
- the **journal** gets the result, outcome and R-multiple
- the **governor** counts the P&L toward today's loss cap
- the **setup blacklist** and **loss streak** are updated
- the **ML gate** gets a training sample and an online update, and retrains every `META_RETRAIN_EVERY` closed trades. This uses the market state saved at entry, so it survives server restarts.
- a **Telegram** alert is sent (if enabled)
- a **lesson** is written for the trades worth learning from (the reflection agent's filter: stop losses, wins of 1.5R+, any move of 0.3R+, and confident calls that lost):
  - `MCP_LESSONS = "client"`: the trade waits in a review queue. Ask your AI app to "review my closed trades". It calls `get_trades_to_review`, which shows the result, the reason you gave at entry and the market state then, and saves the lesson with `save_lesson`. Lessons under `MEMORY_USEFULNESS_THRESHOLD` aren't stored.
  - `MCP_LESSONS = "llm"`: the bot's reflection agent writes it straight away.

Saved lessons are recalled by `bot.py` before its own trades and by `recall_lessons`. Pass `reason` and `confidence` to `place_trade`: they are what the review sees.

Lesson text is also embedded for meaning-based search through NVIDIA's embedding API, using `LLM_API_KEY`, so that part only works when `LLM_API_KEY` is an NVIDIA key. Without it, lessons are still saved and recalled by market fingerprint and feature similarity.

**Don't let `bot.py` and the MCP server trade the same account at the same time.** Both write the governor state file, and a running `bot.py` won't manage (trail or time-exit) positions the MCP server opened until it restarts.

## Control bot.py from your AI app
The MCP server can also run and supervise `bot.py`, the autonomous bot, and train it:

| Tool | What it does |
|---|---|
| `bot_status` | Is `bot.py` running: instance, pid, demo/live, last heartbeat and tick, paused, tightened settings |
| `get_bot_log(lines, which)` | The end of `bot.log`, or `which="console"` for start-up errors and crashes |
| `start_bot(live)` | Starts `bot.py` in the background (it keeps running if your app closes). Demo by default: it analyses and logs but sends no orders. `live=True` needs `MCP_TRADING_MODE = "live"` in config.py |
| `stop_bot(force, wait_seconds)` | The bot finishes its current step and exits. Positions stay open with their SL/TP, and a restarted bot adopts them. `force=True` ends it if it hasn't stopped in time |
| `restart_bot(live)` | Stop, then start (to load new settings) |
| `pause_trading` / `resume_trading` | No new trades, from the bot or from MCP. Open positions are still managed. The bot confirms the pause |
| `get_settings` / `update_settings(changes)` / `reset_settings(keys)` | Limits the AI may change, **only to make them stricter** (see below) |
| `retrain_ml_model` | Retrains the ML gate now. While `bot.py` runs, the bot retrains its own model |
| `run_training(kind, ...)` / `get_training_job(job_id)` | Runs a trainer in the background: `mt5_history` (ML samples from MT5 price history), `tick_csv` (from tick-export CSVs), `trade_csv_lessons` (lessons from a CSV of your past trades; uses the LLM key). Then call `retrain_ml_model` |

**How the two share one account.** These rules are proven in `specs/BotControl.tla` and tested in `tests/test_bot_control.py`:
- **One trader at a time.** A running `bot.py` holds an account lock (`observability/bot.lock`). While it runs, MCP `place_trade` is refused, and `close_position` is sent to the bot to carry out. The operating system frees the lock if the bot crashes, so the MCP server can trade again.
- **Each closed trade is booked once:** governor P&L, journal, ML sample, lesson. Only the lock holder books. The bot also books trades that closed while nothing was running.
- **A second `bot.py` won't start** on the same account. It exits with "already running".
- **Stop requests name one bot instance.** An old request can never stop a bot you start later.
- **A confirmed pause is real.** The bot confirms only between orders, so once `pause_trading` reports `bot_confirmed: true`, nothing is in flight.
- **Settings only tighten.** The AI can lower `RISK_PER_TRADE_PCT`, `MAX_LOT_SIZE`, `MAX_OPEN_POSITIONS`, `MAX_ORDERS_PER_DAY`, `DAILY_LOSS_CAP_PCT` and `MAX_DRAWDOWN_PCT`. It can raise `META_LABEL_MIN_PROB` and `MIN_CONFIDENCE_TO_TRADE`, and drop symbols from `SYMBOLS`. It can't loosen any of them, and it can't change `MCP_TRADING_MODE`, `MCP_HTTP_ALLOW_TRADING` or any key. The value in use is always the stricter of config.py and the override, so if you lower config.py later, that wins. `bot.py` applies overrides when it starts.
- **The kill switch stays yours.** `resume_trading` does not clear a daily-loss kill switch or a drawdown halt.
- **Over HTTP (ChatGPT)**, `start_bot`, `restart_bot`, `resume_trading` and `reset_settings` need `MCP_HTTP_ALLOW_TRADING = True`. Pausing, stopping and tightening always work.

`bot.py` still needs its own LLM key for its agent desk. The AI app supervises the bot; it doesn't do the bot's reasoning.

## Tools

| Tool | What it does |
|---|---|
| `get_account_status` | Equity, balance, open positions, governor state, trading mode |
| `analyze_market(symbol)` | Closed-candle indicators, structure, patterns, regime, ATR, sessions, live bid, spread |
| `get_news(symbol)` | Ranked headlines, live macro (VIX, DXY, yields, gold, fear & greed), calendar blackout and next event |
| `get_ml_probability(symbol, direction)` | The bot's meta-label model: chance this BUY/SELL wins |
| `recall_lessons(symbol)` | Lessons saved from similar past trades |
| `get_recent_trades(count)` | Recent trades from the journal |
| `preview_trade(symbol, direction, size_multiplier)` | All checks, plus lot, stop loss, take profit and money at risk. No order. |
| `place_trade(symbol, direction, size_multiplier, reason, confidence)` | Same checks, then an order (live mode only) |
| `close_position(ticket)` | Closes one of the bot's positions (live mode only) |
| `get_trades_to_review(count)` | Closed trades waiting for a lesson, with a guide to writing one |
| `save_lesson(ticket, lesson, usefulness_score, category, pattern_name, should_save)` | Saves the lesson into the bot's memory (or skips it) |

## Connect your app

Replace `C:\path\to\ai-forex-trading-bot` with your folder.

**Using the Windows app (ForexBot.exe)?** Wherever an example below starts the server with `py ...\mcp_server.py`, use the app instead: the command is `C:\path\to\ForexBot\ForexBot.exe` and the arguments are `mcp`. For example, Claude Code:
```
claude mcp add --transport stdio forex-bot -- C:\path\to\ForexBot\ForexBot.exe mcp
```
and Claude Desktop: `"command": "C:\\path\\to\\ForexBot\\ForexBot.exe", "args": ["mcp"]`.

### Claude Code
```
claude mcp add --transport stdio forex-bot -- py C:\path\to\ai-forex-trading-bot\mcp_server.py
```
Add `--scope user` before `forex-bot` to make it available in every project.

### Claude Desktop
Settings → Developer → Edit Config (`%APPDATA%\Claude\claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "forex-bot": {
      "command": "py",
      "args": ["C:\\path\\to\\ai-forex-trading-bot\\mcp_server.py"]
    }
  }
}
```
Restart Claude Desktop.

### OpenAI Codex (CLI and IDE extension)
```
codex mcp add forex-bot -- py C:\path\to\ai-forex-trading-bot\mcp_server.py
```
Or in `~/.codex/config.toml`:
```toml
[mcp_servers.forex-bot]
command = "py"
args = ["C:\\path\\to\\ai-forex-trading-bot\\mcp_server.py"]
```

### ChatGPT (web, developer mode)
ChatGPT connects only to a **remote HTTPS** MCP server, and its authentication options are OAuth or none, so it can't send an API-key header. This server instead requires a long secret token in the URL.

1. Make a token, then start the server:
   ```
   py -c "import secrets; print(secrets.token_urlsafe(32))"
   set MCP_HTTP_TOKEN=<the token>
   py mcp_server.py --http --port 8765 --public-host <your-tunnel-host>
   ```
2. Expose it over HTTPS with a tunnel, e.g. `ngrok http 8765` or `cloudflared tunnel --url http://localhost:8765`. Pass the tunnel's host name as `--public-host` (requests for any other host are rejected).
3. In ChatGPT: enable **Developer mode**. OpenAI lists it for Pro, Plus, Business, Enterprise and Education accounts on the web. Then create an app with:
   - **URL:** `https://<your-tunnel-host>/mcp?token=<the token>`
   - **Authentication:** No authentication. The token in the URL is the check.
4. Trading over HTTP stays off until you set `MCP_HTTP_ALLOW_TRADING = True`. Anyone who has that URL can use the tools, so treat it like a password.

## Try it
- "Use forex-bot: show my account status."
- "Analyse EURUSD and GBPUSD, check the news and the ML probability, then preview the better trade."
- "Place it if every check passes. Reason: H1 trend continuation." (dry run unless live mode is on)
- "Review my closed trades and save the lessons worth keeping."
- "Start the bot in demo mode and tell me when it's ticking."
- "Pause trading, then show me the last 50 lines of the bot's log."
- "Lower the risk per trade to 0.25% and restart the bot."
- "Pre-train the ML gate from 5000 bars of EURUSD history, then retrain it."

## Autonomous bot vs MCP
- **`bot.py`** trades on its own every candle with its 5-agent LLM desk. It needs an LLM API key, or a local model through Ollama.
- **The MCP server** acts only when you ask your AI app. Your app's model does the reasoning, so no extra key is needed.

## Tests
`py -m pytest tests/test_phase3_mcp.py -v` runs every tool against a fake MT5 terminal. It also starts the server over stdio exactly as Claude/Codex do, and over HTTP with the token check.

`py -m pytest tests/test_bot_control.py -v` tests the bot controls. It covers real processes (start, pause, close, retrain, stop, forced stop, a crash) and the real `bot.py` loop (pause before and during a tick, stop, close and retrain requests, trades booked once). It also runs random action sequences that mirror `specs/BotControl.tla`.
