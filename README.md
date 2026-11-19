# AI Forex Trading Bot — 5-Agent Adversarial Architecture

An autonomous forex trading system for MetaTrader 5 that combines an LLM-based
adversarial agent desk, a machine-learning meta-label gate, a formally-designed
risk governor, and an NVIDIA NeMo-based alpha signal discovery pipeline.

> ⚠️ **RISK DISCLAIMER** — This software places real trades with real money.
> Forex trading carries substantial risk of loss. This project is provided for
> education and research; **nothing here is financial advice, and no claim of
> profitability is made**. Run it exclusively on a demo account until you have
> weeks of forward-test evidence you personally trust. You are solely
> responsible for any losses.

---

## How it works

Every 60-second tick, per symbol on the watch list:

```
MT5 candles (H1 + H4)
   │
   ▼
Regime classifier ──── ADX + Hurst exponent + Kaufman ER + ATR percentile
   │                   (skip symbol if regime not allowed)
   ▼
Gates: spread filter → CUSUM event filter → economic-calendar blackout
   │
   ▼
Context: 65-pattern candlestick scan, S/R zones, pivots, COT positioning,
         economic-surprise index, live macro (VIX/DXY/yields), news RSS,
         memory recall (fingerprint + embedding similarity, win/loss balanced)
   │
   ▼
┌──────────────────── 5-AGENT ADVERSARIAL DESK (LLM) ────────────────────┐
│ 1. Technical Analyst  — builds the bull case from indicators           │
│ 2. Bear Analyst       — attacks the bull case, scores risk 0-10        │
│ 3. News Analyst       — macro tone, event risk, session bias           │
│ 4. Fund Manager       — arbitrates the debate → BUY / SELL / HOLD      │
│ 5. Risk Manager       — position sizing multiplier + veto gate         │
└─────────────────────────────────────────────────────────────────────────┘
   │
   ▼
Confidence adjustments (each small, ±3-8%): seasonality, key levels,
Fibonacci, COT alignment, econ surprise, S/R zones, pivots, intermarket
bias, RSI/MACD divergence, stochastic, pattern miner, discovered alpha
signals (see below)
   │
   ▼
ML meta-label gate ─── calibrated LightGBM (López de Prado meta-labeling):
   │                   P(win | 29 features) must clear META_LABEL_MIN_PROB
   ▼
Hard gates: M15 entry timing → risk-manager veto → bear veto →
duplicate-position → correlation guard → RISK GOVERNOR
   │
   ▼
Order placed (ATR-adaptive SL/TP, ¼-Kelly position sizing)
   │
   ▼ (when the trade closes)
Real P&L from MT5 deal history → R-multiple outcome → LLM reflection →
lesson saved to memory → SGD online update → periodic LightGBM retrain
```

### The agent desk

Five LLM agents (NVIDIA NIM, OpenAI-compatible; provider configurable in
`config.py`) argue each trade adversarially. The Bear Analyst exists to kill
weak setups; a bear risk score above the symbol-specific threshold vetoes the
trade outright. All LLM output is schema-validated and clamped
(`agents/base.py`) — a malformed or failed response degrades to a safe
default (`HOLD`, `trade_qualified=False`), never to an exception in the
trading loop.

### Machine learning layer

* **Meta-labeling** (`ml/meta_label.py`) — the agents choose *direction*; a
  calibrated LightGBM classifier decides *whether the trade is worth taking*
  (López de Prado, *Advances in Financial ML*, ch. 3). Features: 21 technical
  + time-of-day/day-of-week cyclical encoding + regime + Fibonacci proximity
  + COT z-score + economic surprise.
* **Sample-uniqueness weighting** (AFML ch. 4) — overlapping same-symbol
  trades share a price path and are down-weighted accordingly; concurrency is
  computed **per symbol** so cross-pair trades at the same hour count as
  independent evidence.
* **Purged time-series CV** — `TimeSeriesSplit` with a 24-bar embargo gap
  prevents look-ahead leakage from overlapping H1 label windows; Platt
  calibration (correct at small N).
* **Regime-conditional models** — separate TRENDING and RANGING models train
  when ≥15 samples per regime exist.
* **SGD online companion** — updated after every closed trade, blended at 35%
  into predictions between batch retrains.
* **¼-Kelly sizing** — position size capped by quarter-Kelly computed from the
  ML win probability and the actual trade's TP/SL payoff ratio.

### Risk management (defense in depth)

1. **Risk Governor** (`risk_governor.py`, TLA+-spec'd design, Decimal
   arithmetic for all money): daily loss kill-switch (realised *and*
   mark-to-market), max drawdown halt, max positions, max orders/day, session
   filter, overnight-gap detection at startup.
2. **Position Manager**: break-even move, ATR trailing stop, R-based TP
   ladder (scale out at 1R/2R, restart-safe), time-based exit, per-position
   emergency loss cap.
3. **Correlation guard**: never stacks same-direction exposure on >0.85
   correlated pairs (e.g. EURUSD+GBPUSD, AUDUSD+NZDUSD).
4. **Loss-streak tilt guard**: halves size after N consecutive losses on a
   symbol; setup blacklist benches a symbol+regime combo after repeated losses.
5. **Error boundaries**: a failure in one symbol's scan skips that symbol,
   never kills the loop; an unhandled crash restarts the bot via a supervisor,
   re-adopting open positions from MT5 on startup.
6. **Process isolation**: yfinance/curl_cffi calls (known to hard-crash the
   interpreter on Windows network errors) run in a disposable child process.

### Alpha signal discovery (NVIDIA NeMo Agent Toolkit)

`signal-discovery/` (vendored from NVIDIA's
[quantitative-signal-discovery-agent](https://github.com/NVIDIA-AI-Blueprints/quantitative-signal-discovery-agent),
Apache-2.0, with local modifications) runs a closed loop: an LLM proposes
signal formulas → generates executable code → a Rank-IC evaluator scores them
cross-sectionally against forward returns on *your broker's own data* →
failed signals get optimization feedback and retry.

Accepted signals (|IC| ≥ 0.015, p ≤ 0.05) are stored as portable JSON and
picked up automatically by the live bot (`strategy/discovered_signals.py`):
re-quality-gated at load, computed on a live 23-symbol panel, converted to a
cross-sectional z-score per symbol, and applied as a small confidence tilt
(sign-inverted automatically when IC is negative).

```
# one-time: export your tick data as panels
py export_discovery_panel.py --dir <folder with SYMBOL_TICK_*.csv>

# discover signals (needs NVIDIA_API_KEY)
cd signal-discovery
.\run-discovery.ps1
```

### Memory system

SQLite store of trade "lessons": every closed trade is reflected on by an LLM,
scored for usefulness, and saved with its market fingerprint, feature vector,
regime, and embedding. Recall is balanced (wins *and* losses — negative
exemplars measurably improve LLM decisions), similarity-ranked, and
time-decayed. Two bootstrap paths pre-seed memory without LLM cost:
`mt5_historical_trainer.py` (live MT5 bar history) and
`csv_historical_trainer.py` (broker tick-CSV exports).

---

## Setup

**Requirements**: Windows (MetaTrader5 Python API is Windows-only), Python
3.11+, an MT5 terminal logged into a broker account (use a **demo** account),
and a free NVIDIA NIM API key from [build.nvidia.com](https://build.nvidia.com).

```powershell
git clone <this repo>
cd trading-bot
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# configure your key
copy .env.example .env          # then edit .env with your real key
# load it (or set LLM_API_KEY as a user environment variable once):
Get-Content .env | ForEach-Object { $n,$v = $_ -split '=',2; [Environment]::SetEnvironmentVariable($n,$v) }

# optional: pre-seed the ML gate from MT5 history (no LLM cost)
python mt5_historical_trainer.py --bars 8000

# run — ALWAYS start with demo mode
python bot.py --demo
python bot.py                    # live trading, only when you're ready
```

Key `config.py` settings:

| Setting | Default | Meaning |
|---|---|---|
| `SYMBOLS` | 23 pairs | watch list (majors, JPY/GBP crosses, metals, exotics) |
| `LOT_SIZE` | 0.01 | base lot (micro) — sizing multipliers scale this down only |
| `DAILY_LOSS_CAP_PCT` | 2.0 | kill switch — halts trading for the day |
| `MAX_DRAWDOWN_PCT` | 10.0 | halt below peak equity (manual reset by design) |
| `MAX_OPEN_POSITIONS` / `MAX_ORDERS_PER_DAY` | 3 / 8 | hard caps |
| `META_LABEL_MIN_PROB` | 0.52 | ML gate threshold |
| `MIN_CONFIDENCE_TO_TRADE` | 50 | fund-manager confidence floor (walk-forward adaptive) |
| `ALLOWED_REGIMES` | trending + ranging | VOLATILE always skipped |
| `DISCOVERED_SIGNALS_ENABLED` | True | alpha-signal confidence tilt |

Optional extras: Telegram alerts (`TELEGRAM_BOT_TOKEN`/`CHAT_ID`), a
localhost-only Flask dashboard (`dashboard/app.py`), daily reports, and a
backtester (`backtest/`).

---

## Use it from Claude, Codex or ChatGPT (MCP)

`mcp_server.py` lets Claude Code, Claude Desktop, OpenAI Codex, ChatGPT or any MCP client analyse the market and trade through the bot's risk governor and guards. The client's own model does the reasoning, so no LLM API key is needed in this mode. Trading is dry-run by default. Setup: **[MCP.md](MCP.md)**.

## Project layout

```
bot.py                    main loop (tick → gates → agent desk → order)
config.py                 every tunable, heavily commented
mt5_bridge.py             all MT5 I/O: candles, orders, deal history
risk_governor.py          hard limits (TLA+-spec'd, Decimal money math)
position_manager.py       trailing SL, break-even, TP ladder, time exits
agents/                   the 5 LLM agents + validated safe-default layer
strategy/                 indicators, regime, S/R, pivots, discovered signals
ml/                       meta-label gate, CUSUM filter, Hurst, fracdiff
memory/                   SQLite lesson store + embedding recall
data/                     calendar, news RSS, COT, macro context (isolated)
observability/            trade journal, reporter, pattern miner
signal-discovery/         NVIDIA NeMo alpha discovery (vendored, Apache-2.0)
tests/                    pytest suite
specs/                    TLA+ specifications
```

## Security notes

* **No secrets in this repo.** Keys live in `.env` (gitignored) or user
  environment variables. `.env.example` contains placeholders only.
* Logs, databases, and equity snapshots are gitignored — they contain account
  numbers and trading history.
* `strategy/discovered_signals.py` executes signal code stored in
  `signal-discovery/.../output/` — files produced on *your* machine by *your*
  discovery runs (same trust level as the codebase itself). Review any signal
  JSON a third party gives you before dropping it in that folder.
* The dashboard binds to 127.0.0.1 only.

## Contributing

Issues and PRs welcome. The most valuable contributions right now:

* **Backtesting** the full pipeline (agents + gates + sizing) with realistic
  spread/slippage costs
* More months of validation data and walk-forward studies
* Broker-compatibility reports (symbol suffixes, filling modes)
* Additional TLA+ specs for the execution path

Run `pytest tests/ -q` before submitting; all 42 tests must pass.

## License

MIT for this project's code (see `LICENSE`). The vendored
`signal-discovery/` directory retains NVIDIA's Apache-2.0 license
(`signal-discovery/LICENSE.txt`) with local modifications noted in file
headers.
