# Changelog

## v1.0.0 — 2026-11-19

First public release of the AI forex trading bot for MetaTrader 5.

## Download
- **Windows app:** `ForexBot-1.0.0-windows-x64.zip` (no Python needed). Unzip it, then read `START_HERE.txt`. *(Attached when the Windows build runs.)*
- **Source code:** the zip / tar.gz below (Python 3.10+, `pip install -r requirements.txt`).

Trading involves risk of loss. Start with `ForexBot run --demo` on an MT5 demo account.

## What it does
- **5-agent LLM desk** (technical, bear case, macro, risk, fund manager). Works with NVIDIA NIM, DeepSeek, OpenAI, Claude, Ollama (local, free) or any OpenAI-compatible API.
- **Hard risk governor**: daily loss cap, kill switch, drawdown halt, max positions and orders.
- **Safety and decision checks**: spread, session, calendar and news blackout filters; correlation guard.
- **ML meta-label gate**: walk-forward CV, Kelly sizing, online learning from every closed trade.
- **Position management**: trailing stop, break-even, TP ladder (never widens a stop).
- **Memory of lessons** from past trades, recalled before new ones.
- **MCP server** (`ForexBot mcp`): Claude Code, Claude Desktop, Codex, ChatGPT or any MCP client can analyse, trade (dry-run by default), review trades and write lessons, and **start, stop, pause, retrain and supervise the bot**. Only one trader per account at a time; settings can only be tightened from MCP. See MCP.md.
- **Dashboard** at http://127.0.0.1:5000, plus trainers for MT5 history, tick data and past trades.

## Verified
- **TLA+ specs, checked with TLC:**
  - risk governor: 46,136 states
  - order state machine: 4,246,401
  - position lifecycle: 5,550
  - bot control: 10,029,140 (safety) and 751,896 (liveness)
- **167 automated tests**, including real processes and the real bot loop against a simulated MT5 terminal.

## Build it yourself
Run `build_exe.bat` on Windows. It builds `dist\ForexBot\` and the zip, and smoke-tests every command in the built app.
