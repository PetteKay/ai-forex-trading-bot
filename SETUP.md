# AI Forex Trading Bot — Setup Guide

## What makes this different from a basic bot
This bot has **three AI agents** and a **selective memory system**:

1. **Technical Analyst agent** — reads the chart (EMA, RSI, MACD, Bollinger Bands, ATR)
   and gives a structured view of what the market is doing.

2. **Portfolio Manager agent** — reads the technical view AND looks up similar past
   situations from memory. Decides BUY / SELL / HOLD with a confidence score.
   Only trades if confidence exceeds your threshold.

3. **Reflection agent** — after every significant trade closes, reviews what happened,
   extracts a lesson, and saves it to memory IF it's genuinely useful.
   Mediocre/obvious trades are ignored. Only real insights are saved.

The bot starts knowing nothing. After each trade it learns. After 50-100 trades,
it has a real knowledge base of what works on your symbol.

---

## Step 1 — Get an LLM API key (the brain)

**Option A: DeepSeek (recommended — cheap and smart)**
- Sign up at platform.deepseek.com
- Get your API key
- Cost: ~$0.001 per trade decision (very cheap)

**Option B: OpenAI (GPT-4o-mini)**
- Sign up at platform.openai.com
- Cost: ~$0.003 per trade decision

**Option C: Ollama (free, runs on your PC, no internet)**
- Download from ollama.com
- Run: `ollama pull llama3.2`
- Set LLM_PROVIDER = "ollama" in config.py

## Step 2 — Edit config.py
Open `config.py` and set:
```python
LLM_PROVIDER = "deepseek"                    # or "openai" or "ollama"
LLM_API_KEY  = "your-actual-api-key-here"
LLM_MODEL    = "deepseek-chat"               # or "gpt-4o-mini" or "llama3.2"
LLM_BASE_URL = "https://api.deepseek.com"    # None for OpenAI; "http://localhost:11434/v1" for Ollama

SYMBOL       = "EURUSD"   # Change to what you want to trade
```

## Step 3 — Install MetaTrader 5
- Download from your broker (IC Markets, Exness, Pepperstone, OANDA, etc.)
- Log in with a **demo account first**
- Keep MT5 open while the bot runs

## Step 4 — Install Python dependencies
```
cd C:\Users\Kusi\trading-bot
pip install -r requirements.txt
```

## Step 5 — Pre-train with sample data (optional but recommended)
This gives the bot initial knowledge before its first live trade:
```
python trainer.py --demo
```

Or if you have your own past trade history as a CSV:
```
python trainer.py --file data/my_trades.csv
```

## Step 6 — Test in demo mode
```
python bot.py --demo
```
Runs the full agent pipeline every 60 seconds and logs what it WOULD trade —
but places no real orders. Watch it for a day to see how it reasons.

## Step 7 — Go live
```
python bot.py
```

---

## How the learning loop works

```
Every 60 seconds:
  ├─ Fetch candles from MT5
  ├─ Compute 8 indicators
  ├─ Query memory: "have I seen a situation like this before?"
  ├─ Technical Analyst reads the chart → gives view
  ├─ Portfolio Manager: view + memories → decision
  ├─ Risk Governor checks safety limits
  └─ If approved → place order

When a trade closes:
  ├─ Reflection Agent reviews what happened
  ├─ If genuinely useful (score > 0.55) → save lesson to memory
  └─ Next similar situation → Portfolio Manager now knows this
```

## Memory is stored in
`memory/trading_memory.db` (SQLite — you can open it with DB Browser for SQLite
to see what the bot has learned)

---

## Common questions

**How many trades until it starts being smart?**
After 20-30 trades with significant outcomes, it starts having meaningful context.
After 100 trades, the memory is rich. Pre-training with --demo or your CSV gives
it a head start.

**What if I want it to trade gold (XAU/USD)?**
Set `SYMBOL = "XAUUSD"` in config.py. Make sure the symbol name matches exactly
what appears in your MT5 Market Watch panel.

**How much does the LLM cost per month?**
With DeepSeek at ~$0.001 per decision, 4 trades/day × 30 days × 2 (entry + reflection)
= ~$0.24/month. Negligible.

**How do I stop it?**
Press Ctrl+C in the terminal window. It shuts down cleanly.
