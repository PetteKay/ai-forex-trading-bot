# ============================================================
# AI Forex Trading Bot — Full Configuration
# ============================================================

import os

# --- LLM (the brain) ---
# Pick ONE provider block. Comment out the others.
#
# PROVIDER   MODEL examples                    API KEY FROM                 COST
# deepseek   deepseek-chat                     platform.deepseek.com        ~$0.001/decision
# openai     gpt-4o-mini, gpt-4o              platform.openai.com          ~$0.003/decision
# claude     claude-haiku-4-5-20251001        console.anthropic.com        ~$0.001/decision
#            claude-sonnet-4-6                                              ~$0.005/decision
#            claude-opus-4-8                                                ~$0.015/decision
# nvidia     meta/llama-3.1-70b-instruct      build.nvidia.com             free tier + credits
#            meta/llama-3.3-70b-instruct
#            deepseek-ai/deepseek-r1
#            nvidia/llama-3.1-nemotron-70b-instruct
#            mistralai/mixtral-8x22b-instruct-v0.1
# ollama     llama3.2, mistral, etc.          (free, runs on your PC)      free

# ── DeepSeek (recommended to start — cheapest, ~$0.001/decision) ──
# LLM_PROVIDER = "deepseek"
# LLM_API_KEY  = "your-deepseek-api-key-here"
# LLM_MODEL    = "deepseek-chat"
# LLM_BASE_URL = "https://api.deepseek.com"

# ── NVIDIA NIM — free credits, huge model choice (uncomment to use) ──
# Get API key at: build.nvidia.com → Get API Key
LLM_PROVIDER = "nvidia"
LLM_API_KEY  = os.environ.get("LLM_API_KEY", "")
LLM_MODEL    = "meta/llama-3.3-70b-instruct"   # upgraded: better instruction following than 3.1
LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ── Claude / Anthropic (uncomment to use) ──
# LLM_PROVIDER = "claude"
# LLM_API_KEY  = "your-anthropic-api-key-here"
# LLM_MODEL    = "claude-haiku-4-5-20251001"   # fast + cheap
# LLM_MODEL    = "claude-sonnet-4-6"           # smarter, moderate cost
# LLM_MODEL    = "claude-opus-4-8"             # most capable
# LLM_BASE_URL = None

# ── OpenAI (uncomment to use) ──
# LLM_PROVIDER = "openai"
# LLM_API_KEY  = "your-openai-api-key-here"
# LLM_MODEL    = "gpt-4o-mini"
# LLM_BASE_URL = None

# ── Ollama — free, runs on your PC, no internet (uncomment to use) ──
# LLM_PROVIDER = "ollama"
# LLM_API_KEY  = "ollama"
# LLM_MODEL    = "llama3.2"
# LLM_BASE_URL = "http://localhost:11434/v1"

# --- LLM Reproducibility ---
LLM_SEED           = 42     # Fixed seed for reproducible LLM outputs (where supported)
LLM_MAX_TOKENS     = 1000   # Max tokens in LLM response (1000 is ample for JSON; 2048 caused 6-min hangs)
LLM_TIMEOUT_SECONDS = 30    # Hard timeout per LLM call — prevents MT5 IPC disconnect during slow responses

# --- Per-agent model overrides (optional) ---
# Use a stronger model for the most critical decision-makers.
# Leave empty {} to use LLM_MODEL for all agents.
# NOTE: nvidia/llama-3.3-nemotron-super-49b-v1 was causing multi-minute response times.
# All agents now use the default llama-3.3-70b-instruct — fast (~5-15s per call),
# reliable, and avoids MT5 IPC timeouts. Switch individual agents back to nemotron
# only after confirming consistent sub-30s response times from NVIDIA NIM.
MODEL_OVERRIDES = {}

# --- Instrument ---
SYMBOL       = "EURUSD"      # What to trade: EURUSD, XAUUSD (gold), USOIL, GBPUSD...
TIMEFRAME    = "H1"          # Primary chart candle size: M15, H1, H4, D1
HTF_TIMEFRAME = "H4"         # Higher timeframe for trend confirmation
MAGIC_NUMBER = 20260001      # Identifies this bot's trades in MT5

# --- Multi-symbol watch list — 25 broker-verified instruments ---
# Cleaned 2026-07-01: removed symbols not found at MetaQuotes demo broker
# (USOIL, NATGAS, NAS100, GER40, BTCUSD, ETHUSD, JPN225).
# Added user-requested: TSLA, NVDA, MSFT, AMD (US stock CFDs).
#
# Session coverage:
#   Asian  (22:00–08:00): USDJPY AUDUSD NZDUSD AUDJPY NZDJPY CADJPY CHFJPY XAUUSD USDCNH
#   London (08:00–16:00): EURUSD GBPUSD GBPJPY EURJPY GBPNZD GBPAUD GBPCAD GBPCHF EURNZD XAUUSD USDCNH USDSEK
#   NY     (13:00–22:00): EURUSD GBPUSD USDCAD USDCHF XAGUSD
#
# CORRELATION WARNINGS — never hold these in the same direction simultaneously:
#   AUDUSD + NZDUSD (+0.90) | EURUSD + GBPUSD (+0.85) | EURUSD + USDCHF (-0.90)
#   XAUUSD + XAGUSD (+0.80)
#
# Symbol name note: broker names vary. Bot auto-resolves suffixes (.r .pro .a).
SYMBOLS = [
    # ── Forex Majors (7) ──────────────────────────────────────────────────────
    "EURUSD",   # 65–78 pip ADR   | London/NY | ADX>25 ~42–50%
    "GBPUSD",   # 111.5 pip ADR   | London    | ADX>25 ~43–50%; most volatile major
    "USDJPY",   # 75–90 pip ADR   | Asian/NY  | ADX>25 ~48–55%; carry trade
    "AUDUSD",   # 50–65 pip ADR   | Asian     | ADX>25 ~50–58%
    "NZDUSD",   # 45–58 pip ADR   | Asian     | ADX>25 ~50–58%; #1 trending major
    "USDCAD",   # 60–75 pip ADR   | New York  | ADX>25 ~40–48%; oil-linked
    "USDCHF",   # 60–72 pip ADR   | London/NY | ADX>25 ~38–45%; safe-haven

    # ── JPY Crosses (6) ───────────────────────────────────────────────────────
    "GBPJPY",   # 125–175 pip ADR | London/Asian | ADX>25 ~58–65%; #1 trending cross
    "EURJPY",   # 105–145 pip ADR | London/Asian | ADX>25 ~55–63%
    "AUDJPY",   # 85–110 pip ADR  | Asian        | ADX>25 ~48–56%; risk-appetite gauge
    "NZDJPY",   # 72–95 pip ADR   | Asian        | ADX>25 ~48–56%
    "CADJPY",   # 75–98 pip ADR   | Asian/NY     | ADX>25 ~42–50%
    "CHFJPY",   # 65–85 pip ADR   | Asian        | ADX>25 ~38–46%

    # ── GBP Crosses (4) ───────────────────────────────────────────────────────
    "GBPNZD",   # 165–215 pip ADR | London | #1 most volatile pair 2014–2025
    "GBPAUD",   # 145–200 pip ADR | London | #2 most volatile
    "GBPCAD",   # 110–165 pip ADR | London | BoE+BoC+oil
    "GBPCHF",   # 100–135 pip ADR | London | BoE+SNB

    # ── EUR Cross (1) ─────────────────────────────────────────────────────────
    "EURNZD",   # 110–145 pip ADR | London | ECB + RBNZ divergence

    # ── AUD Cross (1) ─────────────────────────────────────────────────────────
    "AUDCAD",   # 55–75 pip ADR   | Asian/NY | ADX>25 ~40–48%; commodity cross (AUD+oil-linked CAD)

    # ── Metals (2) ────────────────────────────────────────────────────────────
    "XAUUSD",   # ADR 350–530 pt  | All sessions | ADX>25 ~65–75%; #1 trending
    "XAGUSD",   # ADR ~120 pt     | NY/London    | ADX>25 ~45–55%

    # ── Exotic Pairs (2) ──────────────────────────────────────────────────────
    "USDCNH",   # 250–450 pip ADR | Asian/London | ADX>25 ~35–42%; PBOC-managed float, low realized vol
    "USDSEK",   # 400–650 pip ADR | London       | ADX>25 ~40–48%; Nordic exotic, wide spreads, thin liquidity
]

# --- Position Sizing ---
LOT_SIZE     = 0.01          # Micro lot = ~$1/pip on EURUSD. Start small.

# --- Risk Governor (SAFETY — never remove) ---
MAX_OPEN_POSITIONS  = 3      # Up to 3 concurrent trades (diversified across uncorrelated pairs)
MAX_ORDERS_PER_DAY  = 8      # Up to 8 trades per day across all symbols
DAILY_LOSS_CAP_PCT  = 2.0    # Kill the bot for the day if account drops 2%
MAX_DRAWDOWN_PCT    = 10.0   # Halt if account drops 10% from peak equity

# --- Session Filter ---
# Only trade during these sessions. Empty list = always trade.
# Options: "LONDON", "NEW_YORK", "ASIAN"
# Asian session added: GBPJPY/EURJPY/AUDJPY/XAUUSD all trend during Asian hours
ALLOWED_SESSIONS = ["LONDON", "NEW_YORK", "ASIAN"]

# --- Economic Calendar Blackout ---
CALENDAR_BLACKOUT_BEFORE_MIN = 30   # Minutes before news event to stop trading
CALENDAR_BLACKOUT_AFTER_MIN  = 60   # Minutes after news event to resume
CALENDAR_CSV_PATH = "data/events.csv"  # Optional custom events file (see data/calendar.py)

# --- Technical Indicators ---
EMA_FAST     = 9
EMA_SLOW     = 21
EMA_TREND    = 200           # Long-term trend filter
RSI_PERIOD   = 14
ATR_PERIOD   = 14
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9
BB_PERIOD    = 20
BB_STD       = 2.0
ATR_SL_MULT  = 1.5           # Stop loss = ATR * 1.5
ATR_TP_MULT  = 2.5           # Take profit = ATR * 2.5

# --- Active Position Management ---
BREAKEVEN_PIPS      = 15     # Move SL to break-even after this many pips profit
TRAIL_START_PIPS    = 20     # Start trailing SL after this many pips profit
ATR_TRAIL_MULT      = 1.0    # Trail at ATR * this multiplier behind price
PARTIAL_CLOSE_PIPS  = 30     # Close 50% of position after this many pips
MAX_HOLD_HOURS      = 48     # Force-close if held this long with < 5 pips profit
MAX_POSITION_LOSS_USD = None  # Emergency close if single position loses this much (None = disabled)

# --- Memory (selective — only genuinely useful lessons) ---
MEMORY_DB_PATH              = "memory/trading_memory.db"
MEMORY_USEFULNESS_THRESHOLD = 0.55   # Only save memories with score > 0.55 (0-1 scale)
MEMORY_RECALL_COUNT         = 5      # Show agents this many past memories per decision
MEMORY_MAX_STORED           = 1000   # Prune oldest/weakest when exceeding this

# --- Meta-Labeling Gate ---
META_LABEL_MIN_PROB       = 0.52   # Minimum meta-label probability to proceed with trade
META_RETRAIN_EVERY        = 20     # Retrain meta-label model every N new memories
ML_TRAINING_LOOKBACK_DAYS = 180    # Only train on memories from the last N days (walk-forward)

# --- Kelly Position Sizing (research: ¼-Kelly maximises growth / minimises drawdown path risk) ---
# Requires meta-label ML to be active (>=30 trades). Falls back to LLM sizing otherwise.
# Formula: kelly = (p × b − q) / b   where b = TP/SL ratio (payoff), p = ML win probability
# Quarter-Kelly: kelly × 0.25  — eliminates 75% of Kelly's variance while keeping ~75% of growth
USE_KELLY_SIZING    = True   # Set False to revert to pure LLM-based sizing
KELLY_FRACTION      = 0.25   # Fraction of full Kelly to use (0.25 = quarter-Kelly)

# --- Agent Decision Gate ---
# 50% = "slight edge" — acceptable given the 5-layer risk system that catches bad trades downstream.
# Was 55% but that combined with the conservative LLM prompt caused a permanent 40-45% lock.
MIN_CONFIDENCE_TO_TRADE  = 50
BEAR_RISK_VETO_THRESHOLD = 8   # Bear analyst risk score >= 8 vetoes the trade

# --- Regime Filter ---
# Which regimes allow trading (comment out any to block that regime)
ALLOWED_REGIMES = [
    "TRENDING_UP",
    "TRENDING_DOWN",
    "RANGING",     # Allowed: bot uses mean-reversion logic in ranging markets
    # "VOLATILE",  # Leave commented — high volatility = skip
]

# --- Observability ---
JOURNAL_DB_PATH    = "observability/trade_journal.db"
# Risk governor safety state (daily P&L, kill switch, drawdown halt, peak equity).
# Survives restarts and crash auto-restarts. Delete only if you know why.
GOVERNOR_STATE_PATH = "observability/governor_state.json"
EQUITY_LOG_EVERY   = 5      # Log equity snapshot every N ticks

# --- Loop ---
CHECK_INTERVAL_SECONDS = 60  # Check for signals every 60 seconds

# --- Backtesting ---
BACKTEST_MAX_HOLD_BARS = 120  # Force-exit in backtest after this many bars

# --- Training ---
TRAIN_DATA_PATH = "data/historical_trades.csv"   # CSV for pre-training memory

# ── UPGRADE FEATURES ─────────────────────────────────────────────────────────

# --- Discovered Alpha Signals (signal-discovery workflow bridge) ---
# Signals found and IC-validated by signal-discovery/ on this broker's own
# tick data (see export_discovery_panel.py + configs/config-bot-forex.yml).
# Accepted signals are re-quality-gated at load (|IC| + p-value below), then
# scored cross-sectionally: each symbol gets a panel z-score used as a
# confidence tilt (BUY favored when z > 0, SELL when z < 0).
DISCOVERED_SIGNALS_ENABLED      = True
DISCOVERED_SIGNALS_CONF_ADJ     = 4      # ± confidence % when |z| clears threshold
DISCOVERED_SIGNALS_MIN_ABS_Z    = 0.5    # ignore weak cross-sectional tilts
DISCOVERED_SIGNALS_IC_THRESHOLD = 0.015  # same acceptance bar as the discovery run
DISCOVERED_SIGNALS_P_THRESHOLD  = 0.05

# --- Spread Filter ---
# Skip the trade if live bid-ask spread exceeds this threshold (in pips).
# Protects against trading during news spikes or illiquid hours.
SPREAD_FILTER_ENABLED = True
# Spread thresholds in broker "pip" units (info.spread * info.point / 0.0001).
# WARNING: Indices and stock CFDs return huge "pip" values because their point
# size (1.0 for US30, 0.01 for stocks) makes the pip-normalized number very large.
# Example: US30 with $1.50 spread → 1.5 * 1.0 / 0.0001 = 15000 "pips".
# Thresholds below account for this — they are NOT forex pip counts.
MAX_SPREAD_PIPS = {
    "DEFAULT":  5.0,        # fallback for FX pairs not listed below
    # Forex majors
    "EURUSD":   2.5,
    "GBPUSD":   3.0,
    "USDJPY":   2.5,
    "AUDUSD":   3.0,
    "NZDUSD":   3.5,
    "USDCAD":   3.0,
    "USDCHF":   3.0,
    # JPY crosses
    "GBPJPY":   5.0,
    "EURJPY":   4.0,
    "AUDJPY":   4.0,
    "NZDJPY":   4.5,
    "CADJPY":   5.0,
    "CHFJPY":   5.0,
    # GBP crosses — wider spreads are normal for volatile pairs
    "GBPNZD":  10.0,
    "GBPAUD":   8.0,
    "GBPCAD":   7.0,
    "GBPCHF":   6.0,
    "EURNZD":   7.0,
    # Metals
    "XAUUSD":  30.0,
    "XAGUSD":  50.0,
    # US Equity Index — point=1.0 makes pip values 10000× larger than FX
    # $3 spread on US30 = 30000 "pips". Cap at $5 spread = 50000.
    "US30":   50000.0,
    # US Stock CFDs — point=0.01 makes pip values 100× larger than FX
    # $0.10 spread on TSLA = 1000 "pips". Cap at $2.00 spread = 20000.
    "TSLA":  20000.0,
    "NVDA":  20000.0,
    "MSFT":  20000.0,
    "AMD":   20000.0,
}

# --- Adaptive ATR Stop-Loss ---
# Adjusts SL width based on the current ATR percentile (volatility regime).
# Low vol → tighter SL (less noise); high vol → wider SL (more room to breathe).
ADAPTIVE_ATR_SL_ENABLED = True
ATR_SL_LOW_VOL_PCT   = 30    # ATR percentile below which "low vol" multiplier applies
ATR_SL_HIGH_VOL_PCT  = 70    # ATR percentile above which "high vol" multiplier applies
ATR_SL_LOW_VOL_MULT  = 1.2   # Use this instead of ATR_SL_MULT in low-vol environments
ATR_SL_HIGH_VOL_MULT = 2.0   # Use this instead of ATR_SL_MULT in high-vol environments
# Normal vol (30th–70th percentile) continues to use ATR_SL_MULT = 1.5

# --- Seasonality ---
# Applies time-of-day and day-of-week confidence multipliers (0.80–1.20).
# London open = +15%; NY open = +10%; Friday close = -15%; Sunday = -20%.
SEASONALITY_ENABLED  = True

# --- M15 Entry Timing ---
# After an H1 signal, check M15 to confirm pullback entry (avoid chasing).
# Returns PROCEED / WAIT. WAIT does not veto — it logs and proceeds after 1 tick.
ENTRY_TIMING_ENABLED = True

# --- Telegram Trade Alerts ---
# Get your bot token from @BotFather and chat ID from @userinfobot on Telegram.
TELEGRAM_ALERTS_ENABLED = False           # Set True once token and chat ID are filled
TELEGRAM_BOT_TOKEN      = ""             # e.g. "7123456789:AAFxyz..."
TELEGRAM_CHAT_ID        = ""             # e.g. "123456789" (your numeric user/group ID)

# --- Live Economic Calendar ---
# Fetches real event times from ForexFactory (unofficial JSON endpoint).
# Falls back to the built-in recurring rules if the API is unreachable.
LIVE_CALENDAR_ENABLED = True

# ── PERFORMANCE ENHANCEMENT PACK v3 ──────────────────────────────────────────

# --- Symbol-Specific Parameters ---
# Each asset class gets calibrated ATR_SL_MULT, ATR_TP_MULT, and BEAR_RISK_VETO.
# Crypto and GBP crosses need wider stops; majors need tighter stops.
# See strategy/symbol_params.py for the full registry.
SYMBOL_PARAMS_ENABLED = True

# --- Key Level Detection ---
# Detects round numbers and previous week high/low.
# Aligned entry (BUY at support, SELL at resistance) boosts confidence.
# Counter-signal entry (BUY at resistance) reduces confidence.
KEY_LEVEL_CHECK_ENABLED     = True
KEY_LEVEL_CONFIDENCE_BOOST  = 8    # Points added to confidence when entry aligns with key level
KEY_LEVEL_COUNTER_PENALTY   = 5    # Points subtracted when entry fights the key level

# --- Loss-Streak Tilt Guard ---
# After N consecutive losses on a symbol, halve the lot size as protection.
# Restores after M consecutive wins.
LOSS_STREAK_ENABLED     = True
LOSS_STREAK_THRESHOLD   = 3      # Consecutive losses before size reduction
LOSS_STREAK_SIZE_MULT   = 0.5    # Lot multiplier during a loss streak (0.5 = half size)
LOSS_STREAK_WIN_RESTORE = 2      # Consecutive wins needed to restore full size

# --- Setup Blacklist ---
# Blocks trading a specific symbol+regime after N consecutive losses.
# Resets automatically after a win. Prevents repeating a losing setup.
SETUP_BLACKLIST_ENABLED   = True
SETUP_BLACKLIST_THRESHOLD = 3    # Consecutive losses to trigger blacklist

# --- Daily Performance Report ---
# Generates a P&L breakdown by symbol and regime each day at midnight UTC
# and on bot shutdown. Sent to Telegram if alerts are enabled.
DAILY_REPORT_ENABLED = True

# --- Regime-Conditional ML ---
# Trains separate meta-label models for TRENDING vs RANGING regimes.
# Requires MIN_SAMPLES_PER_REGIME=15 trades per regime before activating.
# Falls back to combined model if insufficient data per regime.
REGIME_ML_ENABLED = True   # enable the conditional model split (always on when sklearn present)

# --- R-Based TP Ladder ---
# Replaces the single partial-close-at-fixed-pips with a 3-tier exit strategy.
# Tier 1 at 1R: close 33% of position, move SL to breakeven (locks a free trade)
# Tier 2 at 2R: close another 33% (locks profitable exit)
# Remainder (~34%): rides the trailing stop — captures extended moves
# Net effect: dramatically reduces "reverse from full TP to full SL" scenarios.
TP_LADDER_ENABLED   = True
TP_LADDER_TIER1_R   = 1.0   # Close first tranche at 1× risk (1R)
TP_LADDER_TIER2_R   = 2.0   # Close second tranche at 2× risk (2R)
TP_LADDER_TIER1_PCT = 33    # % of original position to close at tier 1
TP_LADDER_TIER2_PCT = 33    # % of original position to close at tier 2
# Remainder (100 - TIER1_PCT - TIER2_PCT = ~34%) rides with trailing stop.

# --- Adaptive Confidence Threshold (Walk-Forward Optimization) ---
# After every N trades, analyzes recent win rates and auto-adjusts
# MIN_CONFIDENCE_TO_TRADE. Prevents system from trading into a cold streak.
ADAPTIVE_CONFIDENCE_ENABLED = True
ADAPTIVE_CONFIDENCE_LOOKBACK = 30   # analyze last N closed trades
ADAPTIVE_CONFIDENCE_MIN      = 50   # never go below this
ADAPTIVE_CONFIDENCE_MAX      = 68   # never go above this

# ── TIER 1 — SIGNAL UPGRADES ─────────────────────────────────────────────────

# --- COT (Commitment of Traders) Institutional Positioning ---
# Free weekly data from CFTC. Large Speculator net position z-score.
# Aligned signal (institutions same direction as trade) +COT_CONFIDENCE_ADJ confidence.
COT_ENABLED        = True
COT_DB_PATH        = "data/cot_history.db"
COT_CONFIDENCE_ADJ = 3      # ±3% confidence for COT alignment/counter

# --- Economic Surprise Index ---
# Tracks actual vs. forecast for HIGH-impact macro releases.
# Recent currency beats → positive surprise score → confidence boost when aligned.
ECON_SURPRISE_ENABLED    = True
ECON_SURPRISE_DB_PATH    = "data/econ_surprise.db"
ECON_SURPRISE_LOOKBACK   = 28   # days of rolling window
ECON_SURPRISE_CONF_ADJ   = 3    # ±3% confidence for economic surprise alignment

# --- Agent Accuracy Tracking ---
# Logs per-agent votes and outcomes. Fund Manager sees accuracy history to
# weight Bear Analyst and Technical Analyst's credibility in real time.
AGENT_ACCURACY_ENABLED  = True
AGENT_ACCURACY_LOOKBACK = 50    # last N trades for accuracy computation

# --- Fibonacci Retracement Confidence Boost ---
# Applied in addition to KEY_LEVEL_CONFIDENCE_BOOST when price is near a
# major Fib level (38.2%, 50%, 61.8%) AND the trade is directionally aligned.
FIB_CONFIDENCE_BOOST    = 4     # extra confidence at major Fib levels (38.2/50/61.8)

# ── TIER 2 — QUALITY UPGRADES ────────────────────────────────────────────────

# --- Multi-Touch Support/Resistance Zone Memory ---
# Identifies price zones tested 3+ times. Being at such a zone + aligned
# trade direction adds significant conviction.
SR_ZONE_ENABLED          = True
SR_ZONE_CONFIDENCE_BOOST = 5    # extra confidence when at strong multi-touch S/R zone
SR_ZONE_MIN_STRENGTH     = 0.4  # only apply boost if zone strength >= this

# --- Volatility-Adjusted Position Sizing ---
# When ATR is elevated (high-vol environment), the SL in pips is wider,
# meaning each lot risks more. Scale lot size down to keep dollar risk constant.
VOL_ADJUSTED_SIZING         = True
VOL_SIZING_HIGH_ATR_PCT     = 80    # ATR percentile above which to reduce lot
VOL_SIZING_HIGH_ATR_MULT    = 0.70  # lot multiplier in high-vol (70% of normal)
VOL_SIZING_EXTREME_ATR_PCT  = 92    # ATR percentile for extreme vol
VOL_SIZING_EXTREME_ATR_MULT = 0.50  # lot multiplier in extreme vol (50% of normal)

# ── TIER 3 — ADVANCED SIGNAL UPGRADES ────────────────────────────────────────

# --- Weekly / Monthly Pivot Levels ---
# Classic floor pivots (PP, R1-R3, S1-S3) watched by institutional desks.
# Aligned entry boosts confidence; counter entry penalizes.
PIVOT_LEVELS_ENABLED      = True
PIVOT_CONFIDENCE_BOOST    = 4     # points added when at aligned pivot
PIVOT_COUNTER_PENALTY     = 3     # points subtracted at counter-trend pivot

# --- Inter-Market Macro Bias ---
# Maps DXY trend, gold, oil, VIX to per-symbol directional bias.
# Adjusts confidence ±2-4 points when intermarket signals align or oppose.
INTERMARKET_ENABLED       = True
INTERMARKET_MAX_ADJ       = 4     # cap on per-signal adjustment (points)

# --- Trade Pattern Miner ---
# Analyzes journal history to find statistically strong / weak setups.
# Setup with ≥+15pp win rate above average gets +4 confidence boost.
PATTERN_MINER_ENABLED     = True
PATTERN_MINER_MIN_TRADES  = 15    # minimum combo trades before applying boost

# --- Divergence Confidence Adjustment ---
# RSI or MACD divergence confirming trade direction → boost.
# RSI or MACD divergence opposing trade direction → penalty.
DIVERGENCE_ENABLED        = True
DIVERGENCE_CONF_ADJ       = 4     # ±points for aligned/opposing divergence

# --- Stochastic Oscillator Confirmation ---
# Overbought + SELL or Oversold + BUY = extra confluence signal.
STOCH_ENABLED             = True
STOCH_CONF_ADJ            = 3     # points added when stoch confirms direction

# ── PERFORMANCE PEAK PACK — Session 5 ────────────────────────────────────────

# --- CUSUM Sequential Filter ---
# Prevents LLM desk from running on information-free ticks.
# Only evaluates a symbol when price has moved ≥ h_pct since last signal,
# OR after CUSUM_MAX_BARS bars without any signal (time fallback).
CUSUM_DISABLED       = False        # Set True to bypass (all ticks evaluated)
CUSUM_H_PCT          = 0.004        # 0.4% price move threshold (≈40 pips EURUSD at 1.08)
CUSUM_MAX_BARS       = 10           # Force evaluation after N ticks (~10 min); was 20 (20 min too slow for first trade)
CUSUM_MIN_BARS       = 1            # Don't re-fire within N ticks of last signal (was 3)

# --- Yang-Zhang Vol Regime (via strategy/vol_estimators.py) ---
# Adds vol_regime_label (QUIET/NORMAL/ELEVATED/EXTREME) to regime_info context.
# Available to all 5 agents via regime dict. EXTREME vol triggers HMM size cut.
VOL_ESTIMATOR_ENABLED = True

# --- HMM Latent Vol Regime (via strategy/hmm_regime.py) ---
# Gaussian HMM with 3 states (LOW_VOL/MED_VOL/HIGH_VOL). Refit daily on H1 bars.
# HIGH_VOL state or high transition probability → position size × 0.70.
HMM_REGIME_ENABLED        = True
HMM_SIZE_REDUCTION        = 0.70    # Lot size multiplier in HIGH_VOL state
HMM_TRANS_TO_HIGH_THRESH  = 0.40   # Transition probability threshold for size cut
