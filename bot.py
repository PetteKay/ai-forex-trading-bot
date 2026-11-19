"""
AI Forex Trading Bot — Main Loop (v2: Full 5-Agent Adversarial Architecture)

Every tick:
  1.  Shared: account equity, detect closed trades, manage open positions
  2.  Per-symbol scan (EURUSD, GBPUSD, XAUUSD, USDJPY …):
      a. Fetch candles (primary + higher timeframe)
      b. Regime classifier — skip if not trending
      c. Economic calendar blackout check
      d. Recall similar past situations from memory
      e. Run 5-agent desk:
           • Technical Analyst  (bull case)
           • Bear Analyst       (adversarial challenge)
           • News Analyst       (macro context)
           • Fund Manager       (debate arbitrator → BUY/SELL/HOLD)
           • Risk Manager       (position sizing + veto gate)
      f. Meta-label gate (ML secondary model)
      g. Risk Governor (TLA+-verified hard limits)
      h. Place order (with risk-manager-adjusted size)
  3.  When trade closes → get REAL P&L from MT5 deal history → reflect → save lesson

Usage:
  python bot.py           (live trading)
  python bot.py --demo    (no real orders — signals only)
"""

import time
import logging
import argparse
import json
import os
import sys
from datetime import datetime, UTC

# Fix Windows console encoding so Unicode box-drawing chars don't crash the logger
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import config
import mt5_bridge as mt5b
from risk_governor import RiskGovernor
from memory.store import TradeMemoryStore
from strategy.indicators import compute, market_fingerprint, feature_vector
from strategy.regime import classify as classify_regime
from data.calendar import EconomicCalendar
from position_manager import PositionManager
from ml.meta_label import MetaLabelGate
from observability.journal import Journal
from agents import base as llm
from agents import technical_analyst, portfolio_manager, reflection
from agents import bear as bear_analyst
from agents import news_analyst
from agents import risk_manager as risk_mgr_agent
from data.news_feed import fetch_headlines
try:
    from data.macro_context import get_macro_context, macro_summary_for_llm
    _MACRO_AVAILABLE = True
except ImportError:
    _MACRO_AVAILABLE = False

try:
    from strategy.seasonality import get_adjustment as seasonality_adjust
    from strategy.entry_timing import check as check_entry_timing
    _SEASONAL_TIMING_AVAILABLE = True
except ImportError:
    _SEASONAL_TIMING_AVAILABLE = False

try:
    from strategy.symbol_params import get as get_symbol_params
    from strategy.key_levels import check as check_key_levels
    _SYMBOL_EXTRAS_AVAILABLE = True
except ImportError:
    _SYMBOL_EXTRAS_AVAILABLE = False

try:
    from observability.reporter import generate_daily_report
    _REPORTER_AVAILABLE = True
except ImportError:
    _REPORTER_AVAILABLE = False

try:
    from optimization.adaptive_params import compute_adaptive_confidence
    _ADAPTIVE_PARAMS_AVAILABLE = True
except ImportError:
    _ADAPTIVE_PARAMS_AVAILABLE = False

try:
    from data.cot_data import get_cot_signal
    from data.economic_surprise import get_symbol_surprise
    _MARKET_SIGNALS_AVAILABLE = True
except ImportError:
    _MARKET_SIGNALS_AVAILABLE = False

try:
    from strategy.sr_zones import find_zones as find_sr_zones, check_sr_zone
    _SR_ZONES_AVAILABLE = True
except ImportError:
    _SR_ZONES_AVAILABLE = False

try:
    from strategy.pivot_levels import check_pivot_levels
    _PIVOT_LEVELS_AVAILABLE = True
except ImportError:
    _PIVOT_LEVELS_AVAILABLE = False

try:
    from data.intermarket import get_intermarket_bias
    _INTERMARKET_AVAILABLE = True
except ImportError:
    _INTERMARKET_AVAILABLE = False

try:
    from observability.pattern_miner import mine_patterns, get_setup_boost
    _PATTERN_MINER_AVAILABLE = True
except ImportError:
    _PATTERN_MINER_AVAILABLE = False

try:
    from ml.cusum_filter import CUSUMFilter
    _CUSUM_AVAILABLE = True
except ImportError:
    _CUSUM_AVAILABLE = False

try:
    from strategy.vol_estimators import regime_from_vol as vol_regime_from_bars
    _VOL_EST_AVAILABLE = True
except ImportError:
    _VOL_EST_AVAILABLE = False

try:
    from strategy.hmm_regime import HMMRegimeModel
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False

try:
    from strategy.discovered_signals import compute_scores as discovered_signal_scores
    _DISCOVERED_SIGNALS_AVAILABLE = True
except ImportError:
    _DISCOVERED_SIGNALS_AVAILABLE = False

from notifications.telegram_alert import TelegramAlert
from concurrent.futures import ThreadPoolExecutor

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("Bot")


# ── Trade record ─────────────────────────────────────────────────────────────
class OpenTradeRecord:
    def __init__(self, ticket: int, symbol: str, direction: str, entry_price: float,
                 entry_context: dict, tech_analysis: dict, pm_decision: dict,
                 entry_time: datetime, atr: float = 0.0,
                 sl: float = 0.0, tp: float = 0.0, volume: float = 0.01,
                 risk_usd: float = 0.0):
        self.ticket        = ticket
        self.symbol        = symbol
        self.direction     = direction
        self.entry_price   = entry_price
        self.entry_context = entry_context
        self.tech_analysis = tech_analysis
        self.pm_decision   = pm_decision
        self.entry_time    = entry_time
        self.atr           = atr
        self.sl            = sl
        self.tp            = tp
        self.risk_usd      = risk_usd  # USD at risk from entry to SL (for R-multiple)
        self.volume        = volume


# ── Outcome classification (R-multiple, account-size invariant) ──────────────
# R = net_pnl / initial_risk_usd.  1.0R = made back exactly what we risked.
def classify_outcome(r_multiple: float) -> str:
    if r_multiple >= 1.5:   return "WIN_STRONG"   # 150%+ of risk recovered
    if r_multiple >= 0.3:   return "WIN"           # at least 30% of risk recovered
    if r_multiple >= -0.15: return "BREAKEVEN"     # flat within spread/commission
    if r_multiple >= -0.8:  return "LOSS"          # partial loss
    return "STOP_LOSS"                             # full or near-full stop hit


# ── Main ─────────────────────────────────────────────────────────────────────
def run(demo_mode: bool = False, reset_drawdown_halt: bool = False):
    # Build watch list — use SYMBOLS if set, else fall back to single SYMBOL
    symbols_config = getattr(config, "SYMBOLS", None) or [config.SYMBOL]

    log.info("=" * 70)
    log.info("AI Forex Trading Bot v2 — 5-Agent Adversarial Architecture")
    log.info(f"Watch list: {', '.join(symbols_config)} | Primary TF: {config.TIMEFRAME} | HTF: {config.HTF_TIMEFRAME}")
    log.info(f"LLM: {config.LLM_PROVIDER} / {config.LLM_MODEL}")
    log.info(f"Demo mode: {demo_mode}")
    log.info("=" * 70)

    if not mt5b.connect():
        log.error("Cannot connect to MetaTrader 5. Is the terminal open and logged in?")
        sys.exit(1)

    # ── Resolve broker symbols (handles .r / .pro / .a suffixes) ─────────
    symbols_list = []
    for sym in symbols_config:
        resolved = mt5b.resolve_symbol(sym)
        symbols_list.append(resolved)
    # Keep config.SYMBOL pointing at the first symbol (used by governor + reflection fallback)
    config.SYMBOL = symbols_list[0]
    log.info(f"Resolved watch list: {symbols_list}")

    starting_equity  = mt5b.get_account_equity()
    starting_balance = mt5b.get_account_balance()
    log.info(f"Starting equity: {starting_equity:.2f} | Balance: {starting_balance:.2f}")

    # ── Load previous balance snapshot for OVERNIGHT gap detection only ───
    _EQUITY_SNAPSHOT_PATH = "observability/last_equity.json"
    prev_balance   = 0.0
    snapshot_date  = None
    try:
        if os.path.exists(_EQUITY_SNAPSHOT_PATH):
            with open(_EQUITY_SNAPSHOT_PATH, "r") as f:
                snap = json.load(f)
                prev_balance  = float(snap.get("balance", snap.get("equity", 0.0)))
                snapshot_date = snap.get("date")
    except Exception:
        pass

    # ── Initialise all subsystems ─────────────────────────────────────────
    llm.init(config)

    _tg = TelegramAlert(
        token   = getattr(config, "TELEGRAM_BOT_TOKEN", ""),
        chat_id = getattr(config, "TELEGRAM_CHAT_ID", ""),
        enabled = getattr(config, "TELEGRAM_ALERTS_ENABLED", False),
    )
    _daily_stats = {"trades": 0, "winners": 0, "total_pnl": 0.0}

    # Loss-streak tilt guard state (per symbol)
    _loss_streak: dict[str, int] = {}   # symbol -> consecutive losses
    _win_streak:  dict[str, int] = {}   # symbol -> consecutive wins after a streak

    # Daily report trigger
    _last_report_date = datetime.now(UTC).date()

    # Adaptive confidence threshold (walk-forward optimization)
    _dynamic_min_confidence: int = getattr(config, "MIN_CONFIDENCE_TO_TRADE", 50)

    memory   = TradeMemoryStore(
        config.MEMORY_DB_PATH,
        config.MEMORY_USEFULNESS_THRESHOLD,
        config.MEMORY_MAX_STORED,
    )
    governor = RiskGovernor(
        starting_equity      = starting_equity,
        daily_loss_cap_pct   = config.DAILY_LOSS_CAP_PCT,
        max_open_positions   = config.MAX_OPEN_POSITIONS,
        max_orders_per_day   = config.MAX_ORDERS_PER_DAY,
        max_drawdown_pct     = config.MAX_DRAWDOWN_PCT,
        allowed_sessions     = config.ALLOWED_SESSIONS,
        state_path           = getattr(config, "GOVERNOR_STATE_PATH", "observability/governor_state.json"),
    )
    if reset_drawdown_halt:
        governor.reset_drawdown_halt(starting_equity)
    calendar      = EconomicCalendar(
        blackout_before_min = config.CALENDAR_BLACKOUT_BEFORE_MIN,
        blackout_after_min  = config.CALENDAR_BLACKOUT_AFTER_MIN,
        csv_path            = config.CALENDAR_CSV_PATH,
        live_enabled        = getattr(config, "LIVE_CALENDAR_ENABLED", True),
    )
    pos_manager   = PositionManager(config)
    meta_gate     = MetaLabelGate()
    journal       = Journal(config.JOURNAL_DB_PATH)

    # CUSUM event filter — prevents re-running the 5-agent desk when price hasn't moved
    _cusum = CUSUMFilter(
        h_pct=getattr(config, "CUSUM_H_PCT", 0.004),
        max_bars_between=getattr(config, "CUSUM_MAX_BARS", 20),
        min_bars_between=getattr(config, "CUSUM_MIN_BARS", 3),
    ) if _CUSUM_AVAILABLE else None

    # HMM latent vol regime model — retrained daily alongside LightGBM
    _hmm_model    = HMMRegimeModel() if _HMM_AVAILABLE else None
    _hmm_fitted   = False
    _hmm_fit_date = None

    # ── Overnight gap check ───────────────────────────────────────────────
    today_str = datetime.now(UTC).strftime("%Y-%m-%d")
    if prev_balance > 0 and snapshot_date and snapshot_date != today_str:
        governor.check_startup_gap(prev_balance, starting_balance)
    elif prev_balance > 0 and not snapshot_date:
        governor.check_startup_gap(prev_balance, starting_balance)

    # Initial meta-label training from existing memories
    all_memories = memory.get_all_for_training()
    if all_memories:
        meta_gate.train(all_memories, training_lookback_days=getattr(config, "ML_TRAINING_LOOKBACK_DAYS", 180))

    memories_since_retrain = 0

    open_trades: dict[int, OpenTradeRecord] = {}
    tick_count = 0

    # Reconcile any positions that were open when the bot last stopped
    _existing = mt5b.get_all_open_positions(config.MAGIC_NUMBER)
    for _pos in _existing:
        _ticket = _pos["ticket"]
        if _ticket not in open_trades:
            open_trades[_ticket] = OpenTradeRecord(
                ticket        = _ticket,
                symbol        = _pos["symbol"],
                direction     = _pos["type"],
                entry_price   = _pos["price"],
                entry_context = {},
                tech_analysis = {},
                pm_decision   = {},
                entry_time    = datetime.now(UTC),  # best available; exact time unknown
                atr           = 0.0,
                sl            = _pos["sl"],
                tp            = _pos["tp"],
                volume        = _pos["volume"],
                risk_usd      = 0.0,
            )
    if _existing:
        log.info(f"Startup reconciliation: adopted {len(_existing)} orphaned position(s) from MT5.")

    log.info(f"Memory on start: {memory.stats()}")

    try:
        while True:
            tick_start = datetime.now(UTC)
            tick_count += 1
            log.info(f"\n{'─'*60}")
            log.info(f"Tick #{tick_count}: {tick_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")

            # ── Midnight daily report ─────────────────────────────────────
            today_date = tick_start.date()
            if (today_date != _last_report_date
                    and getattr(config, "DAILY_REPORT_ENABLED", False)
                    and _REPORTER_AVAILABLE):
                try:
                    generate_daily_report(
                        config.JOURNAL_DB_PATH, days=1,
                        send_telegram=_tg.info,
                    )
                except Exception as _rpt_err:
                    log.debug(f"Daily report error: {_rpt_err}")
                _last_report_date = today_date

            # ── 1. Connection health check + reconnect ────────────────────
            if not mt5b.is_connected():
                log.warning("MT5 connection lost — attempting reconnect...")
                if not mt5b.reconnect():
                    log.error("MT5 reconnect failed. Waiting before retry.")
                    _sleep(tick_start)
                    continue
                log.info("MT5 reconnected — resuming.")

            # ── 2. Account state (ALL symbols combined) ───────────────────
            equity       = mt5b.get_account_equity()
            all_positions = mt5b.get_all_open_positions(config.MAGIC_NUMBER)
            open_count   = len(all_positions)
            current_tickets = {p["ticket"] for p in all_positions}

            unrealised = sum(p["profit"] for p in all_positions)
            governor.record_mark_to_market(unrealised)
            governor.update_equity(equity)

            # Persist balance snapshot for overnight gap detection
            try:
                os.makedirs("observability", exist_ok=True)
                balance_now = mt5b.get_account_balance()
                with open(_EQUITY_SNAPSHOT_PATH, "w") as f:
                    json.dump({
                        "balance": balance_now,
                        "equity":  equity,
                        "date":    tick_start.strftime("%Y-%m-%d"),
                        "ts":      tick_start.isoformat(),
                    }, f)
            except Exception:
                pass

            log.info(
                f"Equity: {equity:.2f} | Open: {open_count} | "
                f"Unrealised: {unrealised:+.2f} | Governor: {governor.status}"
            )

            # ── 3. Detect closed trades → get real P&L → reflect ─────────
            # Runs BEFORE the kill-switch guard so trades closed by the kill switch
            # (or by SL/TP while it is active) still have their real P&L recorded
            # in the governor, journal and memory.
            closed_tickets = set(open_trades.keys()) - current_tickets
            for ticket in closed_tickets:
                record    = open_trades.pop(ticket)
                sym       = record.symbol
                hold_hours = (datetime.now(UTC) - record.entry_time).total_seconds() / 3600

                trade_history = mt5b.get_closed_trade(ticket)
                if trade_history:
                    exit_price = trade_history["exit_price"]
                    net_pnl    = trade_history["net_pnl"]
                    pnl_pct    = (net_pnl / starting_equity) * 100 if starting_equity else 0
                    log.info(
                        f"[{sym}] Trade {ticket} closed: exit={exit_price:.5f} | "
                        f"net_pnl={net_pnl:+.2f} | hold={hold_hours:.1f}h"
                    )
                else:
                    log.warning(f"[{sym}] Could not get MT5 history for ticket {ticket} — estimating P&L")
                    exit_price = record.entry_price
                    net_pnl    = 0.0
                    pnl_pct    = 0.0

                r_multiple = (net_pnl / record.risk_usd) if record.risk_usd > 0 else 0.0
                outcome    = classify_outcome(r_multiple)
                log.info(f"[{sym}] Outcome: {outcome} | P&L: {pnl_pct:+.2f}% | R={r_multiple:+.2f}")

                governor.record_trade_closed(net_pnl)
                _daily_stats["trades"]    += 1
                _daily_stats["total_pnl"] += net_pnl
                if net_pnl > 0:
                    _daily_stats["winners"] += 1

                # ── Agent accuracy outcome update ─────────────────────────
                is_win = net_pnl > 0
                if getattr(config, "AGENT_ACCURACY_ENABLED", False):
                    try:
                        journal.update_agent_outcome(ticket, is_win)
                    except Exception as _acc_err:
                        log.debug(f"Agent outcome update error: {_acc_err}")

                # ── Loss streak + blacklist tracking ──────────────────────
                _regime_at_entry = record.entry_context.get("regime", {})
                if isinstance(_regime_at_entry, dict):
                    _regime_at_entry = _regime_at_entry.get("regime", "")
                if is_win:
                    _win_streak[sym]  = _win_streak.get(sym, 0) + 1
                    _loss_streak[sym] = 0
                    if getattr(config, "SETUP_BLACKLIST_ENABLED", False):
                        journal.reset_blacklist(sym, _regime_at_entry)
                else:
                    _loss_streak[sym] = _loss_streak.get(sym, 0) + 1
                    _win_streak[sym]  = 0
                    if getattr(config, "SETUP_BLACKLIST_ENABLED", False):
                        _bl_count = journal.add_blacklist_loss(sym, _regime_at_entry)
                        _bl_thresh = getattr(config, "SETUP_BLACKLIST_THRESHOLD", 3)
                        if _bl_count >= _bl_thresh:
                            log.warning(
                                f"[{sym}] BLACKLISTED: {_bl_count} consecutive losses "
                                f"in {_regime_at_entry} regime — will skip next occurrence."
                            )
                _cur_streak = _loss_streak.get(sym, 0)
                if _cur_streak >= getattr(config, "LOSS_STREAK_THRESHOLD", 3):
                    log.warning(
                        f"[{sym}] Loss streak: {_cur_streak} consecutive — "
                        f"lot size halved for next trade on this symbol."
                    )

                _tg.trade_close(
                    symbol     = sym,
                    direction  = record.direction,
                    outcome    = outcome,
                    pnl        = net_pnl,
                    r_multiple = r_multiple,
                    hold_hours = hold_hours,
                )

                journal.log_trade_close(
                    ticket     = ticket,
                    exit_price = exit_price,
                    pnl        = net_pnl,
                    pnl_pct    = pnl_pct,
                    outcome    = outcome,
                    hold_hours = hold_hours,
                )

                lesson_record = reflection.reflect(
                    symbol             = sym,
                    direction          = record.direction,
                    entry_price        = record.entry_price,
                    exit_price         = exit_price,
                    pnl_pct            = pnl_pct,
                    r_multiple         = r_multiple,
                    hold_hours         = hold_hours,
                    outcome            = outcome,
                    entry_context      = record.entry_context,
                    technical_analysis = record.tech_analysis,
                    pm_decision        = record.pm_decision,
                    cfg                = config,
                )
                if lesson_record:
                    lesson_record["feature_vector"] = feature_vector(record.entry_context)
                    lesson_record["r_multiple"]     = r_multiple
                    _rc = record.entry_context.get("regime", {})
                    lesson_record["regime"] = _rc.get("regime", "") if isinstance(_rc, dict) else ""
                    # Store Tier 1 signal context for 24-feature ML training
                    lesson_record["fib_proximity"] = record.entry_context.get("fib_proximity", 0.0)
                    lesson_record["cot_z_score"]   = record.entry_context.get("cot_z_score", 0.0)
                    lesson_record["econ_surprise"]  = record.entry_context.get("econ_surprise", 0.0)
                    saved = memory.save(lesson_record)
                    if saved:
                        log.info(f"★ [{sym}] Lesson saved: {lesson_record['lesson'][:100]}")
                        # SGD online update — immediate single-trade adaptation
                        try:
                            _fv_ol = feature_vector(record.entry_context)
                            _rc_ol = record.entry_context.get("regime", {})
                            _re_ol = _rc_ol.get("regime", "") if isinstance(_rc_ol, dict) else ""
                            meta_gate.online_update(
                                feature_vector = _fv_ol,
                                is_win         = net_pnl > 0,
                                regime_str     = _re_ol,
                                fib_proximity  = record.entry_context.get("fib_proximity", 0.0),
                                cot_z_score    = record.entry_context.get("cot_z_score", 0.0),
                                econ_surprise  = record.entry_context.get("econ_surprise", 0.0),
                            )
                        except Exception as _ol_err:
                            log.debug(f"Online update error: {_ol_err}")
                        memories_since_retrain += 1
                        if memories_since_retrain >= config.META_RETRAIN_EVERY:
                            all_m = memory.get_all_for_training()
                            meta_gate.train(all_m, training_lookback_days=getattr(config, "ML_TRAINING_LOOKBACK_DAYS", 180))
                            memories_since_retrain = 0

                            # Walk-forward: adapt confidence threshold based on recent results
                            if (_ADAPTIVE_PARAMS_AVAILABLE
                                    and getattr(config, "ADAPTIVE_CONFIDENCE_ENABLED", False)):
                                _new_conf, _conf_reason = compute_adaptive_confidence(
                                    db_path           = config.JOURNAL_DB_PATH,
                                    lookback_trades   = getattr(config, "ADAPTIVE_CONFIDENCE_LOOKBACK", 30),
                                    current_threshold = _dynamic_min_confidence,
                                )
                                if _new_conf != _dynamic_min_confidence:
                                    log.info(f"AdaptiveConf: {_conf_reason}")
                                    _dynamic_min_confidence = _new_conf
                                else:
                                    log.debug(f"AdaptiveConf: {_conf_reason}")

            # ── Kill-switch guard ─────────────────────────────────────────
            gov_status = governor.status
            if gov_status["kill_active"] or gov_status["drawdown_halt"]:
                halt_reason = "kill switch" if gov_status["kill_active"] else "drawdown halt"
                if open_trades and not demo_mode:
                    log.warning(f"Governor {halt_reason} ACTIVE — force-closing all open positions.")
                    pos_manager.close_all(open_trades, config)
                log.info(f"Governor {halt_reason} — skipping agent desk and sleeping.")
                _sleep(tick_start)
                continue

            # ── 4. Active position management (trailing SL etc.) ─────────
            if open_trades and not demo_mode:
                pos_manager.manage(open_trades, equity)

            # ── Equity snapshot journal ───────────────────────────────────
            if tick_count % config.EQUITY_LOG_EVERY == 0:
                journal.log_equity(equity, governor.status["daily_pnl"], open_count)

            # ── Pre-scan: mine trade patterns (cached — cheap SQLite query) ──
            _trade_patterns: dict = {}
            if _PATTERN_MINER_AVAILABLE and getattr(config, "PATTERN_MINER_ENABLED", False):
                try:
                    _trade_patterns = mine_patterns(
                        config.JOURNAL_DB_PATH,
                        min_trades=getattr(config, "PATTERN_MINER_MIN_TRADES", 15),
                    )
                    if _trade_patterns.get("available") and tick_count % 20 == 1:
                        log.info(
                            f"Pattern miner: {_trade_patterns['total_trades']} trades | "
                            f"Overall win rate: {_trade_patterns['overall_rate']:.0%}"
                        )
                        for s in _trade_patterns.get("top_setups", [])[:3]:
                            log.info(f"  Top setup: {s}")
                except Exception as _pm_err:
                    log.debug(f"Pattern miner error: {_pm_err}")

            # ── Pre-scan: discovered alpha signals (cross-sectional panel) ──
            # One panel fetch per tick covers all symbols; {} = unavailable.
            _alpha_scores: dict = {}
            if _DISCOVERED_SIGNALS_AVAILABLE and getattr(config, "DISCOVERED_SIGNALS_ENABLED", False):
                try:
                    _alpha_scores = discovered_signal_scores(
                        symbols_list, mt5b, timeframe=config.TIMEFRAME,
                        ic_threshold=getattr(config, "DISCOVERED_SIGNALS_IC_THRESHOLD", 0.015),
                        p_threshold=getattr(config, "DISCOVERED_SIGNALS_P_THRESHOLD", 0.05),
                    )
                except Exception as _ds_err:
                    log.debug(f"Discovered signals error: {_ds_err}")

            # ── 5–13. Per-symbol scan ─────────────────────────────────────
            # Filter funnel — counts how many symbols each gate blocks this tick
            _funnel: dict[str, int] = {
                "scanned": 0, "data_err": 0, "regime": 0, "blacklist": 0,
                "spread": 0, "calendar": 0, "hold": 0, "confidence": 0,
                "timing": 0, "risk_veto": 0, "ml_gate": 0, "bear_veto": 0,
                "duplicate": 0, "correlation": 0, "governor": 0, "placed": 0,
            }

            for symbol in symbols_list:
                _funnel["scanned"] += 1

                try:
                    # ── 5. Fetch candles ──────────────────────────────────────
                    try:
                        df      = mt5b.get_candles(symbol, config.TIMEFRAME, count=250)
                        htf_df  = mt5b.get_candles(symbol, config.HTF_TIMEFRAME, count=100)
                        ctx     = compute(df, config, htf_df=htf_df, symbol=symbol)
                    except Exception as e:
                        log.warning(f"[{symbol}] Data error: {e}")
                        _funnel["data_err"] += 1
                        continue

                    m15_df = None
                    if getattr(config, "ENTRY_TIMING_ENABLED", False):
                        try:
                            m15_df = mt5b.get_candles(symbol, "M15", count=60)
                        except Exception:
                            pass

                    if not ctx:
                        log.warning(f"[{symbol}] Not enough candle data.")
                        continue

                    # Signals come from closed candles; level-proximity checks
                    # (S/R zones, pivots, key levels) use the live price, as they
                    # did when the forming candle's close was the "price".
                    try:
                        _lp = mt5b.get_live_price(symbol)
                    except Exception:
                        _lp = None
                    ctx["live_price"] = _lp or ctx["price"]

                    # ── 6. Regime classification ──────────────────────────────
                    regime_info = classify_regime(ctx["_df"], ctx["atr_percentile"])
                    ctx["regime"] = regime_info
                    log.info(
                        f"[{symbol}] Regime: {regime_info['regime']} | "
                        f"ADX={regime_info['adx']} | Hurst={regime_info['hurst']} | "
                        f"ER={regime_info.get('efficiency_ratio', '?')}"
                    )

                    allowed_regimes = getattr(config, "ALLOWED_REGIMES", [
                        "TRENDING_UP", "TRENDING_DOWN", "RANGING"
                    ])
                    if regime_info["regime"] not in allowed_regimes:
                        log.info(f"[{symbol}] Regime {regime_info['regime']} not in allowed list — skipping.")
                        _funnel["regime"] += 1
                        continue

                    # ── 6b. Symbol-specific parameters ────────────────────────
                    _sym_params   = {}
                    _sl_base_mult = config.ATR_SL_MULT
                    _tp_base_mult = config.ATR_TP_MULT
                    _bear_veto    = config.BEAR_RISK_VETO_THRESHOLD
                    if _SYMBOL_EXTRAS_AVAILABLE and getattr(config, "SYMBOL_PARAMS_ENABLED", False):
                        _sym_params   = get_symbol_params(symbol)
                        _sl_base_mult = _sym_params.get("ATR_SL_MULT", config.ATR_SL_MULT)
                        _tp_base_mult = _sym_params.get("ATR_TP_MULT", config.ATR_TP_MULT)
                        _bear_veto    = _sym_params.get("BEAR_RISK_VETO", config.BEAR_RISK_VETO_THRESHOLD)
                        log.debug(
                            f"[{symbol}] Symbol params: SL={_sl_base_mult}x ATR | "
                            f"TP={_tp_base_mult}x ATR | BearVeto={_bear_veto}"
                        )

                    # ── 6c. Setup blacklist check ──────────────────────────────
                    if getattr(config, "SETUP_BLACKLIST_ENABLED", False):
                        _bl_thresh = getattr(config, "SETUP_BLACKLIST_THRESHOLD", 3)
                        if journal.is_blacklisted(symbol, regime_info["regime"], _bl_thresh):
                            log.info(
                                f"[{symbol}] BLACKLISTED: {regime_info['regime']} regime has "
                                f"{_bl_thresh}+ consecutive losses. Skipping until next win."
                            )
                            _funnel["blacklist"] += 1
                            continue

                    # ── 6e. Spread filter ──────────────────────────────────────
                    if getattr(config, "SPREAD_FILTER_ENABLED", False):
                        spread = mt5b.get_spread_pips(symbol)
                        sym_key = symbol.upper().split(".")[0]
                        max_sp  = config.MAX_SPREAD_PIPS.get(sym_key,
                                  config.MAX_SPREAD_PIPS.get("DEFAULT", 5.0))
                        if spread > 0 and spread > max_sp:
                            log.info(f"[{symbol}] Spread too wide: {spread:.1f} pips > max {max_sp:.1f} — skipping.")
                            _funnel["spread"] += 1
                            continue
                        if spread > 0:
                            log.debug(f"[{symbol}] Spread OK: {spread:.1f}/{max_sp:.1f} pips")

                    # ── 6f. CUSUM filter (EARLY) — skip before all expensive operations ──
                    # Must run before embedding/SR/pivot/fingerprint to avoid wasting
                    # 2 NVIDIA API calls per symbol on CUSUM-blocked ticks.
                    _atr_pct = None
                    try:
                        _atr_val = ctx.get("atr")
                        if _atr_val and ctx.get("price"):
                            _atr_pct = float(_atr_val) / float(ctx["price"])
                    except Exception:
                        pass

                    if _cusum is not None and not getattr(config, "CUSUM_DISABLED", False):
                        if not _cusum.should_evaluate(symbol, float(ctx.get("price", 0)), _atr_pct):
                            _funnel["cusum"] = _funnel.get("cusum", 0) + 1
                            continue

                    # ── 7. Economic calendar blackout ─────────────────────────
                    is_blackout, blackout_reason = calendar.is_blackout()
                    if is_blackout:
                        log.info(f"[{symbol}] Calendar blackout: {blackout_reason} — skipping.")
                        _funnel["calendar"] += 1
                        continue

                    # ── 7b. COT Institutional Positioning ────────────────────
                    _cot_signal = {"signal": "NEUTRAL", "z_score": 0.0, "available": False}
                    if _MARKET_SIGNALS_AVAILABLE and getattr(config, "COT_ENABLED", False):
                        try:
                            _cot_signal = get_cot_signal(
                                symbol,
                                db_path=getattr(config, "COT_DB_PATH", "data/cot_history.db"),
                            )
                            if _cot_signal["available"]:
                                log.info(
                                    f"  COT [{symbol}]: {_cot_signal['signal']} | "
                                    f"net={_cot_signal['net_long']:+,} | z={_cot_signal['z_score']:+.2f}"
                                )
                        except Exception as _cot_err:
                            log.debug(f"COT error: {_cot_err}")

                    # ── 7c. Economic Surprise Index ───────────────────────────
                    _econ_signal = {"net_surprise": 0.0, "signal": "NEUTRAL"}
                    if _MARKET_SIGNALS_AVAILABLE and getattr(config, "ECON_SURPRISE_ENABLED", False):
                        try:
                            _econ_signal = get_symbol_surprise(
                                symbol,
                                db_path=getattr(config, "ECON_SURPRISE_DB_PATH", "data/econ_surprise.db"),
                            )
                            if _econ_signal["signal"] != "NEUTRAL":
                                log.info(
                                    f"  EconSurprise [{symbol}]: {_econ_signal['signal']} | "
                                    f"net={_econ_signal['net_surprise']:+.1f}"
                                )
                        except Exception as _es_err:
                            log.debug(f"EconSurprise error: {_es_err}")

                    # ── 7d. Multi-touch S/R Zones ─────────────────────────────
                    _sr_zones  = []
                    _sr_result = {"at_sr_zone": False}
                    if _SR_ZONES_AVAILABLE and getattr(config, "SR_ZONE_ENABLED", False):
                        try:
                            _sr_zones  = find_sr_zones(symbol, ctx["_df"])
                            _sr_result = check_sr_zone(symbol, ctx["live_price"], _sr_zones)
                            if _sr_result["at_sr_zone"]:
                                log.info(
                                    f"  SR Zone [{symbol}]: {_sr_result['zone_type']} @ "
                                    f"{_sr_result['zone_price']:.5f} | "
                                    f"touches={_sr_result['touches']} | "
                                    f"strength={_sr_result['zone_strength']:.0%}"
                                )
                        except Exception as _sr_err:
                            log.debug(f"SR zones error: {_sr_err}")

                    # ── 7e. Agent Accuracy Context ────────────────────────────
                    _agent_acc = None
                    if getattr(config, "AGENT_ACCURACY_ENABLED", False):
                        try:
                            _agent_acc = journal.get_agent_accuracy(
                                lookback=getattr(config, "AGENT_ACCURACY_LOOKBACK", 50)
                            )
                        except Exception:
                            pass

                    # ── 7f. Pivot Levels ──────────────────────────────────────
                    _pivot_result = {"at_pivot": False}
                    if _PIVOT_LEVELS_AVAILABLE and getattr(config, "PIVOT_LEVELS_ENABLED", False):
                        try:
                            _pivot_result = check_pivot_levels(symbol, ctx["live_price"], ctx["_df"])
                            if _pivot_result["at_pivot"]:
                                log.info(
                                    f"  Pivot [{symbol}]: {_pivot_result['pivot_type']} "
                                    f"({_pivot_result['pivot_source']}) @ "
                                    f"{_pivot_result['pivot_price']:.5f} | "
                                    f"{_pivot_result['pivot_signal']}"
                                )
                        except Exception as _pv_err:
                            log.debug(f"Pivot levels error: {_pv_err}")

                    # ── 8. Compute fingerprint + feature vector ───────────────
                    fp  = market_fingerprint(ctx)
                    fv  = feature_vector(ctx)
                    log.info(
                        f"[{symbol}] Fingerprint: {fp} | Structure: {ctx['market_structure']} | "
                        f"Patterns: {ctx.get('candlestick_patterns', [])}"
                    )

                    # ── 9. Recall similar past situations (balanced: wins + losses) ─
                    # Research (arXiv:2507.23211): balanced exemplars ~16% better than wins-only.
                    # recall_balanced() ensures ≥2 losing trades in the few-shot context.
                    memories = memory.recall_balanced(
                        fp, symbol, config.MEMORY_RECALL_COUNT, fv, min_losses=2
                    )

                    try:
                        from memory.semantic import embed_market_context
                        ctx_emb = embed_market_context({**ctx, "regime": regime_info})
                        if ctx_emb:
                            sem_memories = memory.semantic_recall(
                                query  = _ctx_to_query(ctx, regime_info),
                                symbol = symbol,
                                n      = 2,
                            )
                            seen_lessons = {m["lesson"] for m in memories}
                            for sm in sem_memories:
                                if sm["lesson"] not in seen_lessons:
                                    sm["_source"] = "semantic"
                                    memories.append(sm)
                    except Exception as _sem_err:
                        log.debug(f"Semantic recall skipped: {_sem_err}")

                    if memories:
                        log.info(f"[{symbol}] Recalled {len(memories)} memories:")
                        for m in memories:
                            src = " [semantic]" if m.get("_source") == "semantic" else ""
                            log.info(f"  [{m['outcome']}]{src} {m['lesson'][:70]}")
                    else:
                        log.info(f"[{symbol}] No matching memories — operating fresh.")

                    # ── 10a. CUSUM filter — skip if price hasn't moved enough ─
                    # Prevents LLM calls on noisy, information-free ticks.
                    # Fires when cumulative log-return exceeds h_pct threshold.
                    # ── 10b. Enrich regime_info with vol regime + HMM ──────────
                    # Vol regime (Yang-Zhang percentile rank) → QUIET/NORMAL/ELEVATED/EXTREME
                    if _VOL_EST_AVAILABLE:
                        try:
                            _vol_regime = vol_regime_from_bars(df)
                            regime_info["vol_regime"]   = _vol_regime["vol_regime_label"]
                            regime_info["yz_vol"]       = _vol_regime["yz_vol"]
                            regime_info["vol_pct_rank"] = _vol_regime["vol_pct_rank"]
                            log.debug(
                                f"[{symbol}] Vol regime: {_vol_regime['vol_regime_label']} "
                                f"(pct={_vol_regime['vol_pct_rank']:.0f})"
                            )
                        except Exception as _ve:
                            log.debug(f"Vol estimator error: {_ve}")

                    # HMM latent regime — retrain daily alongside LightGBM
                    if _hmm_model is not None:
                        _today_hmm = datetime.now(UTC).date()
                        if not _hmm_fitted or _hmm_fit_date != _today_hmm:
                            if len(df) >= 250:
                                try:
                                    _ok = _hmm_model.fit(df)
                                    if _ok:
                                        _hmm_fitted   = True
                                        _hmm_fit_date = _today_hmm
                                except Exception as _hmm_err:
                                    log.debug(f"HMM fit error: {_hmm_err}")

                        if _hmm_fitted:
                            try:
                                _hmm_out = _hmm_model.predict(df)
                                regime_info["hmm_label"]         = _hmm_out["label"]
                                regime_info["hmm_trans_to_high"] = _hmm_out["transition_to_high"]
                                # Tighten size when HMM says HIGH_VOL regime is imminent
                                if (_hmm_out["label"] == "HIGH_VOL"
                                        or _hmm_out["transition_to_high"] > 0.40):
                                    log.info(
                                        f"[{symbol}] HMM HIGH_VOL signal "
                                        f"(trans_to_high={_hmm_out['transition_to_high']:.2f}) "
                                        f"— will reduce position size"
                                    )
                            except Exception as _hmm_pe:
                                log.debug(f"HMM predict error: {_hmm_pe}")

                    # ── 10. Run 5-agent desk ──────────────────────────────────
                    log.info(f"[{symbol}] ─── Agent Desk ───")

                    log.info(f"[{symbol}] [1/5] Technical Analyst...")
                    tech = technical_analyst.analyse(ctx, config)
                    log.info(
                        f"  Bull: {tech.get('bias','?')} ({tech.get('bias_strength','?')}/10) | "
                        f"{tech.get('summary','')[:80]}"
                    )

                    # Prepare news inputs before parallel execution
                    next_event     = calendar.next_event()
                    upcoming_list  = [next_event] if next_event else []
                    live_headlines = fetch_headlines(symbol)
                    if live_headlines:
                        log.info(f"  Live headlines: {len(live_headlines)} | Top: {live_headlines[0][:80]}")
                    live_macro = {}
                    if _MACRO_AVAILABLE:
                        try:
                            live_macro = get_macro_context(symbol)
                            log.info(
                                f"  Live macro: VIX={live_macro.get('vix','?')} "
                                f"[{live_macro.get('vix_regime','?')}] | "
                                f"DXY={live_macro.get('dxy','?')} [{live_macro.get('dxy_trend','?')}] | "
                                f"F&G={live_macro.get('fear_greed_score','?')}"
                            )
                        except Exception as _me:
                            log.debug(f"Macro fetch skipped: {_me}")

                    log.info(f"[{symbol}] [2+3/5] Bear + News Analysts (parallel)...")
                    with ThreadPoolExecutor(max_workers=2) as _pool:
                        _bear_fut = _pool.submit(bear_analyst.analyse, ctx, tech, config)
                        _news_fut = _pool.submit(lambda: news_analyst.analyse(
                            symbol          = symbol,
                            sessions        = ctx.get("sessions", []),
                            upcoming_events = upcoming_list,
                            cfg             = config,
                            news_snippets   = live_headlines or None,
                            macro           = live_macro or None,
                        ))
                        try:
                            bear = _bear_fut.result()
                        except Exception as _be:
                            log.warning(f"[{symbol}] Bear analyst failed: {_be} — using safe default")
                            bear = {"bear_thesis": "Analysis unavailable.", "risk_score": 5,
                                    "evidence": [], "counter_to_bull": "Unknown", "regime_warning": ""}
                        try:
                            macro = _news_fut.result()
                        except Exception as _ne:
                            log.warning(f"[{symbol}] News analyst failed: {_ne} — using safe default")
                            macro = {"macro_tone": "NEUTRAL", "news_risk": 5, "key_themes": [],
                                     "session_bias": "NEUTRAL", "recommendation": "HOLD"}

                    log.info(
                        f"  Bear risk: {bear.get('risk_score','?')}/10 | "
                        f"{bear.get('bear_thesis','')[:80]}"
                    )
                    log.info(
                        f"  Macro: {macro.get('macro_tone','?')} | "
                        f"News risk: {macro.get('news_risk','?')} | "
                        f"DXY impact: {macro.get('dxy_impact','?')[:60]}"
                    )

                    log.info(f"[{symbol}] [4/5] Fund Manager (arbitrates bull vs bear)...")
                    _extra_signals_for_pm = {
                        "cot_signal":           _cot_signal.get("signal", "NEUTRAL"),
                        "econ_surprise_signal": _econ_signal.get("signal", "NEUTRAL"),
                        "fib_label":            "",   # filled below after key_level check
                        "sr_zone_type":         _sr_result.get("zone_type", ""),
                        "sr_zone_strength":     _sr_result.get("zone_strength", 0.0),
                    }
                    pm = portfolio_manager.decide(
                        technical_analysis = tech,
                        bear_analysis      = bear,
                        news_analysis      = macro,
                        past_memories      = memories,
                        ctx                = ctx,
                        regime             = regime_info,
                        cfg                = config,
                        agent_accuracy     = _agent_acc,
                        extra_signals      = _extra_signals_for_pm,
                    )
                    action     = pm.get("action", "HOLD")
                    confidence = pm.get("confidence", 0)
                    log.info(
                        f"  Decision: {action} ({confidence}%) | "
                        f"Verdict: {pm.get('bull_bear_verdict','?')} | "
                        f"{pm.get('reasoning','')[:100]}"
                    )

                    # ── Seasonality confidence adjustment ─────────────────────
                    if getattr(config, "SEASONALITY_ENABLED", False) and _SEASONAL_TIMING_AVAILABLE:
                        s_mult, s_reason = seasonality_adjust(symbol)
                        if s_mult != 1.0:
                            adj_conf = max(0, round(confidence * s_mult))
                            log.info(f"  Seasonality: {s_mult:.2f}x ({s_reason}) → {confidence}% → {adj_conf}%")
                            confidence = adj_conf
                            pm["confidence"] = adj_conf

                    # ── Key level confidence adjustment ────────────────────────
                    if (_SYMBOL_EXTRAS_AVAILABLE
                            and getattr(config, "KEY_LEVEL_CHECK_ENABLED", False)
                            and action in ("BUY", "SELL")):
                        try:
                            kl = check_key_levels(symbol, ctx["live_price"], ctx["_df"])
                            if kl["at_key_level"]:
                                aligned = (
                                    (action == "BUY"  and kl["signal_alignment"] == "SUPPORT") or
                                    (action == "SELL" and kl["signal_alignment"] == "RESISTANCE")
                                )
                                if aligned:
                                    _kl_boost = getattr(config, "KEY_LEVEL_CONFIDENCE_BOOST", 8)
                                    confidence = min(100, confidence + _kl_boost)
                                    pm["confidence"] = confidence
                                    log.info(
                                        f"  Key level ALIGNED: {kl['level_type']} @ "
                                        f"{kl['level_price']:.5f} ({kl['distance_pips']:.1f} pips) "
                                        f"[{kl['signal_alignment']}] +{_kl_boost}% → {confidence}%"
                                    )
                                else:
                                    _kl_pen = getattr(config, "KEY_LEVEL_COUNTER_PENALTY", 5)
                                    confidence = max(0, confidence - _kl_pen)
                                    pm["confidence"] = confidence
                                    log.info(
                                        f"  Key level COUNTER: {kl['level_type']} @ "
                                        f"{kl['level_price']:.5f} [{kl['signal_alignment']}] "
                                        f"-{_kl_pen}% → {confidence}%"
                                    )
                            else:
                                log.debug(
                                    f"  Key level: {kl['distance_pips']:.1f} pips from "
                                    f"nearest {kl['level_type'] or 'round level'}"
                                )

                            # Fibonacci boost (independent of round number — adds on top)
                            if kl.get("at_fib_level") and action in ("BUY", "SELL"):
                                is_major_fib = kl.get("fib_is_major", False)
                                _fib_boost   = getattr(config, "FIB_CONFIDENCE_BOOST", 4)
                                if is_major_fib:
                                    confidence = min(100, confidence + _fib_boost)
                                    pm["confidence"] = confidence
                                    log.info(
                                        f"  Fibonacci MAJOR {kl['fib_label']} @ "
                                        f"{kl['fib_price']:.5f} ({kl['fib_distance_pips']:.1f}p) "
                                        f"+{_fib_boost}% → {confidence}%"
                                    )
                                _extra_signals_for_pm["fib_label"] = kl.get("fib_label", "")

                            # Store Fib proximity in ctx for ML features (24th feature)
                            ctx["fib_proximity"] = kl.get("fib_proximity_score", 0.0)

                        except Exception as _kl_err:
                            log.debug(f"Key level check error: {_kl_err}")

                    # ── COT Confidence Adjustment ─────────────────────────────
                    if getattr(config, "COT_ENABLED", False) and _cot_signal.get("available"):
                        _cot_adj = getattr(config, "COT_CONFIDENCE_ADJ", 3)
                        cot_sig  = _cot_signal["signal"]
                        if action in ("BUY", "SELL"):
                            aligned = (
                                (action == "BUY"  and cot_sig == "BULLISH_INSTITUTIONAL") or
                                (action == "SELL" and cot_sig == "BEARISH_INSTITUTIONAL")
                            )
                            counter = (
                                (action == "BUY"  and cot_sig == "BEARISH_INSTITUTIONAL") or
                                (action == "SELL" and cot_sig == "BULLISH_INSTITUTIONAL")
                            )
                            if aligned:
                                confidence = min(100, confidence + _cot_adj)
                                pm["confidence"] = confidence
                                log.info(f"  COT aligned → +{_cot_adj}% → {confidence}%")
                            elif counter:
                                confidence = max(0, confidence - _cot_adj)
                                pm["confidence"] = confidence
                                log.info(f"  COT counter → -{_cot_adj}% → {confidence}%")

                    # ── Economic Surprise Confidence Adjustment ───────────────
                    if getattr(config, "ECON_SURPRISE_ENABLED", False) and action in ("BUY", "SELL"):
                        _es_adj  = getattr(config, "ECON_SURPRISE_CONF_ADJ", 3)
                        econ_sig = _econ_signal.get("signal", "NEUTRAL")
                        aligned  = (
                            (action == "BUY"  and econ_sig == "BASE_BULLISH") or
                            (action == "SELL" and econ_sig == "BASE_BEARISH")
                        )
                        counter  = (
                            (action == "BUY"  and econ_sig == "BASE_BEARISH") or
                            (action == "SELL" and econ_sig == "BASE_BULLISH")
                        )
                        if aligned:
                            confidence = min(100, confidence + _es_adj)
                            pm["confidence"] = confidence
                            log.info(f"  EconSurprise aligned → +{_es_adj}% → {confidence}%")
                        elif counter:
                            confidence = max(0, confidence - _es_adj)
                            pm["confidence"] = confidence
                            log.info(f"  EconSurprise counter → -{_es_adj}% → {confidence}%")

                    # ── SR Zone Confidence Adjustment ─────────────────────────
                    if (getattr(config, "SR_ZONE_ENABLED", False)
                            and _sr_result.get("at_sr_zone")
                            and action in ("BUY", "SELL")):
                        _sr_thresh = getattr(config, "SR_ZONE_MIN_STRENGTH", 0.4)
                        _sr_boost  = getattr(config, "SR_ZONE_CONFIDENCE_BOOST", 5)
                        if _sr_result.get("zone_strength", 0) >= _sr_thresh:
                            aligned = (
                                (action == "BUY"  and _sr_result["zone_type"] == "SUPPORT") or
                                (action == "SELL" and _sr_result["zone_type"] == "RESISTANCE")
                            )
                            if aligned:
                                confidence = min(100, confidence + _sr_boost)
                                pm["confidence"] = confidence
                                log.info(
                                    f"  SR Zone aligned ({_sr_result['touches']} touches) "
                                    f"→ +{_sr_boost}% → {confidence}%"
                                )

                    # ── Pivot Level Confidence Adjustment ────────────────────
                    if (getattr(config, "PIVOT_LEVELS_ENABLED", False)
                            and _PIVOT_LEVELS_AVAILABLE
                            and _pivot_result.get("at_pivot")
                            and action in ("BUY", "SELL")):
                        pv_signal = _pivot_result.get("pivot_signal", "NONE")
                        pv_type   = _pivot_result.get("pivot_type", "")
                        pv_src    = _pivot_result.get("pivot_source", "")
                        aligned   = (
                            (action == "BUY"  and pv_signal == "SUPPORT") or
                            (action == "SELL" and pv_signal == "RESISTANCE")
                        )
                        if aligned:
                            _pv_boost = getattr(config, "PIVOT_CONFIDENCE_BOOST", 4)
                            confidence = min(100, confidence + _pv_boost)
                            pm["confidence"] = confidence
                            log.info(
                                f"  Pivot ALIGNED: {pv_type} ({pv_src}) @ "
                                f"{_pivot_result['pivot_price']:.5f} "
                                f"+{_pv_boost}% → {confidence}%"
                            )
                        else:
                            _pv_pen = getattr(config, "PIVOT_COUNTER_PENALTY", 3)
                            confidence = max(0, confidence - _pv_pen)
                            pm["confidence"] = confidence
                            log.info(
                                f"  Pivot COUNTER: {pv_type} ({pv_src}) "
                                f"-{_pv_pen}% → {confidence}%"
                            )

                    # ── Inter-market Macro Bias Adjustment ───────────────────
                    if (getattr(config, "INTERMARKET_ENABLED", False)
                            and _INTERMARKET_AVAILABLE
                            and action in ("BUY", "SELL")
                            and live_macro):
                        try:
                            _im = get_intermarket_bias(symbol, action, live_macro)
                            _im_adj = _im.get("confidence_adj", 0)
                            if _im_adj != 0:
                                old_conf   = confidence
                                confidence = max(0, min(100, confidence + _im_adj))
                                pm["confidence"] = confidence
                                log.info(
                                    f"  InterMarket: {_im['bias']} → "
                                    f"{'+' if _im_adj > 0 else ''}{_im_adj}% → {confidence}% "
                                    f"({'; '.join(_im['reasons'][:2])})"
                                )
                        except Exception as _im_err:
                            log.debug(f"Intermarket error: {_im_err}")

                    # ── RSI / MACD Divergence Adjustment ─────────────────────
                    if getattr(config, "DIVERGENCE_ENABLED", False) and action in ("BUY", "SELL"):
                        _div_adj = getattr(config, "DIVERGENCE_CONF_ADJ", 4)
                        rsi_div  = ctx.get("rsi_divergence", "NONE")
                        macd_div = ctx.get("macd_divergence", "NONE")
                        # Confirmed: divergence aligns with trade direction
                        rsi_confirms  = (action == "BUY"  and rsi_div  == "BULLISH") or \
                                        (action == "SELL" and rsi_div  == "BEARISH")
                        rsi_opposes   = (action == "BUY"  and rsi_div  == "BEARISH") or \
                                        (action == "SELL" and rsi_div  == "BULLISH")
                        macd_confirms = (action == "BUY"  and macd_div == "BULLISH") or \
                                        (action == "SELL" and macd_div == "BEARISH")
                        macd_opposes  = (action == "BUY"  and macd_div == "BEARISH") or \
                                        (action == "SELL" and macd_div == "BULLISH")
                        if rsi_confirms or macd_confirms:
                            which = ("RSI+MACD" if (rsi_confirms and macd_confirms)
                                     else "RSI" if rsi_confirms else "MACD")
                            confidence = min(100, confidence + _div_adj)
                            pm["confidence"] = confidence
                            log.info(f"  {which} divergence CONFIRMS {action} +{_div_adj}% → {confidence}%")
                        elif rsi_opposes and macd_opposes:
                            confidence = max(0, confidence - _div_adj)
                            pm["confidence"] = confidence
                            log.info(f"  RSI+MACD divergence OPPOSES {action} -{_div_adj}% → {confidence}%")

                    # ── Stochastic Confirmation ───────────────────────────────
                    if getattr(config, "STOCH_ENABLED", False) and action in ("BUY", "SELL"):
                        _stoch_adj = getattr(config, "STOCH_CONF_ADJ", 3)
                        stoch_st   = ctx.get("stoch_state", "NEUTRAL")
                        stoch_xu   = ctx.get("stoch_cross_up", False)
                        stoch_xd   = ctx.get("stoch_cross_down", False)
                        if (action == "BUY"  and (stoch_st == "OVERSOLD"   or stoch_xu)):
                            confidence = min(100, confidence + _stoch_adj)
                            pm["confidence"] = confidence
                            log.info(f"  Stochastic OVERSOLD/cross-up confirms BUY +{_stoch_adj}% → {confidence}%")
                        elif (action == "SELL" and (stoch_st == "OVERBOUGHT" or stoch_xd)):
                            confidence = min(100, confidence + _stoch_adj)
                            pm["confidence"] = confidence
                            log.info(f"  Stochastic OVERBOUGHT/cross-down confirms SELL +{_stoch_adj}% → {confidence}%")

                    # ── Pattern Miner Setup Boost ─────────────────────────────
                    if (_PATTERN_MINER_AVAILABLE
                            and getattr(config, "PATTERN_MINER_ENABLED", False)
                            and _trade_patterns.get("available")
                            and action in ("BUY", "SELL")):
                        try:
                            _current_session = ",".join(ctx.get("sessions", ["?"])).split(",")[0]
                            _pm_boost = get_setup_boost(
                                symbol  = symbol,
                                regime  = regime_info["regime"],
                                session = _current_session,
                                patterns= _trade_patterns,
                                min_trades=getattr(config, "PATTERN_MINER_MIN_TRADES", 15),
                            )
                            if _pm_boost != 0:
                                confidence = max(0, min(100, confidence + _pm_boost))
                                pm["confidence"] = confidence
                                log.info(
                                    f"  Pattern miner: {symbol}|{regime_info['regime']}|{_current_session} "
                                    f"{'▲' if _pm_boost > 0 else '▼'}{_pm_boost:+d}% → {confidence}%"
                                )
                        except Exception as _pm_err:
                            log.debug(f"Pattern miner boost error: {_pm_err}")

                    # ── Discovered Alpha Signal Adjustment ────────────────────
                    # Cross-sectional z-score from signals found and IC-validated
                    # by the signal-discovery workflow on this broker's own data.
                    # Positive z = expected to outperform the panel (BUY tilt).
                    if _alpha_scores and action in ("BUY", "SELL"):
                        _az = _alpha_scores.get(symbol)
                        if _az is not None:
                            ctx["alpha_z"] = _az
                            _az_min = getattr(config, "DISCOVERED_SIGNALS_MIN_ABS_Z", 0.5)
                            _az_adj = getattr(config, "DISCOVERED_SIGNALS_CONF_ADJ", 4)
                            if abs(_az) >= _az_min:
                                _az_aligned = (action == "BUY" and _az > 0) or \
                                              (action == "SELL" and _az < 0)
                                if _az_aligned:
                                    confidence = min(100, confidence + _az_adj)
                                    pm["confidence"] = confidence
                                    log.info(f"  Alpha signal ALIGNED (z={_az:+.2f}) +{_az_adj}% → {confidence}%")
                                else:
                                    confidence = max(0, confidence - _az_adj)
                                    pm["confidence"] = confidence
                                    log.info(f"  Alpha signal COUNTER (z={_az:+.2f}) -{_az_adj}% → {confidence}%")

                    # Store COT/econ signals in ctx for ML feature vector
                    ctx["cot_z_score"]   = _cot_signal.get("z_score", 0.0)
                    ctx["econ_surprise"] = _econ_signal.get("net_surprise", 0.0)

                    log.info(f"[{symbol}] [5/5] Risk Manager...")
                    risk_assessment = risk_mgr_agent.assess(
                        ctx            = ctx,
                        tech_analysis  = tech,
                        bear_analysis  = bear,
                        news_analysis  = macro,
                        pm_decision    = pm,
                        regime         = regime_info,
                        cfg            = config,
                    )
                    log.info(
                        f"  Qualified: {risk_assessment.get('trade_qualified')} | "
                        f"Size: {risk_assessment.get('size_multiplier',1.0):.0%} | "
                        f"Veto: {risk_assessment.get('veto_reason','none')}"
                    )

                    # ── 11. Meta-label gate (24 features) ────────────────────
                    ml_result = meta_gate.predict(
                        fv, ctx,
                        fib_proximity = ctx.get("fib_proximity", 0.0),
                        cot_z_score   = ctx.get("cot_z_score", 0.0),
                        econ_surprise = ctx.get("econ_surprise", 0.0),
                    )
                    ml_prob   = ml_result["probability"]
                    ml_pass   = meta_gate.should_trade(ml_prob, config.META_LABEL_MIN_PROB)
                    log.info(
                        f"  Meta-label: prob={ml_prob:.3f} | "
                        f"{'ML' if ml_result['use_ml'] else 'heuristic'} | "
                        f"{'PASS' if ml_pass else 'FAIL'}"
                    )

                    # ── 12. Journal tick ──────────────────────────────────────
                    journal.log_tick(
                        symbol            = symbol,
                        timeframe         = config.TIMEFRAME,
                        price             = ctx["price"],
                        regime            = regime_info["regime"],
                        tech_bias         = tech.get("bias", "?"),
                        pm_action         = action,
                        pm_confidence     = confidence,
                        governor_approved = False,
                        meta_label_prob   = ml_prob,
                        session           = ",".join(ctx.get("sessions", [])),
                        fingerprint       = fp,
                    )

                    # ── 13. Execution gates ───────────────────────────────────
                    if action not in ("BUY", "SELL"):
                        log.info(f"[{symbol}] Fund Manager says HOLD — no trade.")
                        _funnel["hold"] += 1
                        continue

                    min_conf = _dynamic_min_confidence  # adaptive walk-forward threshold
                    if confidence < min_conf:
                        log.info(
                            f"[{symbol}] Confidence too low ({confidence}% < {min_conf}%) — skipping."
                        )
                        _funnel["confidence"] += 1
                        continue

                    # ── M15 entry timing ──────────────────────────────────────
                    if getattr(config, "ENTRY_TIMING_ENABLED", False) and _SEASONAL_TIMING_AVAILABLE and m15_df is not None:
                        timing = check_entry_timing(m15_df, action, ctx.get("atr", 0.001))
                        log.info(f"  M15 timing: {timing['decision']} (quality={timing['quality']:.2f}) | {timing['reason']}")
                        if timing["decision"] == "WAIT":
                            log.info(f"[{symbol}] M15 says WAIT — skipping this tick, re-evaluates next.")
                            _funnel["timing"] += 1
                            continue

                    if not risk_assessment.get("trade_qualified", False):
                        log.info(f"[{symbol}] Risk Manager VETOED: {risk_assessment.get('veto_reason')}")
                        _funnel["risk_veto"] += 1
                        continue

                    if not ml_pass:
                        log.info(
                            f"[{symbol}] Meta-label gate BLOCKED (prob={ml_prob:.3f} < {config.META_LABEL_MIN_PROB})"
                        )
                        _funnel["ml_gate"] += 1
                        continue

                    bear_risk = bear.get("risk_score", 0)
                    if bear_risk >= _bear_veto:
                        log.info(
                            f"[{symbol}] Bear analyst veto: risk score {bear_risk}/10 "
                            f">= threshold {_bear_veto} (symbol-specific)"
                        )
                        _funnel["bear_veto"] += 1
                        continue

                    # Don't stack same-direction positions on the SAME symbol
                    already_same = any(
                        t.direction == action and t.symbol == symbol
                        for t in open_trades.values()
                    )
                    if already_same:
                        log.info(f"[{symbol}] Already have a {action} position — skipping.")
                        _funnel["duplicate"] += 1
                        continue

                    # Correlation guard — block when a highly correlated pair already has
                    # an open position in the same direction (avoids double exposure).
                    # Research-verified correlations > 0.85 from BIS/FXNX 2026 data.
                    corr_blocked = _check_correlation(symbol, action, open_trades)
                    if corr_blocked:
                        log.info(
                            f"[{symbol}] Correlation guard BLOCKED: {corr_blocked} already open "
                            f"{action} — same economic exposure."
                        )
                        _funnel["correlation"] += 1
                        continue

                    # Risk Governor hard limits (global position count)
                    sessions_now = ctx.get("sessions", [])
                    allowed, reason = governor.approve(
                        current_equity      = equity,
                        open_position_count = open_count,
                        sessions            = sessions_now,
                        is_blackout         = False,
                    )
                    if not allowed:
                        log.info(f"[{symbol}] Governor BLOCKED: {reason}")
                        _funnel["governor"] += 1
                        # If we're at the max-positions limit, no point scanning more symbols
                        if open_count >= config.MAX_OPEN_POSITIONS:
                            break
                        continue

                    # ── 14. Place order ───────────────────────────────────────
                    import MetaTrader5 as mt5
                    sym_info = mt5.symbol_info(symbol)
                    point    = sym_info.point if sym_info else 0.00001

                    # Adaptive ATR stop-loss — widens in high-vol, tightens in low-vol
                    # Uses symbol-specific base multipliers from _sl_base_mult / _tp_base_mult.
                    _atr_pct = ctx.get("atr_percentile", 50)
                    if getattr(config, "ADAPTIVE_ATR_SL_ENABLED", False):
                        if _atr_pct < getattr(config, "ATR_SL_LOW_VOL_PCT", 30):
                            _sl_mult = getattr(config, "ATR_SL_LOW_VOL_MULT", 1.2)
                            log.info(f"  ATR low-vol ({_atr_pct:.0f}th pct) — tighter SL: {_sl_mult}x ATR")
                        elif _atr_pct > getattr(config, "ATR_SL_HIGH_VOL_PCT", 70):
                            _sl_mult = getattr(config, "ATR_SL_HIGH_VOL_MULT", 2.0)
                            log.info(f"  ATR high-vol ({_atr_pct:.0f}th pct) — wider SL: {_sl_mult}x ATR")
                        else:
                            _sl_mult = _sl_base_mult  # symbol-specific normal-vol multiplier
                    else:
                        _sl_mult = _sl_base_mult

                    sl_pips  = round(ctx["atr"] / point * _sl_mult, 1)
                    tp_pips  = round(ctx["atr"] / point * _tp_base_mult, 1)

                    size_mult = float(risk_assessment.get("size_multiplier", 1.0))
                    use_kelly = getattr(config, "USE_KELLY_SIZING", True)
                    if use_kelly and ml_result.get("use_ml"):
                        ml_p     = ml_result["probability"]
                        # Payoff must match the ACTUAL trade geometry (symbol-specific
                        # TP mult / adaptive SL mult), not the global config defaults.
                        payoff_b = _tp_base_mult / _sl_mult if _sl_mult > 0 else config.ATR_TP_MULT / config.ATR_SL_MULT
                        kelly_f  = (ml_p * payoff_b - (1.0 - ml_p)) / payoff_b
                        kelly_f  = max(0.0, kelly_f)
                        frac_k   = kelly_f * getattr(config, "KELLY_FRACTION", 0.25)
                        size_mult = min(size_mult, frac_k) if frac_k > 0 else size_mult
                        log.info(
                            f"¼-Kelly: p={ml_p:.3f}, b={payoff_b:.2f} → kelly={kelly_f:.3f} → "
                            f"size={size_mult:.2f} (fraction={frac_k:.3f})"
                        )

                    # ── HMM HIGH_VOL size reduction ──────────────────────────
                    # When the latent vol regime is HIGH_VOL or has high transition
                    # probability to HIGH_VOL, reduce lot size by 30%.
                    _hmm_lbl     = regime_info.get("hmm_label", "")
                    _hmm_trans   = regime_info.get("hmm_trans_to_high", 0.0)
                    if (_hmm_model is not None and _hmm_fitted
                            and (_hmm_lbl == "HIGH_VOL" or _hmm_trans > 0.40)):
                        _hmm_size_mult = 0.70   # 30% reduction in stressed vol regime
                        old_sm = size_mult
                        size_mult = size_mult * _hmm_size_mult
                        log.info(
                            f"  HMM size reduction: label={_hmm_lbl}, trans_to_high={_hmm_trans:.2f} "
                            f"→ size {old_sm:.2f} → {size_mult:.2f}"
                        )

                    # ── Loss-streak tilt guard ────────────────────────────────
                    if getattr(config, "LOSS_STREAK_ENABLED", False):
                        _sym_streak = _loss_streak.get(symbol, 0)
                        _streak_th  = getattr(config, "LOSS_STREAK_THRESHOLD", 3)
                        if _sym_streak >= _streak_th:
                            _streak_mult = getattr(config, "LOSS_STREAK_SIZE_MULT", 0.5)
                            old_sm = size_mult
                            size_mult = size_mult * _streak_mult
                            log.info(
                                f"  Tilt guard: {_sym_streak} consecutive losses → "
                                f"lot halved {old_sm:.2f}x → {size_mult:.2f}x"
                            )

                    # ── Volatility-adjusted position sizing (Tier 2) ──────────
                    # When ATR is high, SL is wider → each lot risks more USD.
                    # Scale lot down to keep dollar risk roughly constant.
                    if getattr(config, "VOL_ADJUSTED_SIZING", False):
                        _atr_p    = ctx.get("atr_percentile", 50)
                        _ext_pct  = getattr(config, "VOL_SIZING_EXTREME_ATR_PCT", 92)
                        _high_pct = getattr(config, "VOL_SIZING_HIGH_ATR_PCT", 80)
                        _ext_m    = getattr(config, "VOL_SIZING_EXTREME_ATR_MULT", 0.50)
                        _high_m   = getattr(config, "VOL_SIZING_HIGH_ATR_MULT", 0.70)
                        if _atr_p >= _ext_pct:
                            old_sm    = size_mult
                            size_mult = size_mult * _ext_m
                            log.info(
                                f"  Vol-adjust: ATR at {_atr_p:.0f}th pct (EXTREME) → "
                                f"{old_sm:.2f}x → {size_mult:.2f}x"
                            )
                        elif _atr_p >= _high_pct:
                            old_sm    = size_mult
                            size_mult = size_mult * _high_m
                            log.info(
                                f"  Vol-adjust: ATR at {_atr_p:.0f}th pct (HIGH) → "
                                f"{old_sm:.2f}x → {size_mult:.2f}x"
                            )

                    adjusted_lot = round(config.LOT_SIZE * size_mult, 2)
                    adjusted_lot = max(0.01, adjusted_lot)

                    log.info(
                        f"[{symbol}] Placing {action} | SL={sl_pips:.0f}pts | TP={tp_pips:.0f}pts | "
                        f"Lot={adjusted_lot} (base={config.LOT_SIZE} × {size_mult:.2f})"
                    )

                    if demo_mode:
                        log.info(f"[DEMO] Would {action} {adjusted_lot} {symbol}")
                    else:
                        result = mt5b.place_order(
                            symbol    = symbol,
                            direction = action,
                            lot_size  = adjusted_lot,
                            sl_pips   = sl_pips,
                            tp_pips   = tp_pips,
                            magic     = config.MAGIC_NUMBER,
                            comment   = f"AIBot|{confidence}%|{regime_info['regime'][:3]}",
                        )
                        if result:
                            risk_usd = mt5b.get_risk_usd(
                                symbol, result["price"], result["sl"], adjusted_lot
                            )
                            governor.record_order_placed()
                            open_count += 1  # keep local count current for this tick
                            open_trades[result["ticket"]] = OpenTradeRecord(
                                ticket        = result["ticket"],
                                symbol        = symbol,
                                direction     = action,
                                entry_price   = result["price"],
                                entry_context = {k: v for k, v in ctx.items() if k != "_df"},
                                tech_analysis = tech,
                                pm_decision   = pm,
                                entry_time    = datetime.now(UTC),
                                atr           = ctx["atr"],
                                sl            = result["sl"],
                                tp            = result["tp"],
                                volume        = adjusted_lot,
                                risk_usd      = risk_usd,
                            )
                            journal.log_trade_open(
                                ticket          = result["ticket"],
                                symbol          = symbol,
                                direction       = action,
                                entry_price     = result["price"],
                                sl              = result["sl"],
                                tp              = result["tp"],
                                volume          = adjusted_lot,
                                confidence      = confidence,
                                regime          = regime_info["regime"],
                                fingerprint     = fp,
                                size_multiplier = size_mult,
                            )
                            _tg.trade_open(
                                symbol      = symbol,
                                direction   = action,
                                entry_price = result["price"],
                                sl          = result["sl"],
                                tp          = result["tp"],
                                lot         = adjusted_lot,
                                confidence  = confidence,
                                regime      = regime_info["regime"],
                                reasoning   = pm.get("reasoning", "")[:120],
                            )
                            log.info(f"[{symbol}] Order placed: ticket={result['ticket']}")
                            _funnel["placed"] += 1
                            # Reset CUSUM state so next signal requires a fresh price move
                            if _cusum is not None:
                                _cusum.reset_symbol(symbol)

                            # Log agent votes for accuracy tracking
                            if getattr(config, "AGENT_ACCURACY_ENABLED", False):
                                try:
                                    journal.log_agent_votes(
                                        ticket        = result["ticket"],
                                        tech_bias     = tech.get("bias", ""),
                                        tech_strength = int(tech.get("bias_strength", 5)),
                                        bear_risk     = int(bear.get("risk_score", 5)),
                                        pm_action     = action,
                                        pm_confidence = confidence,
                                        regime        = regime_info["regime"],
                                    )
                                except Exception as _av_err:
                                    log.debug(f"Agent vote log error: {_av_err}")

                            # Stop scanning more symbols once we've hit max open positions
                            if open_count >= config.MAX_OPEN_POSITIONS:
                                log.info(f"Max positions ({open_count}) reached — stopping symbol scan.")
                                break

                except Exception as _sym_err:
                    # Error boundary: one symbol failing must never kill the whole
                    # trading loop — log it, count it, move to the next symbol.
                    _funnel["error"] = _funnel.get("error", 0) + 1
                    log.error(
                        f"[{symbol}] Unhandled error in symbol scan — skipping this symbol: {_sym_err}",
                        exc_info=True,
                    )
                    continue
            # ── End of per-symbol scan ────────────────────────────────────
            # Log filter funnel — tells you exactly what blocked each symbol
            _active = {k: v for k, v in _funnel.items() if v > 0}
            if _active:
                _gate_summary = " | ".join(f"{k}={v}" for k, v in _active.items())
                log.info(f"Tick funnel [{_funnel['scanned']} scanned]: {_gate_summary}")
            log.info(f"Memory: {memory.stats()}")
            _sleep(tick_start)

    except KeyboardInterrupt:
        log.info("Stopped by user (Ctrl+C).")
    finally:
        mt5b.disconnect()
        journal.print_summary()
        try:
            _tg.daily_summary(
                trades           = _daily_stats["trades"],
                winners          = _daily_stats["winners"],
                total_pnl        = _daily_stats["total_pnl"],
                equity           = 0.0,
                max_drawdown_pct = governor.status.get("drawdown_pct", 0.0),
            )
        except Exception:
            pass
        if _REPORTER_AVAILABLE and getattr(config, "DAILY_REPORT_ENABLED", False):
            try:
                generate_daily_report(
                    config.JOURNAL_DB_PATH, days=1,
                    send_telegram=_tg.info,
                )
            except Exception:
                pass
        log.info("Bot shut down. Final memory: " + str(memory.stats()))


# Pairs with correlation > 0.85 — verified: BIS 2022, FXNX 2026, Offbeat Forex
# Format: { symbol: [correlated_symbols_with_same_direction_risk] }
_CORR_GROUPS: dict[str, list[str]] = {
    # USD-positive pairs (all move together when USD strengthens)
    "USDCHF": ["USDCAD", "USDJPY"],
    "USDCAD": ["USDCHF", "USDJPY"],
    "USDJPY": ["USDCHF", "USDCAD"],
    # USD-negative pairs (EURUSD and GBPUSD are +0.85 correlated)
    "EURUSD": ["GBPUSD"],
    "GBPUSD": ["EURUSD"],
    # Commodity currencies (+0.90 correlated — never trade same direction)
    "AUDUSD": ["NZDUSD"],
    "NZDUSD": ["AUDUSD"],
    # Metals (+0.80 correlated)
    "XAUUSD": ["XAGUSD"],
    "XAGUSD": ["XAUUSD"],
    # Energy (+0.95 — treat as one instrument)
    "USOIL":  ["UKOIL"],
    "UKOIL":  ["USOIL"],
    # US Equity Indices (+0.85–0.90)
    "NAS100": ["SPX500", "US30"],
    "SPX500": ["NAS100", "US30"],
    "US30":   ["NAS100", "SPX500"],
    # Crypto (+0.88)
    "BTCUSD": ["ETHUSD"],
    "ETHUSD": ["BTCUSD"],
    # AUD/NZD JPY crosses (same commodity-currency driver)
    "AUDJPY": ["NZDJPY"],
    "NZDJPY": ["AUDJPY"],
}


def _check_correlation(symbol: str, action: str, open_trades: dict) -> str | None:
    """
    Return the correlated open symbol if adding `symbol` in `action` direction
    would create duplicate economic exposure. Return None if clear.
    """
    # Strip broker suffix (.r / .pro / .a) — same normalisation the spread filter
    # uses. Without this the guard silently never matches on suffixed brokers.
    corr_peers = _CORR_GROUPS.get(symbol.upper().split(".")[0], [])
    for peer in corr_peers:
        for trade in open_trades.values():
            if trade.symbol.upper().startswith(peer.upper()) and trade.direction == action:
                return trade.symbol
    return None


def _ctx_to_query(ctx: dict, regime_info: dict) -> str:
    """Build a natural-language query string from current market context for semantic recall."""
    parts = [
        f"regime {regime_info.get('regime', '')}",
        f"structure {ctx.get('market_structure', '')}",
    ]
    if ctx.get("candlestick_patterns"):
        parts.append(f"patterns {' '.join(ctx['candlestick_patterns'])}")
    if ctx.get("ema_bias"):
        parts.append(f"bias {ctx['ema_bias']}")
    if ctx.get("rsi") is not None:
        parts.append(f"RSI {ctx['rsi']:.0f}")
    return " | ".join(p for p in parts if p.split()[-1])


def _sleep(tick_start: datetime):
    elapsed = (datetime.now(UTC) - tick_start).total_seconds()
    sleep_s = max(5, config.CHECK_INTERVAL_SECONDS - elapsed)
    log.info(f"Next tick in {sleep_s:.0f}s")
    time.sleep(sleep_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Log signals only, no real trades")
    parser.add_argument("--reset-drawdown-halt", action="store_true",
                        help="Clear a saved drawdown halt and re-anchor peak equity (manual decision)")
    args = parser.parse_args()
    _reset_halt_once = args.reset_drawdown_halt

    # Supervisor: if the main loop dies on an unhandled error, restart instead of
    # exiting silently. Open positions are re-adopted by startup reconciliation.
    _RESTART_DELAY_S = 60
    while True:
        try:
            _reset_now, _reset_halt_once = _reset_halt_once, False   # never on auto-restart
            run(demo_mode=args.demo, reset_drawdown_halt=_reset_now)
            break  # clean exit (Ctrl+C is handled inside run())
        except KeyboardInterrupt:
            break
        except Exception:
            log.critical(
                f"Bot crashed with unhandled error — restarting in {_RESTART_DELAY_S}s "
                f"(open positions will be re-adopted on restart)",
                exc_info=True,
            )
            time.sleep(_RESTART_DELAY_S)
