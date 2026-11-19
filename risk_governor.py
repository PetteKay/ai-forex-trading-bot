"""
Risk Governor — the safety layer between the bot and the broker.

Rules enforced before any order reaches MT5:
  1. Kill switch: daily loss cap (DAILY_LOSS_CAP_PCT)
  2. Max open positions
  3. Max orders per day
  4. Account-level drawdown limit (stops trading if peak equity drops > X%)
  5. Session filter: only trade during approved sessions
  6. News blackout: calendar-based blackout window

All money tracked in Decimal (BCMath-equivalent precision — never floats for money).
Design formally verified with TLA+ (specs/RiskGovernor.tla).
"""

from decimal import Decimal
from datetime import date, datetime, timezone
from typing import Optional
import json
import logging
import os

log = logging.getLogger("RiskGovernor")

# Sessions where trading is allowed (empty set = always allowed)
# Override via config.ALLOWED_SESSIONS
ALWAYS_ALLOWED_SESSIONS = set()


def _persists(method):
    """Save governor state to disk after the wrapped call (no-op without state_path)."""
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            self._save_state()
    wrapper.__name__ = method.__name__
    wrapper.__doc__  = method.__doc__
    return wrapper


class RiskGovernor:
    def __init__(
        self,
        starting_equity: float,
        daily_loss_cap_pct: float,
        max_open_positions: int,
        max_orders_per_day: int,
        max_drawdown_pct: float = 10.0,
        allowed_sessions: Optional[list] = None,
        state_path: Optional[str] = None,
        account_id: Optional[str] = None,
    ):
        """
        state_path: JSON file where safety state is saved after every change and
        restored on startup. Without it, a restart forgets today's losses, the
        kill switch and the drawdown halt (verified bug). None = in-memory only
        (used by tests and backtests).

        account_id: the MT5 account login. Each account gets its own state file
        (e.g. governor_state_12345.json), so switching accounts — demo to live —
        never applies one account's drawdown halt or peak equity to another.
        """
        self._starting_equity    = Decimal(str(starting_equity))
        self._peak_equity        = Decimal(str(starting_equity))
        self._daily_loss_cap_pct = Decimal(str(daily_loss_cap_pct))
        self._max_drawdown_pct   = Decimal(str(max_drawdown_pct))
        self._max_open_pos       = max_open_positions
        self._max_orders_day     = max_orders_per_day
        self._allowed_sessions   = set(allowed_sessions) if allowed_sessions else ALWAYS_ALLOWED_SESSIONS

        self._kill_active        = False
        self._drawdown_halt      = False
        self._orders_today       = 0
        self._daily_pnl          = Decimal("0")
        self._open_positions     = 0
        self._today              = datetime.now(timezone.utc).date()   # UTC
        self._current_equity     = Decimal(str(starting_equity))       # updated every tick

        self._account_id         = str(account_id) if account_id not in (None, "") else None
        if state_path and self._account_id:
            _root, _ext = os.path.splitext(state_path)
            state_path = f"{_root}_{self._account_id}{_ext or '.json'}"
        self._state_path         = state_path
        self._last_saved: Optional[dict] = None
        if state_path:
            self._load_state()
            self._save_state()

    # ── Public API ────────────────────────────────────────────────────────────

    @_persists
    def approve(
        self,
        current_equity: float,
        open_position_count: int,
        sessions: Optional[list] = None,
        is_blackout: bool = False,
    ) -> tuple[bool, str]:
        """
        Returns (True, "") if the next order is approved.
        Returns (False, reason) if any rule blocks it.
        """
        self._maybe_day_reset()
        self._open_positions = open_position_count
        eq = Decimal(str(current_equity))

        # Update peak equity
        if eq > self._peak_equity:
            self._peak_equity = eq

        # ── Economic calendar blackout ────────────────────────────────────
        if is_blackout:
            return False, "Economic calendar blackout — high-impact event nearby."

        # ── Session filter ────────────────────────────────────────────────
        if self._allowed_sessions and sessions:
            active = set(sessions)
            if not active.intersection(self._allowed_sessions):
                return False, (
                    f"Session filter: current sessions {sessions} not in "
                    f"allowed {list(self._allowed_sessions)}."
                )

        # ── Kill switch (daily loss) ──────────────────────────────────────
        if self._kill_active:
            return False, (
                f"Kill switch active — daily loss cap hit "
                f"({self._daily_loss_cap_pct}% of equity). Resets tomorrow."
            )

        daily_loss_cap = self._starting_equity * self._daily_loss_cap_pct / Decimal("100")
        if self._daily_pnl <= -daily_loss_cap:
            self._kill_active = True
            log.warning("KILL SWITCH ACTIVATED — daily loss cap reached.")
            return False, "Kill switch just activated — daily loss cap reached."

        # ── Account drawdown halt ─────────────────────────────────────────
        if self._drawdown_halt:
            return False, "Drawdown halt active — account equity below max drawdown threshold."

        drawdown_cap = self._peak_equity * self._max_drawdown_pct / Decimal("100")
        drawdown     = self._peak_equity - eq
        if drawdown >= drawdown_cap:
            self._drawdown_halt = True
            log.warning(f"DRAWDOWN HALT: peak={self._peak_equity:.2f}, current={eq:.2f}, "
                        f"drawdown={drawdown:.2f} >= cap={drawdown_cap:.2f}")
            return False, (
                f"Drawdown halt: account down {float(drawdown/self._peak_equity*100):.1f}% "
                f"from peak — trading suspended."
            )

        # ── Position count ────────────────────────────────────────────────
        if self._open_positions >= self._max_open_pos:
            return False, f"Max open positions reached ({self._max_open_pos})."

        # ── Order frequency ───────────────────────────────────────────────
        if self._orders_today >= self._max_orders_day:
            return False, f"Max orders/day reached ({self._max_orders_day})."

        return True, ""

    @_persists
    def record_order_placed(self):
        self._orders_today   += 1
        self._open_positions += 1
        log.info(f"Order recorded. Today: {self._orders_today} orders, {self._open_positions} open.")

    @_persists
    def record_trade_closed(self, pnl_amount: float):
        """pnl_amount = realised net P&L (positive for profit, negative for loss)."""
        pnl = Decimal(str(pnl_amount))
        self._daily_pnl      += pnl
        self._open_positions  = max(0, self._open_positions - 1)

        daily_loss_cap = self._starting_equity * self._daily_loss_cap_pct / Decimal("100")
        if self._daily_pnl <= -daily_loss_cap and not self._kill_active:
            self._kill_active = True
            log.warning(f"KILL SWITCH ACTIVATED by trade close. Daily P&L: {self._daily_pnl}")

        log.info(
            f"Trade closed. P&L: {pnl:+.2f}. "
            f"Daily P&L: {self._daily_pnl:+.2f}. Kill: {self._kill_active}"
        )

    @_persists
    def record_mark_to_market(self, unrealised_pnl: float):
        """
        Called every tick with total unrealised P&L.
        Fires kill switch if unrealised losses exceed daily cap.
        """
        self._maybe_day_reset()
        daily_loss_cap = self._starting_equity * self._daily_loss_cap_pct / Decimal("100")
        total = self._daily_pnl + Decimal(str(unrealised_pnl))
        if total <= -daily_loss_cap and not self._kill_active:
            self._kill_active = True
            log.warning(
                f"KILL SWITCH ACTIVATED by mark-to-market. Unrealised P&L: {unrealised_pnl:.2f}"
            )

    @_persists
    def update_equity(self, current_equity: float):
        """Update peak equity tracker (call every tick)."""
        eq = Decimal(str(current_equity))
        self._current_equity = eq
        if eq > self._peak_equity:
            self._peak_equity = eq

    @property
    def status(self) -> dict:
        if self._peak_equity > 0:
            drawdown_pct = float(
                (self._peak_equity - self._current_equity) / self._peak_equity * 100
            )
        else:
            drawdown_pct = 0.0
        return {
            "kill_active":    self._kill_active,
            "drawdown_halt":  self._drawdown_halt,
            "orders_today":   self._orders_today,
            "daily_pnl":      float(self._daily_pnl),
            "open_positions": self._open_positions,
            "peak_equity":    float(self._peak_equity),
            "drawdown_pct":   round(drawdown_pct, 2),
            "today":          str(self._today),
        }

    @_persists
    def check_startup_gap(self, prev_equity: float, current_equity: float) -> None:
        """
        Call once at bot startup to detect overnight gap losses.

        VERIFIED BUG: _maybe_day_reset() clears daily_pnl=0 at midnight, but an
        overnight gap (price jumps while market is closed) can exceed the daily
        loss cap before the first tick is processed. By checking the gap at startup
        against the previous session's equity we can activate the kill switch
        before any order is attempted.

        Args:
            prev_equity:    equity at last bot shutdown (persist this to disk)
            current_equity: current equity from MT5 (first read at startup)
        """
        if prev_equity <= 0:
            return
        gap_loss = Decimal(str(prev_equity)) - Decimal(str(current_equity))
        if gap_loss <= 0:
            return

        daily_loss_cap = self._starting_equity * self._daily_loss_cap_pct / Decimal("100")
        self._daily_pnl -= gap_loss

        if gap_loss >= daily_loss_cap and not self._kill_active:
            self._kill_active = True
            log.warning(
                f"KILL SWITCH ACTIVATED by overnight gap: "
                f"prev_equity={prev_equity:.2f}, current={current_equity:.2f}, "
                f"gap_loss={float(gap_loss):.2f} >= cap={float(daily_loss_cap):.2f}"
            )
        elif gap_loss > 0:
            log.info(
                f"Overnight gap detected: -{float(gap_loss):.2f} "
                f"(daily P&L now {float(self._daily_pnl):+.2f})"
            )

    def reset_drawdown_halt(self, current_equity: float) -> None:
        """Manual reset after a drawdown halt: clears the halt and re-anchors peak equity."""
        self._drawdown_halt  = False
        self._peak_equity    = Decimal(str(current_equity))
        self._current_equity = Decimal(str(current_equity))
        log.warning(f"Drawdown halt manually reset. Peak equity re-anchored to {current_equity:.2f}")
        self._save_state()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _snapshot(self) -> dict:
        return {
            "version":         1,
            "account":         self._account_id,
            "today":           self._today.isoformat(),
            "daily_pnl":       str(self._daily_pnl),
            "kill_active":     self._kill_active,
            "orders_today":    self._orders_today,
            "day_anchor":      str(self._starting_equity),
            "drawdown_halt":   self._drawdown_halt,
            "peak_equity":     str(self._peak_equity),
        }

    def _save_state(self) -> None:
        if not self._state_path:
            return
        snap = self._snapshot()
        if snap == self._last_saved:
            return
        try:
            folder = os.path.dirname(self._state_path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            tmp = self._state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(snap, f, indent=2)
            os.replace(tmp, self._state_path)   # atomic: never leaves a half-written file
            self._last_saved = snap
        except Exception as e:
            log.error(f"Could not save governor state to {self._state_path}: {e}")

    def _load_state(self) -> None:
        if not os.path.exists(self._state_path):
            log.info("Governor: no saved state — starting fresh.")
            return
        try:
            with open(self._state_path, "r") as f:
                st = json.load(f)
        except Exception as e:
            # Fail safe: an unreadable state file must not silently clear a halt.
            self._kill_active = True
            log.error(
                f"Governor state file {self._state_path} unreadable ({e}). "
                "Kill switch ON for today as a precaution."
            )
            return

        saved_acct = st.get("account")
        if self._account_id and saved_acct and str(saved_acct) != self._account_id:
            log.warning(
                f"Governor state file belongs to account {saved_acct}, not {self._account_id} — "
                "ignoring it and starting fresh for this account."
            )
            return

        # Drawdown halt and peak equity carry across days.
        self._drawdown_halt = bool(st.get("drawdown_halt", False))
        try:
            saved_peak = Decimal(str(st.get("peak_equity", "0")))
            if saved_peak > self._peak_equity:
                self._peak_equity = saved_peak
        except Exception:
            pass

        # Daily figures carry over only on the same UTC day.
        if st.get("today") == self._today.isoformat():
            try:
                self._daily_pnl       = Decimal(str(st.get("daily_pnl", "0")))
                self._starting_equity = Decimal(str(st.get("day_anchor", self._starting_equity)))
            except Exception:
                pass
            self._kill_active  = bool(st.get("kill_active", False))
            self._orders_today = int(st.get("orders_today", 0))
        log.info(
            f"Governor state restored: daily_pnl={self._daily_pnl} kill={self._kill_active} "
            f"orders_today={self._orders_today} drawdown_halt={self._drawdown_halt} "
            f"peak={self._peak_equity}"
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _maybe_day_reset(self):
        today = datetime.now(timezone.utc).date()   # UTC — not local clock
        if today != self._today:
            log.info(
                f"Day reset: {self._today} → {today}. "
                "Clearing kill switch, orders, and daily P&L."
            )
            self._today        = today
            self._kill_active  = False
            self._orders_today = 0
            self._daily_pnl    = Decimal("0")
            # Re-anchor the daily loss cap to current equity so 2% always means
            # 2% of *today's* starting balance, not the bot's first-ever equity.
            if self._current_equity > 0:
                self._starting_equity = self._current_equity
                log.info(f"Daily cap re-anchored to current equity: {float(self._starting_equity):.2f}")
            # Drawdown halt persists across days (intentional — require manual reset)
