"""
Active Position Manager — manages open trades tick-by-tick.

Capabilities:
  1. Break-even move (move SL to entry after N pips profit)
  2. Trailing stop loss (ATR-based trail)
  3. R-based TP ladder: scale out at 1R, 2R; let remainder run with trail
     (replaces single partial-close-at-fixed-pips approach)
  4. Time-based exit (close if trade held too long without progress)
  5. Drawdown exit (close if unrealised loss hits position-level limit)

R-BASED TP LADDER:
  Tier 1 at 1R profit → close TP_LADDER_TIER1_PCT% (default 33%), SL → breakeven
  Tier 2 at 2R profit → close TP_LADDER_TIER2_PCT% (default 33%) of original
  Remainder (~34%) rides the trailing stop to capture extended moves.

  Volume-ratio detection is restart-safe: uses pos.volume / record.volume
  to infer which tiers have already fired, so a bot restart mid-trade
  does not re-trigger a tier that already executed.

  Falls back to single PARTIAL_CLOSE_PIPS logic if TP_LADDER_ENABLED=False.
"""

import logging
from datetime import datetime, timezone

import mt5_bridge as mt5b

log = logging.getLogger("PositionManager")


class PositionManager:
    def __init__(self, cfg):
        self._cfg = cfg

    def manage(self, open_trades: dict, equity: float) -> None:
        """Called every tick. Manages all open positions."""
        if not open_trades:
            return
        cfg = self._cfg
        for ticket, record in list(open_trades.items()):
            try:
                self._manage_one(ticket, record, equity, cfg)
            except Exception as e:
                log.warning(f"Position manager error for ticket {ticket}: {e}")

    def _manage_one(self, ticket: int, record, equity: float, cfg) -> None:
        import MetaTrader5 as mt5
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return
        pos = positions[0]

        price_open  = pos.price_open
        price_curr  = pos.price_current
        current_sl  = pos.sl
        volume      = pos.volume
        is_buy      = (pos.type == mt5.ORDER_TYPE_BUY)
        symbol      = pos.symbol
        profit      = pos.profit

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return
        point    = sym_info.point
        # Pip = 10 points on fractional-pip quotes: 5-digit (EURUSD 1.13786)
        # AND 3-digit (USDJPY 162.519, XAGUSD 59.270). The old `digits == 5`
        # check made pips_profit 10x too large on every JPY cross, firing
        # break-even/trailing/partial-close at 1/10 the intended distance.
        pip_size = point * 10 if sym_info.digits in (3, 5) else point

        pips_profit = (
            (price_curr - price_open) / pip_size if is_buy
            else (price_open - price_curr) / pip_size
        )

        atr          = getattr(record, "atr", None) or (50 * pip_size)
        orig_volume  = getattr(record, "volume", volume)
        entry_sl     = getattr(record, "sl", 0.0)
        entry_price  = getattr(record, "entry_price", price_open)
        vol_ratio    = volume / orig_volume if orig_volume > 0 else 1.0

        # ── 1. Break-even: move SL to entry + 1 pip ──────────────────────────
        be_pips = getattr(cfg, "BREAKEVEN_PIPS", 15)
        if pips_profit >= be_pips and vol_ratio > 0.85:
            be_sl = round(
                price_open + pip_size if is_buy else price_open - pip_size,
                sym_info.digits,
            )
            needs_move = (
                (is_buy  and current_sl < be_sl - pip_size) or
                (not is_buy and (current_sl == 0 or current_sl > be_sl + pip_size))
            )
            if needs_move:
                if mt5b.modify_position_sl(ticket, symbol, be_sl):
                    log.info(f"Ticket {ticket}: SL → break-even ({be_sl:.5f})")

        # ── 2. Trailing stop: trail at ATR_TRAIL_MULT × ATR behind price ─────
        trail_mult  = getattr(cfg, "ATR_TRAIL_MULT", 1.0)
        trail_start = getattr(cfg, "TRAIL_START_PIPS", 20)
        if pips_profit >= trail_start:
            trail_dist = atr * trail_mult
            if is_buy:
                new_sl = round(price_curr - trail_dist, sym_info.digits)
                if new_sl > current_sl + pip_size:
                    mt5b.modify_position_sl(ticket, symbol, new_sl)
                    log.info(f"Ticket {ticket}: trailing SL → {new_sl:.5f} (+{pips_profit:.1f}p)")
            else:
                new_sl = round(price_curr + trail_dist, sym_info.digits)
                if current_sl == 0 or new_sl < current_sl - pip_size:
                    mt5b.modify_position_sl(ticket, symbol, new_sl)
                    log.info(f"Ticket {ticket}: trailing SL → {new_sl:.5f} (+{pips_profit:.1f}p)")

        # ── 3. TP Ladder (R-based) or single partial close ────────────────────
        if getattr(cfg, "TP_LADDER_ENABLED", False):
            self._tp_ladder(
                ticket, record, pos, pips_profit, vol_ratio, orig_volume,
                entry_price, entry_sl, pip_size, is_buy, symbol, sym_info, cfg,
            )
        else:
            # Legacy single partial close at fixed pip distance
            partial_pips = getattr(cfg, "PARTIAL_CLOSE_PIPS", 30)
            if (pips_profit >= partial_pips and vol_ratio > 0.85
                    and orig_volume > getattr(cfg, "LOT_SIZE", 0.01)):
                partial_vol = round(orig_volume / 2, 2)
                if partial_vol >= 0.01:
                    pos_type = "BUY" if is_buy else "SELL"
                    if mt5b.close_position(ticket, symbol, partial_vol, pos_type, cfg.MAGIC_NUMBER):
                        log.info(f"Ticket {ticket}: partial close {partial_vol} lots at +{pips_profit:.1f}p")

        # ── 4. Time-based exit: close if held > MAX_HOLD_HOURS with no profit ─
        max_hold   = getattr(cfg, "MAX_HOLD_HOURS", 48)
        hold_hours = (datetime.now(timezone.utc) - record.entry_time).total_seconds() / 3600
        if hold_hours > max_hold and pips_profit < 5:
            pos_type = "BUY" if is_buy else "SELL"
            log.info(f"Ticket {ticket}: time exit after {hold_hours:.1f}h | pips={pips_profit:.1f}")
            mt5b.close_position(ticket, symbol, volume, pos_type, cfg.MAGIC_NUMBER)

        # ── 5. Position-level emergency exit ─────────────────────────────────
        max_pos_loss = getattr(cfg, "MAX_POSITION_LOSS_USD", None)
        if max_pos_loss and profit < -abs(max_pos_loss):
            pos_type = "BUY" if is_buy else "SELL"
            log.warning(f"Ticket {ticket}: emergency exit — loss {profit:.2f} > max {max_pos_loss:.2f}")
            mt5b.close_position(ticket, symbol, volume, pos_type, cfg.MAGIC_NUMBER)

    def _tp_ladder(
        self, ticket, record, pos, pips_profit, vol_ratio,
        orig_volume, entry_price, entry_sl, pip_size,
        is_buy, symbol, sym_info, cfg,
    ) -> None:
        """
        R-based take-profit ladder.

        vol_ratio = current_volume / original_volume (restart-safe state detection):
          > 0.85  → neither tier fired yet
          0.40–0.85 → tier 1 fired; tier 2 pending
          < 0.40  → both tiers fired; remainder riding with trail
        """
        import MetaTrader5 as mt5
        pos_type = "BUY" if is_buy else "SELL"

        # Risk in pips = distance from entry to original SL
        if entry_sl > 0 and entry_price > 0:
            risk_pips = abs(entry_price - entry_sl) / pip_size
        else:
            risk_pips = getattr(cfg, "PARTIAL_CLOSE_PIPS", 30) * 1.0
        if risk_pips < 5:
            risk_pips = 30.0  # safety floor

        t1_r    = getattr(cfg, "TP_LADDER_TIER1_R", 1.0)
        t2_r    = getattr(cfg, "TP_LADDER_TIER2_R", 2.0)
        t1_pct  = getattr(cfg, "TP_LADDER_TIER1_PCT", 33) / 100.0
        t2_pct  = getattr(cfg, "TP_LADDER_TIER2_PCT", 33) / 100.0

        t1_pips = risk_pips * t1_r
        t2_pips = risk_pips * t2_r

        # ── Tier 1: close first tranche + move SL to breakeven ───────────────
        if pips_profit >= t1_pips and vol_ratio > 0.85:
            t1_vol = round(orig_volume * t1_pct, 2)
            t1_vol = max(0.01, t1_vol)
            if t1_vol < pos.volume:
                if mt5b.close_position(ticket, symbol, t1_vol, pos_type, cfg.MAGIC_NUMBER):
                    log.info(
                        f"Ticket {ticket}: TP ladder TIER1 — closed {t1_vol} lots "
                        f"at +{pips_profit:.1f} pips (1R = {t1_pips:.0f} pips)"
                    )
                    # Move SL to break-even
                    be_sl = round(
                        pos.price_open + pip_size if is_buy
                        else pos.price_open - pip_size,
                        sym_info.digits,
                    )
                    mt5b.modify_position_sl(ticket, symbol, be_sl)
                    log.info(f"Ticket {ticket}: SL → break-even after tier1 ({be_sl:.5f})")

        # ── Tier 2: close second tranche ─────────────────────────────────────
        elif pips_profit >= t2_pips and 0.40 < vol_ratio <= 0.85:
            t2_vol = round(orig_volume * t2_pct, 2)
            t2_vol = max(0.01, t2_vol)
            if t2_vol < pos.volume:
                if mt5b.close_position(ticket, symbol, t2_vol, pos_type, cfg.MAGIC_NUMBER):
                    log.info(
                        f"Ticket {ticket}: TP ladder TIER2 — closed {t2_vol} lots "
                        f"at +{pips_profit:.1f} pips (2R = {t2_pips:.0f} pips)"
                    )
                    # Remaining ~34% rides with the trailing stop already active

    def close_all(self, open_trades: dict, cfg) -> None:
        """Force-close ALL open positions. Called when kill switch activates."""
        if not open_trades:
            return
        import MetaTrader5 as mt5
        log.warning(f"KILL SWITCH: force-closing {len(open_trades)} open position(s).")
        for ticket in list(open_trades.keys()):
            try:
                positions = mt5.positions_get(ticket=ticket)
                if not positions:
                    continue
                pos      = positions[0]
                pos_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                success  = mt5b.close_position(
                    ticket, pos.symbol, pos.volume, pos_type, cfg.MAGIC_NUMBER
                )
                if success:
                    log.warning(f"KILL: closed {pos_type} {pos.volume} lots ticket={ticket}")
                    open_trades.pop(ticket, None)
                else:
                    log.error(f"KILL: failed to close ticket={ticket} — requires manual intervention!")
            except Exception as e:
                log.error(f"KILL close error for ticket {ticket}: {e}")
