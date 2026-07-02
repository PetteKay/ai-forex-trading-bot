"""
Telegram trade alerts — real-time push notifications via Telegram Bot API.

Setup:
  1. Message @BotFather on Telegram → /newbot → copy the token
  2. Set TELEGRAM_BOT_TOKEN = "your-token" in config.py
  3. Message @userinfobot or @RawDataBot to get your numeric chat ID
  4. Set TELEGRAM_CHAT_ID = "your-chat-id" in config.py
  5. Set TELEGRAM_ALERTS_ENABLED = True in config.py

Uses stdlib urllib only (no python-telegram-bot package required).
"""

import json
import logging
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError

log = logging.getLogger("TelegramAlert")


def _post(token: str, chat_id: str, text: str) -> bool:
    """Low-level Telegram sendMessage call. Returns True on success."""
    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    body = json.dumps({
        "chat_id":    str(chat_id),
        "text":       text[:4096],  # Telegram message limit
        "parse_mode": "HTML",
    }).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if not result.get("ok"):
                log.warning(f"Telegram API error: {result.get('description', result)}")
                return False
            return True
    except URLError as e:
        log.warning(f"Telegram network error: {e}")
        return False
    except Exception as e:
        log.warning(f"Telegram unexpected error: {e}")
        return False


class TelegramAlert:
    """
    Send trade notifications to a Telegram chat.
    All methods are silent no-ops when disabled or token/chat_id are missing.
    """

    def __init__(self, token: str, chat_id: str, enabled: bool = True):
        self._token   = token or ""
        self._chat_id = str(chat_id) if chat_id else ""
        self._enabled = enabled and bool(self._token) and bool(self._chat_id)
        if self._enabled:
            log.info("Telegram alerts ENABLED")
        else:
            log.info("Telegram alerts disabled (no token/chat_id or TELEGRAM_ALERTS_ENABLED=False)")

    def _send(self, text: str) -> bool:
        if not self._enabled:
            return False
        return _post(self._token, self._chat_id, text)

    def trade_open(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        sl: float,
        tp: float,
        lot: float,
        confidence: int,
        regime: str,
        reasoning: str = "",
    ) -> bool:
        arrow = "▲ LONG" if direction == "BUY" else "▼ SHORT"
        now   = datetime.now(timezone.utc).strftime("%H:%M UTC")
        msg   = (
            f"<b>{arrow} {symbol}</b>  [{now}]\n"
            f"Entry:       <code>{entry_price:.5f}</code>\n"
            f"SL / TP:     <code>{sl:.5f} / {tp:.5f}</code>\n"
            f"Lot:         {lot}\n"
            f"Confidence:  {confidence}%\n"
            f"Regime:      {regime}\n"
        )
        if reasoning:
            msg += f"Reasoning:   {reasoning[:140]}\n"
        return self._send(msg)

    def trade_close(
        self,
        symbol: str,
        direction: str,
        outcome: str,
        pnl: float,
        r_multiple: float,
        hold_hours: float,
    ) -> bool:
        icon = "✅" if pnl >= 0 else "❌"
        now  = datetime.now(timezone.utc).strftime("%H:%M UTC")
        msg  = (
            f"{icon} <b>CLOSE {symbol}</b> ({direction}) [{now}]\n"
            f"Outcome:     {outcome}\n"
            f"P&amp;L:          <code>{pnl:+.2f}</code>\n"
            f"R-multiple:  <code>{r_multiple:+.2f}R</code>\n"
            f"Hold time:   {hold_hours:.1f}h\n"
        )
        return self._send(msg)

    def daily_summary(
        self,
        trades: int,
        winners: int,
        total_pnl: float,
        equity: float,
        max_drawdown_pct: float,
    ) -> bool:
        win_rate = (winners / trades * 100) if trades > 0 else 0.0
        pnl_sign = "+" if total_pnl >= 0 else ""
        now      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        msg      = (
            f"<b>Daily Summary</b>  [{now}]\n"
            f"Trades:      {trades}  ({winners}W / {trades - winners}L)\n"
            f"Win rate:    {win_rate:.0f}%\n"
            f"Total P&amp;L:    <code>{pnl_sign}{total_pnl:.2f}</code>\n"
            f"Equity:      <code>{equity:.2f}</code>\n"
            f"Max DD:      <code>{max_drawdown_pct:.2f}%</code>\n"
        )
        return self._send(msg)

    def error(self, message: str) -> bool:
        now = datetime.now(timezone.utc).strftime("%H:%M UTC")
        msg = f"<b>BOT ERROR</b>  [{now}]\n{message[:400]}"
        return self._send(msg)

    def info(self, message: str) -> bool:
        return self._send(message)
