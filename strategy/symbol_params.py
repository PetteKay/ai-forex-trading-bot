"""
Symbol-Specific Trading Parameters.

Different asset classes have different volatility profiles.
Default ATR SL/TP multipliers from config.py are calibrated for EURUSD.
Crypto, energies, and GBP crosses need wider stops; indices need tighter exits.

BEAR_RISK_VETO: overrides config.BEAR_RISK_VETO_THRESHOLD per symbol.
Lower threshold = more aggressive bear protection (used for volatile/unpredictable assets).
"""

_PARAMS: dict[str, dict] = {
    # ── Forex Majors ──────────────────────────────────────────────────────────
    "EURUSD": {"ATR_SL_MULT": 1.5, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 8},
    "GBPUSD": {"ATR_SL_MULT": 1.6, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 8},
    "USDJPY": {"ATR_SL_MULT": 1.5, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 8},
    "AUDUSD": {"ATR_SL_MULT": 1.5, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 8},
    "NZDUSD": {"ATR_SL_MULT": 1.5, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 8},
    "USDCAD": {"ATR_SL_MULT": 1.5, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 8},
    "USDCHF": {"ATR_SL_MULT": 1.5, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 8},
    # ── JPY Crosses — yen carry unwinds cause spike risk; need wider SL ───────
    "GBPJPY": {"ATR_SL_MULT": 1.8, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 7},
    "EURJPY": {"ATR_SL_MULT": 1.7, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 7},
    "AUDJPY": {"ATR_SL_MULT": 1.6, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 8},
    "NZDJPY": {"ATR_SL_MULT": 1.6, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 8},
    "CADJPY": {"ATR_SL_MULT": 1.7, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 8},
    "CHFJPY": {"ATR_SL_MULT": 1.6, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 8},
    # ── GBP Crosses — widest pip ranges in forex; need 2× ATR room ───────────
    "GBPNZD": {"ATR_SL_MULT": 2.0, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 7},
    "GBPAUD": {"ATR_SL_MULT": 2.0, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 7},
    "GBPCAD": {"ATR_SL_MULT": 1.8, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 7},
    "GBPCHF": {"ATR_SL_MULT": 1.7, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 7},
    "EURNZD": {"ATR_SL_MULT": 1.8, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 8},
    # ── Metals — gap risk and geopolitical spike risk ─────────────────────────
    "XAUUSD": {"ATR_SL_MULT": 1.8, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 7},
    "XAGUSD": {"ATR_SL_MULT": 2.0, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 7},
    # ── Equity Indices — intraday mean-reversion risk; tight but fast ─────────
    "US30":   {"ATR_SL_MULT": 1.5, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 7},
    # ── US Stock CFDs — earnings gap risk; wider stops; tighter bear veto ─────
    "TSLA":   {"ATR_SL_MULT": 2.0, "ATR_TP_MULT": 3.0, "BEAR_RISK_VETO": 7},  # high beta, frequent gaps
    "NVDA":   {"ATR_SL_MULT": 1.8, "ATR_TP_MULT": 3.0, "BEAR_RISK_VETO": 7},  # momentum-driven; earnings volatile
    "MSFT":   {"ATR_SL_MULT": 1.5, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 8},  # large-cap, lower beta
    "AMD":    {"ATR_SL_MULT": 1.8, "ATR_TP_MULT": 3.0, "BEAR_RISK_VETO": 7},  # correlated to NVDA; semiconductor
}

_DEFAULT = {"ATR_SL_MULT": 1.5, "ATR_TP_MULT": 2.5, "BEAR_RISK_VETO": 8}


def get(symbol: str) -> dict:
    """
    Returns symbol-specific ATR_SL_MULT, ATR_TP_MULT, and BEAR_RISK_VETO.
    Falls back to defaults if symbol not registered.
    """
    sym = symbol.upper().split(".")[0]
    return dict(_PARAMS.get(sym, _DEFAULT))
