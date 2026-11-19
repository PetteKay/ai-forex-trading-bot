"""
Correlation guard — blocks a new trade when a highly correlated pair already has
an open position in the same direction (duplicate economic exposure).

Shared by bot.py and mcp_server.py.
"""

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
