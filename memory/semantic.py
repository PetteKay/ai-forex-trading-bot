"""
Semantic memory layer — embeds lesson text via NVIDIA NIM embedding API.

Uses the same API key as the LLM (config.LLM_API_KEY + NVIDIA base URL).
Model: nvidia/nv-embedqa-e5-v5  (free with NVIDIA credits, 512-dim vectors)

Falls back silently if the API is unavailable — the bot continues with
fingerprint-only recall rather than crashing.
"""

import logging

log = logging.getLogger("Semantic")

_EMBED_MODEL  = "nvidia/nv-embedqa-e5-v5"
_EMBED_BASE   = "https://integrate.api.nvidia.com/v1"
_EMBED_DIM    = 512

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        import config
        from openai import OpenAI
        _client = OpenAI(api_key=config.LLM_API_KEY, base_url=_EMBED_BASE)
        log.info(f"Semantic embed client ready: {_EMBED_MODEL}")
    except Exception as e:
        log.warning(f"Semantic embed client failed to init: {e}")
        _client = None
    return _client


def embed_text(text: str) -> list[float] | None:
    """
    Embed a single text string. Returns a float list or None on failure.
    Results are NOT cached — each lesson is unique.
    """
    if not text or not text.strip():
        return None
    client = _get_client()
    if client is None:
        return None
    try:
        response = client.embeddings.create(
            input=[text[:2048]],   # NIM hard limit
            model=_EMBED_MODEL,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "END"},
        )
        return response.data[0].embedding
    except Exception as e:
        log.debug(f"embed_text failed: {e}")
        return None


def embed_market_context(ctx: dict) -> list[float] | None:
    """
    Build a natural-language description of the current market context
    and embed it — used for semantic recall in the bot tick loop.
    """
    parts = []

    regime = ctx.get("regime", {})
    if regime:
        parts.append(
            f"Market regime: {regime.get('regime', 'unknown')}, "
            f"ADX={regime.get('adx', '?')}, Hurst={regime.get('hurst', '?')}"
        )

    structure = ctx.get("market_structure", "")
    if structure:
        parts.append(f"Market structure: {structure}")

    patterns = ctx.get("candlestick_patterns", [])
    if patterns:
        parts.append(f"Candlestick patterns: {', '.join(patterns)}")

    bias = ctx.get("ema_bias", "")
    if bias:
        parts.append(f"EMA trend bias: {bias}")

    rsi = ctx.get("rsi")
    if rsi is not None:
        parts.append(f"RSI: {rsi:.1f}")

    atr_pct = ctx.get("atr_percentile")
    if atr_pct is not None:
        parts.append(f"ATR volatility percentile: {atr_pct:.0f}%")

    if not parts:
        return None

    description = ". ".join(parts) + "."
    log.debug(f"Embedding context: {description[:120]}")
    return embed_text(description)
