"""
Base LLM caller. Supports OpenAI, DeepSeek, NVIDIA NIM, Ollama, and Claude (Anthropic).
All agents call this — swap providers in config.py without touching agent code.

Provider       | Package    | Set LLM_BASE_URL to
-------------- | ---------- | -------------------------------------------
openai         | openai     | None  (uses api.openai.com)
deepseek       | openai     | "https://api.deepseek.com"
nvidia         | openai     | "https://integrate.api.nvidia.com/v1"
ollama         | openai     | "http://localhost:11434/v1"
claude         | anthropic  | None  (uses api.anthropic.com automatically)

Reproducibility: temperature=0 + seed (where API supports it) ensures
identical inputs produce identical outputs. All calls log a prompt-hash
for audit purposes.

Model routing: can override model per agent type via MODEL_OVERRIDES in config.
"""

import hashlib
import json
import logging
import os
import re
import time

log = logging.getLogger("LLM")

_openai_client    = None
_anthropic_client = None
_active_provider  = None
_active_cfg       = None

# Maximum retries on transient failures
_MAX_RETRIES = 2     # 3 was too many: 3 × 30s timeout = 90s per agent, killing MT5 IPC
_RETRY_DELAY = 2.0   # seconds



class ConfigError(RuntimeError):
    """A setting the user must fix (e.g. no API key): retrying won't help."""

def init(cfg):
    global _openai_client, _anthropic_client, _active_provider, _active_cfg

    provider = cfg.LLM_PROVIDER.lower()
    _active_provider = provider
    _active_cfg      = cfg
    _key = str(getattr(cfg, "LLM_API_KEY", "") or "")
    if _key.lower().startswith("your-") or _key.lower().endswith("-here"):
        _key = ""                               # the placeholder from .env.example / SETUP.md
    if not _key and not (
            provider == "claude" and os.environ.get("ANTHROPIC_API_KEY")):
        raise ConfigError(
            "LLM_API_KEY is not set. Put it in a file named .env in the bot's folder "
            "(next to config.py / ForexBot.exe):\n"
            "  LLM_API_KEY=your-key-here\n"
            "or set it once in PowerShell:  setx LLM_API_KEY \"your-key-here\"  "
            "(then open a new window).\n"
            "For Ollama (local, free) any value works, e.g. LLM_API_KEY=ollama."
        )

    if provider == "claude":
        try:
            import anthropic
            _anthropic_client = anthropic.Anthropic(api_key=cfg.LLM_API_KEY or None)  # None: ANTHROPIC_API_KEY
            log.info(f"LLM ready: Claude (Anthropic) | model={cfg.LLM_MODEL}")
        except ImportError:
            raise ImportError("Run: pip install anthropic")
    else:
        try:
            from openai import OpenAI
            kwargs = {"api_key": cfg.LLM_API_KEY}
            if getattr(cfg, "LLM_BASE_URL", None):
                kwargs["base_url"] = cfg.LLM_BASE_URL
            _openai_client = OpenAI(**kwargs)
            log.info(f"LLM ready: {provider} | model={cfg.LLM_MODEL}")
        except ImportError:
            raise ImportError("Run: pip install openai")


def call(prompt: str, system: str, cfg, temperature: float = 0.3,
         agent_name: str = "") -> str:
    """Send a prompt to the LLM and return the text response."""
    if _active_provider is None:
        raise RuntimeError("Call agents.base.init(cfg) before using LLM.")

    model = _resolve_model(cfg, agent_name)
    prompt_hash = hashlib.md5((system + prompt).encode()).hexdigest()[:8]
    log.debug(f"LLM call [{agent_name or 'anon'}] hash={prompt_hash} model={model}")

    for attempt in range(_MAX_RETRIES):
        try:
            if _active_provider == "claude":
                return _call_claude(prompt, system, cfg, temperature, model)
            else:
                return _call_openai(prompt, system, cfg, temperature, model)
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                log.warning(f"LLM call failed (attempt {attempt+1}): {e} — retrying...")
                time.sleep(_RETRY_DELAY * (attempt + 1))
            else:
                log.error(f"LLM call failed after {_MAX_RETRIES} attempts ({_active_provider}): {e}")
                return ""

    return ""


def call_json(prompt: str, system: str, cfg, temperature: float = 0.2,
              agent_name: str = "") -> dict:
    """
    Call the LLM, parse the JSON response, and validate domain constraints.
    Handles LLMs that wrap JSON in markdown code blocks.
    Returns a safe default dict on failure — never returns {}.

    VERIFIED: LLMs fail to produce valid JSON ~4.1% of calls; in a 5-agent
    pipeline this compounds to ~19% chance of at least one invalid response
    per tick. Structured retry + validated fallbacks prevent silent bad decisions.
    """
    raw = call(prompt, system, cfg, temperature, agent_name)
    if not raw:
        log.warning(f"LLM returned empty response for agent '{agent_name}'")
        return _safe_default(agent_name)

    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    clean = re.sub(r"\s*```$",          "", clean.strip(), flags=re.MULTILINE)

    parsed = None
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", clean)
        if match:
            try:
                parsed = json.loads(match.group())
            except Exception:
                pass

    if parsed is None:
        log.warning(f"LLM non-JSON for agent '{agent_name}': {raw[:200]}")
        return _safe_default(agent_name)

    validated = _validate_and_sanitize(parsed, agent_name)
    return validated


# ── Per-agent safe defaults (used when LLM fails or returns garbage) ─────────
_DEFAULTS = {
    "technical_analyst": {
        "bias": "NEUTRAL", "bias_strength": 0, "trend": "SIDEWAYS",
        "trend_strength": 0, "momentum": "NEUTRAL",
        "key_support": 0.0, "key_resistance": 0.0,
        "setup_name": None, "risks": [], "summary": "Analysis unavailable.",
    },
    "bear_analyst": {
        "bear_thesis": "No bear analysis available.", "risk_score": 5,
        "evidence": [], "counter_to_bull": "Unknown", "regime_warning": "",
    },
    "news_analyst": {
        "macro_tone": "NEUTRAL", "news_risk": 5, "key_themes": [],
        "session_bias": "NEUTRAL", "recommendation": "HOLD",
    },
    "fund_manager": {
        "action": "HOLD", "confidence": 0, "reasoning": "Analysis unavailable.",
        "key_risks": [], "lessons_applied": [], "bull_bear_verdict": "UNCLEAR",
    },
    "risk_manager": {
        "trade_qualified": False, "size_multiplier": 0.0, "max_risk_pct": 1.0,
        "veto_reason": "Risk analysis unavailable — defaulting to no trade.",
        "wait_for": None, "risk_notes": [],
    },
    "reflection": {
        "lesson": "Trade closed — no analysis available.",
        "usefulness_score": 0.0, "category": "UNKNOWN",
        "pattern_name": None, "agent_reasoning": "",
    },
}

# Fields that must satisfy hard constraints; checked after parsing
_FIELD_CONSTRAINTS = {
    "bias":          (str, {"BUY", "SELL", "NEUTRAL"}),
    "action":        (str, {"BUY", "SELL", "HOLD"}),
    "trend":         (str, {"BULLISH", "BEARISH", "SIDEWAYS"}),
    "macro_tone":    (str, {"BULLISH", "BEARISH", "NEUTRAL"}),
    "recommendation":(str, {"BUY", "SELL", "HOLD", "NEUTRAL"}),
    "bias_strength": (int,  None),
    "trend_strength":(int,  None),
    "confidence":    ((int, float), None),
    "risk_score":    ((int, float), None),
    "news_risk":     ((int, float), None),
    "size_multiplier":((int, float), None),
    "usefulness_score":((int, float), None),
    "trade_qualified": (bool, None),
}

# Numeric fields that must be in [0, 10]
_CLAMP_0_10  = {"bias_strength", "trend_strength", "risk_score", "news_risk"}
# Numeric fields that must be in [0, 1]
_CLAMP_0_1   = {"size_multiplier", "usefulness_score"}
# Numeric fields that must be in [0, 100]
_CLAMP_0_100 = {"confidence"}


def _safe_default(agent_name: str) -> dict:
    return dict(_DEFAULTS.get(agent_name, {"action": "HOLD", "confidence": 0}))


def _validate_and_sanitize(result: dict, agent_name: str) -> dict:
    """
    Sanitize LLM output: coerce types, clamp numbers to valid ranges,
    replace invalid enum values with safe defaults.
    LLMs frequently return e.g. confidence=95.5 (float not int), or
    action="NONE" instead of "HOLD", or risk_score=15 (out of range).
    """
    defaults = _safe_default(agent_name)
    out      = dict(result)

    for field, (expected_type, valid_set) in _FIELD_CONSTRAINTS.items():
        if field not in out:
            continue
        val = out[field]

        # Type coercion
        try:
            if expected_type == bool:
                if isinstance(val, bool):
                    pass  # already correct
                elif isinstance(val, str):
                    # bool("false") == True in Python — must parse string booleans explicitly
                    out[field] = val.lower() in ("true", "1", "yes")
                elif isinstance(val, (int, float)):
                    out[field] = val != 0
                else:
                    out[field] = defaults.get(field, False)
            elif expected_type == int:
                out[field] = int(float(str(val)))
            elif expected_type == (int, float):
                out[field] = float(str(val))
            elif expected_type == str:
                out[field] = str(val).upper().strip()
        except (ValueError, TypeError):
            out[field] = defaults.get(field, val)
            continue

        # Enum validation
        if valid_set and out[field] not in valid_set:
            log.debug(
                f"Agent '{agent_name}' returned invalid {field}='{out[field]}' "
                f"(expected one of {valid_set}) — using default."
            )
            out[field] = defaults.get(field, list(valid_set)[0])

    # Numeric range clamping
    for field in _CLAMP_0_10:
        if field in out:
            try:
                out[field] = max(0, min(10, float(out[field])))
            except (TypeError, ValueError):
                out[field] = defaults.get(field, 5)

    for field in _CLAMP_0_1:
        if field in out:
            try:
                out[field] = max(0.0, min(1.0, float(out[field])))
            except (TypeError, ValueError):
                out[field] = defaults.get(field, 0.0)

    for field in _CLAMP_0_100:
        if field in out:
            try:
                out[field] = max(0, min(100, float(out[field])))
            except (TypeError, ValueError):
                out[field] = defaults.get(field, 0)

    return out


def _resolve_model(cfg, agent_name: str) -> str:
    """
    Resolve which model to use for this agent.
    Checks MODEL_OVERRIDES in config first, falls back to LLM_MODEL.
    """
    overrides = getattr(cfg, "MODEL_OVERRIDES", {})
    return overrides.get(agent_name, cfg.LLM_MODEL)


def _call_openai(prompt: str, system: str, cfg, temperature: float, model: str) -> str:
    timeout = getattr(cfg, "LLM_TIMEOUT_SECONDS", 30)
    kwargs = dict(
        model=model,
        temperature=temperature,
        max_tokens=getattr(cfg, "LLM_MAX_TOKENS", 1000),
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        timeout=timeout,
    )
    # Seed for reproducibility (supported by OpenAI, ignored gracefully by others)
    seed = getattr(cfg, "LLM_SEED", None)
    if seed is not None:
        kwargs["seed"] = seed

    resp = _openai_client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()


def _call_claude(prompt: str, system: str, cfg, temperature: float, model: str) -> str:
    import httpx
    timeout = getattr(cfg, "LLM_TIMEOUT_SECONDS", 30)
    resp = _anthropic_client.messages.create(
        model=model,
        max_tokens=getattr(cfg, "LLM_MAX_TOKENS", 1000),
        temperature=temperature,
        system=system,
        messages=[
            {"role": "user", "content": prompt},
        ],
        timeout=httpx.Timeout(timeout),
    )
    return resp.content[0].text.strip()
