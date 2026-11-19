"""
Meta-Labeling Gate (López de Prado, Advances in Financial ML, Chapter 3).

Primary model: technical analysts give direction (BUY/SELL).
Secondary model (this): decides WHETHER to take the trade and at what size.

VERIFIED FIX: The original implementation used the same feature_vector as the primary
technical model, producing no genuine improvement (verified infinite-regress problem).
This version extends features with time-of-day context that the primary model does NOT
see — giving the meta-label model a genuinely different information set.

TRAIN/SERVE CONSISTENCY: Only timestamp-derivable features (time-of-day, day-of-week)
and regime encoding are included. Session labels are excluded — at training time they
cannot be reliably reconstructed from stored memory metadata.

Extended feature set (primary 16 + 8 meta-only features = 24 total):
  [0-15]  Technical vector from strategy.indicators.feature_vector()
  [16]    sin(hour_of_day * 2pi/24)  — captures intraday seasonality
  [17]    cos(hour_of_day * 2pi/24)  — in cyclical encoding
  [18]    sin(day_of_week * 2pi/7)   — Monday/Friday effects
  [19]    cos(day_of_week * 2pi/7)
  [20]    regime_encoded             — 1.0=TRENDING_UP, -1.0=TRENDING_DOWN,
                                       0.5=RANGING, 0.0=VOLATILE/unknown
  [21]    fib_proximity_score        — 0-1 (1 = at a major Fibonacci level)
  [22]    cot_z_score_norm           — COT net position z-score, clipped [-2,2] → [-1,1]
  [23]    econ_surprise_norm         — Economic surprise, clipped [-30,30] → [-1,1]
  [last]  side                       — +1 BUY / -1 SELL / 0 unknown. Meta-labeling
                                       judges the primary model's chosen side, so the
                                       secondary model must know which side it is.

TIME FEATURES use the trade's ENTRY time at both training and inference (they
previously used the lesson-save time at training = the trade's CLOSE time).

REGIME-CONDITIONAL MODELS:
  When >= MIN_SAMPLES_PER_REGIME trades per regime are available, trains dedicated
  models for TRENDING and RANGING. This catches regime-specific patterns that the
  combined model averages away (e.g., momentum indicators behave oppositely across
  regimes).
"""

import logging
import math
import numpy as np
from datetime import datetime, timedelta, timezone

log = logging.getLogger("MetaLabel")

MIN_SAMPLES           = 30   # minimum memories needed before ML kicks in (combined model)
MIN_SAMPLES_PER_REGIME = 15  # minimum per-regime to train a conditional model

_REGIME_ENCODING: dict[str, float] = {
    "TRENDING_UP":   1.0,
    "TRENDING_DOWN": -1.0,
    "RANGING":        0.5,
    "VOLATILE":       0.0,
}

_TRENDING_REGIMES = {"TRENDING_UP", "TRENDING_DOWN"}
_RANGING_REGIMES  = {"RANGING"}


def _encode_regime(regime_str: str) -> float:
    return _REGIME_ENCODING.get(regime_str, 0.0)


def _encode_side(direction) -> float:
    d = str(direction or "").upper()
    return 1.0 if d == "BUY" else -1.0 if d == "SELL" else 0.0


def _parse_ts(ts) -> datetime | None:
    if isinstance(ts, datetime):
        dt = ts
    else:
        try:
            dt = datetime.fromisoformat(str(ts))
        except Exception:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _time_features(dt: datetime) -> list[float]:
    hour = dt.hour + dt.minute / 60.0
    dow  = dt.weekday()
    return [math.sin(hour * 2 * math.pi / 24), math.cos(hour * 2 * math.pi / 24),
            math.sin(dow * 2 * math.pi / 7),   math.cos(dow * 2 * math.pi / 7)]


_CURRENT_FV_LEN = 21  # strategy.indicators.feature_vector() length after Tier 3 upgrade


def _build_extended_features(
    fv: list[float],
    ctx: dict | None = None,
    regime_override: float | None = None,
    fib_proximity: float = 0.0,
    cot_z_score:   float = 0.0,
    econ_surprise: float = 0.0,
    side:          float = 0.0,
    when:          datetime | None = None,
) -> list[float]:
    """
    Extend the primary technical feature vector with meta-only features.
    when: the trade's entry time (defaults to now, i.e. the moment of entry).
    """
    base = list(fv) if fv else [0.5] * _CURRENT_FV_LEN
    # Pad if the incoming vector is shorter than current standard (e.g. from old memories)
    if len(base) < _CURRENT_FV_LEN:
        base = base + [0.0] * (_CURRENT_FV_LEN - len(base))

    sin_hour, cos_hour, sin_dow, cos_dow = _time_features(
        _parse_ts(when) if when is not None else datetime.now(timezone.utc))

    if regime_override is not None:
        regime_feat = regime_override
    elif ctx:
        regime_str  = ctx.get("regime", {})
        if isinstance(regime_str, dict):
            regime_str = regime_str.get("regime", "")
        regime_feat = _encode_regime(str(regime_str))
    else:
        regime_feat = 0.0

    # Normalise extra features to [-1, 1]
    cot_norm  = max(-1.0, min(1.0, cot_z_score  / 2.0))
    econ_norm = max(-1.0, min(1.0, econ_surprise / 30.0))
    fib_feat  = max(0.0,  min(1.0, fib_proximity))

    return base + [sin_hour, cos_hour, sin_dow, cos_dow, regime_feat, fib_feat, cot_norm, econ_norm,
                   float(side)]


def _build_training_row(m: dict) -> tuple[list[float], float]:
    """
    Build extended feature vector from a stored memory.
    Features [21-23] come from stored context; default to 0.0 for old memories.
    Returns (features, regime_encoded).
    Old memories that predate Tier 3 (16-element fv) are zero-padded to _CURRENT_FV_LEN.
    """
    fv = list(m.get("feature_vector", [0.5] * _CURRENT_FV_LEN))
    # Pad old memories that have fewer features than the current vector length
    if len(fv) < _CURRENT_FV_LEN:
        fv = fv + [0.0] * (_CURRENT_FV_LEN - len(fv))
    dt = _parse_ts(m.get("entry_time") or m.get("timestamp", ""))
    if dt is None:
        dt = datetime(2000, 1, 5, 12, tzinfo=timezone.utc)   # neutral: Wed 12:00
    sin_hour, cos_hour, sin_dow, cos_dow = _time_features(dt)

    regime_str  = m.get("regime", "")
    regime_feat = _encode_regime(str(regime_str))

    # Extra features — populated by bot.py for new trades; 0.0 for old memories
    fib_prox  = float(m.get("fib_proximity", 0.0))
    cot_norm  = max(-1.0, min(1.0, float(m.get("cot_z_score",   0.0)) / 2.0))
    econ_norm = max(-1.0, min(1.0, float(m.get("econ_surprise", 0.0)) / 30.0))

    return (fv + [sin_hour, cos_hour, sin_dow, cos_dow, regime_feat, fib_prox, cot_norm, econ_norm,
                  _encode_side(m.get("direction"))], regime_feat)


_HOLDING_PERIOD_BARS = 24   # hours — concurrency window for uniqueness weighting.

# Assumed label length (entry → exit) when a sample has no recorded exit time.
# Matches the bot's MAX_HOLD_HOURS default (48h time exit), so purging is conservative.
_DEFAULT_LABEL_HOURS = 48
_MIN_TRAIN_PER_FOLD  = 10


def _purged_time_splits(starts: list, ends: list, y: np.ndarray,
                        n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward CV splits purged by TIME, not by row count (AFML ch. 7).

    Samples are sorted by entry time. Each fold tests one contiguous block and
    trains only on earlier samples whose label had finished (exit time) before
    the test block's first entry — so no training label overlaps a test label.

    The old TimeSeriesSplit(gap=24) skipped 24 *trades* instead of 24 hours and
    raised "Too many splits" for every sample count from 30 to ~147, so the ML
    gate silently fell back to the heuristic (verified).

    Folds whose train or test side lacks one of the two classes, or whose train
    side is smaller than _MIN_TRAIN_PER_FOLD, are dropped.
    """
    n = len(starts)
    if n_splits < 2 or n < 2 * n_splits:
        return []
    test_size = n // (n_splits + 1)
    splits = []
    for k in range(n_splits):
        test_lo = n - (n_splits - k) * test_size
        test_hi = test_lo + test_size if k < n_splits - 1 else n
        test_idx = np.arange(test_lo, test_hi)
        t0 = starts[test_lo]
        train_idx = np.array([i for i in range(test_lo)
                              if ends[i] is not None and t0 is not None and ends[i] < t0],
                             dtype=int)
        if len(train_idx) < _MIN_TRAIN_PER_FOLD:
            continue
        if len(set(y[train_idx])) < 2 or len(set(y[test_idx])) < 2:
            continue
        splits.append((train_idx, test_idx))
    return splits


def _compute_sample_weights(timestamps: list[str],
                             pnl_values: list[float],
                             symbols: list[str] | None = None,
                             holding_hours: int = _HOLDING_PERIOD_BARS) -> np.ndarray:
    """
    Sample uniqueness weighting (López de Prado AFML Ch. 4).

    Two trades are "concurrent" if their timestamps are within holding_hours of
    each other ON THE SAME SYMBOL. AFML uniqueness measures overlapping label
    windows on the same instrument's return path — trades on different pairs do
    not share a price path. The earlier cross-symbol version collapsed N_eff to
    ~3% of N once 20+ symbols from the same month were in memory (verified:
    N=222 → N_eff=5.8), effectively erasing the training signal's sample count.

    Weight = 1 / (number of same-symbol concurrent trades at that timestamp),
    multiplied by |pnl| (return attribution), normalized so sum(weights) = N.
    """
    n = len(timestamps)
    if n == 0:
        return np.ones(0)
    if symbols is None or len(symbols) != n:
        symbols = [""] * n   # legacy fallback: treat all as one instrument

    dts = []
    for ts in timestamps:
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dts.append(dt)
        except Exception:
            dts.append(None)

    hold_secs = holding_hours * 3600.0
    concurrency = np.ones(n)
    for i in range(n):
        if dts[i] is None:
            continue
        count = sum(
            1 for j in range(n)
            if dts[j] is not None
            and symbols[j] == symbols[i]
            and abs((dts[i] - dts[j]).total_seconds()) < hold_secs
        )
        concurrency[i] = max(1, count)

    uniqueness = 1.0 / concurrency

    # Return attribution: weight by |pnl| so high-impact trades count more
    abs_pnl = np.abs(np.array(pnl_values, dtype=float))
    abs_pnl = np.where(abs_pnl < 0.01, 0.01, abs_pnl)   # floor to avoid zero weights
    weights  = uniqueness * abs_pnl

    # Normalize so sum = N (LightGBM expects weight sum comparable to N)
    total = weights.sum()
    if total > 0:
        weights = weights * (n / total)

    n_eff = float(uniqueness.sum())
    log.info(f"Uniqueness weighting: N={n}, N_eff={n_eff:.1f} ({n_eff/n:.0%} of N)")
    return weights


def _fit_classifier(X: np.ndarray, y: np.ndarray,
                    clf_name_hint: str = "",
                    sample_weight: np.ndarray | None = None,
                    starts: list | None = None,
                    ends: list | None = None) -> object | None:
    """
    Fit a calibrated LightGBM (or RandomForest fallback) classifier.

    Uses time-purged walk-forward splits (_purged_time_splits): data must be
    sorted by entry time (ensured in train()); a training sample is used only if
    its label ended before the test block starts, so labels never overlap.

    Calibration method: 'sigmoid' (Platt scaling) is the default and correct choice
    at N < 300. Isotonic overfits badly at small N (research: arXiv:2002.10199).

    Returns None on error.
    """
    try:
        from sklearn.calibration import CalibratedClassifierCV
        import warnings
        warnings.filterwarnings("ignore")

        try:
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                min_child_samples=max(3, len(X) // 20),
                learning_rate=0.05,
                num_leaves=15,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            )
            clf_name = f"LightGBM{clf_name_hint}"
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=3,
                random_state=42,
                class_weight="balanced",
            )
            clf_name = f"RandomForest{clf_name_hint}"

        # Time-purged walk-forward splits (see _purged_time_splits).
        if len(set(y)) < 2:
            log.info(f"Meta-label{clf_name_hint}: only one outcome class in data — skipped.")
            return None
        if starts is None or ends is None:
            # No timing info: treat rows as 1h apart with the default label length.
            base   = datetime(2000, 1, 1, tzinfo=timezone.utc)
            starts = [base + timedelta(hours=i) for i in range(len(X))]
            ends   = [t + timedelta(hours=_DEFAULT_LABEL_HOURS) for t in starts]
        splits = []
        for n_try in (min(5, max(2, len(X) // 10)), 3, 2):
            splits = _purged_time_splits(starts, ends, y, n_try)
            if len(splits) >= 2:
                break
        if len(splits) < 2:
            log.info(
                f"Meta-label{clf_name_hint}: {len(X)} samples do not give 2 valid "
                f"purged CV folds yet — keeping heuristic."
            )
            return None
        n_splits = len(splits)

        # CalibratedClassifierCV with Platt (sigmoid) — correct for small N
        calibrated = CalibratedClassifierCV(clf, cv=splits, method="sigmoid")
        fit_kwargs = {}
        if sample_weight is not None and len(sample_weight) == len(X):
            fit_kwargs["sample_weight"] = sample_weight
        calibrated.fit(X, y, **fit_kwargs)

        win_rate = float(y.mean())
        log.info(
            f"Meta-label [{clf_name}] trained on {len(X)} samples | "
            f"base win rate: {win_rate:.1%} | "
            f"CV: time-purged walk-forward, {n_splits} folds"
        )
        return calibrated
    except Exception as e:
        log.warning(f"Meta-label training failed{clf_name_hint}: {e}")
        return None


class MetaLabelGate:
    def __init__(self):
        self._model          = None   # combined model
        self._model_trending = None   # TRENDING_UP / TRENDING_DOWN only
        self._model_ranging  = None   # RANGING only
        self._trained        = False
        self._n_samples      = 0
        # SGD online companion — updated per-trade, blended into predictions
        self._sgd            = None
        self._sgd_n          = 0
        self._sgd_sum        = None   # running sum for online mean normalisation
        self._sgd_sum_sq     = None   # running sum-of-squares for online std

    def train(self, memories: list[dict], training_lookback_days: int = 180) -> bool:
        """
        Train on historical memories from the memory store.
        memories: list of dicts with 'feature_vector', 'pnl_pct', 'outcome'
        training_lookback_days: only use memories from the last N days (walk-forward).

        Trains three models when enough data:
          combined (all regimes), trending-only, ranging-only.

        Returns True if at least the combined model was trained successfully.
        """
        if training_lookback_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=training_lookback_days)
            recent = []
            for m in memories:
                ts = m.get("timestamp", "")
                try:
                    dt = datetime.fromisoformat(ts)
                    # Normalise to UTC-aware so comparison with cutoff is safe
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt >= cutoff:
                        recent.append(m)
                except Exception:
                    recent.append(m)
            if len(recent) < len(memories):
                log.info(
                    f"Walk-forward filter: {len(recent)}/{len(memories)} memories "
                    f"within last {training_lookback_days} days"
                )
            memories = recent

        samples = [
            m for m in memories
            if m.get("feature_vector") and m.get("pnl_pct") is not None
        ]
        if len(samples) < MIN_SAMPLES:
            log.debug(f"Meta-label: only {len(samples)} samples (need {MIN_SAMPLES}). Using heuristic.")
            self._trained = False
            return False

        try:
            # ── Sort by ENTRY time (required by the purged splits) ──
            _far_past = datetime(1970, 1, 1, tzinfo=timezone.utc)
            samples.sort(key=lambda m: _parse_ts(m.get("entry_time") or m.get("timestamp", ""))
                         or _far_past)

            # ── Build combined feature matrix ─────────────────────────────
            X_rows, regimes, outcomes, timestamps, pnl_vals, syms = [], [], [], [], [], []
            for m in samples:
                row, regime_feat = _build_training_row(m)
                X_rows.append(row)
                regimes.append(regime_feat)
                outcomes.append(1 if (m.get("r_multiple", m.get("pnl_pct", 0)) > 0) else 0)
                timestamps.append(m.get("entry_time") or m.get("timestamp", ""))
                pnl_vals.append(abs(float(m.get("pnl_pct", 0.1)) or 0.1))
                syms.append(m.get("symbol", ""))

            X = np.array(X_rows)
            y = np.array(outcomes)
            r = np.array(regimes)

            # Label windows for purging: entry → exit (exit unknown → +48h).
            starts = [_parse_ts(t) for t in timestamps]
            ends   = []
            for m, t0 in zip(samples, starts):
                t1 = _parse_ts(m.get("exit_time")) if m.get("exit_time") else None
                if t1 is None and t0 is not None:
                    t1 = t0 + timedelta(hours=_DEFAULT_LABEL_HOURS)
                ends.append(t1)

            # ── Uniqueness weighting (AFML Ch.4) — correct for time-overlap leakage ──
            # Per-symbol concurrency: only same-instrument overlapping labels dilute
            # each other. Cross-symbol trades at the same hour are separate paths.
            sample_weights = _compute_sample_weights(timestamps, pnl_vals, symbols=syms)

            # ── Combined model ────────────────────────────────────────────
            model = _fit_classifier(X, y, sample_weight=sample_weights,
                                    starts=starts, ends=ends)
            if model is None:
                self._trained = False
                return False

            self._model     = model
            self._trained   = True
            self._n_samples = len(samples)

            # ── Regime-conditional models ─────────────────────────────────
            # TRENDING: TRENDING_UP (1.0) and TRENDING_DOWN (-1.0)
            trending_mask = np.isin(r, [1.0, -1.0])
            n_trending    = int(trending_mask.sum())
            if n_trending >= MIN_SAMPLES_PER_REGIME:
                m_trend = _fit_classifier(
                    X[trending_mask], y[trending_mask], "[TRENDING]",
                    sample_weight=sample_weights[trending_mask],
                    starts=[t for t, k in zip(starts, trending_mask) if k],
                    ends=[t for t, k in zip(ends, trending_mask) if k],
                )
                self._model_trending = m_trend
            else:
                log.info(f"Trending conditional model skipped ({n_trending} < {MIN_SAMPLES_PER_REGIME} samples)")
                self._model_trending = None

            # RANGING: regime_feat == 0.5
            ranging_mask = r == 0.5
            n_ranging    = int(ranging_mask.sum())
            if n_ranging >= MIN_SAMPLES_PER_REGIME:
                m_range = _fit_classifier(
                    X[ranging_mask], y[ranging_mask], "[RANGING]",
                    sample_weight=sample_weights[ranging_mask],
                    starts=[t for t, k in zip(starts, ranging_mask) if k],
                    ends=[t for t, k in zip(ends, ranging_mask) if k],
                )
                self._model_ranging = m_range
            else:
                log.info(f"Ranging conditional model skipped ({n_ranging} < {MIN_SAMPLES_PER_REGIME} samples)")
                self._model_ranging = None

            return True

        except ImportError:
            log.warning("scikit-learn not installed. Run: pip install scikit-learn")
            self._trained = False
            return False
        except Exception as e:
            log.warning(f"Meta-label training failed: {e}")
            self._trained = False
            return False

    def predict(
        self,
        feature_vector: list[float],
        ctx: dict | None = None,
        fib_proximity: float = 0.0,
        cot_z_score:   float = 0.0,
        econ_surprise: float = 0.0,
        direction:     str   = "",
    ) -> dict:
        """
        Predict probability of a profitable trade.
        direction: the side the agent desk chose ("BUY"/"SELL"). The model was
        trained with the side as a feature, so pass it for a meaningful answer.

        Uses regime-conditional model when available, falling back to combined, then heuristic.

        Returns:
            probability: float (0-1)
            use_ml:      bool
            confidence:  str (HIGH / MEDIUM / LOW)
            model_type:  str (combined / trending / ranging / heuristic)
            notes:       list[str]
        """
        ctx = ctx or {}

        regime_info = ctx.get("regime", {})
        if isinstance(regime_info, dict):
            regime_str = regime_info.get("regime", "")
        else:
            regime_str = str(regime_info)

        regime_feat = _encode_regime(regime_str)
        extended    = _build_extended_features(
            feature_vector, ctx,
            regime_override=regime_feat,
            fib_proximity=fib_proximity,
            cot_z_score=cot_z_score,
            econ_surprise=econ_surprise,
            side=_encode_side(direction),
        )

        if self._trained:
            # Select regime-conditional model when available
            model_to_use = None
            model_type   = "combined"

            if regime_str in _TRENDING_REGIMES and self._model_trending is not None:
                model_to_use = self._model_trending
                model_type   = "trending"
            elif regime_str in _RANGING_REGIMES and self._model_ranging is not None:
                model_to_use = self._model_ranging
                model_type   = "ranging"
            elif self._model is not None:
                model_to_use = self._model
                model_type   = "combined"

            if model_to_use is not None:
                try:
                    X    = np.array([extended])
                    prob = float(model_to_use.predict_proba(X)[0][1])

                    # Blend SGD companion when it has enough updates (≥10 trades)
                    notes = []
                    if (self._sgd is not None
                            and self._sgd_n >= 10
                            and self._sgd_sum is not None):
                        try:
                            mean    = self._sgd_sum / self._sgd_n
                            var     = np.maximum(
                                self._sgd_sum_sq / self._sgd_n - mean ** 2, 1e-8)
                            X_norm  = (np.array(extended) - mean) / np.sqrt(var)
                            sgd_p   = float(
                                self._sgd.predict_proba(X_norm.reshape(1, -1))[0][1])
                            prob    = 0.65 * prob + 0.35 * sgd_p
                            notes.append(
                                f"SGD blend (n={self._sgd_n}): tree={prob:.3f},"
                                f" sgd={sgd_p:.3f}"
                            )
                        except Exception:
                            pass

                    if prob > 0.65 or prob < 0.35:
                        conf = "HIGH"
                    elif prob > 0.55 or prob < 0.45:
                        conf = "MEDIUM"
                    else:
                        conf = "LOW"
                    return {
                        "probability": round(prob, 3),
                        "use_ml":      True,
                        "confidence":  conf,
                        "model_type":  model_type,
                        "n_samples":   self._n_samples,
                        "notes":       notes,
                    }
                except Exception as e:
                    log.warning(f"Meta-label prediction failed: {e}")

        # Heuristic fallback
        prob = self._heuristic_probability(feature_vector, ctx)
        return {
            "probability": round(prob, 3),
            "use_ml":      False,
            "confidence":  "LOW",
            "model_type":  "heuristic",
            "n_samples":   0,
            "notes":       [f"Heuristic only — need {MIN_SAMPLES} trades to train ML model"],
        }

    def online_update(
        self,
        feature_vector: list[float],
        is_win: bool,
        regime_str: str  = "",
        fib_proximity:  float = 0.0,
        cot_z_score:    float = 0.0,
        econ_surprise:  float = 0.0,
        direction:      str   = "",
        entry_time      = None,
    ) -> None:
        """
        Update the SGD companion model with a single completed trade result.
        Called immediately after each trade closes — no label-lag because
        we have the actual MT5 outcome in real time.

        The SGD companion learns directional bias shifts between batch retrains.
        Its probability is blended into predict() at 35% weight when ≥ 10 updates.
        """
        regime_feat = _encode_regime(regime_str)
        extended    = _build_extended_features(
            feature_vector,
            regime_override=regime_feat,
            fib_proximity=fib_proximity,
            cot_z_score=cot_z_score,
            econ_surprise=econ_surprise,
            side=_encode_side(direction),
            when=entry_time,      # entry time, not the close time this runs at
        )
        X = np.array(extended, dtype=float)

        # Running online mean/std for normalisation (never refit on future data)
        if self._sgd_sum is None:
            self._sgd_sum    = np.zeros(len(X))
            self._sgd_sum_sq = np.zeros(len(X))

        self._sgd_n      += 1
        self._sgd_sum    += X
        self._sgd_sum_sq += X * X

        mean   = self._sgd_sum / self._sgd_n
        var    = np.maximum(self._sgd_sum_sq / self._sgd_n - mean ** 2, 1e-8)
        X_norm = (X - mean) / np.sqrt(var)

        if self._sgd is None:
            try:
                from sklearn.linear_model import SGDClassifier
                self._sgd = SGDClassifier(
                    loss='log_loss', penalty='l2', alpha=1e-4,
                    learning_rate='adaptive', eta0=0.01,
                    class_weight='balanced', random_state=42,
                )
            except ImportError:
                log.debug("SGD online update skipped: scikit-learn not installed")
                return

        y = np.array([1 if is_win else 0])
        try:
            self._sgd.partial_fit(X_norm.reshape(1, -1), y, classes=np.array([0, 1]))
            log.debug(f"SGD online update #{self._sgd_n}: {'WIN' if is_win else 'LOSS'}")
        except Exception as e:
            log.debug(f"SGD partial_fit error: {e}")

    def should_trade(self, probability: float, min_prob: float = 0.52) -> bool:
        """Returns True if the meta-label probability clears the minimum threshold."""
        return probability >= min_prob

    @staticmethod
    def _heuristic_probability(fv: list[float], ctx: dict) -> float:
        """
        Heuristic probability based on indicator confluence.
        fv layout matches strategy.indicators.feature_vector() (16 floats).
        """
        if len(fv) < 10:
            return 0.5

        score = 0.5
        if fv[0] > 0.5:    score += 0.05   # above trend EMA
        if fv[1] > 0.5:    score += 0.04   # fast above slow
        rsi_norm = fv[4]
        if rsi_norm > 0.75: score -= 0.06  # overbought
        if rsi_norm < 0.25: score -= 0.06  # oversold extreme
        if fv[9] > 0.5:    score -= 0.04   # BB squeeze
        if fv[12] > 0.5 or fv[13] > 0.5: score += 0.05  # HTF aligned
        if fv[14] > 0.5 or fv[15] > 0.5: score -= 0.07  # RSI divergence
        vol_signal = ctx.get("volume_analysis", {}).get("signal", "")
        if vol_signal == "HIGH_VOLUME": score += 0.04
        atr_pct = fv[6] * 100
        if atr_pct > 85: score -= 0.05
        if atr_pct < 15: score -= 0.03

        sessions = set(ctx.get("sessions", []))
        if "LONDON" in sessions or "NEW_YORK" in sessions: score += 0.03

        # Regime bonus: trending regime improves trend-following probability
        regime_str = ""
        regime_info = ctx.get("regime", {})
        if isinstance(regime_info, dict):
            regime_str = regime_info.get("regime", "")
        if regime_str in _TRENDING_REGIMES:
            score += 0.03
        elif regime_str == "RANGING":
            score -= 0.02  # ranging slightly penalised for momentum signals

        return float(np.clip(score, 0.1, 0.9))
