"""
Gaussian Hidden Markov Model — Latent Volatility Regime Detector.

Complements the ADX+Hurst+ER regime classifier (which detects TREND direction)
with a VOLATILITY STRESS axis that is orthogonal to directional regime:

  ADX/Hurst/ER  → "is the market trending or ranging?" (direction)
  HMM states    → "is the market calm, normal, or stressed?" (vol stress)

Why HMM over a simpler percentile threshold?
  → Provides STATE TRANSITION PROBABILITIES (how likely a vol regime change next bar)
  → Handles serial dependence in vol (GARCH-like clustering)
  → 3 stable states (LOW/MED/HIGH_VOL) with regime-aware position sizing

Production design (stability-critical):
  - 3 states: LOW_VOL, MED_VOL, HIGH_VOL
  - Feature vector: [log_return, abs_log_return, yz_vol_rank] (3 columns)
  - n_init=10 random starts → picks highest log-likelihood → avoids local optima
  - States sorted by mean abs_return → stable IDs across refits (no label-switching)
  - Refit daily (not in 60s loop — inference only in the loop, <1ms)
  - Requires hmmlearn: pip install hmmlearn

Downstream use in bot.py:
  - hmm_result["label"] → passed to LLM agents as additional context
  - hmm_result["transition_to_high_vol"] > 0.3 → tighten SL, reduce size
  - hmm_result["label"] == "HIGH_VOL" and regime == "VOLATILE" → skip trade
"""

import numpy as np
import logging
import os
import json

log = logging.getLogger("HMMRegime")

_N_STATES   = 3
_COV_TYPE   = "diag"   # full overfits; diag is robust at small N
_N_ITER     = 100
_N_INIT     = 10       # multiple random restarts → avoid EM local optima
_MIN_BARS   = 250      # minimum bars to fit reliably
_STATE_LABELS = ["LOW_VOL", "MED_VOL", "HIGH_VOL"]   # sorted by ascending vol


def _build_features(df) -> tuple:
    """
    Build (X, mean, std) feature matrix from price DataFrame.
    Returns (None, None, None) if insufficient data or vol estimator unavailable.
    """
    try:
        from strategy.vol_estimators import yang_zhang, vol_percentile_rank
    except ImportError:
        # Fallback: use ATR-based vol if yang_zhang not available
        yang_zhang = None

    close  = df["close"].values.astype(float)
    log_ret = np.diff(np.log(close + 1e-10))
    abs_ret = np.abs(log_ret)

    # Append one NaN at the start to match length
    log_ret = np.concatenate([[np.nan], log_ret])
    abs_ret = np.concatenate([[np.nan], abs_ret])

    if yang_zhang is not None:
        try:
            yz   = yang_zhang(df, window=20)
            rank = vol_percentile_rank(yz, window=min(len(df), 200)) / 100.0
            vol_feat = rank.fillna(0.5).values
        except Exception:
            vol_feat = pd.Series(abs_ret).rolling(20).rank(pct=True).fillna(0.5).values \
                if hasattr(abs_ret, "__len__") else np.full(len(log_ret), 0.5)
    else:
        import pandas as pd
        vol_feat = pd.Series(abs_ret).rolling(20).rank(pct=True).fillna(0.5).values

    X = np.column_stack([log_ret, abs_ret, vol_feat])

    # Drop NaN rows (first row)
    mask = ~np.isnan(X).any(axis=1)
    X    = X[mask]

    if len(X) < _MIN_BARS:
        return None, None, None

    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std  = np.where(std < 1e-10, 1.0, std)
    return (X - mean) / std, mean, std


import pandas as pd   # needed for vol_feat fallback


class HMMRegimeModel:
    """Wraps GaussianHMM with stability conventions for production use."""

    def __init__(self):
        self._model    = None
        self._mean     = None
        self._std      = None
        self._fitted   = False
        self._n_bars   = 0
        # maps raw HMM state int → stable label
        self._state_map: dict[int, str] = {0: "LOW_VOL", 1: "MED_VOL", 2: "HIGH_VOL"}

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, df) -> bool:
        """Fit HMM on a price DataFrame. Returns True on success."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            log.debug("HMMRegime skipped: pip install hmmlearn to activate")
            return False

        X, mean, std = _build_features(df)
        if X is None:
            return False

        best_model, best_score = None, -np.inf
        for seed in range(_N_INIT):
            try:
                m = GaussianHMM(
                    n_components    = _N_STATES,
                    covariance_type = _COV_TYPE,
                    n_iter          = _N_ITER,
                    random_state    = seed,
                    verbose         = False,
                )
                m.fit(X)
                s = m.score(X)
                if s > best_score:
                    best_score, best_model = s, m
            except Exception:
                continue

        if best_model is None:
            log.warning("HMM: all n_init attempts failed")
            return False

        self._model  = best_model
        self._mean   = mean
        self._std    = std
        self._n_bars = len(X)

        # Sort states by mean abs_return (feature[1]) → stable IDs
        state_vols = [float(self._model.means_[s, 1]) for s in range(_N_STATES)]
        order      = np.argsort(state_vols)   # ascending → LOW=0, MED=1, HIGH=2
        self._state_map = {int(order[i]): _STATE_LABELS[i] for i in range(_N_STATES)}

        self._fitted = True
        log.info(
            f"HMM fitted on {self._n_bars} bars | "
            f"log-likelihood={best_score:.2f} | state_map={self._state_map}"
        )
        return True

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, df) -> dict:
        """
        Return current HMM regime for the last bar in df.

        Keys:
          label                — LOW_VOL | MED_VOL | HIGH_VOL | UNKNOWN
          probs                — [p_LOW, p_MED, p_HIGH] posteriors for last bar
          transition_to_high   — P(next bar = HIGH_VOL)
          fitted               — bool
        """
        _default = {
            "label": "UNKNOWN", "probs": [1/3, 1/3, 1/3],
            "transition_to_high": 0.33, "fitted": False,
        }

        if not self._fitted or self._model is None:
            return _default

        try:
            X, _, _ = _build_features(df)
            if X is None or len(X) < 2:
                return _default

            # Normalize with training stats
            X_norm = (X - self._mean) / self._std

            proba     = self._model.predict_proba(X_norm)   # shape (T, n_states)
            last_proba = proba[-1]                           # posteriors for last bar

            # Reorder to [LOW_VOL, MED_VOL, HIGH_VOL] using state_map
            inv_map = {v: k for k, v in self._state_map.items()}
            ordered_probs = [
                float(last_proba[inv_map[lbl]])
                for lbl in _STATE_LABELS
                if lbl in inv_map
            ]
            if len(ordered_probs) != _N_STATES:
                ordered_probs = last_proba.tolist()

            # Current state
            curr_raw   = int(np.argmax(last_proba))
            curr_label = self._state_map.get(curr_raw, "UNKNOWN")

            # P(next = HIGH_VOL) = Σ_s P(current=s) × transmat[s, HIGH_VOL]
            high_state_raw = inv_map.get("HIGH_VOL", _N_STATES - 1)
            trans_to_high  = float(sum(
                float(last_proba[s]) * float(self._model.transmat_[s, high_state_raw])
                for s in range(_N_STATES)
            ))

            return {
                "label":              curr_label,
                "probs":              [round(p, 3) for p in ordered_probs],
                "transition_to_high": round(trans_to_high, 3),
                "fitted":             True,
            }

        except Exception as e:
            log.debug(f"HMM predict error: {e}")
            return _default

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, path: str = "data/hmm_state.json") -> None:
        if not self._fitted:
            return
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            payload = {
                "means":     self._model.means_.tolist(),
                "covars":    self._model.covars_.tolist(),
                "transmat":  self._model.transmat_.tolist(),
                "startprob": self._model.startprob_.tolist(),
                "mean":      self._mean.tolist(),
                "std":       self._std.tolist(),
                "state_map": {str(k): v for k, v in self._state_map.items()},
                "n_bars":    self._n_bars,
            }
            with open(path, "w") as f:
                json.dump(payload, f)
            log.debug(f"HMM state saved → {path}")
        except Exception as e:
            log.debug(f"HMM save error: {e}")
