# lambdas/feature_engineer/feature_builder.py
"""
Feature Engineering Module.
Builds the joint feature tensor F(t) for each cryptocurrency:

  F(t) = [P(t), V(t), TA(t), S_twitter(t), S_reddit(t), S_news(t), FG(t), CrossCorr(t)]

Where:
  - P(t): price and returns at multiple intervals
  - V(t): volume and volume z-score
  - TA(t): RSI, MACD, BB position, VWAP deviation
  - S_*(t): platform sentiment scores and momentum deltas
  - FG(t): fear and greed regime features
  - CrossCorr(t): rolling Pearson correlation with BTC (for non-BTC coins)

All features are normalized. Stationarity is enforced by using log-returns
instead of raw prices (first-difference of log prices).
"""
from __future__ import annotations
import math
import logging
from datetime import datetime, timezone
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _log_return(prices: list[float]) -> list[float]:
    """Compute log-returns: r_t = log(P_t / P_{t-1})."""
    if len(prices) < 2:
        return [0.0] * len(prices)
    returns = [0.0]  # first return is 0
    for i in range(1, len(prices)):
        p_prev = prices[i - 1]
        p_curr = prices[i]
        if p_prev > 0 and p_curr > 0:
            returns.append(math.log(p_curr / p_prev))
        else:
            returns.append(0.0)
    return returns


def _z_score(values: list[float]) -> list[float]:
    """Standardize a series to zero mean, unit variance."""
    arr = np.array(values, dtype=float)
    if np.isnan(arr).all():
        return [0.0] * len(values)
    mean = np.nanmean(arr)
    std  = np.nanstd(arr)
    if std == 0:
        return [0.0] * len(values)
    return list((arr - mean) / std)


def _rolling_mean(values: list[float], window: int) -> list[float]:
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        subset = [v for v in values[start:i+1] if not math.isnan(v)]
        result.append(sum(subset) / len(subset) if subset else 0.0)
    return result


def _rolling_std(values: list[float], window: int) -> list[float]:
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        subset = [v for v in values[start:i+1] if not math.isnan(v)]
        if len(subset) < 2:
            result.append(0.0)
        else:
            mean = sum(subset) / len(subset)
            variance = sum((x - mean) ** 2 for x in subset) / len(subset)
            result.append(math.sqrt(variance))
    return result


def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient between two series."""
    n = min(len(x), len(y))
    if n < 5:
        return 0.0
    x_arr = np.array(x[:n], dtype=float)
    y_arr = np.array(y[:n], dtype=float)
    # mask NaN
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    if mask.sum() < 5:
        return 0.0
    x_valid = x_arr[mask]
    y_valid = y_arr[mask]
    if np.nanstd(x_valid) == 0 or np.nanstd(y_valid) == 0:
        return 0.0
    corr = np.corrcoef(x_valid, y_valid)
    return float(corr[0, 1]) if not np.isnan(corr[0, 1]) else 0.0


def _sentiment_momentum(scores: list[float], window: int = 4) -> float:
    """
    Sentiment momentum delta: difference between recent mean and
    older mean sentiment. Positive = improving sentiment trend.
    """
    if len(scores) < window * 2:
        return 0.0
    recent = scores[-window:]
    older  = scores[-window * 2 : -window]
    recent_mean = sum(recent) / len(recent)
    older_mean  = sum(older) / len(older)
    return recent_mean - older_mean


class FeatureBuilder:
    """
    Builds feature tensors from a 24-hour window of DynamoDB metrics records.
    Handles missing values through forward-fill interpolation.
    """

    WINDOWS_H = [1, 4, 12, 24]   # rolling window sizes in records (each = 15 min)
                                   # 1h=4 records, 4h=16, 12h=48, 24h=96 records

    def __init__(self, btc_records: Optional[list[dict]] = None):
        """
        btc_records: 24h metrics records for BTC-USD, used for cross-asset correlation.
        Pass None if building features for BTC itself.
        """
        self._btc_records = btc_records

    def build(self, coin_id: str, records: list[dict]) -> dict[str, float]:
        """
        Build the complete feature vector for the most recent time step.
        Returns a flat dict of feature_name → float value.
        """
        if len(records) < 2:
            logger.warning("Insufficient records for %s: %d", coin_id, len(records))
            return {}

        records = sorted(records, key=lambda r: r.get("timestamp_bucket", ""))
        features: dict[str, float] = {}

        # ── Price series ─────────────────────────────────────────────────────
        closes  = [_safe_float(r.get("close", 0)) for r in records]
        highs   = [_safe_float(r.get("high", r.get("close", 0))) for r in records]
        lows    = [_safe_float(r.get("low",  r.get("close", 0))) for r in records]
        volumes = [_safe_float(r.get("volume", 0)) for r in records]

        log_returns = _log_return(closes)
        vol_zscore  = _z_score(volumes)

        features["close"]       = closes[-1]
        features["log_return"]  = log_returns[-1]
        features["volume"]      = volumes[-1]
        features["volume_zscore"] = vol_zscore[-1]

        # Multi-horizon return features
        horizons_map = {"1h": 4, "4h": 16, "12h": 48}
        for label, n_periods in horizons_map.items():
            if len(closes) > n_periods:
                p_now   = closes[-1]
                p_prev  = closes[-n_periods - 1]
                if p_prev > 0 and p_now > 0:
                    features[f"return_{label}"] = math.log(p_now / p_prev)
                else:
                    features[f"return_{label}"] = 0.0
            else:
                features[f"return_{label}"] = 0.0

        # ── Rolling volatility (realized) ────────────────────────────────────
        for label, n_periods in horizons_map.items():
            recent_returns = log_returns[-n_periods:] if len(log_returns) >= n_periods else log_returns
            features[f"volatility_{label}"] = float(np.std(recent_returns)) if recent_returns else 0.0

        # ── Technical indicators (latest bar) ────────────────────────────────
        ta_fields = [
            "rsi", "macd", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower", "bb_position", "vwap",
        ]
        for field in ta_fields:
            val = None
            # walk backward to find most recent non-None value (forward-fill)
            for r in reversed(records):
                v = r.get(field)
                if v is not None:
                    val = _safe_float(v)
                    break
            features[field] = val if val is not None else 0.0

        # VWAP deviation: (close - vwap) / vwap
        vwap = features.get("vwap", 0.0)
        if vwap > 0:
            features["vwap_deviation"] = (closes[-1] - vwap) / vwap
        else:
            features["vwap_deviation"] = 0.0

        # RSI overbought/oversold flags
        features["rsi_overbought"] = 1.0 if features.get("rsi", 50) > 70 else 0.0
        features["rsi_oversold"]   = 1.0 if features.get("rsi", 50) < 30 else 0.0

        # MACD crossover signal
        macd_hist = features.get("macd_histogram", 0.0)
        prev_hists = [_safe_float(r.get("macd_histogram", 0)) for r in records[-5:]]
        features["macd_bullish_cross"] = 1.0 if (len(prev_hists) >= 2
            and prev_hists[-2] < 0 and macd_hist > 0) else 0.0
        features["macd_bearish_cross"] = 1.0 if (len(prev_hists) >= 2
            and prev_hists[-2] > 0 and macd_hist < 0) else 0.0

        # ── Sentiment features ───────────────────────────────────────────────
        for platform in ("twitter", "news"):
            field = f"sentiment_{platform}"
            count_field = f"{platform}_count"
            scores_series = []
            count_series = []
            for r in records:
                v = r.get(field)
                scores_series.append(_safe_float(v) if v is not None else float("nan"))
                count_series.append(_safe_float(r.get(count_field, 0)))

            observed_mask = [not math.isnan(s) for s in scores_series]
            filled = []
            last_valid = None
            for s in scores_series:
                if not math.isnan(s):
                    last_valid = s
                filled.append(last_valid if last_valid is not None else float("nan"))

            observed_values = [s for s in scores_series if not math.isnan(s)]
            recent_filled = [0.0 if math.isnan(s) else s for s in filled]
            recent_counts = count_series[-16:]

            features[f"sentiment_{platform}"] = observed_values[-1] if observed_values else 0.0
            features[f"sentiment_{platform}_momentum"] = _sentiment_momentum(recent_filled)
            features[f"sentiment_{platform}_std"] = (
                float(np.std(recent_filled[-16:])) if len(recent_filled) >= 4 else 0.0
            )
            features[f"sentiment_{platform}_available"] = 1.0 if observed_mask[-1] else 0.0
            features[f"sentiment_{platform}_coverage_4h"] = (
                sum(1 for seen in observed_mask[-16:] if seen) / min(len(observed_mask), 16)
            )
            features[count_field] = count_series[-1] if count_series else 0.0
            features[f"{platform}_count_4h"] = float(sum(recent_counts))
            if observed_values:
                age = 0
                for seen in reversed(observed_mask):
                    if seen:
                        break
                    age += 1
                features[f"sentiment_{platform}_age_buckets"] = float(age)
            else:
                features[f"sentiment_{platform}_age_buckets"] = float(len(records))

        # Composite sentiment (equal-weighted average of available platforms)
        sentiment_vals = [
            features[f"sentiment_{p}"]
            for p in ("twitter", "news")
            if features.get(f"sentiment_{p}_available", 0.0) > 0
        ]
        features["sentiment_composite"] = (
            sum(sentiment_vals) / len(sentiment_vals) if sentiment_vals else 0.0
        )

        # ── Cross-asset correlation (non-BTC coins) ───────────────────────────
        fg_series = []
        for r in records:
            v = r.get("fear_greed_value")
            fg_series.append(_safe_float(v) if v is not None else float("nan"))

        fg_filled = []
        last_fg = 50.0
        for value in fg_series:
            if not math.isnan(value):
                last_fg = value
            fg_filled.append(last_fg)

        fg_value = fg_filled[-1]
        fg_counts = [_safe_float(r.get("fear_greed_count", 0)) for r in records]
        features["fear_greed_value"] = fg_value
        features["fear_greed_normalized"] = (fg_value - 50.0) / 50.0
        features["fear_greed_momentum"] = _sentiment_momentum(fg_filled)
        features["fear_greed_std"] = float(np.std(fg_filled[-16:])) if len(fg_filled) >= 4 else 0.0
        features["fear_greed_extreme_fear"] = 1.0 if fg_value <= 25 else 0.0
        features["fear_greed_extreme_greed"] = 1.0 if fg_value >= 75 else 0.0
        features["fear_greed_available"] = 1.0 if not math.isnan(fg_series[-1]) else 0.0
        features["fear_greed_coverage_4h"] = (
            sum(1 for value in fg_series[-16:] if not math.isnan(value)) / min(len(fg_series), 16)
        )
        features["fear_greed_count"] = fg_counts[-1] if fg_counts else 0.0

        gt_series = []
        for r in records:
            v = r.get("google_trends_value")
            gt_series.append(_safe_float(v) if v is not None else float("nan"))

        gt_filled = []
        last_gt = 0.0
        for value in gt_series:
            if not math.isnan(value):
                last_gt = value
            gt_filled.append(last_gt)

        gt_value = gt_filled[-1]
        gt_counts = [_safe_float(r.get("google_trends_count", 0)) for r in records]
        features["google_trends_value"] = gt_value
        features["google_trends_normalized"] = gt_value / 100.0
        features["google_trends_momentum"] = _sentiment_momentum(gt_filled)
        features["google_trends_std"] = float(np.std(gt_filled[-16:])) if len(gt_filled) >= 4 else 0.0
        features["google_trends_high_interest"] = 1.0 if gt_value >= 75 else 0.0
        features["google_trends_available"] = 1.0 if not math.isnan(gt_series[-1]) else 0.0
        features["google_trends_coverage_4h"] = (
            sum(1 for value in gt_series[-16:] if not math.isnan(value)) / min(len(gt_series), 16)
        )
        features["google_trends_count"] = gt_counts[-1] if gt_counts else 0.0

        if self._btc_records and coin_id != "BTC-USD":
            btc_closes = [_safe_float(r.get("close", 0)) for r in self._btc_records]
            btc_returns = _log_return(btc_closes)
            # align lengths
            min_len = min(len(log_returns), len(btc_returns))
            features["btc_correlation_1h"]  = _pearson_correlation(
                log_returns[-min(4, min_len):],
                btc_returns[-min(4, min_len):],
            )
            features["btc_correlation_4h"]  = _pearson_correlation(
                log_returns[-min(16, min_len):],
                btc_returns[-min(16, min_len):],
            )
            features["btc_correlation_24h"] = _pearson_correlation(
                log_returns[-min_len:],
                btc_returns[-min_len:],
            )
        else:
            features["btc_correlation_1h"]  = 1.0 if coin_id == "BTC-USD" else 0.0
            features["btc_correlation_4h"]  = 1.0 if coin_id == "BTC-USD" else 0.0
            features["btc_correlation_24h"] = 1.0 if coin_id == "BTC-USD" else 0.0

        # ── Time features (cyclical encoding) ────────────────────────────────
        # Use sine/cosine encoding to preserve cyclical nature of time
        last_ts_str = records[-1].get("timestamp_bucket", "")
        try:
            last_dt = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))
        except ValueError:
            last_dt = datetime.now(timezone.utc)

        hour_of_day = last_dt.hour + last_dt.minute / 60.0
        day_of_week = last_dt.weekday()

        features["hour_sin"] = math.sin(2 * math.pi * hour_of_day / 24)
        features["hour_cos"] = math.cos(2 * math.pi * hour_of_day / 24)
        features["day_sin"]  = math.sin(2 * math.pi * day_of_week / 7)
        features["day_cos"]  = math.cos(2 * math.pi * day_of_week / 7)
        features["is_weekend"] = 1.0 if day_of_week >= 5 else 0.0

        # ── Clean up: replace NaN/Inf with 0 ────────────────────────────────
        for k in list(features.keys()):
            v = features[k]
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                features[k] = 0.0

        return features
