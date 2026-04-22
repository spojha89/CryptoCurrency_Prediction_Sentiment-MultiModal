# lambdas/stream_processor/technical_analysis.py
"""
Technical Analysis module.
Computes RSI, MACD, Bollinger Bands, and VWAP from OHLCV data.
All computations are pure Python + numpy to keep Lambda size small.
"""
from __future__ import annotations
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_ohlcv_records(ohlcv_records: list[dict]) -> list[dict[str, float]]:
    normalized: list[dict[str, float]] = []
    skipped = 0

    for record in ohlcv_records:
        close = _safe_float(record.get("close"))
        if close is None:
            skipped += 1
            continue

        high = _safe_float(record.get("high", close), close)
        low = _safe_float(record.get("low", close), close)
        open_price = _safe_float(record.get("open", close), close)
        volume = _safe_float(record.get("volume", 0), 0.0)

        normalized.append({
            "open": open_price if open_price is not None else close,
            "close": close,
            "high": high if high is not None else close,
            "low": low if low is not None else close,
            "volume": volume if volume is not None else 0.0,
        })

    if skipped:
        logger.warning("Skipped %d OHLCV records missing close price", skipped)

    return normalized


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average using Wilder's smoothing method."""
    result = np.full_like(values, np.nan)
    k = 2.0 / (period + 1)
    # seed with SMA of first `period` values
    if len(values) < period:
        return result
    result[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


def _sma(values: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average using convolution."""
    result = np.full_like(values, np.nan)
    # Not enough data points for SMA with this period
    if len(values) < period:
        return result
    kernel = np.ones(period) / period
    sma = np.convolve(values, kernel, mode="valid")
    result[period - 1 :] = sma
    return result


def compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index (RSI).
    Formula: RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss over `period` bars.
    Uses Wilder's smoothing (EMA with alpha = 1/period).
    """
    if len(closes) < period + 1:
        return np.full(len(closes), np.nan)

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi = np.full(len(closes), np.nan)
    alpha = 1.0 / period

    # seed averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
        rs = avg_gain / avg_loss if avg_loss > 0 else 1e9
        rsi[i + 1] = 100 - (100 / (1 + rs))

    return rsi


def compute_macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD (Moving Average Convergence Divergence).
    Returns: (macd_line, signal_line, histogram)
    macd_line   = EMA(fast) - EMA(slow)
    signal_line = EMA(macd_line, signal)
    histogram   = macd_line - signal_line
    """
    ema_fast   = _ema(closes, fast)
    ema_slow   = _ema(closes, slow)
    macd_line  = ema_fast - ema_slow

    # mask NaN before computing signal EMA
    valid_mask = ~np.isnan(macd_line)
    signal_line = np.full_like(macd_line, np.nan)
    if valid_mask.sum() >= signal:
        valid_idx = np.where(valid_mask)[0]
        sig = _ema(macd_line[valid_idx], signal)
        signal_line[valid_idx] = sig

    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    closes: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands.
    Returns: (upper, middle, lower, bb_position)
    bb_position = (close - lower) / (upper - lower)  → 0 = at lower, 1 = at upper
    """
    middle = _sma(closes, period)
    rolling_std = np.full_like(closes, np.nan)
    for i in range(period - 1, len(closes)):
        rolling_std[i] = np.std(closes[i - period + 1 : i + 1])

    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std

    band_width = upper - lower
    bb_position = np.full_like(closes, 0.5)
    np.divide(
        closes - lower,
        band_width,
        out=bb_position,
        where=band_width > 0,
    )

    return upper, middle, lower, bb_position


def compute_vwap(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """
    Volume-Weighted Average Price (VWAP).
    VWAP = Σ(Typical Price × Volume) / Σ(Volume)
    Typical Price = (High + Low + Close) / 3
    Uses cumulative calculation from start of available data.
    """
    typical_price = (highs + lows + closes) / 3.0
    cum_tp_vol = np.cumsum(typical_price * volumes)
    cum_vol = np.cumsum(volumes)
    vwap = np.array(closes, copy=True)
    np.divide(cum_tp_vol, cum_vol, out=vwap, where=cum_vol > 0)
    return vwap


def compute_all_indicators(ohlcv_records: list[dict]) -> dict:
    """
    Given a list of OHLCV dicts sorted ascending by timestamp,
    compute all technical indicators and return the latest values.

    Returns a flat dict with the most recent bar's indicator values.
    """
    if not ohlcv_records:
        return {}

    normalized_records = _normalize_ohlcv_records(ohlcv_records)
    if not normalized_records:
        return {}

    closes  = np.array([r["close"] for r in normalized_records], dtype=float)
    highs   = np.array([r["high"] for r in normalized_records], dtype=float)
    lows    = np.array([r["low"] for r in normalized_records], dtype=float)
    volumes = np.array([r["volume"] for r in normalized_records], dtype=float)

    # RSI
    rsi_arr = compute_rsi(closes)

    # MACD
    macd_line, signal_line, histogram = compute_macd(closes)

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower, bb_pos = compute_bollinger_bands(closes)

    # VWAP
    vwap_arr = compute_vwap(closes, highs, lows, volumes)

    def _last(arr: np.ndarray) -> Optional[float]:
        """Return last non-nan value as Python float, or None."""
        valid = arr[~np.isnan(arr)]
        return round(float(valid[-1]), 6) if len(valid) > 0 else None

    return {
        "close":           round(float(closes[-1]), 6),
        "high":            round(float(highs[-1]), 6),
        "low":             round(float(lows[-1]), 6),
        "volume":          round(float(volumes[-1]), 2),
        "rsi":             _last(rsi_arr),
        "macd":            _last(macd_line),
        "macd_signal":     _last(signal_line),
        "macd_histogram":  _last(histogram),
        "bb_upper":        _last(bb_upper),
        "bb_middle":       _last(bb_middle),
        "bb_lower":        _last(bb_lower),
        "bb_position":     _last(bb_pos),
        "vwap":            _last(vwap_arr),
    }
