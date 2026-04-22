"""
Enhanced Sentiment Aggregation Module
Combines VADER lexicon scoring with weighted composite indices for robust sentiment signals.

Features:
  1. VADER sentiment analysis for Twitter/text data
  2. Weighted composite sentiment index (WCSI)
  3. Platform-specific confidence scoring
  4. Trend-based sentiment smoothing
  5. Category-based sentiment breakdown (bullish/bearish/neutral)
"""
from __future__ import annotations
import logging
import math
from typing import Optional
import numpy as np

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

logger = logging.getLogger(__name__)

# ─── VADER Setup ──────────────────────────────────────────────────────────────
_vader = None

def _get_vader():
    global _vader
    if _vader is None and VADER_AVAILABLE:
        try:
            _vader = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.warning("VADER initialization failed: %s", e)
    return _vader


# ─── Sentiment Normalization ──────────────────────────────────────────────────
def normalize_sentiment(
    positive: float,
    negative: float,
    neutral: float = None,
) -> dict[str, float]:
    """
    Normalize sentiment scores to [0, 1] range with probabilistic interpretation.
    
    Args:
        positive: bullish probability (higher = more positive)
        negative: bearish probability (higher = more negative)
        neutral: neutral probability (if None, computed as 1 - positive - negative)
    
    Returns:
        dict with normalized {positive, negative, neutral} probabilities
    """
    total = positive + negative + (neutral or 0)
    if total <= 0:
        return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
    
    if neutral is None:
        # Derived: neutral = 1 - positive - negative
        neutral = 1.0 - (positive + negative)
    
    # Clip to valid range
    p = max(0, min(1, positive / total))
    n = max(0, min(1, negative / total))
    u = max(0, min(1, neutral / total))
    
    # Renormalize to ensure sum = 1
    total_norm = p + n + u
    if total_norm > 0:
        return {
            "positive": round(p / total_norm, 4),
            "negative": round(n / total_norm, 4),
            "neutral": round(u / total_norm, 4),
        }
    else:
        return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}


# ─── VADER Sentiment Scoring ──────────────────────────────────────────────────
def vader_sentiment(text: str) -> dict[str, float]:
    """
    Score text using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    VADER is optimized for social media and crypto text.
    
    Args:
        text: Input text to analyze
    
    Returns:
        dict with {positive, negative, neutral, compound} scores
    """
    if not VADER_AVAILABLE:
        logger.warning("VADER not available, returning neutral sentiment")
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
    
    vader = _get_vader()
    if vader is None:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
    
    try:
        scores = vader.polarity_scores(text)
        return {
            "positive": round(scores["pos"], 4),
            "negative": round(scores["neg"], 4),
            "neutral": round(scores["neu"], 4),
            "compound": round(scores["compound"], 4),  # -1 to +1
        }
    except Exception as e:
        logger.warning("VADER scoring error: %s", e)
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}


# ─── Crypto-Specific Sentiment Indicators ─────────────────────────────────────
def extract_crypto_signals(text: str, platform: str = "twitter") -> dict[str, float]:
    """
    Extract crypto-specific bullish/bearish signals from text.
    
    Args:
        text: Input text
        platform: "twitter", "news", or "reddit"
    
    Returns:
        dict with signal indicators
    """
    text_lower = text.lower()
    
    # Bullish signals
    bullish_keywords = {
        "bullish", "bull run", "moon", "lambo", "pump", "gains",
        "breakout", "surge", "rally", "strong", "bullish",
        "buy", "hodl", "hold", "long", "bull", "crypto boom",
        "parabolic", "uptrend", "support", "go up",
    }
    
    # Bearish signals
    bearish_keywords = {
        "bearish", "bear market", "crash", "dump", "losses", "rekt",
        "short", "downtrend", "resistance", "sell", "panic",
        "bubble", "collapse", "plunge", "decline", "drop",
        "fear", "crisis", "hack", "scam", "dead",
    }
    
    bullish_count = sum(1 for keyword in bullish_keywords if keyword in text_lower)
    bearish_count = sum(1 for keyword in bearish_keywords if keyword in text_lower)
    
    # Platform-specific weighting
    platform_weights = {
        "twitter": 1.0,
        "reddit": 0.8,
        "news": 1.2,
    }
    weight = platform_weights.get(platform, 1.0)
    
    return {
        "bullish_signals": bullish_count * weight,
        "bearish_signals": bearish_count * weight,
        "signal_ratio": (bullish_count - bearish_count) / max(bullish_count + bearish_count, 1),
    }


# ─── Weighted Composite Sentiment Index (WCSI) ────────────────────────────────
class WeightedCompositeIndex:
    """
    Combines multiple sentiment sources into a single robust index.
    Weights based on:
      - Historical accuracy/reliability
      - Recency
      - Data quality/availability
      - Platform-specific characteristics
    """
    
    # Platform reliability weights (learned from historical correlation with prices)
    PLATFORM_WEIGHTS = {
        "twitter": 0.35,      # Higher alpha in price prediction
        "news": 0.35,         # Strong lagging indicator
        "fear_greed": 0.20,   # Market regime indicator
        "google_trends": 0.10, # Weak but complementary
    }
    
    def __init__(self):
        self.platform_scores = {}
        self.last_update = None
    
    def update(
        self,
        twitter_score: float,
        twitter_count: int,
        news_score: float,
        news_count: int,
        fear_greed_score: float,
        google_trends_score: float,
    ) -> dict[str, float]:
        """
        Update composite index with latest platform scores.
        
        Args:
            twitter_score: [-1, 1] sentiment from Twitter
            twitter_count: number of tweets in window
            news_score: [-1, 1] sentiment from news
            news_count: number of news articles in window
            fear_greed_score: [0, 100] Fear & Greed index
            google_trends_score: [0, 100] Google Trends normalied score
        
        Returns:
            dict with composite index and component scores
        """
        # Confidence scores based on data availability
        twitter_confidence = min(1.0, twitter_count / 50) if twitter_count >= 0 else 0.0
        news_confidence = min(1.0, news_count / 20) if news_count >= 0 else 0.0
        
        # Normalize to [-1, 1] range
        twitter_norm = twitter_score if -1 <= twitter_score <= 1 else 0.0
        news_norm = news_score if -1 <= news_score <= 1 else 0.0
        fg_norm = (fear_greed_score - 50) / 50 if 0 <= fear_greed_score <= 100 else 0.0
        gt_norm = (google_trends_score - 50) / 50 if 0 <= google_trends_score <= 100 else 0.0
        
        # Weighted average with confidence adjustment
        weighted_sum = (
            self.PLATFORM_WEIGHTS["twitter"] * twitter_norm * twitter_confidence +
            self.PLATFORM_WEIGHTS["news"] * news_norm * news_confidence +
            self.PLATFORM_WEIGHTS["fear_greed"] * fg_norm +
            self.PLATFORM_WEIGHTS["google_trends"] * gt_norm
        )
        
        # Total weight (accounting for missing data)
        total_weight = (
            self.PLATFORM_WEIGHTS["twitter"] * twitter_confidence +
            self.PLATFORM_WEIGHTS["news"] * news_confidence +
            self.PLATFORM_WEIGHTS["fear_greed"] +
            self.PLATFORM_WEIGHTS["google_trends"]
        )
        
        # Composite score: [-1, 1] with 0 = neutral
        composite = weighted_sum / total_weight if total_weight > 0 else 0.0
        composite = max(-1.0, min(1.0, composite))
        
        # Convert to probability space [0, 1]
        composite_prob = (composite + 1.0) / 2.0
        
        result = {
            "composite": round(composite, 4),
            "composite_prob": round(composite_prob, 4),
            "twitter_contribution": round(
                self.PLATFORM_WEIGHTS["twitter"] * twitter_norm * twitter_confidence / total_weight * 100 if total_weight > 0 else 0,
                2
            ),
            "news_contribution": round(
                self.PLATFORM_WEIGHTS["news"] * news_norm * news_confidence / total_weight * 100 if total_weight > 0 else 0,
                2
            ),
            "fear_greed_contribution": round(
                self.PLATFORM_WEIGHTS["fear_greed"] * fg_norm / total_weight * 100 if total_weight > 0 else 0,
                2
            ),
            "google_trends_contribution": round(
                self.PLATFORM_WEIGHTS["google_trends"] * gt_norm / total_weight * 100 if total_weight > 0 else 0,
                2
            ),
            "data_confidence": round(total_weight, 2),
        }
        
        self.platform_scores = result
        return result
    
    def get_signal_strength(self) -> float:
        """
        Get confidence in current sentiment signal based on data availability.
        Returns [0, 1] where 1 = high confidence (all platforms available)
        """
        return self.platform_scores.get("data_confidence", 0.0) / sum(self.PLATFORM_WEIGHTS.values())


# ─── Sentiment Smoothing ──────────────────────────────────────────────────────
def smooth_sentiment(
    scores: list[float],
    window: int = 4,
    method: str = "exponential",
) -> list[float]:
    """
    Apply smoothing to reduce noise in sentiment time series.
    
    Args:
        scores: List of sentiment scores
        window: Smoothing window size
        method: "exponential", "moving_average", or "ema"
    
    Returns:
        Smoothed sentiment scores
    """
    if len(scores) < 2:
        return scores
    
    scores_arr = np.array(scores, dtype=float)
    
    if method == "moving_average":
        # Simple moving average
        smoothed = []
        for i in range(len(scores_arr)):
            start = max(0, i - window + 1)
            window_data = scores_arr[start:i+1]
            smoothed.append(float(np.nanmean(window_data)))
        return smoothed
    
    elif method == "exponential" or method == "ema":
        # Exponential moving average
        alpha = 2 / (window + 1)
        smoothed = [scores_arr[0]]
        for i in range(1, len(scores_arr)):
            ema = alpha * scores_arr[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(ema)
        return smoothed
    
    else:
        return scores


# ─── Sentiment Momentum ───────────────────────────────────────────────────────
def compute_momentum(
    scores: list[float],
    short_window: int = 4,
    long_window: int = 12,
) -> dict[str, float]:
    """
    Compute sentiment momentum using momentum indicators.
    
    Args:
        scores: Time series of sentiment scores
        short_window: Short-term window (e.g., 1 hour)
        long_window: Long-term window (e.g., 3 hours)
    
    Returns:
        dict with momentum metrics
    """
    if len(scores) < long_window:
        return {"momentum": 0.0, "trend": "neutral", "acceleration": 0.0}
    
    scores_arr = np.array(scores, dtype=float)
    
    # Recent vs historical mean
    recent_mean = float(np.mean(scores_arr[-short_window:]))
    older_mean = float(np.mean(scores_arr[-long_window:-short_window]))
    
    momentum = recent_mean - older_mean
    
    # Trend classification
    if momentum > 0.1:
        trend = "improving"
    elif momentum < -0.1:
        trend = "deteriorating"
    else:
        trend = "neutral"
    
    # Acceleration: rate of change of momentum
    if len(scores_arr) >= long_window + short_window:
        prev_recent = float(np.mean(scores_arr[-long_window:-short_window*2]))
        prev_older = float(np.mean(scores_arr[-long_window*2:-long_window]))
        prev_momentum = prev_recent - prev_older
        acceleration = momentum - prev_momentum
    else:
        acceleration = 0.0
    
    return {
        "momentum": round(momentum, 4),
        "trend": trend,
        "acceleration": round(acceleration, 4),
    }
