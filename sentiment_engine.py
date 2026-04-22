# lambdas/stream_processor/sentiment_engine.py
"""
Sentiment Engine.
Two parallel pipelines:
  1. Bedrock (Claude Haiku) - primary transformer-based scoring
  2. VADER - lexicon-based baseline for comparison

Implements the weighted sentiment score formula from the capstone:
  s(t) = Σ(w_i × (P(pos_i) − P(neg_i))) / Σ(w_i)
  where w_i = log(1 + likes + retweets + comments)
"""
from __future__ import annotations
import json
import logging
import re
import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Bedrock client — reused across invocations
_bedrock = None

def _get_bedrock():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-2"))
    return _bedrock


# ── Text cleaning ──────────────────────────────────────────────────────────────
_URL_RE       = re.compile(r"https?://\S+")
_MENTION_RE   = re.compile(r"@\w+")
_HASHTAG_RE   = re.compile(r"#(\w+)")
_MULTI_WS_RE  = re.compile(r"\s{2,}")

def preprocess_text(text: str) -> str:
    """
    NLP preprocessing pipeline:
    1. URL removal
    2. @mention normalization
    3. Hashtag normalization (keep word, remove #)
    4. Whitespace normalization
    5. Lowercase
    Emojis are kept — Bedrock handles them well.
    """
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r"\1", text)   # "#bitcoin" → "bitcoin"
    text = _MULTI_WS_RE.sub(" ", text)
    return text.strip().lower()


# ── Engagement weight ──────────────────────────────────────────────────────────
import math

DEFAULT_SENTIMENT = {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
BEDROCK_BATCH_SIZE = 5

def engagement_weight(likes: int, retweets: int, comments: int) -> float:
    """
    Compute engagement weight: log(1 + total_engagement).
    Logarithmic scaling prevents viral posts from dominating.
    Minimum weight = 1.0 (log(1 + 0) + 1 = 1).
    """
    total = likes + retweets + comments
    return math.log1p(total) + 1.0  # +1 ensures all posts have positive weight


# ── Bedrock sentiment scoring ──────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a financial sentiment analyzer specializing in cryptocurrency markets.
For the given text, output ONLY a JSON object with this exact structure:
{"positive": 0.0, "negative": 0.0, "neutral": 0.0}
where the three values sum to 1.0.
- positive: probability the text expresses bullish/positive crypto market sentiment
- negative: probability the text expresses bearish/negative crypto market sentiment
- neutral: probability the text is neutral or unrelated to market direction
Output ONLY the JSON. No explanation."""


def _default_sentiment() -> dict[str, float]:
    return dict(DEFAULT_SENTIMENT)


def _chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[i:i + size] for i in range(0, len(values), size)]


def _normalize_scores(scores: dict) -> dict[str, float]:
    if not isinstance(scores, dict):
        raise ValueError("Sentiment payload is not an object")

    normalized = {}
    for key in ("positive", "negative", "neutral"):
        value = scores.get(key)
        if value is None:
            raise ValueError(f"Missing sentiment key: {key}")
        normalized[key] = max(0.0, float(value))

    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("Sentiment probabilities sum to zero")

    return {
        key: round(value / total, 6)
        for key, value in normalized.items()
    }


def _parse_single_response(raw: str) -> dict[str, float]:
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    return _normalize_scores(json.loads(cleaned))


def _build_batch_prompt(texts: list[str]) -> str:
    items = [
        {"id": idx, "text": text[:400]}
        for idx, text in enumerate(texts)
    ]
    return (
        "Classify each crypto-related text.\n"
        "Return ONLY a JSON array.\n"
        "Each element must be an object with keys: id, positive, negative, neutral.\n"
        "Probabilities must be numeric and sum to 1.0.\n"
        f"Inputs: {json.dumps(items, ensure_ascii=True)}"
    )


def _parse_batch_response(raw: str, batch_size: int) -> list[dict[str, float]]:
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    payload = json.loads(cleaned)
    if not isinstance(payload, list):
        raise ValueError("Batch response is not a list")

    by_id: dict[int, dict[str, float]] = {}
    for item in payload:
        if not isinstance(item, dict) or "id" not in item:
            continue
        try:
            by_id[int(item["id"])] = _normalize_scores(item)
        except (TypeError, ValueError, KeyError):
            continue

    return [by_id.get(i, _default_sentiment()) for i in range(batch_size)]

def score_with_bedrock(texts: list[str]) -> list[dict]:
    """
    Batch sentiment scoring using Bedrock Claude Haiku.
    Processes texts in small batches to reduce latency and cost.
    Returns list of {"positive": float, "negative": float, "neutral": float} dicts.
    """
    bedrock = _get_bedrock()
    results = []

    for batch in _chunked(texts, BEDROCK_BATCH_SIZE):
        active_indices = []
        active_texts = []
        batch_results = [_default_sentiment() for _ in batch]

        for idx, text in enumerate(batch):
            if text and len(text.strip()) >= 5:
                active_indices.append(idx)
                active_texts.append(text)

        if not active_texts:
            results.extend(batch_results)
            continue

        try:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 220,
                "temperature": 0,
                "system": _SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": _build_batch_prompt(active_texts)}],
            }
            response = bedrock.invoke_model(
                modelId="arn:aws:bedrock:us-east-2:848820662735:inference-profile/global.anthropic.claude-haiku-4-5-20251001-v1:0",
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            resp_body = json.loads(response["body"].read())
            logger.debug("Bedrock batch response: %s", resp_body)
            raw = resp_body["content"][0]["text"].strip()
            parsed = _parse_batch_response(raw, len(active_texts))
            for original_idx, scores in zip(active_indices, parsed):
                batch_results[original_idx] = scores
        except (ClientError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning("Bedrock batch scoring error: %s", e)
            for original_idx, text in zip(active_indices, active_texts):
                try:
                    body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 80,
                        "temperature": 0,
                        "system": _SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": text[:400]}],
                    }
                    response = bedrock.invoke_model(
                        modelId="arn:aws:bedrock:us-east-2:848820662735:inference-profile/global.anthropic.claude-haiku-4-5-20251001-v1:0",
                        body=json.dumps(body),
                        contentType="application/json",
                        accept="application/json",
                    )
                    resp_body = json.loads(response["body"].read())
                    batch_results[original_idx] = _parse_single_response(
                        resp_body["content"][0]["text"].strip()
                    )
                except (ClientError, json.JSONDecodeError, KeyError, TypeError, ValueError) as inner_e:
                    logger.warning("Bedrock single-text fallback error: %s", inner_e)

        results.extend(batch_results)

    return results


# ── VADER baseline scoring ─────────────────────────────────────────────────────
_vader_analyzer = None

def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader_analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("vaderSentiment not installed; VADER scoring disabled")
    return _vader_analyzer

def score_with_vader(text: str) -> dict:
    """
    VADER lexicon-based sentiment scoring.
    Returns {"positive": float, "negative": float, "neutral": float}.
    """
    vader = _get_vader()
    if vader is None:
        return _default_sentiment()
    scores = vader.polarity_scores(text)
    return {
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral":  scores["neu"],
    }


# ── Aggregate sentiment score for a batch ─────────────────────────────────────
def compute_aggregate_sentiment(
    posts: list[dict],
    use_bedrock: bool = True,
) -> dict:
    """
    Given a list of post dicts with 'text', 'likes', 'retweets', 'comments' fields,
    compute the weighted aggregate sentiment score s(t).

    s(t) = Σ(w_i × (P(pos_i) − P(neg_i))) / Σ(w_i)

    Returns:
      {
        "score": float,           # [-1, +1] weighted composite score
        "post_count": int,
        "avg_engagement_weight": float,
        "positive_fraction": float,
        "negative_fraction": float,
        "method": "bedrock" | "vader",
      }
    """
    if not posts:
        return {"score": 0.0, "post_count": 0, "avg_engagement_weight": 1.0,
                "positive_fraction": 0.33, "negative_fraction": 0.33, "method": "none"}

    texts = [preprocess_text(p.get("text", "")) for p in posts]

    if use_bedrock:
        try:
            sentiment_scores = score_with_bedrock(texts)
            method = "bedrock"
        except Exception as e:
            logger.warning("Bedrock batch scoring failed, falling back to VADER: %s", e)
            sentiment_scores = [score_with_vader(t) for t in texts]
            method = "vader_fallback"
    else:
        sentiment_scores = [score_with_vader(t) for t in texts]
        method = "vader"

    total_weight = 0.0
    weighted_polarity_sum = 0.0
    total_positive = 0.0
    total_negative = 0.0

    for post, scores in zip(posts, sentiment_scores):
        w = engagement_weight(
            post.get("likes", 0),
            post.get("retweets", 0),
            post.get("comments", 0),
        )
        polarity = scores["positive"] - scores["negative"]
        weighted_polarity_sum += w * polarity
        total_weight += w
        total_positive += scores["positive"]
        total_negative += scores["negative"]

    n = len(posts)
    final_score = weighted_polarity_sum / total_weight if total_weight > 0 else 0.0
    final_score = max(-1.0, min(1.0, final_score))   # clamp to [-1, 1]

    return {
        "score":                  round(final_score, 4),
        "post_count":             n,
        "avg_engagement_weight":  round(total_weight / n, 4),
        "positive_fraction":      round(total_positive / n, 4),
        "negative_fraction":      round(total_negative / n, 4),
        "method":                 method,
    }
