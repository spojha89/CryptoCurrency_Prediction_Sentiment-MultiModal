"""
load_historical_data.py
=======================
Loads historical OHLCV data from Binance (no API key needed)
and Fear & Greed sentiment from Alternative.me into DynamoDB.
Also supports LunarCrush social sentiment and Spambots CSV sentiment.

Usage:
  python load_historical_data.py --days 365
  python load_historical_data.py --days 90 --dry-run
  python load_historical_data.py --days 365 --table crypto_metrics
  python load_historical_data.py --run-only ohlcv  # Loads 2021-2023 OHLCV
  python load_historical_data.py --run-only spambots --csv-file Spambots.csv
  python load_historical_data.py --run-only lunarcrush --lunarcrush-key YOUR_KEY
  python load_historical_data.py --run-only fear-greed

Run-only options:
  ohlcv
  spambots
  lunarcrush
  fear-greed

Data sources:
  Binance
  Fear & Greed
  LunarCrush
  Spambots

Cleanup modes:
  all
  coins
  dates

Note:

  Spambots CSV requires a custom sentiment CSV file.

  Binance requires a Binance API key.

  LunarCrush requires a LunarCrush API key.


Binance provides 15-min OHLCV candles free, no auth required.
Fear & Greed provides daily sentiment score back to 2018.
LunarCrush provides social sentiment with API key.
Spambots CSV loads custom sentiment data.
"""
import argparse
import boto3
import csv
import json
import logging
import math
import os
import requests
import sys
import time
import re
from botocore.exceptions import ClientError
from datetime import datetime, timezone, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("historical_loader")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

# ── Config ────────────────────────────────────────────────────────────────────
AWS_REGION    = "us-east-2"
METRICS_TABLE = "crypto_metrics"
BUCKET_MINS   = 15
TTL_DAYS      = 30

# Binance symbol → our coin_id
# BINANCE_SYMBOLS = {
#     "BTCUSDT": "BTC-USD",
#     "ETHUSDT": "ETH-USD",
#     "BNBUSDT": "BNB-USD",
#     "XRPUSDT": "XRP-USD",
#     "LTCUSDT": "LTC-USD",
# }

BINANCE_SYMBOLS = {
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "BNBUSD": "BNB-USD",
    "XRPUSD": "XRP-USD",
    "LTCUSD": "LTC-USD",
}

def get_dynamodb(region_name: str = AWS_REGION):
    return boto3.resource("dynamodb", region_name=region_name)


def get_table(table_name: str, region_name: str = AWS_REGION):
    return get_dynamodb(region_name).Table(table_name)


def normalize_date_str(value: str | None) -> str | None:
    """Best-effort conversion of timestamps/date strings to YYYY-MM-DD."""
    if not value:
        return None
    value = str(value).strip()
    if not value:
        return None
    if len(value) >= 10 and value[4] == "-" and value[7] == "-":
        return value[:10]
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except ValueError:
        return value.split("T")[0].split(" ")[0][:10]


def classify_fear_greed(value: int) -> str:
    """Map Fear & Greed value to the usual textual classification."""
    if value <= 24:
        return "Extreme Fear"
    if value <= 44:
        return "Fear"
    if value <= 54:
        return "Neutral"
    if value <= 74:
        return "Greed"
    return "Extreme Greed"


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS  (pure Python, no TA-Lib needed)
# ══════════════════════════════════════════════════════════════════════════════

def compute_rsi(closes: list[float], period: int = 14) -> list[float]:
    rsi = [None] * len(closes)
    if len(closes) < period + 1:
        return rsi
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 1e9
        rsi[i + 1] = round(100 - (100 / (1 + rs)), 4)
    return rsi


def compute_ema(closes: list[float], period: int) -> list[float]:
    ema = [None] * len(closes)
    if len(closes) < period:
        return ema
    k = 2.0 / (period + 1)
    ema[period - 1] = sum(closes[:period]) / period
    for i in range(period, len(closes)):
        ema[i] = closes[i] * k + ema[i-1] * (1 - k)
    return ema


def compute_macd(closes: list[float]):
    ema12  = compute_ema(closes, 12)
    ema26  = compute_ema(closes, 26)
    macd   = [
        round(ema12[i] - ema26[i], 6)
        if ema12[i] is not None and ema26[i] is not None else None
        for i in range(len(closes))
    ]
    # Signal line = EMA9 of MACD
    valid_macd  = [(i, v) for i, v in enumerate(macd) if v is not None]
    signal_full = [None] * len(closes)
    hist_full   = [None] * len(closes)
    if len(valid_macd) >= 9:
        idxs   = [x[0] for x in valid_macd]
        vals   = [x[1] for x in valid_macd]
        sig    = compute_ema(vals, 9)
        for j, idx in enumerate(idxs):
            signal_full[idx] = sig[j]
            if sig[j] is not None and macd[idx] is not None:
                hist_full[idx] = round(macd[idx] - sig[j], 6)
    return macd, signal_full, hist_full


def compute_bollinger(closes: list[float], period: int = 20):
    upper = [None] * len(closes)
    mid   = [None] * len(closes)
    lower = [None] * len(closes)
    bb_pos = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1: i + 1]
        sma    = sum(window) / period
        std    = math.sqrt(sum((x - sma) ** 2 for x in window) / period)
        u      = sma + 2 * std
        l      = sma - 2 * std
        upper[i] = round(u, 6)
        mid[i]   = round(sma, 6)
        lower[i] = round(l, 6)
        bb_pos[i] = round((closes[i] - l) / (u - l), 6) if (u - l) > 0 else 0.5
    return upper, mid, lower, bb_pos


def compute_vwap(closes, highs, lows, volumes):
    vwap = []
    cum_tv = 0.0
    cum_v  = 0.0
    for i in range(len(closes)):
        typical = (highs[i] + lows[i] + closes[i]) / 3.0
        cum_tv += typical * volumes[i]
        cum_v  += volumes[i]
        vwap.append(round(cum_tv / cum_v, 6) if cum_v > 0 else closes[i])
    return vwap


# ══════════════════════════════════════════════════════════════════════════════
# BINANCE HISTORICAL OHLCV
# ══════════════════════════════════════════════════════════════════════════════

def fetch_binance_candles(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    interval: str = "15m",
) -> list[dict]:
    """
    Fetch 15-minute OHLCV candles from Binance.
    No API key required. Returns up to 1000 candles per call.
    Handles pagination automatically.
    """
    url      = "https://api.binance.us/api/v3/klines"
    all_candles = []
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)
    limit    = 1000   # Binance max per request

    logger.info("Fetching %s from %s to %s",
                symbol, start_dt.date(), end_dt.date())

    while start_ms < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": start_ms,
            "endTime":   end_ms,
            "limit":     limit,
        }
        try:
            resp = requests.get(url, params=params, timeout=15)

            if resp.status_code == 429:
                logger.warning("Binance rate limit — sleeping 60s")
                time.sleep(60)
                continue

            resp.raise_for_status()
            candles = resp.json()

            if not candles:
                break

            all_candles.extend(candles)
            # next page starts after last candle's close time
            start_ms = candles[-1][6] + 1   # index 6 = close time ms
            time.sleep(0.1)   # polite delay

        except requests.RequestException as e:
            logger.error("Binance fetch error for %s: %s", symbol, e)
            time.sleep(5)
            break

    logger.info("Fetched %d raw candles for %s", len(all_candles), symbol)
    return all_candles


def parse_binance_candles(
    candles: list,
    coin_id: str,
    fear_greed_by_date: dict,
) -> list[dict]:
    """
    Convert raw Binance candles to DynamoDB items with TA indicators.
    Merges Fear & Greed sentiment where available.
    """
    if not candles:
        return []

    # Extract price series for TA computation
    opens   = [float(c[1]) for c in candles]
    highs   = [float(c[2]) for c in candles]
    lows    = [float(c[3]) for c in candles]
    closes  = [float(c[4]) for c in candles]
    volumes = [float(c[5]) for c in candles]

    # Compute all indicators over full series
    rsi_series              = compute_rsi(closes)
    macd_s, sig_s, hist_s   = compute_macd(closes)
    bb_u, bb_m, bb_l, bb_p  = compute_bollinger(closes)
    vwap_series             = compute_vwap(closes, highs, lows, volumes)

    items = []
    ttl   = int(datetime.now(timezone.utc).timestamp()) + TTL_DAYS * 86400

    for i, candle in enumerate(candles):
        # Candle open time in ms → datetime
        open_time_ms = candle[0]
        dt = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc)

        # Truncate to 15-min bucket
        bucket_dt = dt.replace(
            minute=(dt.minute // BUCKET_MINS) * BUCKET_MINS,
            second=0,
            microsecond=0,
        )
        bucket_str = bucket_dt.isoformat()
        date_str   = bucket_dt.strftime("%Y-%m-%d")

        # Fear & Greed for this date
        fg = fear_greed_by_date.get(date_str)
        fg_value = int(fg) if fg is not None else None
        fg_score = str(round((fg_value - 50) / 50.0, 4)) if fg_value is not None else None

        item = {
            "coin_id":          coin_id,
            "timestamp_bucket": bucket_str,
            "ttl":              ttl,
            "open":             str(round(opens[i],   6)),
            "high":             str(round(highs[i],   6)),
            "low":              str(round(lows[i],    6)),
            "close":            str(round(closes[i],  6)),
            "volume":           str(round(volumes[i], 2)),
            "source":           "binance_historical",
        }

        # Add TA indicators if computed
        def _add(key, val):
            if val is not None:
                item[key] = str(val)

        _add("rsi",           rsi_series[i])
        _add("macd",          macd_s[i])
        _add("macd_signal",   sig_s[i])
        _add("macd_histogram", hist_s[i])
        _add("bb_upper",      bb_u[i])
        _add("bb_middle",     bb_m[i])
        _add("bb_lower",      bb_l[i])
        _add("bb_position",   bb_p[i])
        _add("vwap",          vwap_series[i])

        # Add Fear & Greed as sentiment_composite for this date
        if fg_score:
            item["sentiment_composite"] = fg_score
            item["sentiment_news"]      = fg_score   # proxy
            item["sentiment_method"]    = "fear_greed_historical"
            item["sentiment_platform"]  = "fear_greed"
            item["fear_greed_value"] = str(fg_value)
            item["fear_greed_count"] = 1
            item["fear_greed_classification"] = classify_fear_greed(fg_value)

        items.append(item)

    return items


# ══════════════════════════════════════════════════════════════════════════════
# FEAR & GREED HISTORICAL
# ══════════════════════════════════════════════════════════════════════════════

def fetch_fear_greed_historical(
    days: int | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict:
    """
    Fetch historical Fear & Greed Index from Alternative.me.
    Returns dict of {date_str: score_0_to_100}.
    Free, no API key, goes back to 2018.
    """
    try:
        limit = 0 if days is None else days
        url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])

        by_date = {}
        start_day = start_date.date() if start_date else None
        end_day = end_date.date() if end_date else None
        for entry in data:
            ts       = int(entry["timestamp"])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            if start_day and dt.date() < start_day:
                continue
            if end_day and dt.date() > end_day:
                continue
            date_str = dt.strftime("%Y-%m-%d")
            by_date[date_str] = entry["value"]

        logger.info(
            "Fear & Greed: %d days of historical data loaded%s",
            len(by_date),
            f" for {start_day} to {end_day}" if start_day or end_day else "",
        )
        return by_date

    except Exception as e:
        logger.error("Fear & Greed historical fetch failed: %s", e)
        return {}



# ══════════════════════════════════════════════════════════════════════════════
# LUNARCRUSH HISTORICAL SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════


def fetch_lunarcrush_historical(
    coins: list[str],
    days: int = 365,
    api_key: str | None = None,
) -> dict[str, dict[str, float]]:
    """
    LunarCrush API — crypto social sentiment.
    Register free at https://lunarcrush.com/developers
    Returns a mapping of coin_id -> {date: sentiment_score}.
    """
    coin_map = {
        "BTC-USD": "BTC",
        "ETH-USD": "ETH",
        "BNB-USD": "BNB",
        "XRP-USD": "XRP",
        "LTC-USD": "LTC",
    }
    sentiment_by_coin = {}
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        logger.warning(
            "LunarCrush API key not configured; requests may return 401 Unauthorized."
        )

    for coin in coins:
        symbol = coin_map.get(coin, coin.split("-")[0])
        daily_scores = {}
        retries = 0
        backoff = 5

        while retries < 5:
            try:
                resp = requests.get(
                    f"https://lunarcrush.com/api4/public/coins/{symbol}/time-series/v2",
                    params={
                        "bucket": "day",
                        "interval": f"{days}d",
                    },
                    headers=headers,
                    timeout=15,
                )

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait = int(retry_after) if retry_after and retry_after.isdigit() else backoff
                    logger.warning(
                        "LunarCrush rate limited for %s: sleeping %ds before retry %d/5",
                        coin, wait, retries + 1,
                    )
                    time.sleep(wait)
                    retries += 1
                    backoff = min(backoff * 2, 60)
                    continue

                resp.raise_for_status()
                data = resp.json().get("data", [])
                for entry in data:
                    ts = entry.get("time")
                    if ts is None:
                        continue
                    date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                    sentiment_raw = entry.get("sentiment", 3)
                    score = (sentiment_raw - 3) / 2.0
                    daily_scores[date_str] = round(score, 4)

                logger.info("LunarCrush: %s historical records loaded: %d", coin, len(daily_scores))
                break

            except requests.RequestException as e:
                if hasattr(e, 'response') and e.response is not None and e.response.status_code == 401:
                    logger.warning("LunarCrush unauthorized for %s: %s", coin, e)
                    break
                logger.warning("LunarCrush fetch failed for %s: %s", coin, e)
                retries += 1
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
        else:
            logger.warning("LunarCrush fetch failed for %s after retries", coin)

        # Avoid hitting the API too quickly between coins
        time.sleep(2)
    return sentiment_by_coin


# ══════════════════════════════════════════════════════════════════════════════
# SPAMBOTS CSV HISTORICAL SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_spambots_csv(
    csv_path: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, dict[str, float]]:
    """
    Load historical sentiment from Spambots.csv.
    Extract coin-level daily social sentiment from raw social posts.
    Returns a mapping of coin_id -> {date: {"score": ..., "count": ...}}.
    """
    sentiment_by_coin = {}
    csv_path = Path(csv_path)
    start_date = start_date or ""
    end_date = end_date or ""

    if not csv_path.exists():
        logger.warning("Spambots CSV file not found: %s", csv_path)
        return sentiment_by_coin

    logger.info("Loading Spambots CSV from: %s", csv_path)

    # Coin keywords to look for in text
    coin_keywords = {
        "BTC-USD": ["btc", "bitcoin", "#btc", "#bitcoin"],
        "ETH-USD": ["eth", "ethereum", "#eth", "#ethereum"],
        "BNB-USD": ["bnb", "binance coin", "binancecoin", "#bnb"],
        "XRP-USD": ["xrp", "ripple", "#xrp"],
        "LTC-USD": ["ltc", "litecoin", "#ltc", "#litecoin"],
    }
    keyword_patterns = {
        coin_id: [
            re.compile(rf"(?<![A-Za-z0-9]){re.escape(keyword.lower())}(?![A-Za-z0-9])")
            for keyword in keywords
        ]
        for coin_id, keywords in coin_keywords.items()
    }

    analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
    if analyzer is None:
        logger.warning("VADER sentiment analyzer unavailable; falling back to engagement-only scoring.")

    positive_words = {
        "bull", "bullish", "buy", "moon", "pump", "breakout", "surge",
        "green", "gain", "rally", "uptrend", "strong",
    }
    negative_words = {
        "bear", "bearish", "sell", "dump", "crash", "drop", "red",
        "loss", "downtrend", "weak", "panic", "scam",
    }

    def _parse_number(value, default=0.0):
        try:
            if value in (None, ""):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _text_sentiment(text: str) -> float:
        if not text:
            return 0.0
        if analyzer:
            return float(analyzer.polarity_scores(text)["compound"])

        tokens = re.findall(r"[A-Za-z#]+", text.lower())
        if not tokens:
            return 0.0
        pos_hits = sum(1 for token in tokens if token in positive_words)
        neg_hits = sum(1 for token in tokens if token in negative_words)
        if pos_hits == 0 and neg_hits == 0:
            return 0.0
        return max(-1.0, min(1.0, (pos_hits - neg_hits) / max(pos_hits + neg_hits, 1)))

    def _find_coins(text: str, hashtags: str, url: str) -> set[str]:
        haystacks = [text.lower(), hashtags.lower(), url.lower()]
        mentioned = set()
        for coin_id, patterns in keyword_patterns.items():
            for pattern in patterns:
                if any(pattern.search(h) for h in haystacks if h):
                    mentioned.add(coin_id)
                    break
        return mentioned

    def _normalize_score(weighted_sum: float, total_weight: float) -> float:
        if total_weight <= 0:
            return 0.0
        return round(max(-1.0, min(1.0, weighted_sum / total_weight)), 4)

    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=";,")
            except csv.Error:
                dialect = csv.excel
                dialect.delimiter = ";" if sample.count(";") >= sample.count(",") else ","

            reader = csv.DictReader(f, dialect=dialect)
            logger.info("CSV headers found: %s", reader.fieldnames)

            row_count = 0
            processed_count = 0
            aggregated = {}

            for row in reader:
                row_count += 1

                date_only = normalize_date_str(row.get("date", ""))
                text = str(row.get("text", "") or "").strip()
                hashtags = str(row.get("hashtags", "") or "").strip()
                url = str(row.get("url", "") or "").strip()
                n_likes = _parse_number(row.get("n_likes"))
                n_retweets = _parse_number(row.get("n_retweets"))
                n_replies = _parse_number(row.get("n_replies"))
                followers = _parse_number(row.get("n_followers"))
                is_retweet = str(row.get("is_retweet", "") or "").strip().lower() in {
                    "1", "true", "t", "yes", "y"
                }

                if not date_only or not text:
                    continue
                if start_date and date_only < start_date:
                    continue
                if end_date and date_only > end_date:
                    continue

                mentioned_coins = _find_coins(text, hashtags, url)
                if not mentioned_coins:
                    continue

                engagement = n_likes + (2.0 * n_retweets) + (1.5 * n_replies)
                engagement_weight = 1.0 + math.log1p(max(engagement, 0.0))
                reach_weight = 1.0 + min(math.log1p(max(followers, 0.0)) / 4.0, 2.0)
                retweet_penalty = 0.85 if is_retweet else 1.0
                total_weight = engagement_weight * reach_weight * retweet_penalty

                sentiment_score = _text_sentiment(text)
                if abs(sentiment_score) < 0.05:
                    engagement_bias = min(math.log1p(max(engagement, 0.0)) / 6.0, 0.35)
                    sentiment_score = engagement_bias

                for coin_id in mentioned_coins:
                    daily = aggregated.setdefault(coin_id, {}).setdefault(
                        date_only,
                        {
                            "weighted_sum": 0.0,
                            "weight_total": 0.0,
                            "count": 0,
                            "engagement_total": 0.0,
                        },
                    )
                    daily["weighted_sum"] += sentiment_score * total_weight
                    daily["weight_total"] += total_weight
                    daily["count"] += 1
                    daily["engagement_total"] += engagement
                processed_count += 1

                # Debug first few processed rows
                if processed_count <= 3:
                    logger.info(
                        "Processed row %d: date=%s, coins=%s, engagement=%.2f, sentiment=%.4f",
                        row_count, date_only, list(mentioned_coins), engagement, sentiment_score
                    )

            for coin_id, by_date in aggregated.items():
                sentiment_by_coin[coin_id] = {}
                for date_only, stats in by_date.items():
                    score = _normalize_score(stats["weighted_sum"], stats["weight_total"])
                    count = int(stats["count"])
                    sentiment_by_coin[coin_id][date_only] = {
                        "score": score,
                        "count": count,
                        "engagement_total": round(stats["engagement_total"], 2),
                    }

            logger.info("Spambots CSV: processed %d rows, extracted %d sentiment scores for %d coins",
                       row_count, processed_count, len(sentiment_by_coin))

    except Exception as e:
        logger.error("Failed to load Spambots CSV: %s", e)

    return sentiment_by_coin


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMODB WRITER
# ══════════════════════════════════════════════════════════════════════════════

def write_items_to_dynamodb(
    items: list[dict],
    table,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> tuple[int, int]:
    """
    Write items to DynamoDB using batch_writer.
    Returns (written_count, skipped_count).
    skip_existing=True avoids overwriting your live collected records.
    """
    if dry_run:
        logger.info("[DRY RUN] Would write %d items", len(items))
        return len(items), 0

    written = 0
    skipped = 0
    batch_size = 25   # DynamoDB batch_writer handles up to 25

    for i in range(0, len(items), batch_size):
        batch = items[i: i + batch_size]

        if skip_existing:
            # Check which items already exist before writing
            filtered = []
            for item in batch:
                try:
                    resp = table.get_item(
                        Key={
                            "coin_id":          item["coin_id"],
                            "timestamp_bucket": item["timestamp_bucket"],
                        },
                        ProjectionExpression="coin_id",
                    )
                    if "Item" not in resp:
                        filtered.append(item)
                    else:
                        skipped += 1
                except ClientError:
                    filtered.append(item)
            batch = filtered

        if not batch:
            continue

        try:
            with table.batch_writer() as writer:
                for item in batch:
                    writer.put_item(Item=item)
            written += len(batch)
        except ClientError as e:
            logger.error("DynamoDB batch write error: %s", e)

        # Progress log every 500 items
        if (i // batch_size) % 20 == 0 and i > 0:
            logger.info("  Progress: %d written, %d skipped so far...",
                        written, skipped)
        time.sleep(0.05)   # avoid DynamoDB throttling

    return written, skipped



# ══════════════════════════════════════════════════════════════════════════════
# DYNAMODB UPDATER — merge sentiment into existing OHLCV records
# ══════════════════════════════════════════════════════════════════════════════

def update_sentiment_in_dynamodb(
    coin_id: str,
    date_sentiment: dict,
    table,
    platform: str = "twitter",
    dry_run: bool = False,
) -> int:
    """
    For each date in date_sentiment, update ALL 15-min buckets
    for that coin on that date with the daily sentiment score.
    Uses UpdateItem to avoid overwriting OHLCV data.
    Returns count of records updated.
    """
    updated = 0

    for date_str, score in date_sentiment.items():
        # Query all 15-min buckets for this coin on this date
        try:
            resp = table.query(
                KeyConditionExpression=(
                    boto3.dynamodb.conditions.Key("coin_id").eq(coin_id) &
                    boto3.dynamodb.conditions.Key("timestamp_bucket").begins_with(date_str)
                )
            )
            items = resp.get("Items", [])

            for item in items:
                if dry_run:
                    updated += 1
                    continue
                try:
                    table.update_item(
                        Key={
                            "coin_id":          item["coin_id"],
                            "timestamp_bucket": item["timestamp_bucket"],
                        },
                        UpdateExpression=(
                            "SET sentiment_twitter = :st, "
                            "sentiment_composite = :sc, "
                            "sentiment_method = :sm"
                        ),
                        ExpressionAttributeValues={
                            ":st": str(score),
                            ":sc": str(score),
                            ":sm": platform,
                        },
                        # Only update if record doesn't already have real sentiment
                        ConditionExpression=(
                            "attribute_not_exists(sentiment_twitter) OR "
                            "sentiment_method = :sm"
                        ),
                    )
                    updated += 1
                except ClientError as e:
                    if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                        pass   # already has real sentiment — skip
                    else:
                        logger.warning("Update failed: %s", e)

        except ClientError as e:
            logger.error("Query failed for %s %s: %s", coin_id, date_str, e)

    return updated


def _update_sentiment_for_day(
    coin_id: str, date_str: str, daily_value: float | dict, table, platform: str,
    dry_run: bool = False,
) -> int:
    """
    Batch update sentiment data for multiple coins and dates.
    Updates existing OHLCV records with sentiment fields.
    Returns total count of records updated.
    """
    updated_count = 0
    items_for_day = []

    # Query all 15-min buckets for this coin on this specific date
    try:
        response = table.query(
            KeyConditionExpression=(
                boto3.dynamodb.conditions.Key("coin_id").eq(coin_id) &
                boto3.dynamodb.conditions.Key("timestamp_bucket").begins_with(date_str)
            )
        )
        items_for_day = response.get("Items", [])
        while "LastEvaluatedKey" in response:
            response = table.query(
                KeyConditionExpression=(
                    boto3.dynamodb.conditions.Key("coin_id").eq(coin_id) &
                    boto3.dynamodb.conditions.Key("timestamp_bucket").begins_with(date_str)
                ),
                ExclusiveStartKey=response["LastEvaluatedKey"]
            )
            items_for_day.extend(response.get("Items", []))
    except ClientError as e:
        logger.error("Query failed for %s %s: %s", coin_id, date_str, e)
        return 0

    if not items_for_day:
        return 0

    if isinstance(daily_value, dict):
        score = daily_value.get("score")
        post_count = int(daily_value.get("count", 0) or 0)
    else:
        score = daily_value
        post_count = 1

    if score is None:
        return 0

    for item in items_for_day:
        if dry_run:
            updated_count += 1
            continue
        try:
            update_expression_parts = []
            expression_attribute_values = {}
            expression_attribute_names = {}

            if platform == "fear_greed":
                fg_value = int(round(float(score)))
                update_expression_parts.extend([
                    "sentiment_composite = :sc",
                    "sentiment_method = :sm",
                    "sentiment_platform = :sp",
                    "fear_greed_value = :fgv",
                    "fear_greed_count = :fgc",
                    "fear_greed_classification = :fgclass"
                ])
                expression_attribute_values.update({
                    ":sc": str(round((fg_value - 50) / 50.0, 4)),
                    ":sm": "fear_greed_historical",
                    ":sp": "fear_greed",
                    ":fgv": str(fg_value),
                    ":fgc": 1,
                    ":fgclass": classify_fear_greed(fg_value),
                })
            else: # For other platforms like spambots, lunarcrush
                update_expression_parts.extend([
                    "sentiment_twitter = :st",
                    "sentiment_composite = :sc",
                    "twitter_count = :tc",
                    "sentiment_method = :sm",
                    "sentiment_platform = :sp"
                ])
                expression_attribute_values.update({
                    ":st": str(score),
                    ":sc": str(score),
                    ":tc": post_count,
                    ":sm": platform,
                    ":sp": platform,
                })

            # Condition to update: platform-specific checks
            if platform == "fear_greed":
                # For fear_greed, only update if the value doesn't already exist
                condition_expression = "attribute_not_exists(fear_greed_value)"
            else:
                # For other platforms, update if sentiment_platform doesn't exist OR it's from the current platform
                condition_expression = "attribute_not_exists(#sp_attr) OR #sp_attr = :sp_current_platform"
                expression_attribute_names["#sp_attr"] = "sentiment_platform"
                expression_attribute_values[":sp_current_platform"] = platform

            update_params = {
                "Key": {
                    "coin_id":          item["coin_id"],
                    "timestamp_bucket": item["timestamp_bucket"],
                },
                "UpdateExpression": "SET " + ", ".join(update_expression_parts),
                "ConditionExpression": condition_expression,
                "ExpressionAttributeValues": expression_attribute_values,
            }
            
            # Only include ExpressionAttributeNames if it has content
            if expression_attribute_names:
                update_params["ExpressionAttributeNames"] = expression_attribute_names
            
            table.update_item(**update_params)
            updated_count += 1
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                # This means the item already has sentiment from a different platform,
                # and our condition prevented overwriting it.
                pass
            else:
                logger.warning("Update failed for %s %s: %s", coin_id, item["timestamp_bucket"], e)
    return updated_count


def update_sentiment_in_dynamodb_batch(
    sentiment_data: dict[str, dict[str, float | dict]],
    table,
    platform: str = "spambots",
    dry_run: bool = False,
) -> int:
    """
    Batch update sentiment data for multiple coins and dates.
    Updates existing OHLCV records with sentiment fields.
    Returns total count of records updated.
    """
    if dry_run:
        total_items = sum(len(dates) for dates in sentiment_data.values())
        logger.info("[DRY RUN] Would update sentiment for %d coin-date combinations", total_items)
        return total_items

    if not sentiment_data:
        logger.warning("No sentiment data available to update")
        return 0

    total_updated = 0
    total_matched_days = 0

    logger.info("Starting %s sentiment update batch...", platform)

    for coin_id, daily_sentiment_for_coin in sentiment_data.items():
        coin_updated = 0
        dates_to_process = len(daily_sentiment_for_coin)
        
        for date_idx, (date_str, daily_value) in enumerate(daily_sentiment_for_coin.items(), 1):
            total_matched_days += 1
            
            # Log progress for every date processed
            logger.info("[%s] Updating %s for date %s (%d/%d)...", 
                        platform, coin_id, date_str, date_idx, dates_to_process)
            
            updated_for_day = _update_sentiment_for_day(
                coin_id, date_str, daily_value, table, platform, dry_run
            )
            
            total_updated += updated_for_day
            coin_updated += updated_for_day
            
            if updated_for_day > 0:
                logger.info("  -> Successfully updated %d records for %s", updated_for_day, coin_id)

        logger.info("Completed sentiment update for coin %s: %d total records updated.", coin_id, coin_updated)

    logger.info(
        "%s updater processed %d coin-date combinations, updated %d total records.",
        platform, total_matched_days, total_updated
    )

    if total_matched_days > 0 and total_updated == 0:
        logger.warning(
            "Processed %d dates but no records were updated. "
            "Ensure historical OHLCV data exists in the table first.",
            total_matched_days
        )

    return total_updated


def cleanup_dynamodb_table(
    table,
    mode: str,
    coins: list = None,
    start_date: str = None,
    end_date: str = None,
    platform: str = None,
    force: bool = False,
) -> int:
    """
    Cleanup DynamoDB table based on specified criteria.
    Returns number of items deleted.
    """
    deleted_count = 0

    # Safety check
    if not force:
        confirm = input(f"⚠️  DANGER: This will delete data from table '{table.table_name}' in region '{table.meta.client.meta.region_name}'.\n"
                       f"Mode: {mode}\n"
                       f"Continue? Type 'yes' to confirm: ")
        if confirm.lower() != 'yes':
            logger.info("Cleanup cancelled by user")
            return 0

    logger.info("Starting cleanup mode: %s", mode)

    if mode == "all":
        # Delete all items from table
        try:
            # Scan all items (be careful with large tables!)
            response = table.scan()
            items = response.get('Items', [])

            while 'LastEvaluatedKey' in response:
                response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
                items.extend(response.get('Items', []))

            logger.info("Found %d items to delete", len(items))

            # Delete in batches
            batch_size = 25
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                with table.batch_writer() as writer:
                    for item in batch:
                        writer.delete_item(
                            Key={
                                'coin_id': item['coin_id'],
                                'timestamp_bucket': item['timestamp_bucket']
                            }
                        )
                deleted_count += len(batch)
                logger.info("Deleted batch of %d items", len(batch))

        except ClientError as e:
            logger.error("Failed to cleanup table: %s", e)

    elif mode == "coins" and coins:
        # Delete items for specific coins
        for coin_id in coins:
            try:
                # Query all items for this coin
                response = table.query(
                    KeyConditionExpression=boto3.dynamodb.conditions.Key("coin_id").eq(coin_id)
                )
                items = response.get('Items', [])

                # Delete items
                with table.batch_writer() as writer:
                    for item in items:
                        writer.delete_item(
                            Key={
                                'coin_id': item['coin_id'],
                                'timestamp_bucket': item['timestamp_bucket']
                            }
                        )
                deleted_count += len(items)
                logger.info("Deleted %d items for coin %s", len(items), coin_id)

            except ClientError as e:
                logger.error("Failed to cleanup coin %s: %s", coin_id, e)

    elif mode == "dates" and start_date and end_date:
        # Delete items within date range
        try:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)

            # Scan for items in date range (this is inefficient for large tables)
            response = table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr('timestamp_bucket').between(
                    start_dt.isoformat(), end_dt.isoformat()
                )
            )
            items = response.get('Items', [])

            while 'LastEvaluatedKey' in response:
                response = table.scan(
                    ExclusiveStartKey=response['LastEvaluatedKey'],
                    FilterExpression=boto3.dynamodb.conditions.Attr('timestamp_bucket').between(
                        start_dt.isoformat(), end_dt.isoformat()
                    )
                )
                items.extend(response.get('Items', []))

            logger.info("Found %d items in date range to delete", len(items))

            # Delete in batches
            batch_size = 25
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                with table.batch_writer() as writer:
                    for item in batch:
                        writer.delete_item(
                            Key={
                                'coin_id': item['coin_id'],
                                'timestamp_bucket': item['timestamp_bucket']
                            }
                        )
                deleted_count += len(batch)
                logger.info("Deleted batch of %d items", len(batch))

        except ClientError as e:
            logger.error("Failed to cleanup date range: %s", e)

    elif mode == "platform" and platform:
        # Delete items by platform/source
        platform_field_map = {
            "binance": "source",  # binance records have source="binance_historical"
            "lunarcrush": "sentiment_platform",
            "spambots": "sentiment_platform",
            "fear_greed": "sentiment_platform"
        }

        field_name = platform_field_map.get(platform)
        if not field_name:
            logger.error("Unknown platform: %s", platform)
            return 0

        try:
            # Scan for items with this platform
            filter_value = "binance_historical" if platform == "binance" else platform

            response = table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr(field_name).eq(filter_value)
            )
            items = response.get('Items', [])

            while 'LastEvaluatedKey' in response:
                response = table.scan(
                    ExclusiveStartKey=response['LastEvaluatedKey'],
                    FilterExpression=boto3.dynamodb.conditions.Attr(field_name).eq(filter_value)
                )
                items.extend(response.get('Items', []))

            logger.info("Found %d items with platform %s to delete", len(items), platform)

            # Delete in batches
            batch_size = 25
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                with table.batch_writer() as writer:
                    for item in batch:
                        writer.delete_item(
                            Key={
                                'coin_id': item['coin_id'],
                                'timestamp_bucket': item['timestamp_bucket']
                            }
                        )
                deleted_count += len(batch)
                logger.info("Deleted batch of %d items", len(batch))

        except ClientError as e:
            logger.error("Failed to cleanup platform %s: %s", platform, e)

    logger.info("Cleanup completed: %d items deleted", deleted_count)
    return deleted_count


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Load historical crypto OHLCV + sentiment into DynamoDB"
    )
    parser.add_argument("--days",           type=int,  default=365,
                        help="How many days of history to load (default: 365)")
    parser.add_argument("--table",          type=str,  default="crypto_metrics",
                        help="DynamoDB table name")
    parser.add_argument("--region",         type=str,  default="us-east-2")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Print counts without writing to DynamoDB")
    parser.add_argument("--skip-existing",  action="store_true", default=True,
                        help="Skip records that already exist (protects live data)")
    parser.add_argument("--coins",          type=str,  default="all",
                        help="Comma-separated coins e.g. BTC-USD,ETH-USD or 'all'")
    parser.add_argument("--lunarcrush-key", type=str,
                        default=os.environ.get("LUNARCRUSH_API_KEY"),
                        help="LunarCrush API token or set LUNARCRUSH_API_KEY")
    parser.add_argument("--csv-file",       type=str,
                        default="Spambots.csv",
                        help="Path to Spambots CSV file for sentiment loading")
    parser.add_argument("--run-only",       type=str,  default="all",
                        choices=["all", "fear-greed", "lunarcrush", "ohlcv", "spambots"],
                        help="Which stage to run: all, fear-greed, lunarcrush, ohlcv, spambots")
    parser.add_argument("--cleanup",        type=str,
                        choices=["all", "coins", "dates", "platform"],
                        help="Cleanup mode: all (delete everything), coins (by coin list), dates (by date range), platform (by data source)")
    parser.add_argument("--cleanup-coins",  type=str,
                        help="Comma-separated coins to cleanup (used with --cleanup coins)")
    parser.add_argument("--cleanup-start",  type=str,
                        help="Start date for cleanup YYYY-MM-DD (used with --cleanup dates)")
    parser.add_argument("--cleanup-end",    type=str,
                        help="End date for cleanup YYYY-MM-DD (used with --cleanup dates)")
    parser.add_argument("--cleanup-platform", type=str,
                        choices=["binance", "lunarcrush", "spambots", "fear_greed"],
                        help="Platform to cleanup (used with --cleanup platform)")
    parser.add_argument("--force",          action="store_true",
                        help="Force deletion without confirmation (dangerous!)")
    args = parser.parse_args()

    # Handle cleanup mode
    if args.cleanup:
        table = get_table(args.table, args.region)

        # Parse cleanup parameters
        cleanup_coins = None
        if args.cleanup_coins:
            cleanup_coins = [c.strip() for c in args.cleanup_coins.split(",")]

        deleted = cleanup_dynamodb_table(
            table=table,
            mode=args.cleanup,
            coins=cleanup_coins,
            start_date=args.cleanup_start,
            end_date=args.cleanup_end,
            platform=args.cleanup_platform,
            force=args.force
        )

        logger.info("Cleanup completed: %d items deleted", deleted)
        return

    # Set date range for historical backfill (2021-2023)
    if args.run_only in ("all", "ohlcv", "spambots"):
        start_dt = datetime(2021, 1, 1, tzinfo=timezone.utc)
        end_dt = datetime(2023, 12, 31, tzinfo=timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=args.days)

    # Filter coins
    if args.coins == "all":
        symbols_to_load = BINANCE_SYMBOLS
    else:
        requested = [c.strip() for c in args.coins.split(",")]
        symbols_to_load = {
            k: v for k, v in BINANCE_SYMBOLS.items() if v in requested
        }

    table = get_table(args.table, args.region)
    run_only = args.run_only
    logger.info("=" * 60)
    logger.info("Historical Loader")
    logger.info("  Days:      %d  (%s → %s)", args.days,
                start_dt.date(), end_dt.date())
    logger.info("  Coins:     %s", list(symbols_to_load.values()))
    logger.info("  Table:     %s", args.table)
    logger.info("  Region:    %s", args.region)
    logger.info("  Dry run:   %s", args.dry_run)
    logger.info("  Run only:  %s", run_only)
    logger.info("=" * 60)

    fear_greed = {}
    if run_only in ("all", "ohlcv", "fear-greed"):
        logger.info("Loading Fear & Greed historical sentiment...")
        fear_greed = fetch_fear_greed_historical(
            days=None,
            start_date=start_dt,
            end_date=end_dt,
        )

    total_written = 0
    total_skipped = 0
    total_updated = 0

    if run_only == "fear-greed":
        logger.info("\n[1/1] Updating Fear & Greed sentiment only...")
        fg_by_coin = {
            coin_id: dict(fear_greed)
            for coin_id in symbols_to_load.values()
        }
        n = update_sentiment_in_dynamodb_batch(
            fg_by_coin,
            table,
            platform="fear_greed",
            dry_run=args.dry_run,
        )
        total_updated += n
        logger.info("  Fear & Greed: %d total records updated across all coins", n)

    if run_only in ("all", "lunarcrush"):
        logger.info("\n[2/3] Loading LunarCrush social sentiment...")
        lc_data = fetch_lunarcrush_historical(
            list(symbols_to_load.values()),
            days=args.days,
            api_key=args.lunarcrush_key,
        )
        # Use batch update for better performance
        n = update_sentiment_in_dynamodb_batch(
            lc_data,
            table,
            platform="lunarcrush",
            dry_run=args.dry_run,
        )
        total_updated += n
        logger.info("  LunarCrush: %d total sentiment records updated", n)

    if run_only in ("all", "spambots"):
        logger.info("\n[3/4] Loading Spambots CSV sentiment...")
        spambots_data = load_spambots_csv(
            args.csv_file,
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
        )
        # Use batch update for better performance
        n = update_sentiment_in_dynamodb_batch(
            spambots_data,
            table,
            platform="spambots",
            dry_run=args.dry_run,
        )
        total_updated += n
        logger.info("  Spambots: %d total sentiment records updated across all coins", n)

    if run_only in ("all", "ohlcv"):
        # Step 4 — load OHLCV per coin
        for symbol, coin_id in symbols_to_load.items():
            logger.info("\n── %s (%s) ──────────────────────────────────", coin_id, symbol)

            candles = fetch_binance_candles(symbol, start_dt, end_dt)
            if not candles:
                logger.warning("No candles returned for %s — skipping", symbol)
                continue

            items = parse_binance_candles(candles, coin_id, fear_greed)
            logger.info("Parsed %d DynamoDB items for %s", len(items), coin_id)

            written, skipped = write_items_to_dynamodb(
                items,
                table,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
            )
            total_written += written
            total_skipped += skipped
            logger.info("%s: %d written, %d skipped (already existed)",
                        coin_id, written, skipped)

            # Polite delay between coins
            time.sleep(1)

    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE")
    logger.info("  Total written: %d", total_written)
    logger.info("  Total skipped: %d (live records protected)", total_skipped)
    logger.info("  Total updated: %d", total_updated)

    if run_only in ("all", "ohlcv"):
        logger.info("  Approx records in table now: %d",
                    total_written + 1500)

    if not args.dry_run and run_only == "all":
        logger.info("\nNext steps:")
        logger.info("  1. python granger_causality.py --data-path <export_csv>")
        logger.info("  2. python train_xgboost.py --data-path <export_csv> --all-coins")
        total_written += written
        total_skipped += skipped
        logger.info("%s: %d written, %d skipped (already existed)",
                    coin_id, written, skipped)

        # Polite delay between coins
        time.sleep(1)

    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE")
    logger.info("  Total written: %d", total_written)
    logger.info("  Total skipped: %d (live records protected)", total_skipped)
    logger.info("  Approx records in table now: %d",
                total_written + 1500)

    if not args.dry_run:
        logger.info("\nNext steps:")
        logger.info("  1. python granger_causality.py --data-path <export_csv>")
        logger.info("  2. python train_xgboost.py --data-path <export_csv> --all-coins")


if __name__ == "__main__":
    main()
