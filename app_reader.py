from __future__ import annotations
import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import Any

import boto3
import requests
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# AWS Clients
_kinesis   = boto3.client("kinesis")
_sqs       = boto3.client("sqs")

PRICE_STREAM  = os.environ["PRICE_STREAM_NAME"]
SOCIAL_STREAM = os.environ["SOCIAL_STREAM_NAME"]
DLQ_URL       = os.environ.get("DLQ_URL", "")

COINGECKO_SECRET = os.environ.get("COINGECKO_SECRET")
NEWSAPI_SECRET   = os.environ.get("NEWSAPI_SECRET")
TWITTER_API_KEY  = os.environ.get("TWITTER_API_KEY")
GOOGLE_TRENDS_GEO = os.environ.get("GOOGLE_TRENDS_GEO", "")
GOOGLE_TRENDS_INTERVAL_MINUTES = max(1, int(os.environ.get("GOOGLE_TRENDS_INTERVAL_MINUTES", "60")))
NEWSAPI_INTERVAL_MINUTES = max(5, int(os.environ.get("NEWSAPI_INTERVAL_MINUTES", "30")))

COINS = {
    "bitcoin":      "BTC-USD",
    "ethereum":     "ETH-USD",
    "binancecoin":  "BNB-USD",
    "ripple":       "XRP-USD",
    "litecoin":     "LTC-USD",
}

GLOBAL_CRYPTO_KEYWORDS = [
    "crypto",
    "cryptocurrency",
    "crypto market",
    "altcoin",
]

COIN_SEARCH_KEYWORDS = {
    "BTC-USD": ["bitcoin", "btc", "btcusd", "bitcoin price"],
    "ETH-USD": ["ethereum", "eth", "ethusd", "ethereum price"],
    "BNB-USD": ["bnb", "binance coin", "bnbusd"],
    "XRP-USD": ["xrp", "ripple", "xrpusd"],
    "LTC-USD": ["litecoin", "ltc", "ltcusd"],
}

# ── Kinesis helpers ───────────────────────────────────────────
def put_kinesis_records(stream_name: str, records: list[dict], partition_key: str) -> int:
    failed = 0
    for i in range(0, len(records), 500):
        chunk = records[i:i+500]
        try:
            response = _kinesis.put_records(
                Records=[
                    {
                        "Data": json.dumps(r, default=str).encode("utf-8"),
                        "PartitionKey": partition_key,
                    } for r in chunk
                ],
                StreamName=stream_name,
            )
            failed_count = response.get("FailedRecordCount", 0)
            failed += failed_count
            if failed_count > 0:
                _send_to_dlq("Kinesis partial failure", chunk)
        except ClientError as e:
            logger.error("Kinesis error: %s", e)
            failed += len(chunk)
            _send_to_dlq(str(e), chunk)
    return failed


def _send_to_dlq(error: str, payload: Any):
    if not DLQ_URL:
        return
    try:
        _sqs.send_message(
            QueueUrl=DLQ_URL,
            MessageBody=json.dumps({"error": error, "payload": payload}, default=str),
        )
    except Exception:
        pass


def _dedupe_terms(terms: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for term in terms:
        normalized = term.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(term.strip())
    return deduped


def _twitter_query_terms(limit: int = 12) -> list[str]:
    terms = list(GLOBAL_CRYPTO_KEYWORDS)
    for keywords in COIN_SEARCH_KEYWORDS.values():
        terms.extend(keywords)
    return _dedupe_terms(terms)[:limit]


def _news_query() -> str:
    terms = []
    for coin_id in ("BTC-USD", "ETH-USD", "BNB-USD"):
        terms.append(COIN_SEARCH_KEYWORDS[coin_id][0])
    terms.append(GLOBAL_CRYPTO_KEYWORDS[0])
    return " OR ".join(_dedupe_terms(terms))


# ── CoinGecko fetcher ─────────────────────────────────────────
def fetch_prices(api_key: str) -> list[dict]:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": ",".join(COINS.keys()),
        "order": "market_cap_desc",
        "per_page": 10,
        "sparkline": False,
    }
    headers = {"x-cg-demo-api-key": api_key}

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)

            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue

            resp.raise_for_status()
            now = datetime.now(timezone.utc).isoformat()

            records = []
            for coin in resp.json():
                coin_id = COINS.get(coin["id"])
                if not coin_id:
                    continue

                records.append({
                    "event_type": "price",
                    "coin_id": coin_id,
                    "timestamp": now,
                    "close": coin.get("current_price", 0),
                    "volume": coin.get("total_volume", 0),
                    "source": "coingecko",
                })

            logger.info("Fetched %d price records", len(records))
            return records

        except requests.RequestException as e:
            logger.error("Price fetch error: %s", e)
            time.sleep(2 ** attempt)

    return []


# ── TwitterAPI.io fetcher ─────────────────────────────────────
def fetch_tweets(api_key: str) -> list[dict]:
    url = "https://api.twitterapi.io/twitter/tweet/advanced_search"

    query_terms = _twitter_query_terms()
    query = "(" + " OR ".join(f'"{term}"' if " " in term else term for term in query_terms) + ") lang:en -is:retweet"

    params = {
        "query": query,
        "limit": 100
    }

    headers = {
        "X-API-Key": api_key
    }

    records = []
    seen_ids = set()

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)

            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue

            resp.raise_for_status()

            data = resp.json()
            tweets = data.get("tweets") or data.get("data") or []

            for tweet in tweets:
                tweet_id = str(tweet.get("id"))
                if tweet_id in seen_ids:
                    continue
                seen_ids.add(tweet_id)

                text = tweet.get("text", "")
                coin_id = _classify_coin(text)

                records.append({
                    "event_type": "social",
                    "platform": "twitter",
                    "coin_id": coin_id,
                    "timestamp": tweet.get("created_at"),
                    "text": text[:512],
                    "author_id": tweet.get("user", {}).get("username"),
                    "likes": tweet.get("favorite_count", 0),
                    "retweets": tweet.get("retweet_count", 0),
                    "comments": tweet.get("reply_count", 0),
                    "tweet_id": tweet_id,
                })

            logger.info("Fetched %d tweets", len(records))
            return records

        except requests.RequestException as e:
            logger.error("TwitterAPI error: %s", e)
            time.sleep(5 * (attempt + 1))

    return []


# ── NewsAPI fetcher ───────────────────────────────────────────
def fetch_news(news_api_key: str) -> list[dict]:
    now = datetime.now(timezone.utc)
    interval_seconds = NEWSAPI_INTERVAL_MINUTES * 60
    current_epoch = int(now.timestamp())
    if current_epoch % interval_seconds >= 60:
        logger.info(
            "Skipping NewsAPI fetch; configured interval is every %d minute(s).",
            NEWSAPI_INTERVAL_MINUTES,
        )
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": _news_query(),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 25,
        "apiKey": news_api_key,
    }

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=10)

            if resp.status_code == 429:
                wait_seconds = 15 * (attempt + 1)
                logger.warning("NewsAPI rate limited (429). Backing off for %d seconds.", wait_seconds)
                time.sleep(wait_seconds)
                continue

            resp.raise_for_status()

            records = []
            for article in resp.json().get("articles", []):
                text = f"{article.get('title','')} {article.get('description','')}"
                coin_id = _classify_coin(text)

                records.append({
                    "event_type": "social",
                    "platform": "news",
                    "coin_id": coin_id,
                    "timestamp": article.get("publishedAt"),
                    "text": text[:512],
                    "url": article.get("url", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "likes": 0,
                    "retweets": 0,
                    "comments": 0,
                })

            logger.info("Fetched %d news records", len(records))
            return records

        except Exception as e:
            logger.error("News fetch error: %s", e)
            time.sleep(5 * (attempt + 1))

    return []


def fetch_fear_greed_index() -> list[dict]:
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": 1, "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("data", [])
        if not items:
            return []

        item = items[0]
        value = int(item.get("value", 0))
        value_classification = item.get("value_classification", "unknown")
        timestamp = item.get("timestamp")
        if timestamp:
            event_time = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()
        else:
            event_time = datetime.now(timezone.utc).isoformat()

        text = (
            f"Crypto Fear and Greed Index is {value} "
            f"which is classified as {value_classification}."
        )

        records = []
        for coin_id in COINS.values():
            records.append({
                "event_type": "social",
                "platform": "fear_greed",
                "coin_id": coin_id,
                "timestamp": event_time,
                "text": text,
                "index_value": value,
                "index_classification": value_classification,
                "source": "alternative_me",
                "likes": 0,
                "retweets": 0,
                "comments": 0,
            })

        logger.info("Fetched Fear & Greed Index=%d (%s)", value, value_classification)
        return records

    except Exception as e:
        logger.error("Fear & Greed fetch error: %s", e)
        return []


def fetch_google_trends(geo: str = "") -> list[dict]:
    now = datetime.now(timezone.utc)
    interval_seconds = GOOGLE_TRENDS_INTERVAL_MINUTES * 60
    current_epoch = int(now.timestamp())
    if current_epoch % interval_seconds >= 60:
        logger.info(
            "Skipping Google Trends fetch; configured interval is every %d minute(s).",
            GOOGLE_TRENDS_INTERVAL_MINUTES,
        )
        return []

    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.warning("pytrends is not installed; skipping Google Trends fetch.")
        return []

    try:
        trend = TrendReq(hl="en-US", tz=360)
        keywords = list(COINS.keys())
        trend.build_payload(keywords, timeframe="now 1-d", geo=geo)
        interest = trend.interest_over_time()
        if interest.empty:
            return []

        latest = interest.iloc[-1]
        event_time = latest.name
        if hasattr(event_time, "to_pydatetime"):
            event_time = event_time.to_pydatetime()
        if isinstance(event_time, datetime):
            timestamp = event_time.astimezone(timezone.utc).isoformat()
        else:
            timestamp = datetime.now(timezone.utc).isoformat()

        records = []
        for keyword in keywords:
            value = int(latest.get(keyword, 0))
            records.append({
                "event_type": "social",
                "platform": "google_trends",
                "coin_id": COINS[keyword],
                "timestamp": timestamp,
                "text": f"Google Trends interest for {keyword} is {value}.",
                "trend_value": value,
                "keyword": keyword,
                "source": "google_trends",
                "likes": 0,
                "retweets": 0,
                "comments": 0,
            })

        logger.info("Fetched %d Google Trends records", len(records))
        return records
    except Exception as e:
        logger.error("Google Trends fetch error: %s", e)
        return []


# ── Coin classifier ───────────────────────────────────────────
_COIN_KEYWORDS = {
    "BTC-USD": ["bitcoin", "btc", "btcusd", "bitcoin price"],
    "ETH-USD": ["ethereum", "eth", "ethusd", "ethereum price"],
    "BNB-USD": ["bnb", "binance", "binance coin", "bnbusd"],
    "XRP-USD": ["xrp", "ripple", "xrpusd"],
    "LTC-USD": ["litecoin", "ltc", "ltcusd"],
}

def _classify_coin(text: str) -> str:
    text = text.lower()
    scores = {coin: 0 for coin in _COIN_KEYWORDS}

    for coin, kws in _COIN_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                scores[coin] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "BTC-USD"


# ── Lambda handler ────────────────────────────────────────────
def lambda_handler(event, context):

    total_price_failed = 0
    total_social_failed = 0

    # PRICE
    price_records = fetch_prices(COINGECKO_SECRET)
    if price_records:
        failed = put_kinesis_records(PRICE_STREAM, price_records, "price")
        total_price_failed += failed

    # SOCIAL (Twitter + News + Fear & Greed + Google Trends)
    social_records = []

    tweets = fetch_tweets(TWITTER_API_KEY)
    social_records.extend(tweets)

    news = fetch_news(NEWSAPI_SECRET)
    social_records.extend(news)

    fear_greed = fetch_fear_greed_index()
    social_records.extend(fear_greed)

    google_trends = fetch_google_trends(GOOGLE_TRENDS_GEO)
    social_records.extend(google_trends)

    if social_records:
        failed = put_kinesis_records(SOCIAL_STREAM, social_records, "social")
        total_social_failed += failed

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Price + Social collection complete",
            "price_records": len(price_records),
            "tweets": len(tweets),
            "news": len(news),
            "fear_greed": len(fear_greed),
            "google_trends": len(google_trends),
            "total_social_records": len(social_records),
            "price_failed": total_price_failed,
            "social_failed": total_social_failed
        })
    }
