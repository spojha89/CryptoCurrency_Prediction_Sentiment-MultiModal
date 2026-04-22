from __future__ import annotations
import base64
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from boto3.dynamodb.conditions import Key

from technical_analysis import compute_all_indicators
from sentiment_engine import compute_aggregate_sentiment

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import boto3
_dynamodb = boto3.resource("dynamodb")

METRICS_TABLE = os.environ.get("CRYPTO_METRICS_TABLE")
if not METRICS_TABLE:
    # fail fast when environment is misconfigured
    raise RuntimeError("CRYPTO_METRICS_TABLE environment variable is required")

BUCKET_MINUTES = 15  # 15-minute sentiment/TA aggregation window
INDICATOR_LOOKBACK_BUCKETS = 96

def _bucket(ts_str: str, minutes: int = BUCKET_MINUTES) -> str:
    """Truncate an ISO timestamp string to the nearest N-minute bucket."""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        dt = datetime.now(timezone.utc)
    truncated = dt.replace(
        minute=(dt.minute // minutes) * minutes,
        second=0,
        microsecond=0,
    )
    return truncated.isoformat()

def _decode_record(record: dict) -> dict | None:
    """Base64-decode and JSON-parse a Kinesis record."""
    try:
        raw = base64.b64decode(record["kinesis"]["data"]).decode("utf-8")
        return json.loads(raw)
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        logger.warning("Failed to decode Kinesis record: %s", e)
        return None

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _aggregate_price_records(price_records: list[dict]) -> dict[str, float] | None:
    """Aggregate all raw price updates in a bucket into a single OHLCV bar."""
    if not price_records:
        return None

    ordered = sorted(price_records, key=lambda r: r.get("timestamp", ""))
    first = ordered[0]
    last = ordered[-1]
    highs = [_safe_float(r.get("high", r.get("close", 0))) for r in ordered]
    lows = [_safe_float(r.get("low", r.get("close", 0))) for r in ordered]
    volumes = [_safe_float(r.get("volume", 0)) for r in ordered]

    return {
        "open": _safe_float(first.get("open", first.get("close", 0))),
        "close": _safe_float(last.get("close", 0)),
        "high": max(highs) if highs else 0.0,
        "low": min(lows) if lows else 0.0,
        "volume": sum(volumes),
    }


def _get_recent_price_history(
    coin_id: str,
    before_bucket: str,
    limit: int = INDICATOR_LOOKBACK_BUCKETS - 1,
) -> list[dict]:
    """Fetch recent historical OHLCV bars before the current bucket."""
    table = _dynamodb.Table(METRICS_TABLE)
    try:
        response = table.query(
            KeyConditionExpression=(
                Key("coin_id").eq(coin_id) &
                Key("timestamp_bucket").lt(before_bucket)
            ),
            ScanIndexForward=False,
            Limit=limit,
        )
    except Exception as e:
        logger.error("DynamoDB history query error for %s/%s: %s", coin_id, before_bucket, e)
        return []

    items = response.get("Items", [])
    items.reverse()
    return [item for item in items if item.get("close") is not None]


def _write_metrics_record(coin_id: str, bucket: str, indicators: dict, ohlcv: dict):
    """Update price and indicator fields without overwriting social fields."""
    if not ohlcv:
        logger.warning("_write_metrics_record called with empty ohlcv for %s/%s", coin_id, bucket)
        return

    table = _dynamodb.Table(METRICS_TABLE)
    expr_names = {
        "#open": "open",
        "#close": "close",
        "#high": "high",
        "#low": "low",
        "#volume": "volume",
        "#ttl": "ttl",
    }
    expr_values = {
        ":open": str(round(_safe_float(ohlcv.get("open", 0)), 6)),
        ":close": str(round(_safe_float(ohlcv.get("close", 0)), 6)),
        ":high": str(round(_safe_float(ohlcv.get("high", 0)), 6)),
        ":low": str(round(_safe_float(ohlcv.get("low", 0)), 6)),
        ":volume": str(round(_safe_float(ohlcv.get("volume", 0)), 6)),
        ":ttl": int(datetime.now(timezone.utc).timestamp() + 30 * 86400),
    }
    update_parts = [
        "#open = :open",
        "#close = :close",
        "#high = :high",
        "#low = :low",
        "#volume = :volume",
        "#ttl = :ttl",
    ]

    for k, v in indicators.items():
        if v is not None:
            if k in {"open", "close", "high", "low", "volume"}:
                continue
            expr_names[f"#{k}"] = k
            expr_values[f":{k}"] = str(v)
            update_parts.append(f"#{k} = :{k}")

    try:
        table.update_item(
            Key={"coin_id": coin_id, "timestamp_bucket": bucket},
            UpdateExpression="SET " + ", ".join(update_parts),
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values,
        )
    except Exception as e:
        logger.error("DynamoDB write_metrics error for %s/%s: %s", coin_id, bucket, e)


def _update_sentiment_record(
    coin_id: str,
    bucket: str,
    platform: str,
    sentiment_result: dict,
):
    """Update sentiment fields on an existing DynamoDB metrics record."""
    table = _dynamodb.Table(METRICS_TABLE)
    score = sentiment_result["score"]
    count = sentiment_result["post_count"]
    try:
        table.update_item(
            Key={"coin_id": coin_id, "timestamp_bucket": bucket},
            UpdateExpression=(
                "SET #s = :score, #c = :count, "
                "sentiment_method = :method"
            ),
            ExpressionAttributeNames={
                "#s": f"sentiment_{platform}",
                "#c": f"{platform}_count",
            },
            ExpressionAttributeValues={
                ":score":  str(round(score, 4)),
                ":count":  count,
                ":method": sentiment_result.get("method", "unknown"),
            },
        )
    except Exception as e:
        logger.error(
            "DynamoDB update_sentiment error for %s/%s/%s: %s",
            coin_id, bucket, platform, e,
        )


def _update_fear_greed_record(
    coin_id: str,
    bucket: str,
    posts: list[dict],
):
    """Update Fear & Greed index fields on an existing DynamoDB metrics record."""
    if not posts:
        return

    values = []
    classifications = []
    for post in posts:
        try:
            values.append(float(post.get("index_value", 0)))
        except (TypeError, ValueError):
            continue
        classification = post.get("index_classification")
        if classification:
            classifications.append(str(classification))

    if not values:
        return

    avg_value = sum(values) / len(values)
    latest_classification = classifications[-1] if classifications else "unknown"

    table = _dynamodb.Table(METRICS_TABLE)
    try:
        table.update_item(
            Key={"coin_id": coin_id, "timestamp_bucket": bucket},
            UpdateExpression=(
                "SET fear_greed_value = :value, "
                "fear_greed_classification = :classification, "
                "fear_greed_count = :count"
            ),
            ExpressionAttributeValues={
                ":value": str(round(avg_value, 4)),
                ":classification": latest_classification,
                ":count": len(values),
            },
        )
    except Exception as e:
        logger.error(
            "DynamoDB update_fear_greed error for %s/%s: %s",
            coin_id, bucket, e,
        )


def _update_google_trends_record(
    coin_id: str,
    bucket: str,
    posts: list[dict],
):
    """Update Google Trends fields on an existing DynamoDB metrics record."""
    if not posts:
        return

    values = []
    for post in posts:
        try:
            values.append(float(post.get("trend_value", 0)))
        except (TypeError, ValueError):
            continue

    if not values:
        return

    avg_value = sum(values) / len(values)
    table = _dynamodb.Table(METRICS_TABLE)
    try:
        table.update_item(
            Key={"coin_id": coin_id, "timestamp_bucket": bucket},
            UpdateExpression=(
                "SET google_trends_value = :value, "
                "google_trends_count = :count"
            ),
            ExpressionAttributeValues={
                ":value": str(round(avg_value, 4)),
                ":count": len(values),
            },
        )
    except Exception as e:
        logger.error(
            "DynamoDB update_google_trends error for %s/%s: %s",
            coin_id, bucket, e,
        )

def _process_price_events(price_events: list[dict]) -> int:
    """
    Group price events by (coin_id, bucket), compute TA indicators,
    and write to DynamoDB. Returns number of records written.
    """
    # Group: coin_id → bucket → list of OHLCV records
    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for ev in price_events:
        coin_id = ev.get("coin_id", "BTC-USD")
        bucket  = _bucket(ev.get("timestamp", ""))
        grouped[coin_id][bucket].append(ev)

    # logger.info("price_events grouped: %s", grouped)

    written = 0
    for coin_id, buckets in grouped.items():
        for bucket, records in buckets.items():
            current_bar = _aggregate_price_records(records)
            if current_bar is None:
                continue

            history = _get_recent_price_history(coin_id, bucket)
            indicator_input = history + [{"timestamp_bucket": bucket, **current_bar}]
            indicators = compute_all_indicators(indicator_input)
            _write_metrics_record(coin_id, bucket, indicators, current_bar)
            written += 1
    
    logger.info("Completed processing price events: %d records → %d DynamoDB writes", len(price_events), written)
    return written

def _process_social_events(social_events: list[dict]) -> int:
    """
    Group social events by (coin_id, platform, bucket), compute weighted
    sentiment score, and update DynamoDB records. Returns updates count.
    """
    # Group: coin_id → platform → bucket → list of posts
    grouped: dict[str, dict[str, dict[str, list]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for ev in social_events:
        coin_id  = ev.get("coin_id", "BTC-USD")
        platform = ev.get("platform", "news")
        bucket   = _bucket(ev.get("timestamp", ""))
        grouped[coin_id][platform][bucket].append(ev)

    updated = 0
    for coin_id, platforms in grouped.items():
        for platform, buckets in platforms.items():
            for bucket, posts in buckets.items():
                if platform == "fear_greed":
                    _update_fear_greed_record(coin_id, bucket, posts)
                    updated += 1
                    continue

                if platform == "google_trends":
                    _update_google_trends_record(coin_id, bucket, posts)
                    updated += 1
                    continue

                sentiment_result = compute_aggregate_sentiment(
                    posts,
                    use_bedrock=(os.environ.get("USE_BEDROCK", "true").lower() == "true"),
                )
                _update_sentiment_record(coin_id, bucket, platform, sentiment_result)
                updated += 1

    return updated

def lambda_handler(event: dict, context: Any) -> dict:
    """
    Kinesis trigger handler.
    Separates price and social records from the batch, processes each.
    """
    # print("Received event:", json.dumps(event))  # log the raw event for debugging
    records = event.get("Records", [])
    if not records:
        return {"statusCode": 200, "body": "No records"}

    price_events:  list[dict] = []
    social_events: list[dict] = []

    for raw_record in records:
        decoded = _decode_record(raw_record)
        # print("Decoded record:", decoded)  # log the decoded record for debugging
        if not decoded:
            continue
        if decoded.get("event_type") == "price":
            price_events.append(decoded)
        elif decoded.get("event_type") == "social":
            social_events.append(decoded)

    # print(f"Price_events: {price_events}")  # log price events for debugging
    price_written  = _process_price_events(price_events)   if price_events  else 0
    social_updated = _process_social_events(social_events) if social_events else 0

    logger.info(
        "Processed batch: %d price records (→ %d DDB writes)",
        len(price_events), price_written,
    )

    return {
        "statusCode": 200,
        "body": json.dumps({
            "price_records":    len(price_events),
            "social_records":   len(social_events),
            "ddb_writes":       price_written,
            "ddb_updates":      social_updated,
        }),
    }
