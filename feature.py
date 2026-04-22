# lambdas/feature_engineer/handler.py
"""
Lambda – Feature Engineer
Triggered every 5 minutes by EventBridge.
Reads the last 24 hours of metrics from DynamoDB for all coins,
builds the joint feature tensor, and writes it to DynamoDB for
the Predictor Lambda to consume.
"""
from __future__ import annotations
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from feature_builder import FeatureBuilder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_dynamodb = boto3.resource("dynamodb")

METRICS_TABLE    = os.environ.get("CRYPTO_METRICS_TABLE", "crypto_metrics")
FEATURES_TABLE   = os.environ.get("CRYPTO_FEATURES_TABLE", "crypto_features")
LOOKBACK_HOURS   = int(os.environ.get("LOOKBACK_HOURS", "24"))
BUCKET_MINUTES   = 15

COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "LTC-USD"]


def _get_metrics_window(coin_id: str, hours: int = LOOKBACK_HOURS) -> list[dict]:
    """Query DynamoDB for the last N hours of metrics for a coin."""
    table = _dynamodb.Table(METRICS_TABLE)
    now   = datetime.now(timezone.utc)
    start = (now - timedelta(hours=hours)).replace(second=0, microsecond=0)
    # truncate to bucket
    start_bucket = start.replace(
        minute=(start.minute // BUCKET_MINUTES) * BUCKET_MINUTES
    ).isoformat()

    try:
        response = table.query(
            KeyConditionExpression=(
                Key("coin_id").eq(coin_id) &
                Key("timestamp_bucket").gte(start_bucket)
            ),
            ScanIndexForward=True,
        )
        return response.get("Items", [])
    except ClientError as e:
        logger.error("DynamoDB query error for %s: %s", coin_id, e)
        return []


def _write_features(coin_id: str, features: dict, timestamp: str):
    """Write computed feature vector to features table."""
    table = _dynamodb.Table(FEATURES_TABLE)
    item = {
        "coin_id":   coin_id,
        "timestamp": timestamp,
        "features":  {k: str(v) for k, v in features.items()},
        "feature_count": len(features),
        "ttl": int(datetime.now(timezone.utc).timestamp() + 3600),  # 1h TTL
    }
    try:
        table.put_item(Item=item)
    except ClientError as e:
        logger.error("Failed to write features for %s: %s", coin_id, e)


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Main handler: fetches metrics for all coins, builds feature vectors,
    writes to features table. BTC features are built first so they can
    be used as cross-correlation inputs for other coins.
    """
    now_str = datetime.now(timezone.utc).isoformat()
    results = {}

    # ── Step 1: fetch BTC records first (needed for cross-asset correlation) ──
    btc_records = _get_metrics_window("BTC-USD")
    logger.info("BTC records fetched: %d", len(btc_records))

    # ── Step 2: build features for each coin ─────────────────────────────────
    for coin_id in COINS:
        try:
            if coin_id == "BTC-USD":
                records = btc_records
                builder = FeatureBuilder(btc_records=None)
            else:
                records = _get_metrics_window(coin_id)
                builder = FeatureBuilder(btc_records=btc_records)

            if len(records) < 4:
                logger.warning("Insufficient data for %s (%d records)", coin_id, len(records))
                results[coin_id] = {"status": "insufficient_data", "records": len(records)}
                continue

            features = builder.build(coin_id, records)
            if not features:
                results[coin_id] = {"status": "feature_build_failed"}
                continue

            _write_features(coin_id, features, now_str)
            results[coin_id] = {
                "status": "ok",
                "feature_count": len(features),
                "records_used": len(records),
            }
            logger.info(
                "Features built for %s: %d features from %d records",
                coin_id, len(features), len(records),
            )

        except Exception as e:
            logger.error("Feature build error for %s: %s", coin_id, e, exc_info=True)
            results[coin_id] = {"status": "error", "message": str(e)}

    return {
        "statusCode": 200,
        "body": json.dumps({
            "timestamp": now_str,
            "coins": results,
        }),
    }
