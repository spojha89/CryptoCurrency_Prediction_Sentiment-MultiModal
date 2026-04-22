"""
training/export_metrics.py
==========================
Exports crypto_metrics DynamoDB table to CSV for model training.
Can export raw metrics only or build training-ready engineered features.

Usage:
  python export_metrics.py
  python export_metrics.py --days 7         # last 7 days only
  python export_metrics.py --all-history
  python export_metrics.py --start-date 2021-01-01 --end-date 2023-12-31
  python export_metrics.py --upload-s3      # also upload to S3
"""
from __future__ import annotations
import argparse
import csv
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("export_metrics")

AWS_REGION    = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
METRICS_TABLE = os.environ.get("CRYPTO_METRICS_TABLE", "crypto_metrics")
MODEL_BUCKET  = os.environ.get("MODEL_BUCKET", "")

COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "LTC-USD"]

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
s3       = boto3.client("s3", region_name=AWS_REGION)

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_EXTRACTOR_DIR = SCRIPT_DIR.parent / "feature_extractor"
if str(FEATURE_EXTRACTOR_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_EXTRACTOR_DIR))

from feature_builder import FeatureBuilder  # noqa: E402


# Preferred columns first; any additional fields found in DynamoDB rows
# will be appended automatically so the export stays forward-compatible.
BASE_COLUMNS = [
    "coin_id", "timestamp_bucket",
    # OHLCV
    "open", "high", "low", "close", "volume",
    # Technical indicators
    "rsi", "macd", "macd_signal", "macd_histogram",
    "bb_upper", "bb_middle", "bb_lower", "bb_position", "vwap",
    # Sentiment
    "sentiment_composite", "sentiment_twitter", "sentiment_news",
    "twitter_count", "news_count", "sentiment_method",
    # Regime / macro attention signals
    "fear_greed_value", "fear_greed_classification", "fear_greed_count",
    "google_trends_value", "google_trends_count",
]

LOOKBACK_RECORDS = 96  # 24h of 15-minute buckets


def query_coin(
    coin_id: str,
    days: int = 30,
    start_date: str | None = None,
    end_date: str | None = None,
    all_history: bool = False,
) -> list[dict]:
    """Query metrics records for a coin across the requested date range."""
    table = dynamodb.Table(METRICS_TABLE)

    if all_history:
        start = None
        end = None
    else:
        start = (
            datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).isoformat()
            if start_date
            else (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        )
        end = (
            datetime.fromisoformat(end_date).replace(
                hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc
            ).isoformat()
            if end_date
            else None
        )

    items = []
    try:
        key_expr = Key("coin_id").eq(coin_id)
        if start and end:
            key_expr = key_expr & Key("timestamp_bucket").between(start, end)
        elif start:
            key_expr = key_expr & Key("timestamp_bucket").gte(start)

        response = table.query(KeyConditionExpression=key_expr, ScanIndexForward=True)
        items.extend(response.get("Items", []))

        # Handle DynamoDB pagination
        while "LastEvaluatedKey" in response:
            response = table.query(
                KeyConditionExpression=key_expr,
                ScanIndexForward=True,
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            items.extend(response.get("Items", []))

    except ClientError as e:
        logger.error("Query error for %s: %s", coin_id, e)

    logger.info("Fetched %d records for %s", len(items), coin_id)
    return items


def clean_value(v):
    """Convert DynamoDB types to plain Python for CSV writing."""
    if isinstance(v, Decimal):
        return float(v)
    return v


def _normalize_row(item: dict) -> dict:
    return {k: clean_value(v) for k, v in item.items()}


def _build_training_rows_for_coin(coin_id: str, raw_rows_by_coin: dict[str, list[dict]]) -> list[dict]:
    """Build training rows for a single coin."""
    coin_rows = raw_rows_by_coin.get(coin_id, [])
    if not coin_rows:
        logger.warning("No rows available for feature building: %s", coin_id)
        return []

    built_rows = []
    skipped_short_window = 0
    skipped_missing_btc = 0
    btc_rows = raw_rows_by_coin.get("BTC-USD", [])
    btc_timestamps = [row.get("timestamp_bucket") for row in btc_rows]
    btc_pointer = 0

    for idx, row in enumerate(coin_rows):
        coin_window = coin_rows[max(0, idx - LOOKBACK_RECORDS + 1): idx + 1]
        if len(coin_window) < 2:
            skipped_short_window += 1
            continue

        if coin_id == "BTC-USD":
            builder = FeatureBuilder(btc_records=None)
        else:
            row_ts = row.get("timestamp_bucket")
            btc_idx = None
            if row_ts:
                while btc_pointer + 1 < len(btc_timestamps):
                    next_ts = btc_timestamps[btc_pointer + 1]
                    if not next_ts or next_ts > row_ts:
                        break
                    btc_pointer += 1
                current_ts = btc_timestamps[btc_pointer] if btc_timestamps else None
                if current_ts and current_ts <= row_ts:
                    btc_idx = btc_pointer
            if btc_idx is None:
                skipped_missing_btc += 1
                builder = FeatureBuilder(btc_records=None)
            else:
                btc_window = btc_rows[max(0, btc_idx - LOOKBACK_RECORDS + 1): btc_idx + 1]
                builder = FeatureBuilder(btc_records=btc_window)

        features = builder.build(coin_id, coin_window)
        if not features:
            continue

        merged = _normalize_row(row)
        merged.update({k: clean_value(v) for k, v in features.items()})
        built_rows.append(merged)

    logger.info(
        "Built %d training rows for %s from %d raw rows (short_window=%d, missing_btc_alignment=%d)",
        len(built_rows),
        coin_id,
        len(coin_rows),
        skipped_short_window,
        skipped_missing_btc,
    )
    return built_rows


def export_to_csv(
    output_path: str,
    days: int = 90,
    start_date: str | None = None,
    end_date: str | None = None,
    all_history: bool = False,
    raw_only: bool = False,
) -> int:
    """Export all coins to a single CSV file. Returns total row count."""
    discovered_columns = set(BASE_COLUMNS)
    total_rows = 0
    normalized_rows_by_coin: dict[str, list[dict]] = {}
    built_rows_by_coin: dict[str, list[dict]] = {}

    for coin_id in COINS:
        items = query_coin(
            coin_id,
            days=days,
            start_date=start_date,
            end_date=end_date,
            all_history=all_history,
        )
        normalized_rows = [_normalize_row(item) for item in items]
        normalized_rows_by_coin[coin_id] = normalized_rows

        source_rows = normalized_rows
        if not raw_only:
            source_rows = _build_training_rows_for_coin(coin_id, normalized_rows_by_coin)
            built_rows_by_coin[coin_id] = source_rows

        for item in source_rows:
            discovered_columns.update(item.keys())

    ordered_extra_columns = sorted(discovered_columns - set(BASE_COLUMNS))
    csv_columns = list(BASE_COLUMNS) + ordered_extra_columns

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

        for coin_id in COINS:
            rows_to_write = normalized_rows_by_coin[coin_id] if raw_only else built_rows_by_coin[coin_id]
            for row in rows_to_write:
                normalized = {col: row.get(col, "") for col in csv_columns}
                writer.writerow(normalized)
            total_rows += len(rows_to_write)

    logger.info(
        "Exported %d total rows with %d columns to %s",
        total_rows, len(csv_columns), output_path,
    )
    if not raw_only:
        engineered_columns = sorted(discovered_columns - set(BASE_COLUMNS))
        logger.info(
            "Engineered export includes %d additional feature columns.",
            len(engineered_columns),
        )
    return total_rows


def upload_to_s3(local_path: str, bucket: str):
    """Upload CSV to S3 exports/ prefix."""
    filename  = os.path.basename(local_path)
    s3_key    = f"exports/{filename}"
    try:
        s3.upload_file(local_path, bucket, s3_key)
        logger.info("Uploaded to s3://%s/%s", bucket, s3_key)
    except ClientError as e:
        logger.error("S3 upload failed: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Export crypto_metrics to CSV")
    parser.add_argument("--days",      type=int, default=90,
                        help="Number of days to export (default: 90)")
    parser.add_argument("--start-date", type=str,
                        help="Export from YYYY-MM-DD")
    parser.add_argument("--end-date", type=str,
                        help="Export through YYYY-MM-DD")
    parser.add_argument("--all-history", action="store_true",
                        help="Export the full history in the table")
    parser.add_argument("--raw-only", action="store_true",
                        help="Export raw DynamoDB metrics only without engineered features")
    parser.add_argument("--output",    type=str,
                        default=f"crypto_metrics_training_{datetime.now().strftime('%Y%m%d')}.csv",
                        help="Output CSV filename")
    parser.add_argument("--upload-s3", action="store_true",
                        help="Upload to S3 MODEL_BUCKET after export")
    args = parser.parse_args()

    row_count = export_to_csv(
        args.output,
        days=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
        all_history=args.all_history,
        raw_only=args.raw_only,
    )

    if row_count == 0:
        logger.warning("No data exported. Check that pipeline has been running.")
        sys.exit(1)

    if args.upload_s3:
        bucket = MODEL_BUCKET or os.environ.get("MODEL_BUCKET", "")
        if not bucket:
            logger.error("MODEL_BUCKET env var not set. Skipping S3 upload.")
        else:
            upload_to_s3(args.output, bucket)

    logger.info("Done. Use this file for training:")
    logger.info("  python granger_causality.py --data-path %s", args.output)
    logger.info("  python train_xgboost.py --data-path %s", args.output)
    logger.info("  python train_tft.py --data-path %s", args.output)


if __name__ == "__main__":
    main()
