"""
Backfill technical indicators in the crypto_metrics DynamoDB table.

This script reads existing OHLCV bucket records for each coin, recomputes
technical indicators over the historical series, and updates the stored rows
without touching sentiment or macro fields.

Usage:
  python backfill_indicators.py
  python backfill_indicators.py --days 14
  python backfill_indicators.py --coin BTC-USD
  python backfill_indicators.py --start-date 2021-01-01 --end-date 2023-12-31
  python backfill_indicators.py --all-history
  python backfill_indicators.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("backfill_indicators")

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
METRICS_TABLE = os.environ.get("CRYPTO_METRICS_TABLE", "crypto_metrics")
COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "LTC-USD"]
INDICATOR_FIELDS = [
    "rsi",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_position",
    "vwap",
]

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(METRICS_TABLE)

SCRIPT_DIR = Path(__file__).resolve().parent
CRYPTO_PROCESSOR_DIR = SCRIPT_DIR.parent / "crypto_processor"
if str(CRYPTO_PROCESSOR_DIR) not in sys.path:
    sys.path.insert(0, str(CRYPTO_PROCESSOR_DIR))

from technical_analysis import compute_all_indicators  # noqa: E402


def query_coin_rows(
    coin_id: str,
    days: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    all_history: bool = False,
) -> list[dict]:
    items: list[dict] = []

    if all_history:
        start = None
        end = None
    else:
        start = (
            datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).isoformat()
            if start_date
            else (datetime.now(timezone.utc) - timedelta(days=days or 30)).isoformat()
        )
        end = (
            datetime.fromisoformat(end_date).replace(
                hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc
            ).isoformat()
            if end_date
            else None
        )

    try:
        key_expr = Key("coin_id").eq(coin_id)
        if start and end:
            key_expr = key_expr & Key("timestamp_bucket").between(start, end)
        elif start:
            key_expr = key_expr & Key("timestamp_bucket").gte(start)

        response = table.query(KeyConditionExpression=key_expr, ScanIndexForward=True)
        items.extend(response.get("Items", []))

        while "LastEvaluatedKey" in response:
            response = table.query(
                KeyConditionExpression=key_expr,
                ScanIndexForward=True,
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            items.extend(response.get("Items", []))
    except ClientError as e:
        logger.error("Query failed for %s: %s", coin_id, e)
        return []

    valid_items = [item for item in items if item.get("close") is not None]
    logger.info("Fetched %d rows for %s (%d with price data)", len(items), coin_id, len(valid_items))
    return valid_items


def update_indicator_row(coin_id: str, timestamp_bucket: str, indicators: dict, dry_run: bool) -> bool:
    expr_names = {}
    expr_values = {}
    update_parts = []

    for field in INDICATOR_FIELDS:
        value = indicators.get(field)
        if value is None:
            continue
        expr_names[f"#{field}"] = field
        expr_values[f":{field}"] = str(value)
        update_parts.append(f"#{field} = :{field}")

    if not update_parts:
        return False

    if dry_run:
        logger.info("DRY RUN update %s/%s -> %s", coin_id, timestamp_bucket, update_parts)
        return True

    try:
        table.update_item(
            Key={"coin_id": coin_id, "timestamp_bucket": timestamp_bucket},
            UpdateExpression="SET " + ", ".join(update_parts),
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values,
        )
        return True
    except ClientError as e:
        logger.error("Update failed for %s/%s: %s", coin_id, timestamp_bucket, e)
        return False


def row_has_missing_indicators(row: dict) -> bool:
    return any(row.get(field) in (None, "") for field in INDICATOR_FIELDS)


def backfill_coin(coin_id: str, rows: list[dict], dry_run: bool, missing_only: bool = True) -> int:
    updated = 0
    history: list[dict] = []
    skipped_populated = 0

    for row in rows:
        history.append(row)
        if missing_only and not row_has_missing_indicators(row):
            skipped_populated += 1
            continue
        indicators = compute_all_indicators(history)
        if update_indicator_row(coin_id, row["timestamp_bucket"], indicators, dry_run=dry_run):
            updated += 1

    logger.info(
        "Backfilled %d rows for %s (%d already had indicator values)",
        updated,
        coin_id,
        skipped_populated,
    )
    return updated


def main():
    parser = argparse.ArgumentParser(description="Backfill technical indicators in crypto_metrics")
    parser.add_argument("--days", type=int, default=30, help="Number of days to backfill")
    parser.add_argument("--coin", choices=COINS, help="Backfill a single coin only")
    parser.add_argument("--start-date", type=str, help="Backfill from YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, help="Backfill through YYYY-MM-DD")
    parser.add_argument("--all-history", action="store_true", help="Backfill the full coin history")
    parser.add_argument(
        "--include-populated",
        action="store_true",
        help="Recalculate rows even when indicator fields are already populated",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview updates without writing")
    args = parser.parse_args()

    coins = [args.coin] if args.coin else COINS
    total_updated = 0

    for coin_id in coins:
        rows = query_coin_rows(
            coin_id,
            days=args.days,
            start_date=args.start_date,
            end_date=args.end_date,
            all_history=args.all_history,
        )
        if not rows:
            logger.warning("Skipping %s because no valid OHLCV rows were found", coin_id)
            continue
        total_updated += backfill_coin(
            coin_id,
            rows,
            dry_run=args.dry_run,
            missing_only=not args.include_populated,
        )

    logger.info("Done. Total rows processed: %d", total_updated)


if __name__ == "__main__":
    main()
