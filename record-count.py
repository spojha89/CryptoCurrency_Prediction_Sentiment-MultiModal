import os
import boto3
from boto3.dynamodb.conditions import Key


def get_dynamodb(region_name=None):
    return boto3.resource(
        "dynamodb",
        region_name=region_name or os.environ.get("AWS_REGION", "us-east-2"),
    )


def get_total_count(table, coin_id):
    if isinstance(table, str):
        ddb = get_dynamodb()
        table = ddb.Table(table)

    total = 0
    last_evaluated_key = None

    while True:
        params = {
            "KeyConditionExpression": Key("coin_id").eq(coin_id),
            "Select": "COUNT",
        }

        if last_evaluated_key:
            params["ExclusiveStartKey"] = last_evaluated_key

        response = table.query(**params)
        total += response.get("Count", 0)

        last_evaluated_key = response.get("LastEvaluatedKey")
        if not last_evaluated_key:
            break

    return total


def get_counts_for_currencies(table, coin_ids):
    return {coin_id: get_total_count(table, coin_id) for coin_id in coin_ids}


if __name__ == "__main__":
    metrics_table = os.environ.get("METRICS_TABLE", "crypto_metrics")
    coin_ids = [
        "BTC-USD",
        "ETH-USD",
        "BNB-USD",
        "XRP-USD",
        "LTC-USD",
    ]

    counts = get_counts_for_currencies(metrics_table, coin_ids)

    print("Total record counts by symbol:")
    for coin_id, total in counts.items():
        print(f"{coin_id}: {total}")
