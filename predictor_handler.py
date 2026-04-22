"""
Lambda - XGBoost predictor.

Triggered after feature extraction, reads the latest feature vector for each
coin, runs binary XGBoost inference, and writes predictions to DynamoDB and
optionally Redis/SNS.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import boto3
import numpy as np
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from model_loader import load_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_dynamodb = boto3.resource("dynamodb")
_sns = boto3.client("sns")

FEATURES_TABLE = os.environ.get(
    "CRYPTO_FEATURES_TABLE",
    os.environ.get("FEATURES_TABLE", "crypto_features"),
)
PREDICTIONS_TABLE = os.environ.get("PREDICTIONS_TABLE", "crypto_predictions")
REDIS_ENDPOINT = os.environ.get("REDIS_ENDPOINT", "")
SNS_TOPIC_ARN = os.environ.get("SNS_ALERT_TOPIC_ARN", "")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "xgboost-v1")
HIGH_CONFIDENCE = float(os.environ.get("HIGH_CONFIDENCE", "0.85"))

COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "LTC-USD"]
CLASS_LABELS = {
    0: "DOWN",
    1: "UP",
}

_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not REDIS_ENDPOINT:
        return None
    try:
        import redis

        _redis_client = redis.Redis(
            host=REDIS_ENDPOINT,
            port=6379,
            decode_responses=True,
            socket_timeout=2,
            socket_connect_timeout=2,
        )
        return _redis_client
    except Exception as e:
        logger.warning("Redis init failed: %s", e)
        return None


def _get_latest_features(coin_id: str) -> dict | None:
    """Fetch the most recently written feature vector for a coin."""
    table = _dynamodb.Table(FEATURES_TABLE)
    try:
        response = table.query(
            KeyConditionExpression=Key("coin_id").eq(coin_id),
            ScanIndexForward=False,
            Limit=1,
        )
        items = response.get("Items", [])
        if not items:
            return None
        raw_features = items[0].get("features", {})
        return {k: float(v) for k, v in raw_features.items()}
    except ClientError as e:
        logger.error("Feature fetch error for %s: %s", coin_id, e)
        return None


def _build_feature_vector(features: dict, feature_names: list[str]) -> np.ndarray:
    """Align features to the exact feature order expected by the model."""
    vector = np.array(
        [features.get(name, 0.0) for name in feature_names],
        dtype=np.float32,
    )
    return vector.reshape(1, -1)


def _signal_strength(confidence: float) -> int:
    """Map confidence score to signal strength 1-5."""
    if confidence >= 0.90:
        return 5
    if confidence >= 0.80:
        return 4
    if confidence >= 0.70:
        return 3
    if confidence >= 0.60:
        return 2
    return 1


def _write_prediction(record: dict):
    """Write prediction to DynamoDB."""
    table = _dynamodb.Table(PREDICTIONS_TABLE)
    try:
        record["ttl"] = int(datetime.now(timezone.utc).timestamp() + 7 * 86400)
        table.put_item(Item=record)
    except ClientError as e:
        logger.error("DynamoDB prediction write error: %s", e)


def _cache_prediction(coin_id: str, prediction: dict):
    """Write latest prediction to Redis for dashboard reads."""
    r = _get_redis()
    if r is None:
        return
    try:
        key = f"prediction:{coin_id}:latest"
        r.setex(key, 300, json.dumps(prediction, default=str))
    except Exception as e:
        logger.warning("Redis cache error for %s: %s", coin_id, e)


def _trigger_alert_if_needed(coin_id: str, prediction: dict):
    """Send SNS alert if prediction confidence exceeds threshold."""
    if not SNS_TOPIC_ARN:
        return
    confidence = float(prediction.get("confidence", 0))
    if confidence < HIGH_CONFIDENCE:
        return
    try:
        direction = prediction["predicted_direction"]
        message = (
            f"HIGH CONFIDENCE PREDICTION\n"
            f"Coin: {coin_id}\n"
            f"Direction: {direction}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Signal Strength: {prediction['signal_strength']}/5\n"
            f"Timestamp: {prediction['timestamp']}"
        )
        _sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=f"[Crypto Alert] {coin_id}: {direction} @ {confidence:.0%}",
            Message=message,
        )
        logger.info("Alert sent for %s: %s @ %.2f", coin_id, direction, confidence)
    except ClientError as e:
        logger.error("SNS publish error: %s", e)


def _resolve_binary_prediction(proba: np.ndarray) -> tuple[int, float, float, float]:
    """Normalize XGBoost binary probabilities into class/confidence outputs."""
    flat = np.asarray(proba, dtype=np.float32).reshape(-1)
    if flat.size == 1:
        prob_up = float(flat[0])
        prob_down = 1.0 - prob_up
    elif flat.size >= 2:
        prob_down = float(flat[0])
        prob_up = float(flat[1])
    else:
        raise ValueError("Empty probability array returned by model")

    class_idx = 1 if prob_up >= prob_down else 0
    confidence = max(prob_down, prob_up)
    return class_idx, confidence, prob_down, prob_up


def lambda_handler(event: dict, context: Any) -> dict:
    """Main prediction handler."""
    now_str = datetime.now(timezone.utc).isoformat()
    results = {}

    for coin_id in COINS:
        try:
            model, metadata = load_model(coin_id)
            feature_names = metadata.get("feature_names", [])
            label_classes = metadata.get("label_classes", [0, 1])
            model_version = metadata.get("version", MODEL_VERSION)

            if not feature_names:
                raise ValueError(f"No feature names found in metadata for {coin_id}")
            if list(label_classes) != [0, 1]:
                logger.warning(
                    "Unexpected label classes for %s: %s. Continuing with binary mapping.",
                    coin_id,
                    label_classes,
                )

            features = _get_latest_features(coin_id)
            if not features:
                results[coin_id] = {"status": "no_features"}
                continue

            X = _build_feature_vector(features, feature_names)
            proba = model.predict_proba(X)[0]
            class_idx, confidence, prob_down, prob_up = _resolve_binary_prediction(proba)
            direction = CLASS_LABELS[class_idx]

            prediction = {
                "prediction_id": str(uuid.uuid4()),
                "coin_id": coin_id,
                "timestamp": now_str,
                "predicted_direction": direction,
                "confidence": str(round(confidence, 4)),
                "signal_strength": _signal_strength(confidence),
                "prob_down": str(round(prob_down, 4)),
                "prob_up": str(round(prob_up, 4)),
                "model_version": model_version,
                "feature_count": len(feature_names),
                "horizon_h": metadata.get("horizon_h", 1),
            }

            _write_prediction(prediction)
            _cache_prediction(coin_id, prediction)
            _trigger_alert_if_needed(coin_id, prediction)

            results[coin_id] = {
                "status": "ok",
                "direction": direction,
                "confidence": round(confidence, 4),
            }
            logger.info(
                "Prediction for %s: %s (%.2f%%) signal=%d",
                coin_id,
                direction,
                confidence * 100,
                prediction["signal_strength"],
            )

        except RuntimeError as e:
            logger.error("Model load failed for %s: %s", coin_id, e)
            results[coin_id] = {"status": "model_error", "message": str(e)}
        except Exception as e:
            logger.error("Prediction error for %s: %s", coin_id, e, exc_info=True)
            results[coin_id] = {"status": "error", "message": str(e)}

    return {
        "statusCode": 200,
        "body": json.dumps({"timestamp": now_str, "predictions": results}),
    }
