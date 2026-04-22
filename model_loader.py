"""
Model loader for the predictor Lambda.

Loads per-coin XGBoost models from S3 and caches them in /tmp for warm
invocations so inference stays fast across repeated runs.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_s3 = boto3.client("s3")

MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "crypto-prediction-models")
MODEL_KEY_TEMPLATE = os.environ.get(
    "MODEL_KEY_TEMPLATE",
    "models/xgboost/{coin_id}/model.pkl",
)
METADATA_KEY_TEMPLATE = os.environ.get(
    "METADATA_KEY_TEMPLATE",
    "models/xgboost/{coin_id}/metadata.json",
)
CACHE_DIR = Path("/tmp/model_cache")

_model_cache: dict[str, tuple[str, Any]] = {}
_metadata_cache: dict[str, dict] = {}


def _coin_path_token(coin_id: str) -> str:
    """Convert coin ids like BTC-USD to safe S3 path segments."""
    return coin_id.replace("-", "_")


def _build_s3_keys(coin_id: str) -> tuple[str, str]:
    token = _coin_path_token(coin_id)
    model_key = MODEL_KEY_TEMPLATE.format(coin_id=token)
    metadata_key = METADATA_KEY_TEMPLATE.format(coin_id=token)
    return model_key, metadata_key


def _get_s3_etag(bucket: str, key: str) -> Optional[str]:
    """Get the ETag of an S3 object for cache invalidation."""
    try:
        response = _s3.head_object(Bucket=bucket, Key=key)
        return response.get("ETag", "").strip('"')
    except ClientError:
        return None


def load_model(coin_id: str, force_reload: bool = False) -> tuple[Any, dict]:
    """
    Load a coin-specific XGBoost model from S3 with local /tmp caching.
    Returns a `(model, metadata)` tuple.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_key, metadata_key = _build_s3_keys(coin_id)
    cache_prefix = _coin_path_token(coin_id).lower()

    current_etag = _get_s3_etag(MODEL_BUCKET, model_key) or "unknown"

    if not force_reload and model_key in _model_cache:
        cached_etag, cached_model = _model_cache[model_key]
        if cached_etag == current_etag:
            logger.info("Using in-memory cached model for %s", coin_id)
            return cached_model, _metadata_cache.get(model_key, {})

    model_path = CACHE_DIR / f"{cache_prefix}_model.pkl"
    etag_path = CACHE_DIR / f"{cache_prefix}_model.etag"
    metadata_path = CACHE_DIR / f"{cache_prefix}_metadata.json"

    if not force_reload and model_path.exists() and etag_path.exists():
        stored_etag = etag_path.read_text().strip()
        if stored_etag == current_etag:
            logger.info("Loading model for %s from /tmp cache", coin_id)
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                _model_cache[model_key] = (current_etag, model)
                _metadata_cache[model_key] = metadata
                return model, metadata
            except Exception as e:
                logger.warning("Failed to load /tmp cache for %s: %s", coin_id, e)

    logger.info("Downloading model for %s from s3://%s/%s", coin_id, MODEL_BUCKET, model_key)
    try:
        _s3.download_file(MODEL_BUCKET, model_key, str(model_path))
        etag_path.write_text(current_etag)

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        try:
            _s3.download_file(MODEL_BUCKET, metadata_key, str(metadata_path))
            with open(metadata_path) as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning("Could not load model metadata for %s: %s", coin_id, e)
            metadata = {"version": "unknown", "feature_names": [], "label_classes": [0, 1]}

        _model_cache[model_key] = (current_etag, model)
        _metadata_cache[model_key] = metadata
        return model, metadata

    except ClientError as e:
        logger.error("Failed to load model for %s from S3: %s", coin_id, e)
        raise RuntimeError(f"Model load failed: {e}") from e


def get_feature_names(coin_id: str) -> list[str]:
    """Return the ordered list of feature names the model expects."""
    _, metadata = load_model(coin_id)
    return metadata.get("feature_names", [])
