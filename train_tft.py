# models/train_tft.py
"""
Temporal Fusion Transformer (TFT) Training Script
Primary deep learning model for multi-horizon cryptocurrency prediction.

Architecture (Lim et al., 2021):
  - Variable Selection Networks: learn feature importance per timestep
  - Gated Residual Networks (GRN): adaptive nonlinear transformations
  - LSTM encoder-decoder: captures sequential patterns
  - Multi-head self-attention: long-range temporal dependencies
  - Quantile outputs: prediction intervals (10th, 50th, 90th percentile)

This implementation uses PyTorch Forecasting library which provides
a production-grade TFT implementation.

Usage:
  pip install pytorch-forecasting pytorch-lightning
  python train_tft.py --data-path data/merged_features.csv --coin BTC-USD
"""
from __future__ import annotations
import argparse
import json
import logging
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)
try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COINS       = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "LTC-USD"]
MAX_ENCODER = 32   # 8 hours of history with 15-minute buckets
MAX_PRED    = 16   # predict next 4 hours by default
BATCH_SIZE  = 32
MAX_EPOCHS  = 20
DEFAULT_NUM_WORKERS = 0
MONTHLY_ROW_THRESHOLD = 5000
FAST_MODE_EPOCHS = 8
FAST_MODE_BATCH_SIZE = 64

CPU_FRIENDLY_HPARAMS = {
    "hidden_size": 16,
    "attention_head_size": 1,
    "dropout": 0.1,
    "hidden_continuous_size": 8,
    "learning_rate": 3e-3,
}

# Continuous known features (observed at each time step, values known in future)
TIME_FEATURES = ["hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend"]

# Continuous unknown features (only known up to the present)
OBSERVED_FEATURES = [
    "log_return", "volume_zscore", "return_1h", "return_4h",
    "volatility_1h", "volatility_4h",
    "rsi", "macd", "macd_histogram", "bb_position", "vwap_deviation",
    "rsi_overbought", "rsi_oversold", "macd_bullish_cross",
    "sentiment_composite", "sentiment_twitter", "sentiment_news",
    "sentiment_twitter_momentum",
    "fear_greed_value", "fear_greed_normalized", "fear_greed_momentum",
    "fear_greed_extreme_fear", "fear_greed_extreme_greed",
    "google_trends_value", "google_trends_normalized", "google_trends_momentum",
    "google_trends_high_interest",
    "btc_correlation_1h", "btc_correlation_4h",
]


def safe_log_return(series: pd.Series) -> pd.Series:
    previous = series.shift(1)
    valid_mask = (series > 0) & (previous > 0)
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    result.loc[valid_mask] = np.log(series.loc[valid_mask] / previous.loc[valid_mask])
    return result


def audit_input_dataset(df: pd.DataFrame) -> dict:
    available_time = [feature for feature in TIME_FEATURES if feature in df.columns]
    available_observed = [feature for feature in OBSERVED_FEATURES if feature in df.columns]
    observed_ratio = len(available_observed) / len(OBSERVED_FEATURES) if OBSERVED_FEATURES else 0.0

    audit = {
        "column_count": int(len(df.columns)),
        "available_time_features": available_time,
        "available_observed_features": available_observed,
        "observed_feature_ratio": round(observed_ratio, 3),
    }
    if audit["column_count"] <= 35 or observed_ratio < 0.5 or len(available_time) < 3:
        logger.warning(
            "Input dataset looks weak for TFT. Columns=%d observed_ratio=%.2f time_features=%d. "
            "Use the engineered export from export_metrics.py for best results.",
            audit["column_count"],
            observed_ratio,
            len(available_time),
        )
    return audit


def filter_sparse_features(
    df: pd.DataFrame,
    train_cutoff: int,
    zero_rate_threshold: float = 0.98,
) -> tuple[pd.DataFrame, list[str], dict]:
    candidate_features = [feature for feature in TIME_FEATURES + OBSERVED_FEATURES if feature in df.columns]
    train_df = df[df["time_idx"] <= train_cutoff]
    dropped_features: list[str] = []

    for feature in candidate_features:
        if feature in TIME_FEATURES:
            continue
        series = train_df[feature]
        zero_rate = float((series == 0).mean())
        nunique = int(series.nunique(dropna=False))
        variance = float(series.var(ddof=0)) if len(series) > 1 else 0.0
        if zero_rate >= zero_rate_threshold or nunique < 2 or variance == 0.0:
            dropped_features.append(feature)

    retained_features = [feature for feature in candidate_features if feature not in dropped_features]
    audit = {
        "input_feature_count": int(len(candidate_features)),
        "selected_feature_count": int(len(retained_features)),
        "dropped_feature_count": int(len(dropped_features)),
        "dropped_features": dropped_features,
        "zero_rate_threshold": zero_rate_threshold,
    }

    if dropped_features:
        logger.info("Dropped %d sparse TFT features: %s", len(dropped_features), dropped_features)
    else:
        logger.info("TFT feature audit kept all %d candidate features.", len(retained_features))

    return df, retained_features, audit


def classify_returns(values: np.ndarray, threshold: float) -> np.ndarray:
    labels = np.full(values.shape, 1, dtype=int)
    labels[values > threshold] = 2
    labels[values < -threshold] = 0
    return labels


def summarize_attention(interpretation: dict) -> dict:
    attention = interpretation.get("attention")
    if attention is None:
        return {}

    if hasattr(attention, "detach"):
        attention = attention.detach().cpu().numpy()
    attention = np.asarray(attention, dtype=float)
    if attention.size == 0:
        return {}

    reduced = attention
    while reduced.ndim > 1:
        reduced = reduced.mean(axis=0)

    top_indices = np.argsort(reduced)[::-1][:5]
    return {
        "mean_attention_by_lag": [round(float(v), 6) for v in reduced.tolist()],
        "top_attention_lags": [
            {"lag_index": int(idx), "weight": round(float(reduced[idx]), 6)}
            for idx in top_indices
        ],
    }


def build_horizon_table(
    actual_np: np.ndarray,
    pred_np: np.ndarray,
    direction_threshold: float,
) -> list[dict]:
    horizon_count = actual_np.shape[1]
    candidate_indices = [0, horizon_count // 2, horizon_count - 1]
    horizon_indices = []
    for idx in candidate_indices:
        if idx not in horizon_indices:
            horizon_indices.append(idx)

    horizon_names = {}
    if horizon_count == 1:
        horizon_names[horizon_indices[0]] = "short"
    else:
        if len(horizon_indices) >= 1:
            horizon_names[horizon_indices[0]] = "short"
        if len(horizon_indices) >= 2:
            horizon_names[horizon_indices[1]] = "mid"
        if len(horizon_indices) >= 3:
            horizon_names[horizon_indices[2]] = "long"

    rows: list[dict] = []
    for idx in horizon_indices:
        actual_h = actual_np[:, idx]
        pred_h = pred_np[:, idx]
        valid_mask = np.isfinite(actual_h) & np.isfinite(pred_h)
        if not valid_mask.any():
            continue

        actual_valid = actual_h[valid_mask]
        pred_valid = pred_h[valid_mask]
        actual_cls = classify_returns(actual_valid, direction_threshold)
        pred_cls = classify_returns(pred_valid, direction_threshold)

        directional_acc = float(np.mean(np.sign(actual_valid) == np.sign(pred_valid)))
        try:
            bal_acc = float(balanced_accuracy_score(actual_cls, pred_cls))
        except ValueError:
            bal_acc = 0.0
        try:
            mcc = float(matthews_corrcoef(actual_cls, pred_cls))
        except ValueError:
            mcc = 0.0

        rows.append({
            "horizon_name": horizon_names.get(idx, f"h{idx + 1}"),
            "horizon_step": int(idx + 1),
            "samples": int(valid_mask.sum()),
            "mae": round(float(np.mean(np.abs(actual_valid - pred_valid))), 6),
            "rmse": round(float(np.sqrt(np.mean((actual_valid - pred_valid) ** 2))), 6),
            "directional_accuracy": round(directional_acc, 4),
            "accuracy": round(float(accuracy_score(actual_cls, pred_cls)), 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_f1": round(float(f1_score(actual_cls, pred_cls, average="macro", zero_division=0)), 4),
            "mcc": round(mcc, 4),
        })
    return rows


def log_horizon_table(coin_id: str, horizon_rows: list[dict]) -> None:
    if not horizon_rows:
        logger.warning("No valid horizon metrics were computed for %s.", coin_id)
        return

    logger.info("TFT horizon metrics for %s:", coin_id)
    logger.info(
        "%-8s  %-7s  %-8s  %-10s  %-10s  %-10s  %-10s  %-10s  %-10s",
        "Horizon", "Step", "Samples", "Dir.Acc", "Bal.Acc", "MacroF1", "MCC", "MAE", "RMSE",
    )
    logger.info("-" * 100)
    for row in horizon_rows:
        logger.info(
            "%-8s  %-7d  %-8d  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-10.6f  %-10.6f",
            row["horizon_name"],
            row["horizon_step"],
            row["samples"],
            row["directional_accuracy"],
            row["balanced_accuracy"],
            row["macro_f1"],
            row["mcc"],
            row["mae"],
            row["rmse"],
        )


def prepare_tft_dataframe(
    df: pd.DataFrame,
    coin_id: str,
    max_prediction_length: int,
    zero_fill_missing: bool = True,
) -> pd.DataFrame:
    """
    Prepare dataframe for pytorch-forecasting TimeSeriesDataSet.
    Requires:
      - 'time_idx': integer time index (monotonically increasing)
      - 'group_id': coin identifier
      - target column: 'log_return' (we predict future returns)
    """
    df = df[df["coin_id"] == coin_id].sort_values("timestamp").reset_index(drop=True)
    df["time_idx"]  = np.arange(len(df))
    df["group_id"]  = coin_id
    df["target"]    = df["log_return"].shift(-max_prediction_length)  # future return is target

    # drop rows without future targets (last max_prediction_length rows)
    df = df.iloc[:-max_prediction_length]

    # fill missing features
    all_features = TIME_FEATURES + OBSERVED_FEATURES
    for col in all_features:
        if col not in df.columns:
            if zero_fill_missing:
                df[col] = 0.0
            else:
                logger.warning("Missing TFT feature %s; creating as zeros.", col)
                df[col] = 0.0
        df[col] = df[col].fillna(0.0).astype(float)

    return df


def create_datasets(
    df: pd.DataFrame,
    train_cutoff: int,
    val_cutoff: int,
    max_encoder_length: int,
    max_prediction_length: int,
    retained_features: list[str] | None = None,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    Create pytorch-forecasting TimeSeriesDataSet objects.
    Strict chronological splits based on time_idx.
    """
    retained = set(retained_features or (TIME_FEATURES + OBSERVED_FEATURES))
    available_known   = [f for f in TIME_FEATURES if f in df.columns and f in retained]
    available_unknown = [f for f in OBSERVED_FEATURES if f in df.columns and f in retained]

    training_dataset = TimeSeriesDataSet(
        df[df.time_idx <= train_cutoff],
        time_idx            = "time_idx",
        target              = "target",
        group_ids           = ["group_id"],
        max_encoder_length  = max_encoder_length,
        max_prediction_length = max_prediction_length,
        time_varying_known_reals  = available_known,
        time_varying_unknown_reals = available_unknown,
        target_normalizer   = GroupNormalizer(groups=["group_id"]),
        add_relative_time_idx    = True,
        add_target_scales        = True,
        add_encoder_length       = True,
        allow_missing_timesteps  = True,
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df[df.time_idx <= val_cutoff],
        min_prediction_idx=train_cutoff + 1,
        predict=False,
        stop_randomization=True,
    )

    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df,
        min_prediction_idx=val_cutoff + 1,
        predict=False,
        stop_randomization=True,
    )

    return training_dataset, validation_dataset, test_dataset


def train_tft(
    df: pd.DataFrame,
    coin_id: str,
    n_trials: int = 0,
    output_dir: Path = Path("outputs"),
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    num_workers: int = DEFAULT_NUM_WORKERS,
    max_encoder_length: int = MAX_ENCODER,
    max_prediction_length: int = MAX_PRED,
    hidden_size: int = CPU_FRIENDLY_HPARAMS["hidden_size"],
    attention_head_size: int = CPU_FRIENDLY_HPARAMS["attention_head_size"],
    hidden_continuous_size: int = CPU_FRIENDLY_HPARAMS["hidden_continuous_size"],
    dropout: float = CPU_FRIENDLY_HPARAMS["dropout"],
    learning_rate: float = CPU_FRIENDLY_HPARAMS["learning_rate"],
    direction_threshold: float = 0.0,
    zero_rate_threshold: float = 0.98,
    fast_mode: bool = False,
    precision: str = "32",
    limit_rows: int | None = None,
) -> dict:
    """Full TFT training pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if limit_rows is not None and limit_rows > 0:
        df = (
            df[df["coin_id"] == coin_id]
            .sort_values("timestamp")
            .tail(limit_rows)
            .reset_index(drop=True)
        )

    if fast_mode:
        max_epochs = min(max_epochs, FAST_MODE_EPOCHS)
        batch_size = max(batch_size, FAST_MODE_BATCH_SIZE)
        hidden_size = min(hidden_size, 12)
        attention_head_size = 1
        hidden_continuous_size = min(hidden_continuous_size, 8)
        n_trials = 0
        logger.info(
            "Fast mode enabled for %s: max_epochs=%d batch_size=%d hidden_size=%d",
            coin_id,
            max_epochs,
            batch_size,
            hidden_size,
        )

    # ── Prepare data ──────────────────────────────────────────────────────────
    prepared_df = prepare_tft_dataframe(df, coin_id, max_prediction_length=max_prediction_length)
    n = len(prepared_df)
    train_cutoff = int(n * 0.80)
    val_cutoff   = int(n * 0.90)
    _, retained_features, feature_audit = filter_sparse_features(
        prepared_df,
        train_cutoff=train_cutoff,
        zero_rate_threshold=zero_rate_threshold,
    )

    logger.info("TFT dataset: %d rows, train≤%d, val≤%d", n, train_cutoff, val_cutoff)
    if n <= MONTHLY_ROW_THRESHOLD:
        logger.info(
            "Monthly-sized TFT dataset detected for %s (%d rows). Using compact defaults is recommended.",
            coin_id,
            n,
        )

    if n < (max_encoder_length + max_prediction_length) * 3:
        raise ValueError(
            f"Not enough rows for TFT windows: got {n}, need at least "
            f"{(max_encoder_length + max_prediction_length) * 3}."
        )

    if platform.system() == "Windows" and num_workers > 0:
        logger.warning("Windows + num_workers>0 can stall dataloaders; forcing num_workers=0.")
        num_workers = 0

    if not torch.cuda.is_available() and precision != "32":
        logger.warning("Mixed precision requested without CUDA support; falling back to precision=32.")
        precision = "32"

    training_ds, val_ds, test_ds = create_datasets(
        prepared_df,
        train_cutoff,
        val_cutoff,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        retained_features=retained_features,
    )

    train_loader = training_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    val_loader   = val_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)
    test_loader  = test_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    logger.info(
        "Rolling windows: train=%d, val=%d, test=%d",
        len(training_ds),
        len(val_ds),
        len(test_ds),
    )

    # ── Hyperparameter optimization ───────────────────────────────────────────
    if n_trials > 0:
        logger.info("Running TFT hyperparameter optimization (%d trials)...", n_trials)
        study = optimize_hyperparameters(
            train_loader,
            val_loader,
            model_path     = str(output_dir / "tft_trials"),
            n_trials       = n_trials,
            max_epochs     = 20,
            # gradient_clip_val = 0.1,
            use_learning_rate_finder = False,
        )
        best_hparams = study.best_params
        logger.info("Best TFT hparams: %s", best_hparams)
    else:
        best_hparams = {
            "hidden_size":              hidden_size,
            "attention_head_size":      attention_head_size,
            "dropout":                  dropout,
            "hidden_continuous_size":   hidden_continuous_size,
            "learning_rate":            learning_rate,
        }

    # ── Build TFT model ───────────────────────────────────────────────────────
    tft = TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate              = best_hparams.get("learning_rate", 1e-3),
        hidden_size                = best_hparams.get("hidden_size", 64),
        attention_head_size        = best_hparams.get("attention_head_size", 4),
        dropout                    = best_hparams.get("dropout", 0.1),
        hidden_continuous_size     = best_hparams.get("hidden_continuous_size", 32),
        output_size                = 7,       # 7 quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        loss                       = QuantileLoss(),
        log_interval               = 10,
        reduce_on_plateau_patience = 4,
    )

    logger.info("TFT model parameter count: %d", sum(p.numel() for p in tft.parameters()))

    # ── Training ──────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath   = str(output_dir),
        filename  = f"tft_{coin_id.replace('-', '_')}_best",
        monitor   = "val_loss",
        mode      = "min",
        save_top_k = 1,
    )
    early_stop_cb = EarlyStopping(
        monitor   = "val_loss",
        patience  = 5,
        mode      = "min",
    )

    trainer = pl.Trainer(
        max_epochs         = max_epochs,
        accelerator        = "gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val  = 0.1,
        callbacks          = [checkpoint_cb, early_stop_cb],
        enable_progress_bar = True,
        log_every_n_steps  = 1,
        num_sanity_val_steps = 0,
        enable_model_summary = False,
        precision          = precision,
    )

    trainer.fit(
        tft,
        train_dataloaders = train_loader,
        val_dataloaders   = val_loader,
    )

    # ── Load best checkpoint ──────────────────────────────────────────────────
    best_val_loss = None
    if checkpoint_cb.best_model_score is not None:
        best_val_loss = float(checkpoint_cb.best_model_score.detach().cpu().item())

    best_epoch = None
    if checkpoint_cb.best_model_path and "epoch=" in checkpoint_cb.best_model_path:
        try:
            best_epoch = int(checkpoint_cb.best_model_path.split("epoch=")[-1].split("-")[0])
        except ValueError:
            best_epoch = None

    logger.info(
        "Training summary for %s: stopped_at_epoch=%d, max_epochs=%d, best_epoch=%s, best_val_loss=%s",
        coin_id,
        trainer.current_epoch + 1,
        max_epochs,
        best_epoch if best_epoch is not None else "unknown",
        f"{best_val_loss:.4f}" if best_val_loss is not None else "unknown",
    )

    best_model_path = checkpoint_cb.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # ── Variable importance (TFT built-in) ───────────────────────────────────
    prediction_output = best_tft.predict(val_loader, return_x=True, mode="raw")
    if hasattr(prediction_output, "output"):
        raw_predictions = prediction_output.output
        x = prediction_output.x
    elif isinstance(prediction_output, tuple):
        raw_predictions, x = prediction_output[:2]
    else:
        raw_predictions = prediction_output
        x = None

    interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")

    encoder_variable_names = list(
        getattr(best_tft, "encoder_variables", [])
        or [f for f in (TIME_FEATURES + OBSERVED_FEATURES) if f in prepared_df.columns]
    )
    decoder_variable_names = list(
        getattr(best_tft, "decoder_variables", [])
        or [f for f in TIME_FEATURES if f in prepared_df.columns]
    )

    encoder_scores = interpretation["encoder_variables"].detach().cpu().numpy()
    decoder_scores = interpretation["decoder_variables"].detach().cpu().numpy()

    importance = {
        "encoder_variables": {
            k: round(float(v), 4)
            for k, v in zip(encoder_variable_names, encoder_scores)
        },
        "decoder_variables": {
            k: round(float(v), 4)
            for k, v in zip(decoder_variable_names, decoder_scores)
        },
    }
    attention_summary = summarize_attention(interpretation)

    logger.info("TFT variable importance (top encoder):")
    sorted_imp = sorted(importance["encoder_variables"].items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_imp[:10]:
        bar = "█" * int(imp * 200)
        logger.info("  %-35s %.4f  %s", feat, imp, bar[:30])

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_prediction_output = best_tft.predict(test_loader, mode="quantiles", return_y=True)
    if hasattr(test_prediction_output, "output"):
        test_predictions = test_prediction_output.output
        actuals = test_prediction_output.y[0] if isinstance(test_prediction_output.y, tuple) else test_prediction_output.y
    elif isinstance(test_prediction_output, tuple):
        test_predictions = test_prediction_output[0]
        actuals = test_prediction_output[1]
    else:
        test_predictions = test_prediction_output
        actuals = None

    if isinstance(test_predictions, torch.Tensor):
        test_predictions = test_predictions.detach().cpu()
    if isinstance(actuals, torch.Tensor):
        actuals = actuals.detach().cpu()

    # p50 is the median prediction (index 3 in 7-quantile output)
    p50 = test_predictions[:, :, 3].numpy()

    directional_accuracy = None
    thresholded_directional_accuracy = None
    thresholded_sample_count = 0
    horizon_metrics: list[dict] = []
    overall_metrics: dict = {}
    if actuals is not None:
        actual_np = actuals.numpy()
        if actual_np.ndim == 1:
            actual_np = actual_np[:, None]
        valid_mask = np.isfinite(actual_np) & np.isfinite(p50)
        if valid_mask.any():
            directional_accuracy = float(
                np.mean(np.sign(actual_np[valid_mask]) == np.sign(p50[valid_mask]))
            )
            threshold_mask = valid_mask & (np.abs(actual_np) >= direction_threshold)
            thresholded_sample_count = int(np.sum(threshold_mask))
            if thresholded_sample_count > 0:
                thresholded_directional_accuracy = float(
                    np.mean(np.sign(actual_np[threshold_mask]) == np.sign(p50[threshold_mask]))
                )
            horizon_metrics = build_horizon_table(actual_np, p50, direction_threshold)
            log_horizon_table(coin_id, horizon_metrics)

            actual_cls_all = classify_returns(actual_np[valid_mask], direction_threshold)
            pred_cls_all = classify_returns(p50[valid_mask], direction_threshold)
            try:
                overall_bal_acc = float(balanced_accuracy_score(actual_cls_all, pred_cls_all))
            except ValueError:
                overall_bal_acc = 0.0
            try:
                overall_mcc = float(matthews_corrcoef(actual_cls_all, pred_cls_all))
            except ValueError:
                overall_mcc = 0.0
            overall_metrics = {
                "samples": int(valid_mask.sum()),
                "mae": round(float(np.mean(np.abs(actual_np[valid_mask] - p50[valid_mask]))), 6),
                "rmse": round(float(np.sqrt(np.mean((actual_np[valid_mask] - p50[valid_mask]) ** 2))), 6),
                "directional_accuracy": round(directional_accuracy, 4),
                "accuracy": round(float(accuracy_score(actual_cls_all, pred_cls_all)), 4),
                "balanced_accuracy": round(overall_bal_acc, 4),
                "macro_f1": round(float(f1_score(actual_cls_all, pred_cls_all, average="macro", zero_division=0)), 4),
                "mcc": round(overall_mcc, 4),
            }

    results = {
        "coin_id":        coin_id,
        "model":          "TemporalFusionTransformer",
        "trained_at":     datetime.now().isoformat(),
        "max_encoder_len": max_encoder_length,
        "max_pred_len":   max_prediction_length,
        "num_train_windows": len(training_ds),
        "num_val_windows": len(val_ds),
        "num_test_windows": len(test_ds),
        "best_val_loss":  float(trainer.callback_metrics.get("val_loss", 0)),
        "directional_accuracy": directional_accuracy,
        "direction_threshold": direction_threshold,
        "thresholded_directional_accuracy": thresholded_directional_accuracy,
        "thresholded_sample_count": thresholded_sample_count,
        "overall_metrics": overall_metrics,
        "horizon_metrics": horizon_metrics,
        "variable_importance": importance,
        "attention_summary": attention_summary,
        "feature_audit": feature_audit,
        "retained_features": retained_features,
        "best_hyperparams": best_hparams,
        "checkpoint_path": best_model_path,
        "fast_mode": fast_mode,
        "precision": precision,
    }

    # save results
    with open(output_dir / f"tft_results_{coin_id.replace('-', '_')}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    if directional_accuracy is not None:
        logger.info(
            "TFT training complete for %s. Directional accuracy: %.2f%%",
            coin_id,
            directional_accuracy * 100,
        )
        if overall_metrics:
            logger.info(
                "Overall metrics for %s: Bal.Acc=%.4f MacroF1=%.4f MCC=%.4f MAE=%.6f RMSE=%.6f",
                coin_id,
                overall_metrics["balanced_accuracy"],
                overall_metrics["macro_f1"],
                overall_metrics["mcc"],
                overall_metrics["mae"],
                overall_metrics["rmse"],
            )
        if thresholded_directional_accuracy is not None:
            logger.info(
                "Thresholded directional accuracy for %s at |return| >= %.4f: %.2f%% (%d samples)",
                coin_id,
                direction_threshold,
                thresholded_directional_accuracy * 100,
                thresholded_sample_count,
            )
    else:
        logger.info("TFT training complete for %s. Best val_loss: %.4f",
                    coin_id, results["best_val_loss"])
    return results


def main():
    parser = argparse.ArgumentParser(description="Train TFT model")
    parser.add_argument("--data-path",  type=str, default="data/merged_features.csv")
    parser.add_argument("--coin",       type=str, default="BTC-USD", choices=COINS)
    parser.add_argument("--all-coins",  action="store_true")
    parser.add_argument("--n-trials",   type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="outputs/tft")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-encoder", type=int, default=MAX_ENCODER)
    parser.add_argument("--max-pred", type=int, default=MAX_PRED)
    parser.add_argument("--hidden-size", type=int, default=CPU_FRIENDLY_HPARAMS["hidden_size"])
    parser.add_argument("--attention-head-size", type=int, default=CPU_FRIENDLY_HPARAMS["attention_head_size"])
    parser.add_argument("--hidden-continuous-size", type=int, default=CPU_FRIENDLY_HPARAMS["hidden_continuous_size"])
    parser.add_argument("--dropout", type=float, default=CPU_FRIENDLY_HPARAMS["dropout"])
    parser.add_argument("--learning-rate", type=float, default=CPU_FRIENDLY_HPARAMS["learning_rate"])
    parser.add_argument("--direction-threshold", type=float, default=0.001)
    parser.add_argument("--zero-rate-threshold", type=float, default=0.98)
    parser.add_argument("--fast-mode", action="store_true", help="Use laptop-friendly training defaults.")
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Lightning precision mode. Mixed precision can be faster on supported hardware.",
    )
    parser.add_argument("--limit-rows", type=int, help="Optional per-coin tail window for faster experiments.")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    if "timestamp_bucket" in df.columns:
        df["timestamp_bucket"] = pd.to_datetime(df["timestamp_bucket"])
        df = df.rename(columns={"timestamp_bucket": "timestamp"})
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError("Input data must contain either 'timestamp_bucket' or 'timestamp'")

    df = df.sort_values(["coin_id", "timestamp"]).reset_index(drop=True)
    audit_input_dataset(df)

    # --- add log_return if missing ---
    if "log_return" not in df.columns:
        df["log_return"] = (
            df.groupby("coin_id", group_keys=False)["close"]
            .apply(safe_log_return)
            .fillna(0.0)
        )
    coins = COINS if args.all_coins else [args.coin]
    all_results: list[dict] = []

    for coin_id in coins:
        result = train_tft(
            df=df,
            coin_id=coin_id,
            n_trials=args.n_trials,
            output_dir=Path(args.output_dir) / coin_id,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            num_workers=args.num_workers,
            max_encoder_length=args.max_encoder,
            max_prediction_length=args.max_pred,
            hidden_size=args.hidden_size,
            attention_head_size=args.attention_head_size,
            hidden_continuous_size=args.hidden_continuous_size,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            direction_threshold=args.direction_threshold,
            zero_rate_threshold=args.zero_rate_threshold,
            fast_mode=args.fast_mode,
            precision=args.precision,
            limit_rows=args.limit_rows,
        )
        overall = result.get("overall_metrics", {})
        all_results.append({
            "coin_id": coin_id,
            "bal_acc": overall.get("balanced_accuracy"),
            "macro_f1": overall.get("macro_f1"),
            "mcc": overall.get("mcc"),
            "dir_acc": overall.get("directional_accuracy", result.get("directional_accuracy")),
            "mae": overall.get("mae"),
            "rmse": overall.get("rmse"),
            "val_loss": result.get("best_val_loss"),
        })

    if all_results:
        logger.info("\nFINAL TFT RESULTS SUMMARY:")
        logger.info(
            "%-12s  %-10s  %-10s  %-10s  %-10s  %-10s  %-10s",
            "Coin", "Dir.Acc", "Bal.Acc", "MacroF1", "MCC", "MAE", "RMSE",
        )
        logger.info("-" * 92)
        for row in all_results:
            logger.info(
                "%-12s  %-10s  %-10s  %-10s  %-10s  %-10s  %-10s",
                row["coin_id"],
                f"{row['dir_acc']:.4f}" if row["dir_acc"] is not None else "n/a",
                f"{row['bal_acc']:.4f}" if row["bal_acc"] is not None else "n/a",
                f"{row['macro_f1']:.4f}" if row["macro_f1"] is not None else "n/a",
                f"{row['mcc']:.4f}" if row["mcc"] is not None else "n/a",
                f"{row['mae']:.6f}" if row["mae"] is not None else "n/a",
                f"{row['rmse']:.6f}" if row["rmse"] is not None else "n/a",
            )


if __name__ == "__main__":
    main()
