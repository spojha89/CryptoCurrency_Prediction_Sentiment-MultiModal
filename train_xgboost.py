# models/train_xgboost.py
"""
XGBoost Model Training Script
Trains a 3-class XGBoost classifier for cryptocurrency direction prediction.

Pipeline:
  1. Load historical features (CSV or S3)
  2. Create target labels from future log-returns
  3. Chronological train/val/test split (80/10/10) - no random shuffling
  4. Bayesian hyperparameter optimization (Optuna)
  5. Final training with best params
  6. SHAP interpretability analysis
  7. Upload model + metadata to S3

Usage:
  python train_xgboost.py --data-path data/merged_features.csv --coin BTC-USD
  python train_xgboost.py --data-path data/merged_features.csv --all-coins
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available, S3 upload will be skipped")

S3_BUCKET = os.environ.get("MODEL_BUCKET", "crypto-prediction-models")
MODEL_S3_KEY = "models/xgboost/latest/model.pkl"
META_S3_KEY = "models/xgboost/latest/metadata.json"

BUCKET_MINUTES = 15
DEFAULT_HORIZON_HOURS = 4  # Increased to 4 hours for more predictable moves
HORIZON_PERIODS = (DEFAULT_HORIZON_HOURS * 60) // BUCKET_MINUTES
DEFAULT_LABEL_MODE = "quantile"
DEFAULT_TARGET_MODE = "auto"
FIXED_FLAT_THRESHOLD = 0.0015
LABEL_QUANTILES = (1 / 3, 2 / 3)
MONTHLY_ROW_THRESHOLD = 5000

COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "LTC-USD"]

TA_FEATURES = [
    "close", "log_return", "volume", "volume_zscore",
    "return_1h", "return_4h", "return_12h",
    "volatility_1h", "volatility_4h", "volatility_12h",
    "rsi", "macd", "macd_signal", "macd_histogram",
    "bb_upper", "bb_middle", "bb_lower", "bb_position", "vwap",
    "vwap_deviation", "rsi_overbought", "rsi_oversold",
    "macd_bullish_cross", "macd_bearish_cross",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend",
    "return_lag1", "return_lag2",
    "volume_change", "volume_ma_ratio",  # Additional volume features
    "price_range", "price_range_zscore",  # Price volatility
    "momentum_1h", "momentum_4h",  # Momentum indicators
]

SENTIMENT_FEATURES = ["sentiment_twitter", "sentiment_news", "fear_greed_value", "google_trends_value", "sentiment_composite"]
SENTIMENT_LAG_PERIODS = (1, 2, 4, 8, 12)  # Increased lags for better temporal context
LABEL_ORDER = ["DOWN", "FLAT", "UP"]
BINARY_LABEL_ORDER = ["DOWN", "UP"]


def create_labels_from_returns(
    future_returns: pd.Series,
    down_threshold: float,
    up_threshold: float,
) -> pd.Series:
    labels = pd.Series("FLAT", index=future_returns.index, dtype="object")
    labels[future_returns > up_threshold] = "UP"
    labels[future_returns < down_threshold] = "DOWN"
    return labels


def create_binary_labels_from_returns(
    future_returns: pd.Series,
    down_threshold: float,
    up_threshold: float,
) -> pd.Series:
    labels = pd.Series(np.nan, index=future_returns.index, dtype="object")
    labels[future_returns <= down_threshold] = "DOWN"
    labels[future_returns >= up_threshold] = "UP"
    return labels


def create_future_returns(df: pd.DataFrame, horizon: int = HORIZON_PERIODS) -> pd.Series:
    future_close = df["close"].shift(-horizon)
    current_close = df["close"]
    valid_mask = (current_close > 0) & (future_close > 0)

    future_returns = pd.Series(np.nan, index=df.index, dtype="float64")
    future_returns.loc[valid_mask] = np.log(
        future_close.loc[valid_mask] / current_close.loc[valid_mask]
    )
    return future_returns


def compute_label_thresholds(
    future_returns: pd.Series,
    label_mode: str = DEFAULT_LABEL_MODE,
    fixed_flat_threshold: float = FIXED_FLAT_THRESHOLD,
) -> tuple[float, float]:
    clean_returns = future_returns.dropna()
    if clean_returns.empty:
        raise ValueError("Cannot compute label thresholds from empty future returns.")

    if label_mode == "fixed":
        return -fixed_flat_threshold, fixed_flat_threshold
    if label_mode == "quantile":
        down_threshold = float(clean_returns.quantile(LABEL_QUANTILES[0]))
        up_threshold = float(clean_returns.quantile(LABEL_QUANTILES[1]))
        if down_threshold < up_threshold:
            return down_threshold, up_threshold

        negative_returns = clean_returns[clean_returns < 0]
        positive_returns = clean_returns[clean_returns > 0]
        if not negative_returns.empty and not positive_returns.empty:
            down_threshold = float(negative_returns.quantile(0.5))
            up_threshold = float(positive_returns.quantile(0.5))
            if down_threshold < up_threshold:
                logger.warning(
                    "Quantile thresholds collapsed on the training split; "
                    "using median negative/positive returns instead "
                    "(down=%.6f, up=%.6f).",
                    down_threshold,
                    up_threshold,
                )
                return down_threshold, up_threshold

        logger.warning(
            "Quantile thresholds collapsed on the training split "
            "(down=%.6f, up=%.6f); falling back to fixed thresholds +/- %.6f.",
            down_threshold,
            up_threshold,
            fixed_flat_threshold,
        )
        return -fixed_flat_threshold, fixed_flat_threshold
    raise ValueError(f"Unsupported label mode: {label_mode}")


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def _sentiment_feature_candidates(
    selected_sentiment: list[str],
    lag_only_sentiment: bool = True,
) -> list[str]:
    candidates: list[str] = []
    for feature in selected_sentiment:
        if lag_only_sentiment:
            candidates.extend([f"{feature}_lag{lag}" for lag in SENTIMENT_LAG_PERIODS])
        else:
            candidates.append(feature)
            candidates.extend([f"{feature}_lag{lag}" for lag in SENTIMENT_LAG_PERIODS])
    return candidates


def prepare_features(
    df: pd.DataFrame,
    selected_sentiment: list[str],
    lag_only_sentiment: bool = True,
    include_sentiment: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    all_features = list(TA_FEATURES)
    if include_sentiment:
        all_features.extend(_sentiment_feature_candidates(selected_sentiment, lag_only_sentiment))

    available = [f for f in all_features if f in df.columns]
    missing = set(all_features) - set(available)
    if missing:
        logger.warning("Missing features (will be omitted): %s", missing)
    return df[available].fillna(0.0), available


def audit_input_dataset(df: pd.DataFrame) -> dict:
    available_ta = [feature for feature in TA_FEATURES if feature in df.columns]
    available_sentiment = [feature for feature in SENTIMENT_FEATURES if feature in df.columns]
    engineered_ratio = len(available_ta) / len(TA_FEATURES) if TA_FEATURES else 0.0
    has_time_features = any(feature in df.columns for feature in ("hour_sin", "hour_cos", "day_sin", "day_cos"))

    audit = {
        "column_count": int(len(df.columns)),
        "available_ta_features": available_ta,
        "available_sentiment_features": available_sentiment,
        "engineered_feature_ratio": round(engineered_ratio, 3),
        "has_time_features": has_time_features,
    }

    if audit["column_count"] <= 35 or engineered_ratio < 0.5 or not has_time_features:
        logger.warning(
            "Input dataset looks closer to a raw metrics export than a full engineered training export. "
            "Columns=%d engineered_ratio=%.2f has_time_features=%s. "
            "Re-export with export_metrics.py without --raw-only for best XGBoost results.",
            audit["column_count"],
            engineered_ratio,
            has_time_features,
        )
    return audit


def select_training_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    zero_rate_threshold: float = 0.95,
    min_unique_values: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict]:
    zero_rate = (X_train == 0).mean()
    nunique = X_train.nunique(dropna=False)
    variance = X_train.var(axis=0, ddof=0)

    keep_mask = (
        (zero_rate < zero_rate_threshold)
        & (nunique >= min_unique_values)
        & (variance > 0)
    )
    selected_features = X_train.columns[keep_mask].tolist()
    dropped_features = X_train.columns[~keep_mask].tolist()

    if not selected_features:
        logger.warning("Feature audit removed all features; falling back to original feature set.")
        selected_features = X_train.columns.tolist()
        dropped_features = []

    audit = {
        "input_feature_count": int(X_train.shape[1]),
        "selected_feature_count": int(len(selected_features)),
        "dropped_feature_count": int(len(dropped_features)),
        "dropped_features": dropped_features,
        "zero_rate_threshold": zero_rate_threshold,
        "min_unique_values": min_unique_values,
    }

    if dropped_features:
        logger.info(
            "Dropped %d low-signal features from training set: %s",
            len(dropped_features),
            dropped_features,
        )
    else:
        logger.info("Feature audit kept all %d features.", len(selected_features))

    return (
        X_train[selected_features],
        X_val[selected_features],
        X_test[selected_features],
        selected_features,
        audit,
    )


def choose_target_mode(
    requested_mode: str,
    y_ternary_train: pd.Series,
    total_rows: int,
    monthly_row_threshold: int = MONTHLY_ROW_THRESHOLD,
) -> str:
    if requested_mode in {"binary", "ternary"}:
        return requested_mode

    class_share = y_ternary_train.value_counts(normalize=True)
    max_share = float(class_share.max()) if not class_share.empty else 1.0
    min_share = float(class_share.min()) if not class_share.empty else 0.0

    if total_rows <= monthly_row_threshold:
        logger.info(
            "Auto target mode selected binary classification because dataset is relatively small (%d rows).",
            total_rows,
        )
        return "binary"

    if max_share >= 0.65 or min_share <= 0.10:
        logger.info(
            "Auto target mode selected binary classification due to ternary class imbalance "
            "(max_share=%.3f min_share=%.3f).",
            max_share,
            min_share,
        )
        return "binary"

    return "ternary"


def build_search_params(
    trial: optuna.Trial,
    sample_count: int,
    n_classes: int,
) -> dict:
    if sample_count <= 4000:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 320),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 3.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 8.0),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        }
    else:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        }

    if n_classes == 2:
        params.update({
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        })
    else:
        params.update({
            "objective": "multi:softprob",
            "num_class": n_classes,
            "eval_metric": "mlogloss",
        })

    params.update({
        "random_state": 42,
        "n_jobs": -1,
    })
    return params


def finalize_model_params(best_params: dict, n_classes: int) -> dict:
    params = dict(best_params)
    if n_classes == 2:
        params.update({
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        })
        params.pop("num_class", None)
    else:
        params.update({
            "objective": "multi:softprob",
            "num_class": n_classes,
            "eval_metric": "mlogloss",
        })
    params.update({
        "random_state": 42,
        "n_jobs": -1,
    })
    return params


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for feature in ("sentiment_twitter", "sentiment_news", "fear_greed_value", "google_trends_value"):
        if feature in df.columns:
            for lag in SENTIMENT_LAG_PERIODS:
                df[f"{feature}_lag{lag}"] = df[feature].shift(lag)

    if "log_return" in df.columns:
        df["return_lag1"] = df["log_return"].shift(1)
        df["return_lag2"] = df["log_return"].shift(2)

    return df


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list,
    title: str,
    output_dir: str = "outputs",
) -> str:
    """Plot and save confusion matrix visualization.
    
    Args:
        y_true: True labels (encoded)
        y_pred: Predicted labels (encoded)
        label_names: List of label names (e.g., ['DOWN', 'FLAT', 'UP'])
        title: Title for the plot
        output_dir: Directory to save the plot
    
    Returns:
        Path to saved figure
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        
        # Setup colorbar
        plt.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=label_names,
               yticklabels=label_names)
        
        # Rotate tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set_ylabel("True label", fontsize=12)
        ax.set_xlabel("Predicted label", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        
        # Save figure
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / f"confusion_matrix_{title.replace(' ', '_').lower()}.png"
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"Saved confusion matrix to {output_path}")
        return str(output_path)
    except Exception as e:
        logger.warning(f"Failed to generate confusion matrix: {e}")
        return None


def evaluate_model(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
    label_encoder: LabelEncoder,
    future_returns: pd.Series | None = None,
    eval_move_threshold: float = 0.0,
) -> dict:
    y_true = label_encoder.transform(y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    def _metric_block(
        y_true_block: np.ndarray,
        y_pred_block: np.ndarray,
        y_proba_block: np.ndarray,
        label: str,
    ) -> dict:
        acc = accuracy_score(y_true_block, y_pred_block)
        balanced_acc = balanced_accuracy_score(y_true_block, y_pred_block)
        macro_f1 = f1_score(y_true_block, y_pred_block, average="macro", zero_division=0)
        mcc = matthews_corrcoef(y_true_block, y_pred_block)
        
        report = classification_report(
            y_true_block,
            y_pred_block,
            labels=np.arange(len(label_encoder.classes_)),
            target_names=list(label_encoder.classes_),
            output_dict=True,
            zero_division=0,
        )

        try:
            if len(label_encoder.classes_) == 2:
                auc = roc_auc_score(y_true_block, y_proba_block[:, 1])
            else:
                auc = roc_auc_score(
                    y_true_block,
                    y_proba_block,
                    multi_class="ovr",
                    average="macro",
                    labels=np.arange(len(label_encoder.classes_)),
                )
        except ValueError:
            auc = 0.0

        logger.info(
            "[%s] Acc=%.4f BalAcc=%.4f MacroF1=%.4f MCC=%.4f AUC=%.4f",
            label, acc, balanced_acc, macro_f1, mcc, auc
        )
        
        # Generate confusion matrix visualization
        cm_path = plot_confusion_matrix(
            y_true_block,
            y_pred_block,
            list(label_encoder.classes_),
            f"Confusion Matrix - {label}",
            output_dir="outputs"
        )
        
        return {
            "split": label,
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(balanced_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "mcc": round(mcc, 4),
            "directional_accuracy": round(acc, 4),
            "auc": round(auc, 4),
            "confusion_matrix_plot": cm_path,
            "classification_report": report,
        }

    metrics = _metric_block(y_true, y_pred, y_proba, split_name)

    if future_returns is not None and eval_move_threshold > 0:
        move_mask = future_returns.abs().to_numpy() >= eval_move_threshold
        large_move_count = int(move_mask.sum())
        metrics["large_move_threshold"] = eval_move_threshold
        metrics["large_move_count"] = large_move_count

        if large_move_count > 0:
            metrics["large_move_metrics"] = _metric_block(
                y_true[move_mask],
                y_pred[move_mask],
                y_proba[move_mask],
                f"{split_name}_large_moves",
            )
        else:
            metrics["large_move_metrics"] = None

    return metrics


def fit_xgb_model(
    params: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> xgb.XGBClassifier:
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    sample_weights = np.array([class_weights[y] for y in y_train])

    fit_signature = inspect.signature(xgb.XGBClassifier.fit)
    fit_params = fit_signature.parameters
    supports_fit_callbacks = "callbacks" in fit_params
    supports_fit_early_stopping = "early_stopping_rounds" in fit_params

    callback_factory = getattr(xgb, "callback", None)
    early_stopping_callback = None
    if callback_factory is not None and hasattr(callback_factory, "EarlyStopping"):
        early_stopping_callback = callback_factory.EarlyStopping(
            rounds=30,
            save_best=True,
            maximize=False,
        )

    model_params = dict(params)
    if not supports_fit_callbacks and early_stopping_callback is not None:
        model_params["callbacks"] = [early_stopping_callback]
    elif not supports_fit_early_stopping:
        model_params["early_stopping_rounds"] = 30

    model = xgb.XGBClassifier(**model_params)

    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "eval_set": [(X_val, y_val)],
        "sample_weight": sample_weights,
        "verbose": False,
    }
    if supports_fit_callbacks and early_stopping_callback is not None:
        fit_kwargs["callbacks"] = [early_stopping_callback]
    elif supports_fit_early_stopping:
        fit_kwargs["early_stopping_rounds"] = 30

    try:
        model.fit(**fit_kwargs)
    except TypeError:
        logger.warning("Falling back to XGBoost training without early stopping due to API mismatch.")
        fallback_model = xgb.XGBClassifier(**params)
        fallback_model.fit(
            X=X_train,
            y=y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weights,
            verbose=False,
        )
        return fallback_model
    return model


def _reduce_shap_importance(shap_values, feature_count: int) -> np.ndarray:
    if isinstance(shap_values, list):
        arrays = [np.asarray(getattr(sv, "values", sv), dtype=float) for sv in shap_values]
        reduced = np.mean([_reduce_shap_importance(arr, feature_count) for arr in arrays], axis=0)
    else:
        reduced = np.asarray(getattr(shap_values, "values", shap_values), dtype=float)
        if reduced.ndim == 1:
            reduced = np.abs(reduced)
        else:
            axes = tuple(i for i, size in enumerate(reduced.shape) if size != feature_count)
            reduced = np.abs(reduced).mean(axis=axes) if axes else np.abs(reduced)

    reduced = np.asarray(reduced, dtype=float).reshape(-1)
    if reduced.shape[0] != feature_count:
        raise ValueError(
            f"Expected {feature_count} SHAP values after reduction, got shape {tuple(reduced.shape)}"
        )
    return reduced


def run_shap_analysis(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
):
    logger.info("Running SHAP analysis on %d test samples...", len(X_test))
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
        mean_shap = _reduce_shap_importance(shap_values, len(feature_names))

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_shap,
        }).sort_values("mean_abs_shap", ascending=False)

        shap_path = output_dir / "shap_importance.csv"
        shap_df.to_csv(shap_path, index=False)

        logger.info("Top 10 features by SHAP importance:")
        for _, row in shap_df.head(10).iterrows():
            logger.info("  %-35s %.4f", row["feature"], row["mean_abs_shap"])

        sentiment_shap = shap_df[shap_df["feature"].str.contains("sentiment")]["mean_abs_shap"].sum()
        technical_shap = shap_df[~shap_df["feature"].str.contains("sentiment")]["mean_abs_shap"].sum()
        total_shap = sentiment_shap + technical_shap

        logger.info(
            "Sentiment features total SHAP: %.4f (%.1f%%)",
            sentiment_shap, 100 * sentiment_shap / total_shap if total_shap > 0 else 0,
        )
        logger.info(
            "Technical features total SHAP: %.4f (%.1f%%)",
            technical_shap, 100 * technical_shap / total_shap if total_shap > 0 else 0,
        )
        return shap_df.to_dict("records")
    except Exception as e:
        logger.warning("SHAP analysis failed: %s", e)
        return []


def train_xgboost(
    df: pd.DataFrame,
    coin_id: str,
    selected_sentiment: list[str],
    n_trials: int = 12,
    lag_only_sentiment: bool = True,
    include_sentiment: bool = True,
    horizon_periods: int = HORIZON_PERIODS,
    eval_move_threshold: float = 0.0,
    label_mode: str = DEFAULT_LABEL_MODE,
    target_mode: str = DEFAULT_TARGET_MODE,
    fixed_flat_threshold: float = FIXED_FLAT_THRESHOLD,
    zero_rate_threshold: float = 0.95,
    output_dir: Path = Path("outputs"),
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = add_lag_features(df.copy())
    df["future_return"] = create_future_returns(df, horizon=horizon_periods)
    df = df.iloc[:-horizon_periods]
    df = df.dropna(subset=["future_return"])

    future_returns_full = df["future_return"]
    future_train_raw, _, _ = chronological_split(future_returns_full)
    down_threshold, up_threshold = compute_label_thresholds(
        future_train_raw,
        label_mode=label_mode,
        fixed_flat_threshold=fixed_flat_threshold,
    )
    df["label"] = create_labels_from_returns(
        future_returns_full,
        down_threshold=down_threshold,
        up_threshold=up_threshold,
    )
    y_ternary_train, _, _ = chronological_split(df["label"])
    effective_target_mode = choose_target_mode(target_mode, y_ternary_train, len(df))
    if effective_target_mode == "binary":
        df["label"] = create_binary_labels_from_returns(
            future_returns_full,
            down_threshold=down_threshold,
            up_threshold=up_threshold,
        )
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        future_returns_full = df["future_return"]
        binary_classes = sorted(df["label"].dropna().unique().tolist())
        if len(binary_classes) < 2:
            logger.warning(
                "Binary labeling collapsed to a single class for %s; reverting to ternary labels.",
                coin_id,
            )
            df["label"] = create_labels_from_returns(
                future_returns_full,
                down_threshold=down_threshold,
                up_threshold=up_threshold,
            )
            label_order = LABEL_ORDER
            effective_target_mode = "ternary"
        else:
            label_order = BINARY_LABEL_ORDER
    else:
        label_order = LABEL_ORDER

    label_dist = df["label"].value_counts()
    logger.info("Label distribution for %s:\n%s", coin_id, label_dist.to_string())
    logger.info(
        "Labeling mode=%s target_mode=%s down_threshold=%.6f up_threshold=%.6f",
        label_mode,
        effective_target_mode,
        down_threshold,
        up_threshold,
    )

    X_full, feature_names = prepare_features(
        df,
        selected_sentiment,
        lag_only_sentiment=lag_only_sentiment,
        include_sentiment=include_sentiment,
    )
    X_baseline_full, baseline_feature_names = prepare_features(
        df,
        selected_sentiment,
        lag_only_sentiment=lag_only_sentiment,
        include_sentiment=False,
    )
    y_full = df["label"]

    X_train, X_val, X_test = chronological_split(X_full)
    X_train_base, X_val_base, X_test_base = chronological_split(X_baseline_full)
    y_train, y_val, y_test = chronological_split(y_full)
    future_train, future_val, future_test = chronological_split(future_returns_full)

    X_train, X_val, X_test, feature_names, feature_audit = select_training_features(
        X_train,
        X_val,
        X_test,
        zero_rate_threshold=zero_rate_threshold,
    )
    X_train_base, X_val_base, X_test_base, baseline_feature_names, baseline_feature_audit = (
        select_training_features(
            X_train_base,
            X_val_base,
            X_test_base,
            zero_rate_threshold=zero_rate_threshold,
        )
    )

    logger.info("Split sizes - Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))

    le = LabelEncoder()
    le.fit(label_order)
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    logger.info("Starting Bayesian hyperparameter optimization (%d trials)...", n_trials)

    def objective(trial: optuna.Trial) -> float:
        params = build_search_params(trial, len(X_train), len(label_order))
        model = fit_xgb_model(params, X_train, y_train_enc, X_val, y_val_enc)
        y_val_pred = model.predict(X_val)
        return balanced_accuracy_score(y_val_enc, y_val_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    logger.info("Best hyperparameters: %s", best_params)
    logger.info("Best validation balanced accuracy: %.4f", study.best_value)

    best_params = finalize_model_params(best_params, len(label_order))

    final_model = fit_xgb_model(best_params, X_train, y_train_enc, X_val, y_val_enc)
    baseline_model = fit_xgb_model(best_params, X_train_base, y_train_enc, X_val_base, y_val_enc)

    baseline_metrics = {
        "train_metrics": evaluate_model(
            baseline_model,
            X_train_base,
            y_train,
            "baseline_train",
            le,
            future_returns=future_train,
            eval_move_threshold=eval_move_threshold,
        ),
        "val_metrics": evaluate_model(
            baseline_model,
            X_val_base,
            y_val,
            "baseline_val",
            le,
            future_returns=future_val,
            eval_move_threshold=eval_move_threshold,
        ),
        "test_metrics": evaluate_model(
            baseline_model,
            X_test_base,
            y_test,
            "baseline_test",
            le,
            future_returns=future_test,
            eval_move_threshold=eval_move_threshold,
        ),
    }

    results = {
        "coin_id": coin_id,
        "timestamp": datetime.now().isoformat(),
        "feature_names": feature_names,
        "baseline_feature_names": baseline_feature_names,
        "label_classes": list(le.classes_),
        "n_features": len(feature_names),
        "baseline_n_features": len(baseline_feature_names),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "horizon_periods": horizon_periods,
        "horizon_hours": round(horizon_periods * BUCKET_MINUTES / 60, 2),
        "label_mode": label_mode,
        "target_mode_requested": target_mode,
        "target_mode_effective": effective_target_mode,
        "label_quantiles": list(LABEL_QUANTILES),
        "down_threshold": down_threshold,
        "up_threshold": up_threshold,
        "fixed_flat_threshold": fixed_flat_threshold if label_mode == "fixed" else None,
        "eval_move_threshold": eval_move_threshold,
        "lag_only_sentiment": lag_only_sentiment,
        "sentiment_lag_periods": list(SENTIMENT_LAG_PERIODS),
        "selected_sentiment_features": selected_sentiment,
        "feature_audit": feature_audit,
        "baseline_feature_audit": baseline_feature_audit,
        "label_distribution": {
            "train": dict(pd.Series(y_train).value_counts()),
            "val": dict(pd.Series(y_val).value_counts()),
            "test": dict(pd.Series(y_test).value_counts()),
        },
        "best_hyperparams": best_params,
        "best_val_balanced_accuracy": study.best_value,
        "train_metrics": evaluate_model(
            final_model,
            X_train,
            y_train,
            "train",
            le,
            future_returns=future_train,
            eval_move_threshold=eval_move_threshold,
        ),
        "val_metrics": evaluate_model(
            final_model,
            X_val,
            y_val,
            "val",
            le,
            future_returns=future_val,
            eval_move_threshold=eval_move_threshold,
        ),
        "test_metrics": evaluate_model(
            final_model,
            X_test,
            y_test,
            "test",
            le,
            future_returns=future_test,
            eval_move_threshold=eval_move_threshold,
        ),
        "baseline_metrics": baseline_metrics,
    }

    results["predictive_lift"] = {
        "test_accuracy_lift": round(
            results["test_metrics"]["accuracy"] - baseline_metrics["test_metrics"]["accuracy"], 4
        ),
        "test_balanced_accuracy_lift": round(
            results["test_metrics"]["balanced_accuracy"] - baseline_metrics["test_metrics"]["balanced_accuracy"], 4
        ),
        "test_mcc_lift": round(
            results["test_metrics"]["mcc"] - baseline_metrics["test_metrics"]["mcc"], 4
        ),
        "test_auc_lift": round(
            results["test_metrics"]["auc"] - baseline_metrics["test_metrics"]["auc"], 4
        ),
        "val_accuracy_lift": round(
            results["val_metrics"]["accuracy"] - baseline_metrics["val_metrics"]["accuracy"], 4
        ),
        "val_macro_f1_lift": round(
            results["val_metrics"]["macro_f1"] - baseline_metrics["val_metrics"]["macro_f1"], 4
        ),
    }

    shap_importance = run_shap_analysis(final_model, X_test, feature_names, output_dir)
    results["shap_importance"] = shap_importance[:20]

    model_path = output_dir / f"model_{coin_id.replace('-', '_')}.pkl"
    meta_path = output_dir / f"metadata_{coin_id.replace('-', '_')}.json"

    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    metadata = {
        "version": f"xgboost-{datetime.now().strftime('%Y%m%d')}",
        "coin_id": coin_id,
        "feature_names": feature_names,
        "label_classes": list(le.classes_),
        "label_mode": label_mode,
        "target_mode_requested": target_mode,
        "target_mode_effective": effective_target_mode,
        "label_quantiles": list(LABEL_QUANTILES),
        "down_threshold": down_threshold,
        "up_threshold": up_threshold,
        "fixed_flat_threshold": fixed_flat_threshold if label_mode == "fixed" else None,
        "eval_move_threshold": eval_move_threshold,
        "horizon_h": round(horizon_periods * BUCKET_MINUTES / 60, 2),
        "trained_at": datetime.now().isoformat(),
        "test_accuracy": results["test_metrics"]["accuracy"],
        "test_balanced_accuracy": results["test_metrics"]["balanced_accuracy"],
        "test_macro_f1": results["test_metrics"]["macro_f1"],
        "test_mcc": results["test_metrics"]["mcc"],
        "test_dir_acc": results["test_metrics"]["directional_accuracy"],
        "test_auc": results["test_metrics"]["auc"],
        "baseline_test_accuracy": baseline_metrics["test_metrics"]["accuracy"],
        "baseline_test_balanced_accuracy": baseline_metrics["test_metrics"]["balanced_accuracy"],
        "baseline_test_mcc": baseline_metrics["test_metrics"]["mcc"],
        "baseline_test_auc": baseline_metrics["test_metrics"]["auc"],
        "predictive_lift": results["predictive_lift"],
        "lag_only_sentiment": lag_only_sentiment,
        "selected_sentiment_features": selected_sentiment,
        "feature_audit": feature_audit,
        "best_hyperparams": best_params,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    try:
        s3 = boto3.client("s3")
        s3.upload_file(str(model_path), S3_BUCKET, f"models/xgboost/{coin_id}/model.pkl")
        s3.upload_file(str(meta_path), S3_BUCKET, f"models/xgboost/{coin_id}/metadata.json")
        s3.upload_file(str(model_path), S3_BUCKET, MODEL_S3_KEY)
        s3.upload_file(str(meta_path), S3_BUCKET, META_S3_KEY)
        logger.info("Model uploaded to s3://%s/models/xgboost/%s/", S3_BUCKET, coin_id)
    except Exception as e:
        logger.warning("S3 upload failed (continuing): %s", e)

    results_path = output_dir / f"results_{coin_id.replace('-', '_')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost cryptocurrency predictor")
    parser.add_argument("--data-path", type=str, default="data/merged_features.csv")
    parser.add_argument("--coin", type=str, default="BTC-USD", choices=COINS)
    parser.add_argument("--all-coins", action="store_true")
    parser.add_argument("--n-trials", type=int, default=20)  # Increased default for better optimization
    parser.add_argument("--granger-file", type=str, default="granger_results.json")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--no-sentiment", action="store_true",
                        help="Train without sentiment features")
    parser.add_argument("--horizon-hours", type=float, default=DEFAULT_HORIZON_HOURS)
    parser.add_argument("--eval-move-threshold", type=float, default=0.0)
    parser.add_argument(
        "--label-mode",
        type=str,
        choices=["fixed", "quantile"],
        default=DEFAULT_LABEL_MODE,
        help="How to map future returns into DOWN/FLAT/UP labels.",
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        choices=["auto", "binary", "ternary"],
        default=DEFAULT_TARGET_MODE,
        help="Classification setup. Auto prefers binary on smaller or imbalanced datasets.",
    )
    parser.add_argument(
        "--fixed-flat-threshold",
        type=float,
        default=FIXED_FLAT_THRESHOLD,
        help="Symmetric threshold used when --label-mode fixed.",
    )
    parser.add_argument(
        "--lag-only-sentiment",
        action="store_true",
        help="Use sentiment lag features only. Default keeps current sentiment plus lags.",
    )
    parser.add_argument(
        "--zero-rate-threshold",
        type=float,
        default=0.95,
        help="Drop features whose training-set zero rate is at or above this threshold.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    lag_only_sentiment = args.lag_only_sentiment
    include_sentiment = not args.no_sentiment
    horizon_periods = max(1, int(round(args.horizon_hours * 60 / BUCKET_MINUTES)))

    granger_path = Path(args.granger_file)
    if granger_path.exists():
        with open(granger_path) as f:
            granger = json.load(f)
        logger.info("Loaded Granger causality results from %s", granger_path)
    else:
        granger = {}
        logger.warning("Granger results not found; using all sentiment features")

    df = pd.read_csv(args.data_path)
    if "timestamp_bucket" in df.columns:
        df["timestamp_bucket"] = pd.to_datetime(df["timestamp_bucket"])
        df = df.rename(columns={"timestamp_bucket": "timestamp"})
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError("Input data must contain either 'timestamp_bucket' or 'timestamp'")
    logger.info("Loaded %d rows from %s", len(df), args.data_path)
    dataset_audit = audit_input_dataset(df)

    coins = COINS if args.all_coins else [args.coin]

    all_results = {}
    for coin_id in coins:
        coin_df = df[df["coin_id"] == coin_id].sort_values("timestamp").reset_index(drop=True)
        if len(coin_df) < 500:
            logger.warning("Skipping %s: only %d rows", coin_id, len(coin_df))
            continue

        coin_granger = granger.get("feature_selection", {}).get(coin_id, [])
        coin_selected = coin_granger if coin_granger else SENTIMENT_FEATURES

        logger.info("\n" + "=" * 60)
        logger.info(
            "Training %s | %d rows | %d sentiment features selected | lag_only_sentiment=%s | target_mode=%s",
            coin_id, len(coin_df), len(coin_selected), lag_only_sentiment, args.target_mode,
        )
        logger.info("=" * 60)

        result = train_xgboost(
            df=coin_df,
            coin_id=coin_id,
            selected_sentiment=coin_selected,
            n_trials=args.n_trials,
            lag_only_sentiment=lag_only_sentiment,
            include_sentiment=include_sentiment,
            horizon_periods=horizon_periods,
            eval_move_threshold=args.eval_move_threshold,
            label_mode=args.label_mode,
            target_mode=args.target_mode,
            fixed_flat_threshold=args.fixed_flat_threshold,
            zero_rate_threshold=args.zero_rate_threshold,
            output_dir=output_dir / f"horizon_{horizon_periods}" / coin_id,
        )
        all_results[coin_id] = {
            "test_accuracy": result["test_metrics"]["accuracy"],
            "test_balanced_accuracy": result["test_metrics"]["balanced_accuracy"],
            "test_directional_acc": result["test_metrics"]["directional_accuracy"],
            "test_macro_f1": result["test_metrics"]["macro_f1"],
            "test_mcc": result["test_metrics"]["mcc"],
            "test_auc": result["test_metrics"]["auc"],
            "baseline_accuracy": result["baseline_metrics"]["test_metrics"]["accuracy"],
            "accuracy_lift": result["predictive_lift"]["test_accuracy_lift"],
            "auc_lift": result["predictive_lift"]["test_auc_lift"],
            "target_mode_effective": result["target_mode_effective"],
        }

    logger.info("\nFINAL RESULTS SUMMARY:")
    logger.info(
        "%-12s  %-8s  %-10s  %-10s  %-10s  %-10s  %-10s  %-10s",
        "Coin", "Mode", "Accuracy", "Bal.Acc", "MacroF1", "MCC", "AUC", "Lift",
    )
    logger.info("-" * 100)
    for coin, metrics in all_results.items():
        logger.info(
            "%-12s  %-8s  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-10.4f",
            coin,
            metrics["target_mode_effective"],
            metrics["test_accuracy"],
            metrics["test_balanced_accuracy"],
            metrics["test_macro_f1"],
            metrics["test_mcc"],
            metrics["test_auc"],
            metrics["accuracy_lift"],
        )


if __name__ == "__main__":
    main()
