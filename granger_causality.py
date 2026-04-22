# models/granger_causality.py
"""
Granger Causality Pre-Validation
Tests whether external signals Granger-cause cryptocurrency price movements.

Usage:
  python granger_causality.py --data-path data/merged_features.csv

Output: granger_results.json - used by train_xgboost.py for feature selection
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "LTC-USD"]
SIGNAL_COLUMNS = {
    "twitter": "sentiment_twitter",
    "news": "sentiment_news",
    "fear_greed": "fear_greed_value",
    "google_trends": "google_trends_value",
}
MAX_LAGS = 12
ALPHA = 0.05


def adf_test(series: pd.Series, name: str = "") -> dict:
    from statsmodels.tsa.stattools import adfuller

    clean = series.dropna()
    if len(clean) < 8:
        logger.warning("ADF [%s]: skipped - only %d non-null values", name, len(clean))
        return {
            "series": name,
            "adf_stat": None,
            "p_value": None,
            "critical_values": {},
            "is_stationary": False,
            "lags_used": 0,
            "skipped": True,
        }
    if clean.max() == clean.min():
        logger.warning("ADF [%s]: skipped - series has no variance", name)
        return {
            "series": name,
            "adf_stat": None,
            "p_value": None,
            "critical_values": {},
            "is_stationary": False,
            "lags_used": 0,
            "skipped": True,
        }

    result = adfuller(clean, autolag="AIC")
    is_stationary = result[1] < ALPHA
    logger.info(
        "ADF [%s]: stat=%.4f, p=%.4f -> %s",
        name,
        result[0],
        result[1],
        "STATIONARY" if is_stationary else "NON-STATIONARY",
    )
    return {
        "series": name,
        "adf_stat": round(result[0], 4),
        "p_value": round(result[1], 4),
        "critical_values": {k: round(v, 4) for k, v in result[4].items()},
        "is_stationary": is_stationary,
        "lags_used": int(result[2]),
        "skipped": False,
    }


def convert_numpy(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def make_stationary(series: pd.Series) -> pd.Series:
    return series.diff().dropna()


def safe_log_return(series: pd.Series) -> pd.Series:
    """Compute log returns while treating non-positive prices as missing."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.where(numeric > 0)
    return np.log(valid / valid.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()


def _sanitize_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    cleaned = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    return cleaned


def _apply_fdr(p_values: dict[str, float], alpha: float = ALPHA) -> dict[str, dict[str, float | bool]]:
    if not p_values:
        return {}

    ranked = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ranked)
    adjusted: dict[str, dict[str, float | bool]] = {}
    running_min = 1.0

    for rank_from_end, (name, p_val) in enumerate(reversed(ranked), start=1):
        rank = m - rank_from_end + 1
        adjusted_p = min(1.0, p_val * m / rank)
        running_min = min(running_min, adjusted_p)
        adjusted[name] = {"adjusted_p": round(running_min, 4), "rejects_h0": running_min <= alpha}

    return adjusted


def granger_test(y: pd.Series, x: pd.Series, max_lags: int = MAX_LAGS) -> dict:
    from statsmodels.tsa.stattools import grangercausalitytests

    combined = pd.concat([_sanitize_series(y), _sanitize_series(x)], axis=1).dropna()
    combined.columns = ["y", "x"]
    if combined.empty:
        return {"error": "insufficient_data", "min_p": 1.0, "best_lag": None, "rejects_h0": False}
    if combined["y"].nunique() < 2 or combined["x"].nunique() < 2:
        return {"error": "constant_series", "min_p": 1.0, "best_lag": None, "rejects_h0": False}

    max_allowed_lag = min(max_lags, (len(combined) - 1) // 3)
    if max_allowed_lag < 1:
        return {"error": "insufficient_data", "min_p": 1.0, "best_lag": None, "rejects_h0": False}
    if len(combined) < max_lags * 4:
        logger.info(
            "Granger test using reduced max_lag=%d due to sample size=%d",
            max_allowed_lag,
            len(combined),
        )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in log",
                category=RuntimeWarning,
            )
            results = grangercausalitytests(combined, maxlag=max_allowed_lag, verbose=False)
        p_values = {lag: result_tuple[0]["ssr_ftest"][1] for lag, result_tuple in results.items()}
        best_lag = min(p_values, key=p_values.get)
        min_p = p_values[best_lag]
        return {
            "min_p": float(round(min_p, 4)),
            "best_lag": int(best_lag),
            "rejects_h0": bool(min_p < ALPHA),
            "all_p_values": {str(k): float(round(v, 4)) for k, v in p_values.items()},
            "tested_lags": int(max_allowed_lag),
        }
    except Exception as e:
        logger.warning("Granger test failed: %s", e)
        return {"error": str(e), "min_p": 1.0, "best_lag": None, "rejects_h0": False}


def mutual_information_score(y: np.ndarray, x: np.ndarray, n_bins: int = 10) -> float:
    try:
        from sklearn.metrics import mutual_info_score

        y_bins = pd.cut(pd.Series(y).dropna(), bins=n_bins, labels=False)
        x_bins = pd.cut(pd.Series(x).dropna(), bins=n_bins, labels=False)
        aligned = pd.concat([y_bins, x_bins], axis=1).dropna()
        if len(aligned) < 20:
            return 0.0
        return round(float(mutual_info_score(aligned.iloc[:, 0], aligned.iloc[:, 1])), 4)
    except Exception:
        return 0.0


def run_analysis(df: pd.DataFrame) -> dict:
    results = {"stationarity": {}, "granger": {}, "mutual_information": {}, "feature_selection": {}}

    for coin_id in COINS:
        coin_df = df[df["coin_id"] == coin_id].set_index("timestamp").sort_index()
        if len(coin_df) < 100:
            logger.warning("Insufficient data for %s: %d rows", coin_id, len(coin_df))
            continue

        coin_df["log_return"] = safe_log_return(coin_df["close"])
        adf_result = adf_test(coin_df["log_return"].dropna(), f"{coin_id}_log_return")
        results["stationarity"][coin_id] = {"log_return": adf_result}

        y = _sanitize_series(coin_df["log_return"])
        if not adf_result["is_stationary"]:
            y = make_stationary(y)
        y = _sanitize_series(y)

        coin_granger = {}
        coin_mi = {}
        selected = []
        raw_p_values = {}

        for signal_name, col in SIGNAL_COLUMNS.items():
            if col not in coin_df.columns:
                continue

            signal_series = _sanitize_series(coin_df[col])
            if len(signal_series) < 8:
                logger.warning("SKIPPED: %s %s - only %d non-null values", coin_id, col, len(signal_series))
                continue

            signal_adf = adf_test(signal_series, f"{coin_id}_{signal_name}")
            results["stationarity"].setdefault(coin_id, {})[col] = signal_adf

            x = signal_series if signal_adf["is_stationary"] else make_stationary(signal_series)
            x = _sanitize_series(x)
            gc_result = granger_test(y, x, max_lags=MAX_LAGS)
            coin_granger[signal_name] = gc_result
            raw_p_values[signal_name] = gc_result.get("min_p", 1.0)

            min_len = min(len(y), len(x))
            coin_mi[signal_name] = mutual_information_score(y.values[-min_len:], x.values[-min_len:])

        fdr_results = _apply_fdr(raw_p_values)
        for signal_name, gc_result in coin_granger.items():
            fdr_info = fdr_results.get(signal_name, {"adjusted_p": 1.0, "rejects_h0": False})
            gc_result["fdr_adjusted_p"] = fdr_info["adjusted_p"]
            gc_result["fdr_rejects_h0"] = fdr_info["rejects_h0"]

            col = SIGNAL_COLUMNS[signal_name]
            if gc_result.get("fdr_rejects_h0", False):
                selected.append(col)
                logger.info(
                    "SELECTED: %s %s (raw_p=%.4f, fdr_p=%.4f, best_lag=%s)",
                    coin_id, col, gc_result.get("min_p", 1.0), gc_result["fdr_adjusted_p"], gc_result.get("best_lag"),
                )
            else:
                logger.info(
                    "REJECTED: %s %s (raw_p=%.4f, fdr_p=%.4f)",
                    coin_id, col, gc_result.get("min_p", 1.0), gc_result["fdr_adjusted_p"],
                )

        results["granger"][coin_id] = coin_granger
        results["mutual_information"][coin_id] = coin_mi
        results["feature_selection"][coin_id] = selected

    results["summary"] = {
        "total_pairs_tested": len(COINS) * len(SIGNAL_COLUMNS),
        "significant_pairs": sum(
            1 for c in results["granger"].values() for p in c.values() if p.get("rejects_h0", False)
        ),
        "alpha": ALPHA,
        "max_lags": MAX_LAGS,
    }
    logger.info("Summary: %s", results["summary"])
    return results


def main():
    parser = argparse.ArgumentParser(description="Granger Causality Analysis")
    parser.add_argument("--data-path", type=str, default="data/merged_features.csv")
    parser.add_argument("--output", type=str, default="granger_results.json")
    parser.add_argument("--from-dynamodb", action="store_true")
    args = parser.parse_args()

    if args.from_dynamodb:
        raise NotImplementedError(
            "DynamoDB export not yet implemented. Export DynamoDB table to CSV first and use --data-path."
        )

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.warning("Data file not found. Generating synthetic demo data.")
        np.random.seed(42)
        n = 2000
        t = pd.date_range("2024-01-01", periods=n, freq="15min")
        rows = []
        for coin in COINS:
            price = 50000 * np.exp(np.cumsum(np.random.normal(0, 0.001, n)))
            sent_tw = np.random.normal(0, 0.3, n)
            sent_nw = np.random.normal(0, 0.2, n)
            fear_greed = np.clip(50 + np.random.normal(0, 10, n), 0, 100)
            google_trends = np.clip(40 + np.random.normal(0, 15, n), 0, 100)
            price[5:] += sent_tw[:-5] * 50
            for i in range(n):
                rows.append({
                    "coin_id": coin,
                    "timestamp": t[i],
                    "close": price[i],
                    "sentiment_twitter": sent_tw[i],
                    "sentiment_news": sent_nw[i],
                    "fear_greed_value": fear_greed[i],
                    "google_trends_value": google_trends[i],
                })
        df = pd.DataFrame(rows)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.info("Synthetic data written to %s", data_path)
    else:
        df = pd.read_csv(data_path)
        if "timestamp_bucket" in df.columns:
            df["timestamp_bucket"] = pd.to_datetime(df["timestamp_bucket"])
            df = df.rename(columns={"timestamp_bucket": "timestamp"})
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            raise ValueError("Input data must contain either 'timestamp_bucket' or 'timestamp'")

    results = run_analysis(df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=convert_numpy)
    logger.info("Granger causality results written to %s", output_path)


if __name__ == "__main__":
    main()
