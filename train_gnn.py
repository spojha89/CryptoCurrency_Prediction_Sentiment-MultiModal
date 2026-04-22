"""
Graph Neural Network training script for cryptocurrency interdependency analysis.

The model treats each timestamp as a graph snapshot:
  - nodes: cryptocurrencies
  - node features: technical + sentiment + simple lagged features
  - edges: training-split return correlations between coins

It performs binary classification (DOWN / UP) for each coin at a chosen forecast horizon
and reports metrics comparable with the XGBoost and TFT pipelines.

Example:
  python models/train_gnn.py ^
      --data-path models/crypto_metrics_20260410_monthly_enh.csv ^
      --horizon-hours 4 ^
      --epochs 60 ^
      --fast-mode
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "LTC-USD"]
BUCKET_MINUTES = 15
DEFAULT_HORIZON_HOURS = 4
DEFAULT_HORIZON_PERIODS = int(DEFAULT_HORIZON_HOURS * 60 / BUCKET_MINUTES)
DEFAULT_RETURN_THRESHOLD = 0.0
DEFAULT_HIDDEN_DIM = 32
DEFAULT_EPOCHS = 80
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT = 0.15
DEFAULT_FAST_EPOCHS = 30

BASE_FEATURES = [
    "close",
    "volume",
    "rsi",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bb_position",
    "vwap",
    "sentiment_composite",
    "sentiment_twitter",
    "sentiment_news",
    "fear_greed_value",
    "google_trends_value",
]

ENGINEERED_FEATURES = [
    "log_return",
    "return_1h",
    "return_4h",
    "volatility_1h",
    "volatility_4h",
    "volume_zscore",
    "vwap_deviation",
    "sentiment_twitter_lag1",
    "sentiment_news_lag1",
    "fear_greed_lag1",
    "google_trends_lag1",
    "sentiment_twitter_change",
    "sentiment_news_change",
]


def safe_log_return(series: pd.Series, periods: int = 1) -> pd.Series:
    previous = series.shift(periods)
    valid_mask = (series > 0) & (previous > 0)
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    result.loc[valid_mask] = np.log(series.loc[valid_mask] / previous.loc[valid_mask])
    return result


def binary_labels_from_returns(future_returns: pd.Series, threshold: float) -> pd.Series:
    labels = pd.Series(np.nan, index=future_returns.index, dtype="object")
    labels[future_returns >= threshold] = "UP"
    labels[future_returns < threshold] = "DOWN"
    return labels


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for coin_id, coin_df in df.groupby("coin_id", sort=False):
        coin_df = coin_df.sort_values("timestamp").copy()
        coin_df["log_return"] = safe_log_return(coin_df["close"]).fillna(0.0)
        coin_df["return_1h"] = safe_log_return(coin_df["close"], periods=4).fillna(0.0)
        coin_df["return_4h"] = safe_log_return(coin_df["close"], periods=16).fillna(0.0)
        coin_df["volatility_1h"] = coin_df["log_return"].rolling(4, min_periods=2).std().fillna(0.0)
        coin_df["volatility_4h"] = coin_df["log_return"].rolling(16, min_periods=4).std().fillna(0.0)

        volume_mean = coin_df["volume"].rolling(16, min_periods=4).mean()
        volume_std = coin_df["volume"].rolling(16, min_periods=4).std()
        coin_df["volume_zscore"] = ((coin_df["volume"] - volume_mean) / volume_std.replace(0, np.nan)).fillna(0.0)

        coin_df["vwap_deviation"] = (
            (coin_df["close"] - coin_df["vwap"]) / coin_df["vwap"].replace(0, np.nan)
        ).fillna(0.0)

        coin_df["sentiment_twitter_lag1"] = coin_df["sentiment_twitter"].shift(1).fillna(0.0)
        coin_df["sentiment_news_lag1"] = coin_df["sentiment_news"].shift(1).fillna(0.0)
        coin_df["fear_greed_lag1"] = coin_df["fear_greed_value"].shift(1).fillna(method="ffill").fillna(50.0)
        coin_df["google_trends_lag1"] = coin_df["google_trends_value"].shift(1).fillna(0.0)
        coin_df["sentiment_twitter_change"] = coin_df["sentiment_twitter"].diff().fillna(0.0)
        coin_df["sentiment_news_change"] = coin_df["sentiment_news"].diff().fillna(0.0)

        frames.append(coin_df)

    return pd.concat(frames, ignore_index=True)


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = BASE_FEATURES + ENGINEERED_FEATURES
    available = [column for column in candidates if column in df.columns]
    missing = sorted(set(candidates) - set(available))
    if missing:
        logger.warning("Missing GNN feature columns (will be omitted): %s", missing)
    return available


@dataclass
class GraphDataset:
    features: torch.Tensor
    labels: torch.Tensor
    mask: torch.Tensor
    timestamps: list[str]
    coin_ids: list[str]
    feature_names: list[str]
    adjacency: torch.Tensor
    edge_summary: list[dict]


def build_edge_summary(adjacency: np.ndarray, coin_ids: list[str]) -> list[dict]:
    rows: list[dict] = []
    for i, source in enumerate(coin_ids):
        for j, target in enumerate(coin_ids):
            if i >= j:
                continue
            rows.append({
                "source": source,
                "target": target,
                "weight": round(float(adjacency[i, j]), 6),
            })
    return sorted(rows, key=lambda row: abs(row["weight"]), reverse=True)


def build_graph_dataset(
    df: pd.DataFrame,
    feature_names: list[str],
    horizon_periods: int,
    return_threshold: float,
) -> GraphDataset:
    df = df.copy()
    df["future_return"] = np.nan
    shifted_close = df.groupby("coin_id", group_keys=False)["close"].shift(-horizon_periods)
    valid_mask = (df["close"] > 0) & (shifted_close > 0)
    df.loc[valid_mask, "future_return"] = np.log(shifted_close[valid_mask] / df.loc[valid_mask, "close"])
    df["label"] = binary_labels_from_returns(df["future_return"], return_threshold)

    pivot_index = "timestamp"
    complete_timestamps = (
        df.groupby(pivot_index)["coin_id"].nunique().loc[lambda s: s == len(COINS)].index
    )
    df = df[df[pivot_index].isin(complete_timestamps)].copy()

    timestamp_strings = [ts.isoformat() for ts in sorted(df[pivot_index].unique())]
    time_count = len(timestamp_strings)
    node_count = len(COINS)
    feature_count = len(feature_names)

    features = np.zeros((time_count, node_count, feature_count), dtype=np.float32)
    labels = np.zeros((time_count, node_count), dtype=np.int64)
    mask = np.zeros((time_count, node_count), dtype=bool)

    timestamp_to_idx = {ts: idx for idx, ts in enumerate(sorted(df[pivot_index].unique()))}
    coin_to_idx = {coin_id: idx for idx, coin_id in enumerate(COINS)}

    for _, row in df.iterrows():
        t_idx = timestamp_to_idx[row[pivot_index]]
        c_idx = coin_to_idx[row["coin_id"]]
        features[t_idx, c_idx, :] = row[feature_names].to_numpy(dtype=np.float32)
        if pd.notna(row["label"]):
            labels[t_idx, c_idx] = 1 if row["label"] == "UP" else 0
            mask[t_idx, c_idx] = True

    train_end = max(1, int(time_count * 0.8))
    train_returns = (
        df[df[pivot_index].isin(sorted(df[pivot_index].unique())[:train_end])]
        .pivot(index=pivot_index, columns="coin_id", values="log_return")
        .reindex(columns=COINS)
        .fillna(0.0)
    )
    adjacency = train_returns.corr().fillna(0.0).to_numpy(dtype=np.float32)
    np.fill_diagonal(adjacency, 1.0)
    adjacency = np.abs(adjacency)
    adjacency = adjacency + np.eye(node_count, dtype=np.float32)
    degree = np.sum(adjacency, axis=1)
    degree_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(degree, 1e-8, None)))
    adjacency = degree_inv_sqrt @ adjacency @ degree_inv_sqrt

    return GraphDataset(
        features=torch.tensor(features, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.long),
        mask=torch.tensor(mask, dtype=torch.bool),
        timestamps=timestamp_strings,
        coin_ids=COINS,
        feature_names=feature_names,
        adjacency=torch.tensor(adjacency, dtype=torch.float32),
        edge_summary=build_edge_summary(adjacency, COINS),
    )


class GraphConvBlock(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        propagated = torch.einsum("ij,bjf->bif", adjacency, x)
        return self.linear(propagated)


class CryptoGNN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = GraphConvBlock(input_dim, hidden_dim)
        self.conv2 = GraphConvBlock(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, 2)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, adjacency)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, adjacency)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.classifier(x)


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.sum() == 0:
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    return torch.nn.functional.cross_entropy(logits[mask], labels[mask])


def compute_metric_block(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    except ValueError:
        auc = 0.0

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["DOWN", "UP"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "mcc": round(float(mcc), 4),
        "auc": round(float(auc), 4),
        "classification_report": report,
    }


def evaluate_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    coin_ids: list[str],
) -> dict:
    probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    predictions = np.argmax(probabilities, axis=-1)
    labels_np = labels.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    overall_true = labels_np[mask_np]
    overall_pred = predictions[mask_np]
    overall_proba = probabilities[mask_np]
    overall_metrics = compute_metric_block(overall_true, overall_pred, overall_proba)

    per_coin_metrics: dict[str, dict] = {}
    for coin_idx, coin_id in enumerate(coin_ids):
        coin_mask = mask_np[:, coin_idx]
        if not coin_mask.any():
            continue
        per_coin_metrics[coin_id] = compute_metric_block(
            labels_np[:, coin_idx][coin_mask],
            predictions[:, coin_idx][coin_mask],
            probabilities[:, coin_idx][coin_mask],
        )
    return {
        "overall_metrics": overall_metrics,
        "per_coin_metrics": per_coin_metrics,
    }


def split_time_indices(time_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(time_count * 0.8)
    val_end = int(time_count * 0.9)
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, time_count)
    return train_idx, val_idx, test_idx


def fit_scaler(
    features: torch.Tensor,
    train_idx: np.ndarray,
) -> StandardScaler:
    scaler = StandardScaler()
    train_view = features[train_idx].reshape(-1, features.shape[-1]).numpy()
    scaler.fit(train_view)
    return scaler


def apply_scaler(features: torch.Tensor, scaler: StandardScaler) -> torch.Tensor:
    reshaped = features.reshape(-1, features.shape[-1]).numpy()
    transformed = scaler.transform(reshaped)
    return torch.tensor(transformed.reshape(features.shape), dtype=torch.float32)


def train_gnn(
    dataset: GraphDataset,
    output_dir: Path,
    horizon_hours: float,
    return_threshold: float,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    dropout: float = DEFAULT_DROPOUT,
    fast_mode: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    time_count = dataset.features.shape[0]
    train_idx, val_idx, test_idx = split_time_indices(time_count)

    scaler = fit_scaler(dataset.features, train_idx)
    features = apply_scaler(dataset.features, scaler)
    labels = dataset.labels
    mask = dataset.mask
    adjacency = dataset.adjacency

    if fast_mode:
        epochs = min(epochs, DEFAULT_FAST_EPOCHS)
        hidden_dim = min(hidden_dim, 16)
        dropout = min(dropout, 0.1)
        logger.info(
            "Fast mode enabled: epochs=%d hidden_dim=%d dropout=%.2f",
            epochs,
            hidden_dim,
            dropout,
        )

    model = CryptoGNN(input_dim=features.shape[-1], hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = None
    best_val_bal_acc = -np.inf
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features[train_idx], adjacency)
        loss = masked_cross_entropy(logits, labels[train_idx], mask[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(features[val_idx], adjacency)
            val_eval = evaluate_predictions(val_logits, labels[val_idx], mask[val_idx], dataset.coin_ids)
            val_bal_acc = val_eval["overall_metrics"]["balanced_accuracy"]

        history.append({
            "epoch": epoch,
            "train_loss": round(float(loss.item()), 6),
            "val_balanced_accuracy": val_bal_acc,
        })
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            logger.info(
                "[epoch=%d] train_loss=%.6f val_bal_acc=%.4f",
                epoch,
                float(loss.item()),
                val_bal_acc,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_logits = model(features[train_idx], adjacency)
        val_logits = model(features[val_idx], adjacency)
        test_logits = model(features[test_idx], adjacency)

    train_metrics = evaluate_predictions(train_logits, labels[train_idx], mask[train_idx], dataset.coin_ids)
    val_metrics = evaluate_predictions(val_logits, labels[val_idx], mask[val_idx], dataset.coin_ids)
    test_metrics = evaluate_predictions(test_logits, labels[test_idx], mask[test_idx], dataset.coin_ids)

    results = {
        "model": "GraphNeuralNetwork",
        "trained_at": datetime.now().isoformat(),
        "coin_ids": dataset.coin_ids,
        "feature_names": dataset.feature_names,
        "horizon_hours": horizon_hours,
        "return_threshold": return_threshold,
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "fast_mode": fast_mode,
        "best_val_balanced_accuracy": round(float(best_val_bal_acc), 4),
        "edge_summary": dataset.edge_summary,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_history": history,
    }

    torch.save(model.state_dict(), output_dir / "gnn_model.pt")
    with open(output_dir / "gnn_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    with open(output_dir / "gnn_edge_summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset.edge_summary, f, indent=2)

    edge_frame = pd.DataFrame(dataset.edge_summary)
    edge_frame.to_csv(output_dir / "gnn_edge_summary.csv", index=False)

    logger.info(
        "GNN overall test metrics: Acc=%.4f BalAcc=%.4f MacroF1=%.4f MCC=%.4f AUC=%.4f",
        test_metrics["overall_metrics"]["accuracy"],
        test_metrics["overall_metrics"]["balanced_accuracy"],
        test_metrics["overall_metrics"]["macro_f1"],
        test_metrics["overall_metrics"]["mcc"],
        test_metrics["overall_metrics"]["auc"],
    )
    return results


def log_results_table(results: dict) -> None:
    logger.info("FINAL GNN RESULTS SUMMARY:")
    logger.info(
        "%-12s  %-10s  %-10s  %-10s  %-10s  %-10s",
        "Coin", "Accuracy", "Bal.Acc", "MacroF1", "MCC", "AUC",
    )
    logger.info("-" * 80)
    for coin_id, metrics in results["test_metrics"]["per_coin_metrics"].items():
        logger.info(
            "%-12s  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-10.4f",
            coin_id,
            metrics["accuracy"],
            metrics["balanced_accuracy"],
            metrics["macro_f1"],
            metrics["mcc"],
            metrics["auc"],
        )
    overall = results["test_metrics"]["overall_metrics"]
    logger.info("-" * 80)
    logger.info(
        "%-12s  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-10.4f",
        "OVERALL",
        overall["accuracy"],
        overall["balanced_accuracy"],
        overall["macro_f1"],
        overall["mcc"],
        overall["auc"],
    )

    logger.info("Top learned interdependency edges:")
    for row in results["edge_summary"][:8]:
        logger.info("  %s <-> %s | weight=%.4f", row["source"], row["target"], row["weight"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight GNN for crypto interdependency classification.")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="models/outputs/gnn")
    parser.add_argument("--horizon-hours", type=float, default=DEFAULT_HORIZON_HOURS)
    parser.add_argument("--return-threshold", type=float, default=DEFAULT_RETURN_THRESHOLD)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--fast-mode", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    df = pd.read_csv(data_path)
    if "timestamp_bucket" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp_bucket"])
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError("Input data must contain either 'timestamp_bucket' or 'timestamp'.")

    df = df[df["coin_id"].isin(COINS)].sort_values(["coin_id", "timestamp"]).reset_index(drop=True)
    df = add_engineered_features(df)
    feature_names = infer_feature_columns(df)
    horizon_periods = max(1, int(round(args.horizon_hours * 60 / BUCKET_MINUTES)))

    dataset = build_graph_dataset(
        df=df,
        feature_names=feature_names,
        horizon_periods=horizon_periods,
        return_threshold=args.return_threshold,
    )
    logger.info(
        "Built graph dataset with %d timestamps, %d nodes, %d features, horizon=%d periods",
        dataset.features.shape[0],
        dataset.features.shape[1],
        dataset.features.shape[2],
        horizon_periods,
    )

    results = train_gnn(
        dataset=dataset,
        output_dir=output_dir / f"horizon_{horizon_periods}",
        horizon_hours=args.horizon_hours,
        return_threshold=args.return_threshold,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        fast_mode=args.fast_mode,
    )
    log_results_table(results)


if __name__ == "__main__":
    main()
