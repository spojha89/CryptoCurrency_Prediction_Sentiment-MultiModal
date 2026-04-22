"""
Create a single dashboard-style figure showing crypto price action,
technical indicators, and sentiment metrics on one timeline.

Example:
  python models/plot_crypto_dashboard.py ^
      --data-path models/crypto_metrics_training_20260410_YearlyEnh.csv ^
      --coin BTC-USD ^
      --last 500
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("models/outputs/charts")
DEFAULT_PRICE_COLUMNS = ["close", "vwap", "bb_upper", "bb_middle", "bb_lower"]
DEFAULT_SENTIMENT_COLUMNS = ["sentiment_composite", "sentiment_twitter", "sentiment_news"]
DEFAULT_MACRO_COLUMNS = ["fear_greed_value", "google_trends_value"]


def load_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if "timestamp_bucket" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp_bucket"])
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError("Input data must contain either 'timestamp_bucket' or 'timestamp'.")
    return df.sort_values(["coin_id", "timestamp"]).reset_index(drop=True)


def prepare_coin_frame(
    df: pd.DataFrame,
    coin_id: str,
    last: int | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    coin_df = df[df["coin_id"] == coin_id].copy()
    if coin_df.empty:
        raise ValueError(f"No rows found for coin {coin_id}.")

    if start:
        coin_df = coin_df[coin_df["timestamp"] >= pd.to_datetime(start)]
    if end:
        coin_df = coin_df[coin_df["timestamp"] <= pd.to_datetime(end)]
    if last:
        coin_df = coin_df.tail(last)

    coin_df = coin_df.reset_index(drop=True)
    if coin_df.empty:
        raise ValueError(f"No rows left for coin {coin_id} after filtering.")
    return coin_df


def _available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _plot_price_panel(ax, df: pd.DataFrame) -> None:
    available_price = _available_columns(df, DEFAULT_PRICE_COLUMNS)
    if "close" in available_price:
        ax.plot(df["timestamp"], df["close"], color="#0f172a", linewidth=1.8, label="Close")
    if "vwap" in available_price:
        ax.plot(df["timestamp"], df["vwap"], color="#0ea5e9", linewidth=1.2, label="VWAP")
    if "bb_upper" in available_price and "bb_lower" in available_price:
        ax.fill_between(
            df["timestamp"],
            df["bb_lower"],
            df["bb_upper"],
            color="#cbd5e1",
            alpha=0.35,
            label="Bollinger Band",
        )
    if "bb_middle" in available_price:
        ax.plot(df["timestamp"], df["bb_middle"], color="#94a3b8", linewidth=1.0, linestyle="--", label="BB Mid")

    ax.set_ylabel("Price", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", ncol=4, fontsize=8, frameon=False)


def _plot_ta_panel(ax, df: pd.DataFrame) -> None:
    rsi_present = "rsi" in df.columns
    macd_present = all(column in df.columns for column in ("macd", "macd_signal"))

    if rsi_present:
        ax.plot(df["timestamp"], df["rsi"], color="#f97316", linewidth=1.4, label="RSI")
        ax.axhline(70, color="#ef4444", linewidth=0.9, linestyle="--", alpha=0.6)
        ax.axhline(30, color="#22c55e", linewidth=0.9, linestyle="--", alpha=0.6)
        ax.set_ylim(0, 100)
        ax.set_ylabel("RSI", fontsize=10)
    else:
        ax.set_ylabel("Indicators", fontsize=10)

    if macd_present:
        macd_ax = ax.twinx()
        if "macd_histogram" in df.columns:
            colors = ["#16a34a" if value >= 0 else "#dc2626" for value in df["macd_histogram"]]
            macd_ax.bar(
                df["timestamp"],
                df["macd_histogram"],
                color=colors,
                alpha=0.28,
                width=0.01,
                label="MACD Hist",
            )
        macd_ax.plot(df["timestamp"], df["macd"], color="#2563eb", linewidth=1.1, label="MACD")
        macd_ax.plot(df["timestamp"], df["macd_signal"], color="#7c3aed", linewidth=1.0, label="MACD Signal")
        macd_ax.set_ylabel("MACD", fontsize=10)

        lines_left, labels_left = ax.get_legend_handles_labels()
        lines_right, labels_right = macd_ax.get_legend_handles_labels()
        ax.legend(lines_left + lines_right, labels_left + labels_right, loc="upper left", ncol=4, fontsize=8, frameon=False)
    elif rsi_present:
        ax.legend(loc="upper left", fontsize=8, frameon=False)

    ax.grid(True, alpha=0.2)


def _plot_sentiment_panel(ax, df: pd.DataFrame) -> None:
    sentiment_cols = _available_columns(df, DEFAULT_SENTIMENT_COLUMNS)
    palette = {
        "sentiment_composite": "#0f766e",
        "sentiment_twitter": "#2563eb",
        "sentiment_news": "#d97706",
    }
    for column in sentiment_cols:
        ax.plot(df["timestamp"], df[column], linewidth=1.2, label=column.replace("_", " ").title(), color=palette.get(column))

    ax.axhline(0, color="#94a3b8", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_ylabel("Sentiment", fontsize=10)
    ax.grid(True, alpha=0.2)
    if sentiment_cols:
        ax.legend(loc="upper left", ncol=3, fontsize=8, frameon=False)


def _plot_macro_panel(ax, df: pd.DataFrame) -> None:
    macro_cols = _available_columns(df, DEFAULT_MACRO_COLUMNS)
    palette = {
        "fear_greed_value": "#dc2626",
        "google_trends_value": "#0284c7",
    }
    for column in macro_cols:
        ax.plot(df["timestamp"], df[column], linewidth=1.3, label=column.replace("_", " ").title(), color=palette.get(column))

    ax.set_ylabel("Macro", fontsize=10)
    ax.set_xlabel("Timestamp", fontsize=10)
    ax.grid(True, alpha=0.2)
    if macro_cols:
        ax.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)


def build_dashboard(
    df: pd.DataFrame,
    coin_id: str,
    output_path: Path,
    title_suffix: str = "",
) -> Path:
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is not installed. Please install matplotlib to generate charts.")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 2.2, 2.0, 1.8], "hspace": 0.08},
    )

    _plot_price_panel(axes[0], df)
    _plot_ta_panel(axes[1], df)
    _plot_sentiment_panel(axes[2], df)
    _plot_macro_panel(axes[3], df)

    start_ts = df["timestamp"].min().strftime("%Y-%m-%d %H:%M")
    end_ts = df["timestamp"].max().strftime("%Y-%m-%d %H:%M")
    title = f"{coin_id} Market + Indicator + Sentiment Dashboard"
    subtitle = f"{start_ts} to {end_ts}"
    if title_suffix:
        subtitle = f"{subtitle} | {title_suffix}"

    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.955, subtitle, ha="center", fontsize=10, color="#475569")

    axes[3].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved dashboard chart to %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot crypto indicators and sentiment on a single dashboard figure.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to CSV file.")
    parser.add_argument("--coin", type=str, default="BTC-USD", help="Coin symbol, for example BTC-USD.")
    parser.add_argument("--last", type=int, default=500, help="Plot the last N rows after filtering.")
    parser.add_argument("--start", type=str, help="Optional start datetime filter.")
    parser.add_argument("--end", type=str, help="Optional end datetime filter.")
    parser.add_argument("--output", type=str, help="Optional output PNG path.")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR / f"{args.coin.replace('-', '_')}_dashboard.png"

    df = load_dataset(data_path)
    coin_df = prepare_coin_frame(df, args.coin, last=args.last, start=args.start, end=args.end)
    title_suffix = f"{len(coin_df)} rows from {data_path.name}"
    build_dashboard(coin_df, args.coin, output_path, title_suffix=title_suffix)


if __name__ == "__main__":
    main()
