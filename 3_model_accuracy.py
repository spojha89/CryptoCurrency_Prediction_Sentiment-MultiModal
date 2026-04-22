# dashboard/pages/3_model_accuracy.py
"""
Page 3: Model Accuracy Tracker
Rolling evaluation of prediction accuracy.
Compares predicted direction to actual price movement,
computing rolling accuracy, directional accuracy, and confusion matrix.
"""
import os
from datetime import datetime, timezone, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import boto3
from boto3.dynamodb.conditions import Key

st.set_page_config(page_title="Model Accuracy", page_icon="🎯", layout="wide")

COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "LTC-USD"]
FLAT_THRESHOLD = 0.005  # ±0.5%
HORIZON_BARS   = 4      # 4 × 15-min = 1 hour

@st.cache_resource
def get_ddb():
    return boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1"))


@st.cache_data(ttl=120)
def load_predictions_and_actuals(coin_id: str, days: int = 7) -> pd.DataFrame:
    """
    Loads predictions and actual price data, computes whether each
    prediction was correct based on the actual 1-hour forward return.
    """
    ddb = get_ddb()
    pred_table    = ddb.Table(os.environ.get("PREDICTIONS_TABLE", "crypto_predictions"))
    metrics_table = ddb.Table(os.environ.get("METRICS_TABLE", "crypto_metrics"))

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    # fetch predictions
    try:
        resp = pred_table.query(
            KeyConditionExpression=Key("coin_id").eq(coin_id) & Key("timestamp").gte(cutoff),
            ScanIndexForward=True,
        )
        preds = resp.get("Items", [])
    except Exception as e:
        st.error(f"Prediction fetch error: {e}")
        return pd.DataFrame()

    if not preds:
        return pd.DataFrame()

    df_preds = pd.DataFrame(preds)
    for col in ["confidence", "prob_up", "prob_flat", "prob_down"]:
        if col in df_preds.columns:
            df_preds[col] = pd.to_numeric(df_preds[col], errors="coerce")
    df_preds["timestamp"] = pd.to_datetime(df_preds["timestamp"])

    # fetch price data to compute actual direction
    try:
        resp = metrics_table.query(
            KeyConditionExpression=Key("coin_id").eq(coin_id) & Key("timestamp_bucket").gte(cutoff),
            ScanIndexForward=True,
        )
        prices = resp.get("Items", [])
        df_prices = pd.DataFrame(prices)
        df_prices["close"] = pd.to_numeric(df_prices["close"], errors="coerce")
        df_prices["timestamp"] = pd.to_datetime(df_prices["timestamp_bucket"])
        df_prices = df_prices.sort_values("timestamp").set_index("timestamp")
    except Exception:
        return df_preds

    # match each prediction to actual 1-hour forward price
    actual_directions = []
    for ts in df_preds["timestamp"]:
        future_ts = ts + pd.Timedelta(hours=1)
        # find closest price record
        future_prices = df_prices[df_prices.index >= future_ts]
        current_prices = df_prices[df_prices.index <= ts]
        if future_prices.empty or current_prices.empty:
            actual_directions.append(None)
            continue
        p_now    = float(current_prices.iloc[-1]["close"])
        p_future = float(future_prices.iloc[0]["close"])
        if p_now <= 0:
            actual_directions.append(None)
            continue
        log_ret = np.log(p_future / p_now)
        if log_ret > FLAT_THRESHOLD:
            actual_directions.append("UP")
        elif log_ret < -FLAT_THRESHOLD:
            actual_directions.append("DOWN")
        else:
            actual_directions.append("FLAT")

    df_preds["actual_direction"] = actual_directions
    df_preds = df_preds.dropna(subset=["actual_direction"])
    df_preds["correct"] = df_preds["predicted_direction"] == df_preds["actual_direction"]

    # directional accuracy (UP/DOWN only, excluding FLAT)
    df_dir = df_preds[df_preds["actual_direction"] != "FLAT"]
    df_preds["correct_directional"] = df_dir["predicted_direction"] == df_dir["actual_direction"]

    return df_preds


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎯 Model Accuracy Tracker")

col_a, col_b = st.columns([2, 1])
with col_a:
    selected_coin = st.selectbox("Coin", COINS)
with col_b:
    days = st.selectbox("Evaluation Window", [3, 7, 14, 30], index=1,
                        format_func=lambda x: f"Last {x} days")

df = load_predictions_and_actuals(selected_coin, days)
if df.empty:
    st.warning("No prediction data available yet.")
    st.stop()

total = len(df)
n_correct = df["correct"].sum()
overall_acc = n_correct / total if total > 0 else 0

df_dir = df[df["actual_direction"] != "FLAT"]
n_dir = len(df_dir)
dir_acc = df_dir["correct"].sum() / n_dir if n_dir > 0 else 0

# ── Summary metrics ───────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Overall Accuracy", f"{overall_acc:.1%}", f"{n_correct}/{total} correct")
with m2:
    st.metric("Directional Accuracy", f"{dir_acc:.1%}", f"(excl. FLAT) n={n_dir}")
with m3:
    avg_conf = df["confidence"].mean()
    st.metric("Avg Confidence", f"{avg_conf:.1%}")
with m4:
    hi_conf = df[df["confidence"] >= 0.75]
    if len(hi_conf) > 0:
        hi_acc = hi_conf["correct"].mean()
        st.metric("High-Conf Accuracy (≥75%)", f"{hi_acc:.1%}", f"n={len(hi_conf)}")

st.markdown("---")

# ── Rolling accuracy chart ────────────────────────────────────────────────────
st.subheader("Rolling Accuracy (20-prediction window)")
df_sorted = df.sort_values("timestamp")
window = 20
df_sorted["rolling_acc"] = df_sorted["correct"].rolling(window, min_periods=5).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_sorted["timestamp"],
    y=df_sorted["rolling_acc"],
    mode="lines",
    name="Rolling Accuracy",
    line=dict(color="#10B981", width=2),
    fill="tozeroy",
    fillcolor="rgba(16,185,129,0.1)",
))
# random baseline
fig.add_hline(y=0.333, line_dash="dash", line_color="#EF4444",
              annotation_text="Random (33%)", annotation_position="right")
fig.add_hline(y=0.50, line_dash="dash", line_color="#F59E0B",
              annotation_text="50% baseline")

fig.update_layout(
    height=300, template="plotly_dark",
    paper_bgcolor="#0A0F1E", plot_bgcolor="#111827",
    yaxis=dict(tickformat=".0%", range=[0, 1]),
    margin=dict(l=40, r=80, t=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# ── Confusion matrix ──────────────────────────────────────────────────────────
st.subheader("Confusion Matrix")
classes = ["DOWN", "FLAT", "UP"]
cm = pd.DataFrame(0, index=classes, columns=classes)
for _, row in df.iterrows():
    pred   = row.get("predicted_direction")
    actual = row.get("actual_direction")
    if pred in classes and actual in classes:
        cm.loc[actual, pred] += 1

fig_cm = px.imshow(
    cm.values,
    x=[f"Pred {c}" for c in classes],
    y=[f"True {c}" for c in classes],
    color_continuous_scale="Blues",
    text_auto=True,
    title="Confusion Matrix (True label vs Predicted)",
)
fig_cm.update_layout(
    height=350, template="plotly_dark",
    paper_bgcolor="#0A0F1E",
    margin=dict(l=40, r=20, t=40, b=20),
)
st.plotly_chart(fig_cm, use_container_width=True)

# ── Confidence calibration ────────────────────────────────────────────────────
st.subheader("Confidence Calibration")
df["conf_bin"] = pd.cut(df["confidence"], bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        labels=["<50%", "50-60%", "60-70%", "70-80%", "80-90%", "90%+"])
calib = df.groupby("conf_bin", observed=True)["correct"].agg(["mean", "count"]).reset_index()
calib.columns = ["Confidence Bin", "Accuracy", "Count"]

fig_cal = go.Figure()
fig_cal.add_trace(go.Bar(
    x=calib["Confidence Bin"], y=calib["Accuracy"],
    name="Actual Accuracy",
    marker_color="#2E74B5",
    text=calib["Count"].apply(lambda x: f"n={x}"),
    textposition="outside",
))
fig_cal.add_hline(y=0.333, line_dash="dash", line_color="#EF4444")
fig_cal.update_layout(
    height=300, template="plotly_dark",
    paper_bgcolor="#0A0F1E", plot_bgcolor="#111827",
    yaxis=dict(tickformat=".0%", range=[0, 1.1]),
    margin=dict(l=40, r=20, t=20, b=20),
)
st.plotly_chart(fig_cal, use_container_width=True)
