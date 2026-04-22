# dashboard/pages/1_live_prices.py
"""
Page 1: Live Prices & Predictions
Real-time price chart with overlaid prediction direction indicators,
technical indicators, and signal strength.
"""
import os
import json
from datetime import datetime, timezone, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import boto3
from boto3.dynamodb.conditions import Key

st.set_page_config(page_title="Live Prices", page_icon="📈", layout="wide")

COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "LTC-USD"]

@st.cache_resource
def get_ddb():
    return boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1"))


@st.cache_data(ttl=30)
def load_metrics(coin_id: str, hours: int = 24) -> pd.DataFrame:
    """Load price and indicator metrics from DynamoDB."""
    ddb = get_ddb()
    table = ddb.Table(os.environ.get("METRICS_TABLE", "crypto_metrics"))
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    try:
        resp = table.query(
            KeyConditionExpression=Key("coin_id").eq(coin_id) & Key("timestamp_bucket").gte(cutoff),
            ScanIndexForward=True,
        )
        items = resp.get("Items", [])
        if not items:
            return pd.DataFrame()
        df = pd.DataFrame(items)
        numeric = ["close", "open", "high", "low", "volume", "rsi", "macd",
                   "macd_signal", "bb_upper", "bb_lower", "bb_middle", "vwap"]
        for col in numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp_bucket"])
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        st.error(f"Failed to load metrics: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_predictions(coin_id: str, limit: int = 50) -> pd.DataFrame:
    """Load recent predictions from DynamoDB."""
    ddb = get_ddb()
    table = ddb.Table(os.environ.get("PREDICTIONS_TABLE", "crypto_predictions"))
    try:
        resp = table.query(
            KeyConditionExpression=Key("coin_id").eq(coin_id),
            ScanIndexForward=False,
            Limit=limit,
        )
        items = resp.get("Items", [])
        if not items:
            return pd.DataFrame()
        df = pd.DataFrame(items)
        for col in ["confidence", "prob_up", "prob_flat", "prob_down"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        return pd.DataFrame()


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📈 Live Prices & Predictions")

col_sel1, col_sel2, col_sel3 = st.columns([2, 1, 1])
with col_sel1:
    selected_coin = st.selectbox("Coin", COINS)
with col_sel2:
    hours = st.selectbox("Window", [6, 12, 24, 48, 72], index=2, format_func=lambda x: f"{x}h")
with col_sel3:
    show_bb = st.checkbox("Bollinger Bands", value=True)
    show_predictions = st.checkbox("Predictions", value=True)

df_metrics = load_metrics(selected_coin, hours)
df_preds   = load_predictions(selected_coin)

if df_metrics.empty:
    st.warning(f"No data available for {selected_coin}. Is the pipeline running?")
    st.stop()

# ── Current metrics summary ───────────────────────────────────────────────────
last_row = df_metrics.iloc[-1]
latest_pred = df_preds.iloc[-1] if not df_preds.empty else None

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Price (USD)", f"${float(last_row.get('close', 0)):,.2f}")
with m2:
    rsi_val = float(last_row.get("rsi", 50) or 50)
    rsi_delta = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
    st.metric("RSI (14)", f"{rsi_val:.1f}", rsi_delta)
with m3:
    macd_val = float(last_row.get("macd", 0) or 0)
    st.metric("MACD", f"{macd_val:.2f}", "↑ Bullish" if macd_val > 0 else "↓ Bearish")
with m4:
    if latest_pred is not None:
        direction = latest_pred.get("predicted_direction", "FLAT")
        conf = float(latest_pred.get("confidence", 0))
        st.metric("Prediction", direction, f"{conf:.0%} confidence")
with m5:
    if latest_pred is not None:
        sig = int(latest_pred.get("signal_strength", 1))
        st.metric("Signal Strength", "⭐" * sig, f"{sig}/5")

st.markdown("---")

# ── Main chart ────────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.6, 0.2, 0.2],
    vertical_spacing=0.03,
    subplot_titles=["Price", "Volume", "RSI"],
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=df_metrics["timestamp"],
    open=df_metrics.get("open", df_metrics["close"]),
    high=df_metrics.get("high", df_metrics["close"]),
    low=df_metrics.get("low", df_metrics["close"]),
    close=df_metrics["close"],
    name=selected_coin,
    increasing_line_color="#10B981",
    decreasing_line_color="#EF4444",
), row=1, col=1)

# VWAP
if "vwap" in df_metrics.columns:
    fig.add_trace(go.Scatter(
        x=df_metrics["timestamp"], y=df_metrics["vwap"],
        name="VWAP", line=dict(color="#F59E0B", width=1, dash="dash"),
    ), row=1, col=1)

# Bollinger Bands
if show_bb and all(c in df_metrics.columns for c in ["bb_upper", "bb_lower", "bb_middle"]):
    fig.add_trace(go.Scatter(
        x=df_metrics["timestamp"], y=df_metrics["bb_upper"],
        name="BB Upper", line=dict(color="#6366F1", width=1),
        showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_metrics["timestamp"], y=df_metrics["bb_lower"],
        name="BB Lower", line=dict(color="#6366F1", width=1),
        fill="tonexty", fillcolor="rgba(99,102,241,0.05)",
    ), row=1, col=1)

# Prediction markers
if show_predictions and not df_preds.empty:
    for _, pred_row in df_preds.iterrows():
        direction = pred_row.get("predicted_direction", "FLAT")
        ts = pred_row.get("timestamp")
        conf = float(pred_row.get("confidence", 0))
        if direction == "UP":
            marker = dict(symbol="triangle-up", color="#10B981", size=10 + int(conf * 8))
        elif direction == "DOWN":
            marker = dict(symbol="triangle-down", color="#EF4444", size=10 + int(conf * 8))
        else:
            continue  # skip FLAT markers to reduce clutter
        # find closest price
        closest = df_metrics[df_metrics["timestamp"] >= ts]
        if closest.empty:
            continue
        price_at = float(closest.iloc[0]["close"])
        fig.add_trace(go.Scatter(
            x=[ts], y=[price_at],
            mode="markers", marker=marker,
            name=f"{direction} ({conf:.0%})",
            showlegend=False,
            hovertemplate=f"Direction: {direction}<br>Confidence: {conf:.1%}<extra></extra>",
        ), row=1, col=1)

# Volume
fig.add_trace(go.Bar(
    x=df_metrics["timestamp"],
    y=df_metrics.get("volume", pd.Series([0] * len(df_metrics))),
    name="Volume",
    marker_color="#2E74B5",
    opacity=0.7,
), row=2, col=1)

# RSI
if "rsi" in df_metrics.columns:
    fig.add_trace(go.Scatter(
        x=df_metrics["timestamp"], y=df_metrics["rsi"],
        name="RSI", line=dict(color="#EC4899", width=1.5),
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#EF4444", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#10B981", row=3, col=1)

fig.update_layout(
    height=700,
    template="plotly_dark",
    paper_bgcolor="#0A0F1E",
    plot_bgcolor="#111827",
    font=dict(color="#F8FAFC"),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.02),
    margin=dict(l=40, r=20, t=40, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# ── Prediction probability bar chart ──────────────────────────────────────────
if latest_pred is not None:
    st.subheader("Latest Prediction Breakdown")
    prob_cols = st.columns(3)
    probs = {
        "DOWN": float(latest_pred.get("prob_down", 0.33)),
        "FLAT": float(latest_pred.get("prob_flat", 0.33)),
        "UP":   float(latest_pred.get("prob_up",   0.33)),
    }
    colors = {"DOWN": "#EF4444", "FLAT": "#F59E0B", "UP": "#10B981"}
    for col, (label, prob) in zip(prob_cols, probs.items()):
        with col:
            st.markdown(f"""
            <div style="text-align:center;background:#111827;border-radius:8px;padding:16px">
                <div style="color:#94A3B8;font-size:14px;">{label}</div>
                <div style="color:{colors[label]};font-size:36px;font-weight:bold">{prob:.0%}</div>
                <div style="background:#1F2937;border-radius:4px;height:8px;margin-top:8px">
                    <div style="background:{colors[label]};width:{prob*100:.1f}%;height:8px;border-radius:4px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
