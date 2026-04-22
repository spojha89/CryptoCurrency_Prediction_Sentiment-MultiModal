# dashboard/pages/2_sentiment_timeline.py
"""
Page 2: Sentiment Timeline
Per-coin sentiment score timelines by platform (Twitter, Reddit, News),
composite sentiment with momentum, and sentiment vs price overlay.
"""
import os
from datetime import datetime, timezone, timedelta

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import boto3
from boto3.dynamodb.conditions import Key

st.set_page_config(page_title="Sentiment Timeline", page_icon="🔍", layout="wide")

COINS     = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "LTC-USD"]
PLATFORM_COLORS = {
    "twitter": "#1DA1F2",
    "reddit":  "#FF4500",
    "news":    "#10B981",
    "composite": "#F59E0B",
}

@st.cache_resource
def get_ddb():
    return boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1"))


@st.cache_data(ttl=60)
def load_sentiment_data(coin_id: str, hours: int = 48) -> pd.DataFrame:
    """Load sentiment and price data from DynamoDB metrics table."""
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
        numeric_cols = [
            "close", "sentiment_twitter", "sentiment_reddit", "sentiment_news",
            "twitter_count", "reddit_count", "news_count",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp_bucket"])

        # compute composite sentiment
        sent_cols = [c for c in ["sentiment_twitter", "sentiment_reddit", "sentiment_news"]
                     if c in df.columns]
        if sent_cols:
            df["sentiment_composite"] = df[sent_cols].mean(axis=1)

        # compute sentiment momentum (4-period rolling delta)
        if "sentiment_composite" in df.columns:
            rolling = df["sentiment_composite"].rolling(4, min_periods=1).mean()
            df["sentiment_momentum"] = rolling.diff()

        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        st.error(f"Data load error: {e}")
        return pd.DataFrame()


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 Sentiment Timeline")

col_a, col_b = st.columns([2, 1])
with col_a:
    selected_coin = st.selectbox("Coin", COINS)
with col_b:
    hours = st.selectbox("Window", [12, 24, 48, 72], index=2, format_func=lambda x: f"{x}h")

df = load_sentiment_data(selected_coin, hours)
if df.empty:
    st.warning("No sentiment data available. Is the sentiment pipeline running?")
    st.stop()

# ── Current sentiment summary ──────────────────────────────────────────────────
st.subheader("Current Sentiment Scores")
m1, m2, m3, m4 = st.columns(4)
for col_ui, platform in zip([m1, m2, m3, m4], ["twitter", "reddit", "news", "composite"]):
    field = f"sentiment_{platform}"
    if field not in df.columns:
        continue
    latest = df[field].dropna()
    score = float(latest.iloc[-1]) if len(latest) > 0 else 0.0
    color = "#10B981" if score > 0.1 else ("#EF4444" if score < -0.1 else "#F59E0B")
    label = "🐦 Twitter" if platform == "twitter" else \
            "🟠 Reddit" if platform == "reddit" else \
            "📰 News" if platform == "news" else "📊 Composite"
    with col_ui:
        st.markdown(f"""
        <div style="background:#111827;border-radius:8px;padding:12px;text-align:center;
                    border-top:3px solid {PLATFORM_COLORS[platform]}">
            <div style="color:#94A3B8;font-size:12px">{label}</div>
            <div style="color:{color};font-size:28px;font-weight:bold">{score:+.3f}</div>
            <div style="color:#94A3B8;font-size:11px">[-1 bearish → +1 bullish]</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Sentiment timeline chart ───────────────────────────────────────────────────
st.subheader("Sentiment Scores Over Time")
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.45, 0.3, 0.25],
    vertical_spacing=0.04,
    subplot_titles=["Platform Sentiment Scores", "Price (USD)", "Post Volume"],
)

# Platform sentiment lines
for platform in ["twitter", "reddit", "news", "composite"]:
    col = f"sentiment_{platform}"
    if col not in df.columns:
        continue
    dash = "solid" if platform != "composite" else "dot"
    width = 1.5 if platform != "composite" else 2.5
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df[col].fillna(0),
        name=platform.capitalize(),
        line=dict(color=PLATFORM_COLORS[platform], width=width, dash=dash),
        hovertemplate=f"{platform}: %{{y:.3f}}<extra></extra>",
    ), row=1, col=1)

# Zero line
fig.add_hline(y=0, line_dash="dash", line_color="#374151", row=1, col=1)
# Bullish/bearish zones
fig.add_hrect(y0=0.1, y1=1.0,  fillcolor="rgba(16,185,129,0.05)", line_width=0, row=1, col=1)
fig.add_hrect(y0=-1.0, y1=-0.1, fillcolor="rgba(239,68,68,0.05)", line_width=0, row=1, col=1)

# Price overlay
if "close" in df.columns:
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["close"],
        name="Price", line=dict(color="#2E74B5", width=1.5),
    ), row=2, col=1)

# Post volume bars (total across platforms)
count_cols = [c for c in ["twitter_count", "reddit_count", "news_count"] if c in df.columns]
if count_cols:
    total_posts = df[count_cols].fillna(0).sum(axis=1)
    fig.add_trace(go.Bar(
        x=df["timestamp"], y=total_posts,
        name="Total Posts",
        marker_color="#6366F1", opacity=0.8,
    ), row=3, col=1)

fig.update_layout(
    height=650,
    template="plotly_dark",
    paper_bgcolor="#0A0F1E",
    plot_bgcolor="#111827",
    font=dict(color="#F8FAFC"),
    legend=dict(orientation="h", y=1.02),
    margin=dict(l=40, r=20, t=50, b=20),
)
fig.update_yaxes(range=[-1, 1], row=1, col=1)
st.plotly_chart(fig, use_container_width=True)

# ── Sentiment momentum heatmap ─────────────────────────────────────────────────
st.subheader("Sentiment Momentum (Hourly Delta)")
if "sentiment_momentum" in df.columns:
    momentum = df[["timestamp", "sentiment_momentum"]].dropna()
    if not momentum.empty:
        fig2 = go.Figure(go.Scatter(
            x=momentum["timestamp"],
            y=momentum["sentiment_momentum"],
            mode="lines+markers",
            fill="tozeroy",
            line=dict(color="#EC4899", width=1.5),
            fillcolor="rgba(236,72,153,0.15)",
            name="Momentum",
        ))
        fig2.update_layout(
            height=200,
            template="plotly_dark",
            paper_bgcolor="#0A0F1E",
            plot_bgcolor="#111827",
            font=dict(color="#F8FAFC"),
            showlegend=False,
            margin=dict(l=40, r=20, t=20, b=20),
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="#374151")
        st.plotly_chart(fig2, use_container_width=True)

# ── Raw data table ────────────────────────────────────────────────────────────
with st.expander("Raw Sentiment Data (last 20 rows)"):
    display_cols = ["timestamp", "close", "sentiment_twitter", "sentiment_reddit",
                    "sentiment_news", "sentiment_composite", "twitter_count", "news_count"]
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available_cols].tail(20).round(4),
        use_container_width=True,
        hide_index=True,
    )
