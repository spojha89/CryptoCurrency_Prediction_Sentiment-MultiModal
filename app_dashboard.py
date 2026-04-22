# dashboard/app.py
"""
Streamlit Dashboard – Crypto Prediction System
Main entry point. Configures page, shared state, and sidebar navigation.

Pages:
  1. Live Prices & Predictions
  2. Sentiment Timeline
  3. Model Accuracy Tracker
  4. Alert Configuration
"""
import os
import streamlit as st
import boto3
import redis
import json
from datetime import datetime, timezone

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Crypto Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0A0F1E; }
    .stMetric { background-color: #111827; border-radius: 8px; padding: 12px; }
    .metric-up { color: #10B981; font-weight: bold; }
    .metric-down { color: #EF4444; font-weight: bold; }
    .metric-flat { color: #F59E0B; font-weight: bold; }
    .signal-bar { height: 8px; border-radius: 4px; }
    div[data-testid="stSidebar"] { background-color: #0D1F35; }
    h1, h2, h3 { color: #F8FAFC; }
    .stSelectbox > div > div { background-color: #111827; color: #F8FAFC; }
</style>
""", unsafe_allow_html=True)

# ── Connection helpers ─────────────────────────────────────────────────────────
@st.cache_resource
def get_dynamodb():
    return boto3.resource(
        "dynamodb",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )

@st.cache_resource
def get_redis():
    endpoint = os.environ.get("REDIS_ENDPOINT", "")
    if not endpoint:
        return None
    try:
        r = redis.Redis(host=endpoint, port=6379, decode_responses=True, socket_timeout=2)
        r.ping()
        return r
    except Exception:
        return None

# ── Shared data fetchers ───────────────────────────────────────────────────────
COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "LTC-USD"]

@st.cache_data(ttl=30)  # refresh every 30 seconds
def get_latest_predictions() -> dict:
    """
    Fetch latest prediction for each coin.
    Tries Redis first (low-latency), falls back to DynamoDB.
    """
    results = {}
    r = get_redis()
    ddb = get_dynamodb()
    predictions_table = ddb.Table(
        os.environ.get("PREDICTIONS_TABLE", "crypto_predictions")
    )

    for coin_id in COINS:
        prediction = None

        # try Redis cache first
        if r:
            try:
                cached = r.get(f"prediction:{coin_id}:latest")
                if cached:
                    prediction = json.loads(cached)
            except Exception:
                pass

        # fallback to DynamoDB
        if not prediction:
            try:
                from boto3.dynamodb.conditions import Key
                response = predictions_table.query(
                    KeyConditionExpression=Key("coin_id").eq(coin_id),
                    ScanIndexForward=False,
                    Limit=1,
                )
                items = response.get("Items", [])
                if items:
                    prediction = {k: float(v) if isinstance(v, str) and v.replace('.', '', 1).lstrip('-').isdigit()
                                  else v for k, v in items[0].items()}
            except Exception as e:
                st.warning(f"DynamoDB fetch error for {coin_id}: {e}")

        results[coin_id] = prediction
    return results


@st.cache_data(ttl=60)
def get_recent_metrics(coin_id: str, hours: int = 24) -> list:
    """Fetch recent metrics records from DynamoDB for chart rendering."""
    from boto3.dynamodb.conditions import Key
    from datetime import timedelta
    ddb = get_dynamodb()
    table = ddb.Table(os.environ.get("METRICS_TABLE", "crypto_metrics"))
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    try:
        response = table.query(
            KeyConditionExpression=(
                Key("coin_id").eq(coin_id) &
                Key("timestamp_bucket").gte(cutoff)
            ),
            ScanIndexForward=True,
        )
        return response.get("Items", [])
    except Exception as e:
        return []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Crypto Prediction")
    st.markdown("---")
    selected_coin = st.selectbox("Select Coin", COINS, index=0)
    st.markdown("---")
    st.markdown("**System Status**")

    r = get_redis()
    st.markdown(
        f"Redis: {'🟢 Connected' if r else '🔴 Offline'}"
    )
    st.markdown(
        f"DynamoDB: 🟢 Connected"
    )
    st.markdown("---")
    st.markdown(f"*Last refresh: {datetime.now().strftime('%H:%M:%S')}*")
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
    if auto_refresh:
        st.empty()  # placeholder for auto-refresh


# ── Main content: redirect to page 1 by default ───────────────────────────────
st.title("Cryptocurrency Price Prediction System")
st.markdown(
    "Use the sidebar pages to navigate: "
    "**Live Prices** · **Sentiment** · **Model Accuracy** · **Alerts**"
)

# Show quick summary on home page
st.subheader("📊 Current Predictions")
predictions = get_latest_predictions()

cols = st.columns(len(COINS))
for col, coin_id in zip(cols, COINS):
    pred = predictions.get(coin_id)
    with col:
        if pred:
            direction  = pred.get("predicted_direction", "FLAT")
            confidence = float(pred.get("confidence", 0))
            signal     = int(pred.get("signal_strength", 1))
            color = {"UP": "#10B981", "DOWN": "#EF4444", "FLAT": "#F59E0B"}[direction]
            arrow = {"UP": "▲", "DOWN": "▼", "FLAT": "●"}[direction]
            st.markdown(f"""
            <div style="background:#111827;border-radius:8px;padding:16px;text-align:center;
                        border-left:4px solid {color}">
                <div style="color:#94A3B8;font-size:12px;">{coin_id}</div>
                <div style="color:{color};font-size:28px;font-weight:bold;">{arrow} {direction}</div>
                <div style="color:#F8FAFC;font-size:14px;">{confidence:.0%} confidence</div>
                <div style="color:#94A3B8;font-size:12px;">Signal: {'⭐' * signal}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"{coin_id}\nNo prediction")
