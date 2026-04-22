# dashboard/pages/4_alert_config.py
"""
Page 4: Alert Configuration
View recent alerts, configure thresholds, subscribe/unsubscribe
from SNS notifications.
"""
import os
import json
from datetime import datetime, timezone, timedelta

import streamlit as st
import pandas as pd
import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError

st.set_page_config(page_title="Alert Config", page_icon="🔔", layout="wide")

COINS        = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "LTC-USD"]
ALERT_TYPES  = ["high_confidence", "sentiment_reversal", "price_anomaly"]

@st.cache_resource
def get_ddb():
    return boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1"))

@st.cache_resource
def get_sns():
    return boto3.client("sns", region_name=os.environ.get("AWS_REGION", "us-east-1"))


@st.cache_data(ttl=30)
def load_recent_alerts(hours: int = 24) -> pd.DataFrame:
    """Load recent alerts from DynamoDB alerts table."""
    ddb = get_ddb()
    table = ddb.Table(os.environ.get("ALERTS_TABLE", "crypto_alerts"))
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    try:
        # scan (alerts table is small, scan is acceptable)
        response = table.scan(
            FilterExpression=Attr("timestamp").gte(cutoff),
        )
        items = response.get("Items", [])
        if not items:
            return pd.DataFrame()
        df = pd.DataFrame(items)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    except Exception as e:
        st.error(f"Alert load error: {e}")
        return pd.DataFrame()


def subscribe_to_sns(email: str) -> bool:
    """Subscribe an email address to the SNS alerts topic."""
    sns = get_sns()
    topic_arn = os.environ.get("SNS_ALERT_TOPIC_ARN", "")
    if not topic_arn:
        return False
    try:
        sns.subscribe(
            TopicArn=topic_arn,
            Protocol="email",
            Endpoint=email,
        )
        return True
    except ClientError as e:
        st.error(f"SNS subscription error: {e}")
        return False


def acknowledge_alert(alert_id: str):
    """Mark an alert as acknowledged in DynamoDB."""
    ddb = get_ddb()
    table = ddb.Table(os.environ.get("ALERTS_TABLE", "crypto_alerts"))
    try:
        # scan to find PK (alert_id)
        resp = table.scan(FilterExpression=Attr("alert_id").eq(alert_id))
        items = resp.get("Items", [])
        if not items:
            return
        item = items[0]
        table.update_item(
            Key={"alert_id": item["alert_id"], "timestamp": item["timestamp"]},
            UpdateExpression="SET acknowledged = :val",
            ExpressionAttributeValues={":val": True},
        )
    except Exception as e:
        st.warning(f"Acknowledge error: {e}")


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔔 Alert Configuration")

tab1, tab2 = st.tabs(["Recent Alerts", "Subscription Settings"])

# ── Tab 1: Recent Alerts ──────────────────────────────────────────────────────
with tab1:
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_coin = st.multiselect("Filter by Coin", COINS, default=COINS)
    with col_f2:
        filter_type = st.multiselect("Filter by Type", ALERT_TYPES, default=ALERT_TYPES)
    with col_f3:
        alert_hours = st.selectbox("Time Window", [6, 12, 24, 48, 72], index=2,
                                   format_func=lambda x: f"Last {x}h")

    df_alerts = load_recent_alerts(alert_hours)

    if df_alerts.empty:
        st.info("No alerts in the selected window. The system is calm. ✅")
    else:
        # apply filters
        if filter_coin:
            df_alerts = df_alerts[df_alerts["coin_id"].isin(filter_coin)]
        if filter_type:
            df_alerts = df_alerts[df_alerts["alert_type"].isin(filter_type)]

        # stats
        n_total = len(df_alerts)
        n_unack  = (~df_alerts.get("acknowledged", pd.Series([False]*n_total)).fillna(False)).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Alerts", n_total)
        c2.metric("Unacknowledged", n_unack)
        c3.metric("By Type", df_alerts["alert_type"].value_counts().to_dict())

        st.markdown("---")

        # alert cards
        for _, alert in df_alerts.head(30).iterrows():
            alert_type = alert.get("alert_type", "unknown")
            ack        = bool(alert.get("acknowledged", False))
            coin_id    = alert.get("coin_id", "?")
            ts         = str(alert.get("timestamp", ""))
            msg        = str(alert.get("message", ""))

            type_colors = {
                "high_confidence":   "#10B981",
                "sentiment_reversal": "#F59E0B",
                "price_anomaly":     "#EF4444",
            }
            color = type_colors.get(alert_type, "#6366F1")
            opacity = "0.5" if ack else "1.0"

            with st.container():
                col_badge, col_content, col_action = st.columns([1, 6, 1])
                with col_badge:
                    st.markdown(f"""
                    <div style="background:{color};color:white;border-radius:6px;
                                padding:6px;text-align:center;opacity:{opacity};font-size:11px">
                        {alert_type.replace('_', ' ').upper()}
                    </div>
                    """, unsafe_allow_html=True)
                with col_content:
                    st.markdown(f"""
                    <div style="background:#111827;border-radius:6px;padding:10px;
                                opacity:{opacity};border-left:3px solid {color}">
                        <span style="color:#94A3B8;font-size:11px">{coin_id} · {ts[:19]}</span><br>
                        <span style="color:#F8FAFC;font-size:13px">{msg[:200]}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with col_action:
                    if not ack:
                        if st.button("✓", key=f"ack_{alert.get('alert_id', ts)}",
                                     help="Acknowledge alert"):
                            acknowledge_alert(str(alert.get("alert_id", "")))
                            st.rerun()

# ── Tab 2: Subscription Settings ─────────────────────────────────────────────
with tab2:
    st.subheader("Email Notifications")
    st.markdown("""
    Subscribe to receive email alerts when:
    - **High Confidence**: Model predicts with >85% confidence
    - **Sentiment Reversal**: Composite sentiment shifts by >0.5 points
    - **Price Anomaly**: 24-hour return z-score exceeds 2.5σ
    """)

    email_input = st.text_input("Email address", placeholder="your@email.com")
    if st.button("Subscribe to Alerts", type="primary"):
        if "@" in email_input:
            success = subscribe_to_sns(email_input)
            if success:
                st.success(
                    f"✅ Subscription request sent to {email_input}. "
                    "Check your inbox to confirm."
                )
        else:
            st.error("Please enter a valid email address.")

    st.markdown("---")
    st.subheader("Alert Thresholds")
    st.info("Thresholds are configured via Lambda environment variables. "
            "Contact your system administrator to adjust these values.")

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.markdown("**High Confidence Threshold**")
        st.code(f"{os.environ.get('HIGH_CONFIDENCE', '0.85')}", language="text")
    with col_t2:
        st.markdown("**Sentiment Reversal Threshold**")
        st.code(f"{os.environ.get('SENTIMENT_REVERSAL_THRESH', '0.50')}", language="text")
    with col_t3:
        st.markdown("**Price Anomaly Z-Score**")
        st.code(f"{os.environ.get('PRICE_ANOMALY_ZSCORE', '2.5')}", language="text")
