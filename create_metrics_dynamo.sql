-- ============================================================
-- crypto_metrics  –  DynamoDB Table DDL (AWS CLI / SDK equiv.)
-- Represented as SQL-style CREATE TABLE for documentation.
-- Actual AWS CLI command shown below.
-- ============================================================

CREATE TABLE crypto_metrics (

    -- ── Keys ────────────────────────────────────────────────
    coin_id             STRING      NOT NULL,   -- Partition Key  e.g. "BTC-USD"
    timestamp_bucket    STRING      NOT NULL,   -- Sort Key       e.g. "2025-01-15T14:00:00+00:00"

    -- ── OHLCV (stored as STRING to preserve float precision) ─
    open                STRING,
    high                STRING,
    low                 STRING,
    close               STRING,
    volume              STRING,

    -- ── Technical Indicators ─────────────────────────────────
    rsi                 STRING,                 -- RSI(14), null if < 14 bars
    macd                STRING,                 -- MACD line (EMA12 - EMA26)
    macd_signal         STRING,                 -- Signal line (EMA9 of MACD)
    macd_histogram      STRING,                 -- MACD line - Signal line
    bb_upper            STRING,                 -- Bollinger Upper (SMA20 + 2σ)
    bb_middle           STRING,                 -- Bollinger Middle (SMA20)
    bb_lower            STRING,                 -- Bollinger Lower (SMA20 - 2σ)
    bb_position         STRING,                 -- (close-lower)/(upper-lower), 0-1
    vwap                STRING,                 -- Volume-Weighted Average Price

    -- ── Sentiment Scores (added via UpdateItem) ───────────────
    sentiment_twitter   STRING,                 -- Weighted score [-1, +1]
    sentiment_reddit    STRING,                 -- Weighted score [-1, +1]
    sentiment_news      STRING,                 -- Weighted score [-1, +1]
    sentiment_composite STRING,                 -- Mean of available platforms
    twitter_count       NUMBER,                 -- Posts scored in this bucket
    reddit_count        NUMBER,
    news_count          NUMBER,
    fear_greed_value    STRING,                 -- Alternative.me Fear & Greed Index [0, 100]
    fear_greed_classification STRING,           -- e.g. "Fear", "Greed", "Neutral"
    fear_greed_count    NUMBER,                 -- Records ingested for this bucket
    sentiment_method    STRING,                 -- "bedrock" | "vader" | "vader_fallback"

    -- ── System ───────────────────────────────────────────────
    ttl                 NUMBER,                 -- Unix epoch, auto-deleted after 30 days

    PRIMARY KEY (coin_id, timestamp_bucket),

    GLOBAL SECONDARY INDEX timestamp_index (
        HASH KEY    timestamp_bucket,
        PROJECTION  INCLUDE (coin_id, close, sentiment_composite, rsi)
    )
)
BILLING_MODE        = PAY_PER_REQUEST
TTL_ATTRIBUTE       = ttl
PITR                = ENABLED
