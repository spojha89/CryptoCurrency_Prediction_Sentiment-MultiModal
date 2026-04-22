CryptoCurrency Prediction using Sentiment Analysis & Multimodal Deep Learning

A research-driven machine learning and deep learning project for cryptocurrency price movement prediction using historical market data, social media sentiment, news sentiment, and Fear & Greed Index.

This project explores how multimodal data fusion can improve prediction accuracy by combining:

📈 Historical OHLCV market data
📰 News sentiment analysis
🐦 Twitter / social media sentiment
😨 Fear & Greed Index
🔍 Technical indicators

The system evaluates multiple models such as:

XGBoost
Temporal Fusion Transformer (TFT)
Graph Neural Networks (GNN)
📌 Problem Statement

Cryptocurrency markets are highly volatile and influenced by multiple external factors such as:

market trends,
investor sentiment,
breaking news,
macroeconomic conditions.

Traditional models relying only on price history often fail to capture these external signals.

This project aims to build a multimodal predictive framework that integrates structured financial data with unstructured sentiment data to improve forecasting performance.

🏗️ Project Architecture
                +----------------------+
                | Historical OHLCV API |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Technical Indicators |
                +----------+-----------+
                           |
                           v
+----------------+    +------------------+    +-------------------+
| Twitter/X Data | -> | Sentiment Engine | <- | News Articles/API |
+----------------+    +------------------+    +-------------------+
                           |
                           v
                +----------------------+
                | Fear & Greed Index   |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Feature Engineering  |
                +----------+-----------+
                           |
                           v
        +----------------+-------------------+----------------+
        |                |                   |                |
        v                v                   v                v
   XGBoost         TFT Model            GNN Model      Model Comparison
📂 Dataset Scenarios

This project experiments with multiple real-world scenarios:

Scenario A: Real-time Short-Term Prediction
1 month real-time crypto market data
Twitter sentiment
News sentiment
Fear & Greed Index
Scenario B: Medium-Term Historical Prediction
1 year historical market data
Fear & Greed Index
Scenario C: Long-Term Multimodal Prediction
4 years historical market data
Kaggle-based Twitter sentiment dataset
Fear & Greed Index
⚙️ Features Used
Market Features
Open
High
Low
Close
Volume
Technical Indicators
RSI
MACD
Bollinger Bands
SMA / EMA
ATR
Momentum
Volatility metrics
Sentiment Features
Average Twitter sentiment
News sentiment score
Sentiment volatility
Market Psychology
Fear & Greed Index
🤖 Models Implemented
1. XGBoost

A tree-based ensemble learning model for classification of:

UP
DOWN
FLAT
Advantages:
Handles tabular structured data efficiently
Provides feature importance
Fast training
2. Temporal Fusion Transformer (TFT)

Deep learning model specialized for multi-horizon time-series forecasting.

Architecture:
Variable Selection Networks
Gated Residual Networks
LSTM Encoder-Decoder
Multi-Head Attention
Quantile Forecasting
Metrics:
Direction Accuracy
Balanced Accuracy
Macro F1
MCC
MAE
RMSE
3. Graph Neural Network (GNN)

Models inter-coin relationships and dependencies.

Example graph relationships:

BTC ↔ ETH
BTC ↔ BNB
ETH ↔ SOL
Metrics:
Accuracy
Balanced Accuracy
Macro F1
MCC
AUC
📊 Evaluation Metrics

This project uses robust evaluation metrics:

Classification Metrics
Accuracy
Balanced Accuracy
Precision
Recall
F1 Score
Macro F1
MCC
ROC-AUC
Regression Metrics
MAE
RMSE
MAPE
📈 Key Findings

Some observations from experiments:

XGBoost performs well with structured historical + engineered features.
TFT captures temporal dependencies but may require larger datasets.
GNN helps when inter-coin relationships are strong.
Sentiment features improve performance in short-term prediction.
Fear & Greed Index contributes to market trend direction.
🛠️ Installation

Clone the repository:

git clone https://github.com/spojha89/CryptoCurrency_Prediction_Sentiment-MultiModal.git
cd CryptoCurrency_Prediction_Sentiment-MultiModal

Install dependencies:

pip install -r requirements.txt
▶️ Usage
Train XGBoost
python models/train_xgboost.py
Train TFT
python models/train_tft.py
Train GNN
python models/train_gnn.py
📁 Project Structure
CryptoCurrency_Prediction_Sentiment-MultiModal/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── models/
│   ├── train_xgboost.py
│   ├── train_tft.py
│   ├── train_gnn.py
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Sentiment_Analysis.ipynb
│
├── utils/
│   ├── indicators.py
│   ├── preprocessing.py
│
├── results/
│   ├── charts/
│   ├── metrics/
│
├── requirements.txt
└── README.md
🔬 Research Contributions

This repository contributes by:

✅ Comparing traditional ML vs advanced DL models
✅ Studying sentiment impact on crypto forecasting
✅ Evaluating multimodal fusion strategies
✅ Investigating inter-coin relationships using GNN

📚 Future Improvements

Potential future enhancements:

Real-time streaming predictions
Reinforcement learning trading agent
LLM-based sentiment extraction
Agentic AI system for automated signal generation
Portfolio optimization integration
