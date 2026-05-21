# FinSentinel — Fine-tuned Financial News Sentiment & Trading Signal System

Fine-tuned **FinBERT** on 9,543 real financial news samples for sentiment classification, served via a **FastAPI** inference endpoint with a **Streamlit** demo correlating sentiment signals against live stock price movements.

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 86.9% |
| Weighted F1 | 0.869 |
| Training Samples | 9,543 |
| Base Model | ProsusAI/finBERT |
| Dataset | Twitter Financial News Sentiment |

## Features

- **Fine-tuned FinBERT** — 3-epoch fine-tuning on T4 GPU with mixed-precision training and learning rate warmup
- **FastAPI inference endpoint** — `/predict` and `/predict_batch` routes with confidence scores per class
- **Streamlit demo** — single headline analysis, batch processing with color-coded sentiment, live stock price charts
- **Live stock correlation** — candlestick chart + volume via yfinance for any public ticker

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | HuggingFace Transformers, FinBERT, PyTorch |
| Training | HuggingFace Trainer, T4 GPU, mixed-precision (fp16) |
| Serving | FastAPI, Uvicorn |
| Frontend | Streamlit, Plotly |
| Market Data | yfinance |

## Architecture

```
Financial News Headline
        │
        ▼
FinBERT Tokenizer (max_length=128)
        │
        ▼
Fine-tuned FinBERT Classification Head
        │
        ▼
Sentiment + Confidence Scores
{negative, neutral, positive}
        │
        ▼
FastAPI Endpoint (/predict, /predict_batch)
        │
        ▼
Streamlit UI + Live Stock Price Correlation
```

## API Endpoints

```bash
# Single prediction
POST /predict
{"text": "Apple reports record quarterly revenue"}

# Response
{"sentiment": "positive", "confidence": 0.94, "scores": {...}}

# Batch prediction
POST /predict_batch
{"texts": ["headline 1", "headline 2", ...]}
```

## Local Setup

```bash
# 1. Clone
git clone https://github.com/saumyg3/finsentinel.git && cd finsentinel

# 2. Install dependencies
pip install fastapi uvicorn transformers torch streamlit yfinance plotly requests

# 3. Download model (fine-tuned weights not included in repo due to size)
#    Run the Colab notebook to fine-tune and save the model to /model

# 4. Start the API
uvicorn api:app --reload --port 8000

# 5. Start the Streamlit app (new terminal)
streamlit run app.py
```

## Training Details

- **Base model:** ProsusAI/finBERT
- **Dataset:** zeroshot/twitter-financial-news-sentiment (9,543 samples)
- **Labels:** Bearish (0), Neutral (1), Bullish (2)
- **Training:** 3 epochs, batch size 16, warmup 100 steps, weight decay 0.01
- **Hardware:** T4 GPU (Google Colab), fp16 mixed precision
- **Best model:** Selected by weighted F1 on validation set
