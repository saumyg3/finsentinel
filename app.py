import streamlit as st
import requests
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta

API_URL = "http://localhost:8000"

st.set_page_config(page_title="FinSentinel", page_icon="📈", layout="wide")

st.title("📈 FinSentinel")
st.markdown("**Fine-tuned FinBERT** for financial news sentiment analysis — correlate sentiment signals with stock price movements")

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
period = st.sidebar.selectbox("Price History", ["1mo", "3mo", "6mo", "1y"], index=1)

# ── Single Prediction ─────────────────────────────────────
st.header("🔍 Analyze a Headline")
headline = st.text_input("Enter a financial news headline:", 
    value="Apple reports record quarterly revenue driven by iPhone sales")

if st.button("Analyze Sentiment", type="primary"):
    with st.spinner("Running inference..."):
        resp = requests.post(f"{API_URL}/predict", json={"text": headline})
        result = resp.json()

    col1, col2, col3 = st.columns(3)
    sentiment = result['sentiment']
    color = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}[sentiment]

    col1.metric("Sentiment", f"{color} {sentiment.upper()}")
    col2.metric("Confidence", f"{result['confidence']*100:.1f}%")
    col3.metric("Model", "FinBERT Fine-tuned")

    scores = result['scores']
    fig = px.bar(
        x=list(scores.keys()), y=list(scores.values()),
        color=list(scores.keys()),
        color_discrete_map={"negative": "#e74c3c", "neutral": "#f39c12", "positive": "#2ecc71"},
        labels={"x": "Sentiment", "y": "Probability"},
        title="Sentiment Score Distribution"
    )
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

# ── Batch Analysis ────────────────────────────────────────
st.header("📰 Batch Headline Analysis")
st.markdown("Enter multiple headlines (one per line):")

default_headlines = """Apple beats earnings estimates as iPhone demand surges globally
Federal Reserve signals further interest rate hikes amid inflation concerns
Microsoft announces major layoffs affecting cloud and gaming divisions
Goldman Sachs reports strong trading revenue in volatile markets
Tesla misses delivery targets for the third consecutive quarter
Amazon Web Services growth accelerates driving overall profitability
Oil prices stabilize after OPEC production cut announcement
Nvidia GPU demand hits record high driven by AI infrastructure buildout"""

headlines_input = st.text_area("Headlines", value=default_headlines, height=200)

if st.button("Analyze All Headlines"):
    headlines = [h.strip() for h in headlines_input.strip().split('\n') if h.strip()]
    with st.spinner(f"Analyzing {len(headlines)} headlines..."):
        resp = requests.post(f"{API_URL}/predict_batch", json={"texts": headlines})
        results = resp.json()['results']

    df_results = pd.DataFrame(results)
    
    # Color code
    def color_sentiment(val):
        colors = {"positive": "background-color: #d4edda", 
                  "neutral": "background-color: #fff3cd",
                  "negative": "background-color: #f8d7da"}
        return colors.get(val, "")

    st.dataframe(
        df_results[['text', 'sentiment', 'confidence']].style.applymap(
            color_sentiment, subset=['sentiment']
        ),
        use_container_width=True, height=300
    )

    # Summary chart
    sentiment_counts = df_results['sentiment'].value_counts()
    fig2 = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        color=sentiment_counts.index,
        color_discrete_map={"negative": "#e74c3c", "neutral": "#f39c12", "positive": "#2ecc71"},
        title="Sentiment Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Stock Price Correlation ───────────────────────────────
st.header(f"📊 Stock Price — {ticker.upper()}")
try:
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    hist.index = hist.index.tz_localize(None)

    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=(f"{ticker.upper()} Price", "Volume"),
                         row_heights=[0.7, 0.3])

    fig3.add_trace(go.Candlestick(
        x=hist.index, open=hist['Open'], high=hist['High'],
        low=hist['Low'], close=hist['Close'], name="Price"
    ), row=1, col=1)

    fig3.add_trace(go.Bar(
        x=hist.index, y=hist['Volume'],
        name="Volume", marker_color='steelblue', opacity=0.5
    ), row=2, col=1)

    fig3.update_layout(height=500, xaxis_rangeslider_visible=False,
                       title=f"{ticker.upper()} — {period} Price History")
    st.plotly_chart(fig3, use_container_width=True)

    # Key stats
    info = stock.info
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
    col2.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
    col3.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")
    col4.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f}B")

except Exception as e:
    st.error(f"Could not fetch stock data: {e}")

st.markdown("---")
st.markdown("**FinSentinel** — Fine-tuned FinBERT on 9,543 real financial news samples | F1: 0.869 | Accuracy: 86.9%")
