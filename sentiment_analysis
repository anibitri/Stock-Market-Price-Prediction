import os
import re
from datetime import date, timedelta
import tensorflow
import pandas as pd
import transformers
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from polygon import RESTClient

# from StockMarketPred import Symobl, Company_name

load_dotenv()
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

tokenizer = transformers.AutoTokenizer.from_pretrained("ProsusAI/finbert")
# Use the sequence classification head to enable sentiment predictions
model = transformers.TFAutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Build a pipeline for efficient, batched inference; use GPU if available
# Use CPU by default to maximize compatibility
device = -1
sentiment_pipe = transformers.pipeline(
    task="sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=device,
    truncation=True
)

client = RESTClient(api_key=POLYGON_API_KEY)

end_date = date.today()
start_date = date.today() - timedelta(days=14)

def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"https?://\S+", "", s)  # strip URLs
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

sentiment_count = []
for day in pd.date_range(start=start_date, end=end_date):
    day_str = day.strftime("%Y-%m-%d")
    daily_sentiment = {'date': day_str, 'positive': 0, 'negative': 0, 'neutral': 0}

    # Fetch news once per day
    daily_news = list(client.list_ticker_news("AAPL", published_utc=day_str, limit=100))

    # Collect text inputs (prefer title + description)
    texts = []
    for a in daily_news:
        title = getattr(a, 'title', None)
        desc = getattr(a, 'description', None) or getattr(a, 'summary', None)
        text = " ".join(t for t in [title, desc] if t)
        text = _clean_text(text)
        if text:
            texts.append(text)

    if not texts:
        sentiment_count.append(daily_sentiment)
        continue

    # Batched inference for efficiency
    batch_size = 32
    confidence_threshold = 0.6  # below this, treat as neutral to reduce noise
    for batch in _chunk(texts, batch_size):
        preds = sentiment_pipe(batch)
        for p in preds:
            # Normalize label names to lower-case keys 'positive'|'negative'|'neutral'
            lbl = p.get('label', '').lower()
            score = float(p.get('score', 0.0))
            if lbl not in daily_sentiment and 'positive' in lbl:
                lbl = 'positive'
            elif lbl not in daily_sentiment and 'negative' in lbl:
                lbl = 'negative'
            elif lbl not in daily_sentiment and 'neutral' in lbl:
                lbl = 'neutral'
            # Confidence gating: low-confidence pos/neg -> neutral
            if lbl in ('positive', 'negative') and score < confidence_threshold:
                lbl = 'neutral'
            if lbl in daily_sentiment:
                daily_sentiment[lbl] += 1

    sentiment_count.append(daily_sentiment)

df_sentiment = pd.DataFrame(sentiment_count)

if not df_sentiment.empty:
    df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
    df_sentiment.set_index('date', inplace=True)

    # Prepare data for plotting
    df_plot = df_sentiment[['positive', 'neutral', 'negative']].copy()
    df_plot = df_plot.sort_index()
    df_plot['net'] = df_plot['positive'] - df_plot['negative']

    # Plot: stacked bars for counts, line for net sentiment
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.bar(df_plot.index, df_plot['negative'], label='Negative', color='#d73027')
    ax1.bar(df_plot.index, df_plot['neutral'], bottom=df_plot['negative'], label='Neutral', color='#ffffbf')
    ax1.bar(df_plot.index, df_plot['positive'], bottom=(df_plot['negative'] + df_plot['neutral']), label='Positive', color='#1a9850')
    ax1.set_ylabel('Article Count')

    ax2 = ax1.twinx()
    ax2.plot(df_plot.index, df_plot['net'], color='#2c7fb8', marker='o', linewidth=2, label='Net Sentiment (pos - neg)')
    ax2.set_ylabel('Net Sentiment')

    # title_symbol = f"{Company_name} ({Symobl})" if Company_name else Symobl
    # ax1.set_title(f"News Sentiment by Day - {title_symbol}")
    fig.autofmt_xdate(rotation=45)

    # Build a combined legend
    bars_labels = [
        plt.Line2D([0], [0], color='#d73027', lw=10),
        plt.Line2D([0], [0], color='#ffffbf', lw=10),
        plt.Line2D([0], [0], color='#1a9850', lw=10),
        plt.Line2D([0], [0], color='#2c7fb8', lw=2)
    ]
    ax1.legend(bars_labels, ['Negative', 'Neutral', 'Positive', 'Net Sentiment'], loc='upper left')

    plt.tight_layout()
    plt.show()
else:
    print('No sentiment data available to plot.')


