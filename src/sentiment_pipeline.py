import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import time
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
from joblib import Parallel, delayed
from src.data_pipeline import read_config, get_universe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_services():
    try:
        cfg = read_config()
        TICKERS = get_universe(cfg)
    except Exception as e:
        logger.error(f"Failed to load config or universe: {e}")
        raise

    load_dotenv()
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        logger.error("NEWSAPI_KEY missing.")
        raise ValueError("NEWSAPI_KEY missing")

    newsapi = NewsApiClient(api_key=api_key)
    
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    if torch.cuda.is_available():
        model = model.cuda()

    return TICKERS, newsapi, tokenizer, model

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_articles(newsapi, ticker, date_str):
    articles = newsapi.get_everything(q=ticker, from_param=date_str, to=date_str, language='en', sort_by='relevancy', page_size=10)
    if articles['status'] != 'ok':
        raise RuntimeError(f"NewsAPI error: {articles.get('message')}")
    return articles['articles']

def analyze_sentiment(batch_texts, tokenizer, model):
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
    return probs[:,0] - probs[:,1]  # Positive - negative

def process_ticker(ticker, raw_dir, sentiment_dir, newsapi, tokenizer, model):
    logger.info(f"Processing {ticker}...")
    raw_file = raw_dir / f"{ticker}.csv"
    if not raw_file.exists():
        logger.warning(f"Raw data missing for {ticker}. Skipping.")
        return

    df_dates = pd.read_csv(raw_file, usecols=["date"])
    df_dates["date"] = pd.to_datetime(df_dates["date"])
    dates = df_dates["date"].unique()

    daily_sentiments = []
    for date in tqdm(dates, desc=ticker):
        date_str = date.strftime('%Y-%m-%d')
        try:
            articles = fetch_articles(newsapi, ticker, date_str)
            texts = [a['title'] for a in articles if a['title']]
            if texts:
                scores = analyze_sentiment(texts, tokenizer, model)
                avg_score = np.mean(scores)
                daily_sentiments.append({'date': date, 'sentiment': avg_score})
            time.sleep(1 / 4)  # Rate limit
        except Exception as e:
            logger.warning(f"Error on {date_str} for {ticker}: {e}")

    if daily_sentiments:
        df_sentiment = pd.DataFrame(daily_sentiments).set_index('date')
        output_path = sentiment_dir / f"{ticker}_sentiment.csv"
        df_sentiment.to_csv(output_path)
        logger.info(f"Saved {output_path}")

def fetch_and_analyze_sentiment():
    TICKERS, newsapi, tokenizer, model = initialize_services()
    raw_dir = Path("data/raw")
    sentiment_dir = Path("data/sentiment")
    sentiment_dir.mkdir(parents=True, exist_ok=True)

    Parallel(n_jobs=4, backend="threading")(delayed(process_ticker)(t, raw_dir, sentiment_dir, newsapi, tokenizer, model) for t in TICKERS)

if __name__ == "__main__":
    fetch_and_analyze_sentiment()
