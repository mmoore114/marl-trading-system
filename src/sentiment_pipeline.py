import os
import yaml
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import time

# --- Configuration and Initialization ---
def initialize_services():
    """Loads config, API keys, and initializes models."""
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        TICKERS = config['data_settings']['tickers']
    except FileNotFoundError:
        print("Error: config.yaml not found.")
        return None, None, None, None

    load_dotenv()
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        print("Error: NEWSAPI_KEY not found in .env file.")
        return None, None, None, None

    newsapi = NewsApiClient(api_key=api_key)
    
    # Load FinBERT model for sentiment analysis
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    return TICKERS, newsapi, tokenizer, model

def analyze_sentiment(text, tokenizer, model):
    """Analyzes the sentiment of a single piece of text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # Score is positive probability - negative probability
    return probs[0][0].item() - probs[0][1].item()

def fetch_and_analyze_sentiment():
    """
    Fetches news for each ticker for each day in its history,
    analyzes sentiment, and saves the daily scores.
    """
    TICKERS, newsapi, tokenizer, model = initialize_services()
    if not TICKERS:
        return

    sentiment_dir = Path("data/sentiment")
    sentiment_dir.mkdir(parents=True, exist_ok=True)
    raw_data_dir = Path("data/raw")

    for ticker in TICKERS:
        print(f"\nProcessing sentiment for {ticker}...")
        
        # Load raw data to get the date range
        raw_file = raw_data_dir / f"{ticker}.csv"
        if not raw_file.exists():
            print(f"Raw data for {ticker} not found. Skipping.")
            continue
        
       # Robustly read just the first column (the date) and convert it
        df_dates = pd.to_datetime(pd.read_csv(raw_file, header=None, usecols=[0], skiprows=3)[0])



        daily_sentiments = []

        # Use tqdm for a progress bar
        for date in tqdm(df_dates, desc=f"Fetching news for {ticker}"):
            date_str = date.strftime('%Y-%m-%d')
            try:
                articles = newsapi.get_everything(q=ticker,
                                                  from_param=date_str,
                                                  to=date_str,
                                                  language='en',
                                                  sort_by='relevancy',
                                                  page_size=10) # Get top 10 articles for the day
                
                daily_scores = []
                if articles['totalResults'] > 0:
                    for article in articles['articles']:
                        if article['title']:
                            score = analyze_sentiment(article['title'], tokenizer, model)
                            daily_scores.append(score)
                
                if daily_scores:
                    avg_score = sum(daily_scores) / len(daily_scores)
                    daily_sentiments.append({'Date': date, 'Sentiment': avg_score})
                
                # IMPORTANT: Pause to respect API rate limits
                time.sleep(1)

            except Exception as e:
                print(f"Error on {date_str} for {ticker}: {e}")
                time.sleep(1) # Wait after an error too

        if daily_sentiments:
            df_sentiment = pd.DataFrame(daily_sentiments)
            df_sentiment.set_index('Date', inplace=True)
            output_path = sentiment_dir / f"{ticker}_sentiment.csv"
            df_sentiment.to_csv(output_path)
            print(f"Sentiment data for {ticker} saved to {output_path}")

if __name__ == "__main__":
    fetch_and_analyze_sentiment()