import yfinance as yf
import pandas as pd
from pathlib import Path
import yaml
import time

# Load config
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    TICKERS = config['data_settings']['tickers']
except FileNotFoundError:
    print("Error: config.yaml not found.")
    exit()

processed_data_dir = Path("data/processed")
processed_data_dir.mkdir(parents=True, exist_ok=True)

def fetch_with_retry(ticker, retries=3, delay=5):
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start="2010-01-01", end="2025-08-09", auto_adjust=True)
            if not df.empty:
                return df
            else:
                print(f"Empty data for {ticker} on attempt {attempt+1}")
        except Exception as e:
            print(f"Error fetching {ticker} on attempt {attempt+1}: {e}")
        time.sleep(delay)
    raise ValueError(f"Failed to fetch data for {ticker} after {retries} attempts.")

print("Fetching historical data via yfinance with retries...")
for ticker in TICKERS:
    df = fetch_with_retry(ticker)
    df = df[['Close', 'Volume']]  # Add Volume
    file_path = processed_data_dir / f"{ticker}_processed.csv"
    df.to_csv(file_path, index_label='Date')
    print(f"Saved {ticker} data: {len(df)} rows.")
print("Data fetch complete.")

