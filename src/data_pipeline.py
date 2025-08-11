import yfinance as yf
import pandas as pd
import yaml
from pathlib import Path

# --- 1. Load Configuration ---
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    TICKERS = config['data_settings']['tickers']
    START_DATE = config['data_settings']['start_date']
    END_DATE = config['data_settings']['end_date']

except FileNotFoundError:
    print("Error: config.yaml not found. Make sure it's in the project root.")
    exit()


# --- 2. Define Paths ---
RAW_DATA_PATH = Path("data/raw")


def fetch_data():
    """
    Fetches historical daily stock data for a list of tickers from the
    config file and saves it to CSV files in the /data/raw directory.
    """
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

    print("Starting data fetch based on config.yaml...")

    for ticker in TICKERS:
        print(f"Fetching data for {ticker}...")
        try:
            data = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                progress=False
            )

            if data.empty:
                print(f"No data found for {ticker}, skipping.")
                continue

            output_path = RAW_DATA_PATH / f"{ticker}.csv"
            data.to_csv(output_path)
            print(f"Successfully saved data for {ticker}")

        except Exception as e:
            print(f"Could not fetch or save data for {ticker}: {e}")

    print("\nData fetch complete for all tickers.")

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    fetch_data()
    