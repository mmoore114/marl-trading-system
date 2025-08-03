import yfinance as yf
import pandas as pd
from pathlib import Path

# Define the list of stock tickers we want to analyze
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# Define the date range for the historical data
START_DATE = "2010-01-01"
END_DATE = "2025-01-01"

# Define the path to save the raw data
RAW_DATA_PATH = Path("./data/raw")

def fetch_data():
    """
    Fetches historical daily stock data for a list of tickers and saves it
    to CSV files in the /data/raw directory.
    """
    # Ensure the output directory exists
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

    print("Starting data fetch...")

    for ticker in TICKERS:
        print(f"Fetching data for {ticker}...")
        try:
            # Download the data using yfinance
            data = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                progress=False
            )

            if data.empty:
                print(f"No data found for {ticker}, skipping.")
                continue

            # Define the output path for the CSV file
            output_path = RAW_DATA_PATH / f"{ticker}.csv"

            # Save the data to a CSV file
            data.to_csv(output_path)
            print(f"Successfully saved data for {ticker} to {output_path}")

        except Exception as e:
            print(f"Could not fetch or save data for {ticker}: {e}")

    print("Data fetch complete.")

if __name__ == "__main__":
    fetch_data()
    