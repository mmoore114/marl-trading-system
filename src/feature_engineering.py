import pandas as pd
import numpy as np
import yaml
from ta.momentum import rsi
from ta.trend import macd, macd_signal, ema_indicator
from ta.volatility import bollinger_hband, bollinger_lband, bollinger_pband, average_true_range
from ta.volume import on_balance_volume
from pathlib import Path

def add_features(df):
    """
    Adds technical analysis features to a stock data DataFrame.
    """
    df['RSI'] = rsi(df['Close'], window=14)
    df['MACD'] = macd(df['Close'], window_fast=12, window_slow=26)
    df['MACD_signal'] = macd_signal(df['Close'], window_fast=12, window_slow=26, window_sign=9)
    df['EMA_20'] = ema_indicator(df['Close'], window=20)
    df['EMA_50'] = ema_indicator(df['Close'], window=50)
    df['EMA_200'] = ema_indicator(df['Close'], window=200)
    df['BB_upper'] = bollinger_hband(df['Close'], window=20, window_dev=2)
    df['BB_lower'] = bollinger_lband(df['Close'], window=20, window_dev=2)
    df['BB_percent'] = bollinger_pband(df['Close'], window=20, window_dev=2)
    df['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['OBV'] = on_balance_volume(df['Close'], df['Volume'])
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

# --- Main Execution Block ---
if __name__ == "__main__":
    # Load configuration
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        TICKERS = config['data_settings']['tickers']
    except FileNotFoundError:
        print("Error: config.yaml not found.")
        exit()

    # Define paths
    raw_data_dir = Path("data/raw")
    processed_data_dir = Path("data/processed")
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Process each ticker from the config file
    for ticker in TICKERS:
        print(f"Processing features for {ticker}...")
        
        raw_data_path = raw_data_dir / f"{ticker}.csv"
        
        if not raw_data_path.exists():
            print(f"  Raw data for {ticker} not found, skipping.")
            continue

        raw_df = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
        
        cols_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols_to_convert:
            raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
        raw_df.dropna(inplace=True)

        processed_df = add_features(raw_df.copy())

        output_file = processed_data_dir / f"{ticker}_processed.csv"
        processed_df.to_csv(output_file)
        
        print(f"  Successfully processed and saved to {output_file}")

    print("\nFeature engineering complete for all tickers.")