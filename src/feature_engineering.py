import pandas as pd
import numpy as np
import yaml
from ta.momentum import rsi
from ta.trend import macd, macd_signal, ema_indicator
from ta.volatility import bollinger_hband, bollinger_lband, bollinger_pband, average_true_range
from ta.volume import on_balance_volume
from pathlib import Path

def calculate_factors(df, factor_config):
    """
    Calculates technical analysis factors based on a configuration dictionary.
    """
    if factor_config.get('RSI', {}).get('enabled', False):
        params = factor_config['RSI']
        df['RSI'] = rsi(df['Close'], window=params.get('window', 14))

    if factor_config.get('MACD', {}).get('enabled', False):
        params = factor_config['MACD']
        macd_df = macd(df['Close'], 
                       window_fast=params.get('fast', 12), 
                       window_slow=params.get('slow', 26))
        macd_signal_df = macd_signal(df['Close'], 
                                     window_fast=params.get('fast', 12), 
                                     window_slow=params.get('slow', 26), 
                                     window_sign=params.get('signal', 9))
        df['MACD'] = macd_df
        df['MACD_signal'] = macd_signal_df

    if factor_config.get('EMA', {}).get('enabled', False):
        params = factor_config['EMA']
        for window in params.get('windows', []):
            df[f'EMA_{window}'] = ema_indicator(df['Close'], window=window)

    # You can continue this pattern for BBands, ATR, OBV etc.

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        TICKERS = config['data_settings']['tickers']
        FACTOR_CONFIG = config['factor_settings']
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading configuration: {e}")
        exit()

    raw_data_dir = Path("data/raw")
    processed_data_dir = Path("data/processed")
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        print(f"Processing factors for {ticker}...")
        raw_data_path = raw_data_dir / f"{ticker}.csv"
        
        if not raw_data_path.exists():
            print(f"  Raw data for {ticker} not found, skipping.")
            continue

        # Define the correct column names, as the file header is malformed
        column_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        # Load the data, skipping junk rows and assigning the correct names
        raw_df = pd.read_csv(raw_data_path, 
                     header=None, 
                     names=column_names, 
                     index_col='Date', 
                     parse_dates=True, 
                     skiprows=3)

        
        cols_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols_to_convert:
            raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
        raw_df.dropna(inplace=True)

        processed_df = calculate_factors(raw_df.copy(), FACTOR_CONFIG)

        output_file = processed_data_dir / f"{ticker}_processed.csv"
        processed_df.to_csv(output_file)
        print(f"  Successfully processed and saved to {output_file}")

    print("\nFactor engineering complete for all tickers.")