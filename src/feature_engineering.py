import pandas as pd
import numpy as np # Import numpy
from ta.momentum import rsi
from ta.trend import macd, macd_signal, ema_indicator
from ta.volatility import bollinger_hband, bollinger_lband, bollinger_pband, average_true_range
from ta.volume import on_balance_volume
from pathlib import Path

def add_features(df):
    """
    Adds technical analysis features to the stock data DataFrame using the 'ta' library.
    """
    # --- Momentum Indicators ---
    df['RSI'] = rsi(df['Close'], window=14)
    
    # --- Trend Indicators ---
    df['MACD'] = macd(df['Close'], window_fast=12, window_slow=26)
    df['MACD_signal'] = macd_signal(df['Close'], window_fast=12, window_slow=26, window_sign=9)
    df['EMA_20'] = ema_indicator(df['Close'], window=20)
    df['EMA_50'] = ema_indicator(df['Close'], window=50)
    df['EMA_200'] = ema_indicator(df['Close'], window=200)

    # --- Volatility Indicators ---
    df['BB_upper'] = bollinger_hband(df['Close'], window=20, window_dev=2)
    df['BB_lower'] = bollinger_lband(df['Close'], window=20, window_dev=2)
    df['BB_percent'] = bollinger_pband(df['Close'], window=20, window_dev=2)
    df['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], window=14)

    # --- Volume Indicators ---
    df['OBV'] = on_balance_volume(df['Close'], df['Volume'])

    # IMPORTANT: Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with any NaN values
    df.dropna(inplace=True)

    return df

# --- Testing Block ---
if __name__ == "__main__":
    # Define paths
    raw_data_path = Path("data/raw/AAPL.csv")
    processed_data_path = Path("data/processed")

    # Ensure the output directory exists
    processed_data_path.mkdir(parents=True, exist_ok=True)

    # Load raw data
    raw_df = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
    
    # Convert columns to numeric, coercing errors
    cols_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_convert:
        raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
    raw_df.dropna(inplace=True)

    # Add features
    processed_df = add_features(raw_df.copy())

    # Save processed data
    output_file = processed_data_path / "AAPL_processed.csv"
    processed_df.to_csv(output_file)

    print("Feature engineering complete. Data cleaned of inf and NaN values.")
    print(f"Processed data saved to {output_file}")