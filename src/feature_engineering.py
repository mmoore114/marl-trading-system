import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from ta import add_all_ta_features
from ta.utils import dropna
from ta.volatility import average_true_range
from ta.momentum import rsi
from ta.trend import ema_indicator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Loading ---
# Use a relative path to ensure the script works from any directory
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

RAW_DIR = ROOT / CONFIG["storage"]["local_data_dir"] / "raw"
PROC_DIR = ROOT / CONFIG["storage"]["local_data_dir"] / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

def calculate_factors(df: pd.DataFrame, factors_config: dict) -> pd.DataFrame:
    """Calculates all factors defined in the config file."""
    df = df.copy()
    
    # Calculate returns first as they are often used in other factors
    df['returns'] = df['close'].pct_change()

    for factor_name, params in factors_config.items():
        window = params.get('window', 20) # Default window if not specified
        logger.debug(f"Calculating factor: {factor_name} with window {window}")
        
        if 'momentum' in factor_name:
            df[factor_name] = df['close'].pct_change(periods=window)
        elif 'rsi' in factor_name:
            df[factor_name] = rsi(close=df['close'], window=window)
        elif 'vol' in factor_name:
            df[factor_name] = df['returns'].rolling(window=window).std() * np.sqrt(252)
        elif 'atr' in factor_name:
            df[factor_name] = average_true_range(high=df['high'], low=df['low'], close=df['close'], window=window)
        elif 'ma_fast' in factor_name:
            df['ma_fast'] = ema_indicator(close=df['close'], window=params.get('window', 10))
        elif 'ma_slow' in factor_name:
            df['ma_slow'] = ema_indicator(close=df['close'], window=params.get('window', 50))

    # Add a MACD-like feature from the moving averages
    if 'ma_fast' in df.columns and 'ma_slow' in df.columns:
        df['ma_ratio'] = df['ma_fast'] / df['ma_slow']
            
    return df

def process_file(file_path: Path, factors_config: dict):
    """Loads a single raw data file, processes it, and saves the result."""
    try:
        logger.info(f"Processing {file_path.name}...")
        df = pd.read_csv(file_path, parse_dates=['date'])
        
        # Standardize column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]

        # Ensure essential columns exist
        required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(df.columns):
            logger.error(f"Skipping {file_path.name}. Missing required columns. Found: {df.columns}")
            return

        # --- FIX STARTS HERE ---
        # Convert OHLCV columns to numeric types, coercing errors
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # --- FIX ENDS HERE ---

        df.sort_values('date', inplace=True)
        
        # Calculate features based on the config
        df_features = calculate_factors(df, factors_config)
        
        # Drop rows with NaN values resulting from window calculations
        df_features.dropna(inplace=True)

        if df_features.empty:
            logger.warning(f"No data left for {file_path.name} after processing. Skipping.")
            return

        # Save to a more efficient format
        output_path = PROC_DIR / f"{file_path.stem}.parquet"
        df_features.to_parquet(output_path, index=False)
        logger.info(f"Successfully saved processed file to {output_path}")

    except Exception as e:
        logger.error(f"Failed to process {file_path.name}. Error: {e}")

        # Save to a more efficient format
        output_path = PROC_DIR / f"{file_path.stem}.parquet"
        df_features.to_parquet(output_path, index=False)
        logger.info(f"Successfully saved processed file to {output_path}")

    except Exception as e:
        logger.error(f"Failed to process {file_path.name}. Error: {e}")

def main():
    """Main function to run the feature engineering pipeline."""
    logger.info("--- Starting Feature Engineering ---")
    
    factors_config = CONFIG.get('factors', {})
    if not factors_config:
        logger.error("No factors found in config.yaml. Exiting.")
        return
        
    raw_files = list(RAW_DIR.glob("*.csv"))
    if not raw_files:
        logger.warning("No raw data files found to process.")
        return

    for file_path in raw_files:
        process_file(file_path, factors_config)
        
    logger.info("--- Feature Engineering Complete ---")

if __name__ == "__main__":
    main()
