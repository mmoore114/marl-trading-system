from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ta.momentum import rsi
from ta.trend import ema_indicator
from ta.volatility import average_true_range

# --- Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuration ---
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

RAW_DIR = ROOT / CONFIG["storage"]["local_data_dir"] / "raw"
PROC_DIR = ROOT / CONFIG["storage"]["local_data_dir"] / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


# --- Factor Calculation ---
def calculate_factors(df: pd.DataFrame, factors_config: dict) -> pd.DataFrame:
    """Calculates all factors defined in the config file."""
    df_out = df.copy()

    # CRITICAL: Use adjusted close for all return-based calculations
    adj_close = df_out["adj_close"]
    
    # Calculate returns first
    df_out["returns"] = adj_close.pct_change()

    for factor_name, params in factors_config.items():
        window = params.get("window", 20)
        logger.debug(f"Calculating factor: {factor_name} with window {window}")

        if "momentum" in factor_name:
            df_out[factor_name] = adj_close.pct_change(periods=window)
        elif "rsi" in factor_name:
            df_out[factor_name] = rsi(close=adj_close, window=window)
        elif "vol" in factor_name:
            # Volatility is calculated on returns, which are already adjusted
            df_out[factor_name] = (
                df_out["returns"].rolling(window=window).std() * np.sqrt(252)
            )
        elif "atr" in factor_name:
            # ATR is a volatility measure based on unadjusted prices
            df_out[factor_name] = average_true_range(
                high=df_out["high"], low=df_out["low"], close=df_out["close"], window=window
            )
        elif "ma_fast" in factor_name:
            df_out["ma_fast"] = ema_indicator(close=adj_close, window=window)
        elif "ma_slow" in factor_name:
            df_out["ma_slow"] = ema_indicator(close=adj_close, window=window)

    # Add a MACD-like feature from the moving averages
    if "ma_fast" in df_out.columns and "ma_slow" in df_out.columns:
        df_out["ma_ratio"] = df_out["ma_fast"] / df_out["ma_slow"]

    return df_out


# --- File Processing ---
def process_file(file_path: Path, factors_config: dict):
    """Loads a single raw data file, processes it, and saves the result."""
    try:
        logger.info(f"Processing {file_path.name}...")
        df = pd.read_csv(file_path, parse_dates=["date"])

        # Standardize column names to lowercase for consistency
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]

        # Ensure essential columns exist
        required_cols = {"date", "open", "high", "low", "close", "adj_close", "volume"}
        if not required_cols.issubset(df.columns):
            logger.error(
                f"Skipping {file_path.name}. Missing required columns. Found: {list(df.columns)}"
            )
            return

        # Convert OHLCV columns to numeric types, coercing errors
        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.sort_values("date", inplace=True)

        # Calculate features based on the config
        df_features = calculate_factors(df, factors_config)

        # Normalize feature columns for stable learning
        df_features = normalize_features(df_features)

        # Drop rows with NaN values resulting from window calculations
        df_features.dropna(inplace=True)

        if df_features.empty:
            logger.warning(
                f"No data left for {file_path.name} after processing. Skipping."
            )
            return

        # Save to a more efficient format (float32)
        float_cols = df_features.select_dtypes(include=[np.floating]).columns
        df_features[float_cols] = df_features[float_cols].astype(np.float32)
        output_path = PROC_DIR / f"{file_path.stem}.parquet"
        df_features.to_parquet(output_path, index=False)
        logger.info(f"Successfully saved processed file to {output_path}")

    except Exception as e:
        logger.error(f"Failed to process {file_path.name}. Error: {e}")


# --- Main Execution ---
def main():
    """Main function to run the feature engineering pipeline."""
    logger.info("--- Starting Feature Engineering ---")

    factors_config = CONFIG.get("factors", {})
    if not factors_config:
        logger.error("No 'factors' defined in config.yaml. Exiting.")
        return

    raw_files = list(RAW_DIR.glob("*.csv"))
    if not raw_files:
        logger.warning(f"No raw CSV data files found in {RAW_DIR}. Exiting.")
        return

    for file_path in raw_files:
        process_file(file_path, factors_config)

    logger.info("--- Feature Engineering Complete ---")


if __name__ == "__main__":
    main()


# --- Helpers ---
def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply simple normalization: RSI->0..1; others rolling z-score then tanh clip."""
    out = df.copy()
    base_cols = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    factor_cols = [c for c in out.columns if c not in base_cols]

    for col in factor_cols:
        series = out[col]
        if series.dtype.kind not in {"f", "i"}:
            continue
        if "rsi" in col:
            out[col] = (series / 100.0).clip(0.0, 1.0)
        else:
            mean = series.rolling(window=252, min_periods=20).mean()
            std = series.rolling(window=252, min_periods=20).std()
            z = (series - mean) / std
            out[col] = np.tanh(z.fillna(0.0))

    return out
