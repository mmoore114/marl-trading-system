from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

# --- Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Constants ---
ROOT = Path(__file__).resolve().parents[1]
CONFIG_FP = ROOT / "config.yaml"
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# --- Configuration Loading ---
def read_config() -> dict:
    """Reads the project configuration from config.yaml."""
    try:
        with open(CONFIG_FP, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {CONFIG_FP}")
        raise


def get_universe(cfg: dict) -> List[str]:
    """Extracts and formats the list of tickers from the config."""
    uni = cfg.get("universe", {}) or {}
    tickers = [str(t).upper().strip() for t in (uni.get("tickers") or [])]
    benchmark = str((uni.get("benchmark") or "SPY")).upper().strip()
    if benchmark and benchmark not in tickers:
        tickers.append(benchmark)
    return list(sorted(set(tickers)))


def get_date_range(cfg: dict) -> Tuple[str, str]:
    """Extracts and validates the date range from the config."""
    data = cfg.get("data", {}) or {}
    start_date = str(data.get("start_date", "2018-01-01"))
    end_date = str(data.get("end_date", "2025-08-01"))
    if pd.to_datetime(start_date) > pd.to_datetime(end_date):
        raise ValueError("start_date cannot be after end_date in config.yaml")
    return start_date, end_date


# --- Data Downloading Function ---
@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def download_one(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """Downloads a single ticker from EODHD and adjusts prices."""
    logger.info(f"Downloading {ticker} from EODHD...")
    ticker_us = f"{ticker}.US"
    url = f"https://eodhd.com/api/eod/{ticker_us}?api_token={api_key}&fmt=json&from={start}&to={end}&period=d"
    
    resp = requests.get(url)
    resp.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
    data = resp.json()

    if not data:
        raise RuntimeError(f"EODHD returned empty data for {ticker}")

    df = pd.DataFrame(data)
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adjusted_close": "Adj Close",
            "volume": "Volume",
        }
    )
    df["date"] = pd.to_datetime(df["date"])

    # Calculate adjustment ratio and apply to OHL prices for consistency
    adj_ratio = df["Adj Close"] / df["Close"]
    df["Open"] = df["Open"] * adj_ratio
    df["High"] = df["High"] * adj_ratio
    df["Low"] = df["Low"] * adj_ratio
    
    return df[["date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]


# --- Main Execution ---
def run():
    """Main function to run the data pipeline."""
    cfg = read_config()
    tickers = get_universe(cfg)
    start_date, end_date = get_date_range(cfg)
    
    api_key_env = (cfg.get("eodhd", {}) or {}).get("api_key_env", "EODHD_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        logger.error(f"EODHD API key environment variable '{api_key_env}' is not set in your .env file.")
        return

    logger.info(f"Starting EODHD download: tickers={len(tickers)}, start={start_date}, end={end_date}")

    ok_tickers, failed_tickers = [], []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(download_one, t, start_date, end_date, api_key): t
            for t in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                df = future.result()
                output_path = RAW_DIR / f"{ticker}.csv"
                df.to_csv(output_path, index=False)
                logger.info(f"Successfully saved {ticker} to {output_path}")
                ok_tickers.append(ticker)
            except Exception as e:
                logger.error(f"Failed to download or process {ticker}: {e}")
                failed_tickers.append((ticker, str(e)))

    logger.info(f"--- Download Summary ---")
    logger.info(f"Success: {len(ok_tickers)}")
    logger.info(f"Failures: {len(failed_tickers)}")
    if failed_tickers:
        for ticker, reason in failed_tickers:
            logger.warning(f"  - {ticker}: {reason}")

if __name__ == "__main__":
    run()