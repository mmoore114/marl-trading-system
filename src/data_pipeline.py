from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import yaml
import yfinance as yf
import requests
import logging
import os
from tenacity import retry, stop_after_attempt, wait_fixed
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FP = ROOT / "config.yaml"
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def read_config() -> dict:
    with open(CONFIG_FP, "r") as f:
        return yaml.safe_load(f) or {}

def universe_and_benchmark(cfg: dict) -> Tuple[List[str], str]:
    uni = cfg.get("universe", {}) or {}
    tickers = [str(t).upper().strip() for t in (uni.get("tickers") or [])]
    benchmark = str((uni.get("benchmark") or "SPY")).upper().strip()
    if benchmark and benchmark not in tickers:
        tickers.append(benchmark)
    return tickers, benchmark

def date_range(cfg: dict) -> Tuple[str, str, str]:
    data = cfg.get("data", {}) or {}
    start = str(data.get("start_date") or "2018-01-01")
    end = str(data.get("end_date") or "2025-08-01")
    freq = str(data.get("price_frequency") or "1d")
    if pd.to_datetime(start) > pd.to_datetime(end):
        raise ValueError("start_date > end_date")
    return start, end, freq

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def download_one(ticker: str, start: str, end: str, interval: str, source: str = "yfinance") -> pd.DataFrame:
    if source == "yfinance":
        df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        if df.empty:
            raise RuntimeError(f"Empty data for {ticker}")
        df = df.reset_index().rename(columns={"Date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "Open", "High", "Low", "Close", "Volume"]]
    elif source == "eodhd":
        api_key = os.getenv("EODHD_API_KEY")
        if not api_key:
            raise ValueError("EODHD_API_KEY missing")
        url = f"https://eodhd.com/api/eod/{ticker}?api_token={api_key}&fmt=json&from={start}&to={end}"
        resp = requests.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"EODHD fail: {resp.text}")
        data = resp.json()
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "open", "high", "low", "close", "volume"]]
    else:
        raise ValueError(f"Unknown source: {source}")

def run():
    cfg = read_config()
    tickers, benchmark = universe_and_benchmark(cfg)
    start, end, freq = date_range(cfg)
    source = cfg["data"].get("source", "yfinance")

    logger.info(f"Fetching: start={start} end={end} freq={freq} source={source}")
    logger.info(f"Tickers ({len(tickers)}): {tickers}")

    ok, fail = [], []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_one, t, start, end, freq, source): t for t in tickers}
        for future in as_completed(futures):
            t = futures[future]
            try:
                df = future.result()
                fp = RAW_DIR / f"{t}.csv"
                df.to_csv(fp, index=False)
                logger.info(f"Saved {fp}")
                ok.append(t)
            except Exception as e:
                logger.error(f"Fail {t}: {e}")
                fail.append((t, str(e)))

    logger.info(f"Done. Success={len(ok)} Failures={len(fail)}")
    if fail:
        for t, msg in fail:
            logger.warning(f"  - {t}: {msg}")
        if benchmark in [t for t, _ in fail]:
            logger.warning("Benchmark failed—backtests may fail.")

if __name__ == "__main__":
    run()





