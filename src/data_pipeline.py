# src/data_pipeline.py
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import yaml
import yfinance as yf

# Optional .env support (safe even if no .env exists)
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

CFG_PATH = Path("config.yaml")


def load_config() -> Dict[str, Any]:
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we end up with columns:
    ['date','open','high','low','close','adj_close','volume', ...]
    and that 'date' is a column (not an index). Handles MultiIndex outputs.
    """
    # 1) Reset index so date is a column
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()

    # 2) Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in tup if x]).strip() for tup in df.columns]

    # 3) Lowercase, underscore
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # 4) Standardize date column
    if "date" not in df.columns:
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "date"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # 5) Standardize adjusted close
    # Common yfinance variants: 'adj_close', 'adj_close_(adjusted)', 'adjclose'
    if "adj_close" not in df.columns:
        if "adj_close_(adjusted)" in df.columns:
            df = df.rename(columns={"adj_close_(adjusted)": "adj_close"})
        elif "adjclose" in df.columns:
            df = df.rename(columns={"adjclose": "adj_close"})
        elif "adj_close" not in df.columns:
            # Fallback: if missing, duplicate 'close'
            if "close" in df.columns:
                df["adj_close"] = df["close"]

    # 6) Ensure required columns exist
    required = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing after normalization. Columns: {list(df.columns)}")

    return df


# -----------------------------
# PROTOTYPE: yfinance downloader
# -----------------------------
def download_yf(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    # Ticker().history is more predictable for single symbols than yf.download
    df = yf.Ticker(symbol).history(start=start, end=end, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()

    df = _normalize_columns(df)
    df["symbol"] = symbol
    cols = ["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"]
    return df[cols].sort_values("date")


# -----------------------------
# FUTURE: EODHD downloader stub
# -----------------------------
def _get_eodhd_key_from_env(env_name: str) -> Optional[str]:
    if load_dotenv:
        load_dotenv()
    return os.getenv(env_name)


def download_eodhd(symbol: str, start: str, end: str, api_key: str, rate_limit_per_sec: float = 4.0) -> pd.DataFrame:
    """
    Minimal EODHD fetch (daily EOD).
    When you upgrade to the All-in-One plan, we'll extend this to include fundamentals and sentiment.
    """
    import requests
    from urllib.parse import urlencode

    base = "https://eodhd.com/api"
    # Throttle between calls
    min_interval = 1.0 / max(rate_limit_per_sec, 1e-6)
    time.sleep(min_interval)

    params = {
        "period": "d",
        "from": start,
        "to": end,
        "api_token": api_key,
        "fmt": "json",
    }
    url = f"{base}/eod/{symbol}.US?{urlencode(params)}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    rows: List[Dict[str, Any]] = r.json() or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).rename(
        columns={
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adjusted_close": "adj_close",
            "volume": "volume",
        }
    )
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"])
    cols = ["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"]
    return df[cols].sort_values("date")

