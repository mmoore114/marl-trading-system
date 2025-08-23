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

try:
    from dotenv import load_dotenv
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
    Make sure we end up with columns:
    ['date','open','high','low','close','adj_close','volume', ...]
    and that 'date' is a column (not an index). Handles MultiIndex outputs.
    """
    # Reset index to expose date as a column (works whether index is Date/DatetimeIndex)
    if df.index.name is not None or not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()

    # Flatten potential MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in tup if x]).strip() for tup in df.columns]

    # Lowercase and replace spaces with underscores
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Unify date column name
    if "date" not in df.columns:
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "date"})

    # Unify adj close
    if "adj_close" not in df.columns:
        if "adj_close_(adjusted)" in df.columns:  # some odd providers
            df = df.rename(columns={"adj_close_(adjusted)": "adj_close"})
        elif "adjclose" in df.columns:
            df = df.rename(columns={"adjclose": "adj_close"})
        elif "adj_close" not in df.columns and "adj_close" not in df.columns and "adj_close" not in df.columns:
            # yfinance often gives 'adj_close' or 'adj_close' via mapping above; else try 'adj_close' from 'adj_close' with space
            if "adj_close" not in df.columns and "adj_close" not in df.columns and "adj_close" not in df.columns:
                if "adj_close" not in df.columns and "adj_close" not in df.columns and "adj_close" not in df.columns:
                    # as a final fallback, if only 'close' exists, duplicate it (not ideal but keeps pipeline running)
                    if "close" in df.columns and "adj_close" not in df.columns:
                        df["adj_close"] = df["close"]

    # Ensure required columns exist
    required = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing after normalization. Columns: {list(df.columns)}")

    return df


# -----------------------------
# PROTOTYPE: yfinance downloader
# -----------------------------
def download_yf(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    # Ticker().history tends to be less quirky for single symbols
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
    import requests
    from urllib.parse import urlencode

    base = "https://eodhd.com/api"
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

    df = pd.DataFrame(rows)
    df = df.rename(
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
    df["symbol"] = symb

