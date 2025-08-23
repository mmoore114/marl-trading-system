# src/data_pipeline.py
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import yaml

# Prototype source
import yfinance as yf

# Optional: EODHD when you're ready (kept local to avoid import errors if not used)
try:
    from dotenv import load_dotenv  # safe even if .env not present
except Exception:
    load_dotenv = None  # type: ignore


CFG_PATH = Path("config.yaml")


def load_config() -> Dict[str, Any]:
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# PROTOTYPE: yfinance downloader
# -----------------------------
def download_yf(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        interval=interval,
        progress=False,
    )
    if df.empty:
        return df

    # Standardize columns & include symbol/date for consistency
    df = df.rename(columns=str.lower).reset_index().rename(columns={"adj close": "adj_close"})
    df["symbol"] = symbol
    cols = ["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"]
    return df[cols]


# -----------------------------
# FUTURE: EODHD downloader stub
# -----------------------------
def _get_eodhd_key_from_env(env_name: str) -> Optional[str]:
    if load_dotenv:
        load_dotenv()
    return os.getenv(env_name)


def download_eodhd(symbol: str, start: str, end: str, api_key: str, rate_limit_per_sec: float = 4.0) -> pd.DataFrame:
    """
    Minimal eodhd fetch (daily EOD). Kept simple here; we'll expand when you activate the All-in-One plan.
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
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    cols = ["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"]
    return df[cols]


def run() -> None:
    cfg = load_config()

    data_dir = Path(cfg["storage"]["local_data_dir"])
    raw_dir = data_dir / "raw"
    ensure_dir(raw_dir)

    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"]
    src = cfg["data"].get("source", "yfinance").lower()
    tickers: List[str] = cfg["universe"]["tickers"]

    manifest: List[Dict[str, Any]] = []

    if src == "yfinance":
        for sym in tickers:
            df = download_yf(sym, start, end, "1d")
            if df.empty:
                print(f"[yf] No data for {sym}")
                continue
            out = raw_dir / f"{sym}.parquet"
            df.sort_values("date").to_parquet(out, index=False)
            manifest.append({"symbol": sym, "path": str(out), "n": int(df.shape[0])})
            print(f"[yf] Saved {sym}: {out}")
    elif src == "eodhd":
        api_key_env = cfg.get("eodhd", {}).get("api_key_env", "EODHD_API_KEY")
        rate = float(cfg.get("eodhd", {}).get("rate_limit_per_sec", 4.0))
        api_key = _get_eodhd_key_from_env(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"EODHD mode selected but no API key found in env '{api_key_env}'. "
                f"Create a .env and set {api_key_env}=<your_key> or export it in your shell."
            )
        for sym in tickers:
            df = download_eodhd(sym, start, end, api_key=api_key, rate_limit_per_sec=rate)
            if df.empty:
                print(f"[eodhd] No data for {sym}")
                continue
            out = raw_dir / f"{sym}.parquet"
            df.sort_values("date").to_parquet(out, index=False)
            manifest.append({"symbol": sym, "path": str(out), "n": int(df.shape[0])})
            print(f"[eodhd] Saved {sym}: {out}")
    else:
        raise ValueError(f"Unknown data.source '{src}'. Use 'yfinance' or 'eodhd'.")

    (data_dir / "raw_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved raw data for {len(manifest)} symbols to {raw_dir}")


if __name__ == "__main__":
    run()
