# src/data_pipeline.py
"""
Fetches raw OHLCV for all tickers in config.yaml (universe.tickers) plus the
benchmark (universe.benchmark) using yfinance, and writes CSVs to data/raw.

Run:
    python -m src.data_pipeline
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FP = ROOT / "config.yaml"
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def read_config() -> dict:
    with open(CONFIG_FP, "r", encoding="utf-8") as f:
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
    return (
        str(data.get("start_date") or "2018-01-01"),
        str(data.get("end_date") or "2025-08-01"),
        str(data.get("price_frequency") or "1d"),
    )


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure simple string columns (no tuples/MultiIndex)."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    else:
        # Some yfinance versions still hand back tuples as column labels
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    return df


def download_one(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",  # ask yfinance not to create symbol-level groups
        threads=True,
    )
    if df is None or len(df) == 0:
        raise RuntimeError(f"yfinance returned empty data for {ticker}")

    df = _flatten_columns(df)

    # Normalize date column
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"Date": "date"})
    elif "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    else:
        # try to coerce the first column to date
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Build a case-insensitive column lookup
    cmap = {str(c).lower(): c for c in df.columns}

    def pick(name: str):
        low = name.lower()
        if low in cmap:
            return df[cmap[low]]
        # common alternates
        alts = {
            "open": ["open", "open*"],
            "high": ["high"],
            "low": ["low"],
            "close": ["close"],
            "volume": ["volume", "vol"],
        }[low]
        for a in alts:
            if a in cmap:
                return df[cmap[a]]
        return pd.NA

    out = pd.DataFrame(
        {
            "date": df["date"],
            "Open": pick("Open"),
            "High": pick("High"),
            "Low": pick("Low"),
            "Close": pick("Close"),
            "Volume": pick("Volume"),
        }
    )

    out = (
        out.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(subset=["date"])
        .reset_index(drop=True)
    )
    return out


def run():
    cfg = read_config()
    tickers, benchmark = universe_and_benchmark(cfg)
    start, end, freq = date_range(cfg)

    print(f"[pipeline] start={start} end={end} freq={freq}")
    print(f"[pipeline] tickers ({len(tickers)}): {tickers}")
    print(f"[pipeline] benchmark: {benchmark}")

    ok, fail = [], []
    for t in tickers:
        try:
            print(f"[pipeline] Downloading {t} ...", end="", flush=True)
            df = download_one(t, start, end, freq)
            fp = RAW_DIR / f"{t}.csv"
            df.to_csv(fp, index=False)
            print(f" saved -> {fp}")
            ok.append(t)
        except Exception as e:
            print(f" ERROR: {e}")
            fail.append((t, str(e)))

    print(f"[pipeline] Done. Success={len(ok)} Failures={len(fail)}")
    if fail:
        for t, msg in fail:
            print(f"  - {t}: {msg}")
        if benchmark and any(t == benchmark for t, _ in fail):
            print("[pipeline] WARNING: Benchmark failed to download; "
                  "backtests that expect it may fail until you re-run.")


if __name__ == "__main__":
    run()





