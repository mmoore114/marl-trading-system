# src/feature_engineering.py
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import sys

CFG_PATH = Path("config.yaml")


def log(msg: str):
    print(f"[features] {msg}", flush=True)


def load_config():
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def _read_raw(fp: Path) -> pd.DataFrame:
    """
    Robust reader:
      - CSV: try normal read; if no 'date' column, re-read forcing col 0 as datetime.
      - Parquet: read as-is.
    """
    if fp.suffix.lower() == ".csv":
        # First try: standard read
        df = pd.read_csv(fp)
        cols_lower = [c.lower() for c in df.columns]
        if "date" in cols_lower:
            df = df.rename(columns={df.columns[cols_lower.index("date")]: "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            # Second try: force first column to be date
            df = pd.read_csv(fp, parse_dates=[0])
            df = df.rename(columns={df.columns[0]: "date"})
        return df
    else:
        return pd.read_parquet(fp)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required columns exist and are numeric:
      date, open, high, low, close, adj_close, volume
    """
    # Normalize headers to lowercase/underscored
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    if "date" not in df.columns:
        raise KeyError(f"Missing 'date'. Columns present: {df.columns.tolist()}")



