# src/feature_engineering.py
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

CFG_PATH = Path("config.yaml")


def load_config():
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)


def _try_promote_date_column(df: pd.DataFrame) -> pd.DataFrame:
    # If index looks like dates, move it to a column named 'date'
    if not isinstance(df.index, pd.RangeIndex):
        tmp = df.reset_index()
        lower = [c.lower() for c in tmp.columns]
        for cand in ("date", "datetime", "timestamp", "index"):
            if cand in lower:
                tmp = tmp.rename(columns={tmp.columns[lower.index(cand)]: "date"})
                tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
                return tmp
        return tmp  # at least expose the old index
    # Otherwise try to find a date-like column
    lower_map = {c.lower(): c for c in df.columns}
    for key in ("date", "datetime", "timestamp"):
        if key in lower_map:
            col = lower_map[key]
            df = df.rename(columns={col: "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df
    # Heuristic: try each column on a small sample
    for col in df.columns:
        sample = df[col].head(50)
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.8:
                df = df.rename(columns={col: "date"})
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                return df
        except Exception:
            pass
    return df


def _normalize_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns: date, open, high, low, close, adj_close, volume (as numeric)."""
    # Lowercase headers
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Promote/standardize date
    df = _try_promote_date_column(df)
    if "date" not in df.columns:
        raise KeyError(
            f"Missing 'date' column after normalization. Columns found: {list(df.columns)}. "
            f"If your CSV has the date as the first column without a header, save it with a header or index label."
        )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Standardize adjusted close
    if "adj_close" not in df.columns:
        if "adjclose" in df.columns:
            df = df.rename(columns={"adjclose": "adj_close"})
        elif "adjusted_close" in df.columns:
            df = df.rename(columns={"adjusted_close": "adj_close"})
        elif "close" in df.columns:
            df["adj_close"] = df["close"]

    # Coerce numeric columns (your CSV has numbers as strings)
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop obvious junk rows (no date or no prices)
    df = df[df["date"].notna()].copy()
    if "adj_close" in df.columns:
        df = df[df["adj_close"].notna()].copy()

    # Some exports have an extra 'price' column — safe to drop if present
    if "price" in df.columns and "adj_close" in df.columns:
        df = df.drop(columns=["price"])

    required = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing}. Got: {list(df.columns)}")

    return df


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolli_



