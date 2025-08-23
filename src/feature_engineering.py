# src/feature_engineering.py
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

CFG_PATH = Path("config.yaml")


def load_config():
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)


def _normalize_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns: date, open, high, low, close, adj_close, volume, symbol."""
    # Expose index as a column (handles Date/DatetimeIndex)
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()

    # Lowercase names and replace spaces
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Standardize date column
    if "date" not in df.columns:
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "date"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Standardize adjusted close
    if "adj_close" not in df.columns:
        if "adjclose" in df.columns:
            df = df.rename(columns={"adjclose": "adj_close"})
        elif "adjusted_close" in df.columns:
            df = df.rename(columns={"adjusted_close": "adj_close"})
        elif "close" in df.columns:
            df["adj_close"] = df["close"]  # fallback

    required = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns after normalization: {missing}. Got: {list(df.columns)}")
    return df


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def run():
    cfg = load_config()
    data_dir = Path(cfg["storage"]["local_data_dir"])
    raw_dir = data_dir / "raw"
    proc_dir = data_dir / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    mom_w = cfg["factors"]["momentum_21d"]["window"]
    rsi_w = cfg["factors"]["rsi_14"]["window"]
    vol_w = cfg["factors"]["vol_21d"]["window"]

    for fp in raw_dir.glob("*.*"):  # handle .csv or .parquet
        # Read
        if fp.suffix.lower() == ".csv":
            df = pd.read_csv(fp)
        else:
            df = pd.read_parquet(fp)

        # Normalize columns
        df = _normalize_raw(df)

        # Sort by date
        df = df.sort_values("date")

        # Basic daily returns
        df["ret_1d"] = df["adj_close"].pct_change()

        # Factors
        df[f"mom_{mom_w}"] = (df["adj_close"] / df["adj_close"].shift(mom_w)) - 1.0
        df[f"rsi_{rsi_w}"] = rsi(df["adj_close"], rsi_w)
        df[f"vol_{vol_w}"] = df["ret_1d"].rolling(vol_w).std() * np.sqrt(252)

        # Specialist views
        df["spec_momentum"] = df[f"mom_{mom_w}"].fillna(0.0)
        df["spec_meanrev"] = (-df["ret_1d"].rolling(5).mean()).fillna(0.0)

        # Output as parquet (standard for the env)
        out = proc_dir / (fp.stem + ".parquet")
        df.to_parquet(out, index=False)
        print(f"[features] Saved {out}")

    print(f"Processed features saved to {proc_dir}")


if __name__ == "__main__":
    run()

