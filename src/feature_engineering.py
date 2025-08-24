# src/feature_engineering.py
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

CFG_PATH = Path("config.yaml")


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
    - CSV: force first column to be 'date' and parse as datetime.
    - Parquet: read as-is.
    """
    if fp.suffix.lower() == ".csv":
        # Load once to know the header, then force first column as date
        df = pd.read_csv(fp)
        # If a 'date' column already exists (any case), normalize its name
        lower = [c.lower() for c in df.columns]
        if "date" in lower:
            df = df.rename(columns={df.columns[lower.index("date")]: "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            # Force first column to be date
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
        raise KeyError(f"Missing 'date' after read. Columns: {list(df.columns)}")

    # Adjusted close fallback
    if "adj_close" not in df.columns:
        if "adjclose" in df.columns:
            df = df.rename(columns={"adjclose": "adj_close"})
        elif "adjusted_close" in df.columns:
            df = df.rename(columns={"adjusted_close": "adj_close"})
        elif "close" in df.columns:
            df["adj_close"] = df["close"]

    # Some exports include an extra 'price' column—drop it if we have adj_close
    if "price" in df.columns and "adj_close" in df.columns:
        df = df.drop(columns=["price"])

    # Coerce numerics
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Final checks
    need = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing}. Got: {list(df.columns)}")

    # Clean
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "adj_close"]).sort_values("date")
    return df


def run():
    cfg = load_config()
    data_dir = Path(cfg["storage"]["local_data_dir"])
    raw_dir = data_dir / "raw"
    proc_dir = data_dir / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    mom_w = int(cfg["factors"]["momentum_21d"]["window"])
    rsi_w = int(cfg["factors"]["rsi_14"]["window"])
    vol_w = int(cfg["factors"]["vol_21d"]["window"])

    files = sorted(list(raw_dir.glob("*.*")))
    if not files:
        print(f"[features] No raw files found in {raw_dir}")
        return

    for fp in files:
        # Read + normalize
        df = _read_raw(fp)
        df = _normalize(df)

        # Daily returns (no implicit forward-fill)
        df["ret_1d"] = df["adj_close"].pct_change(fill_method=None)

        # Factors
        df[f"mom_{mom_w}"] = (df["adj_close"] / df["adj_close"].shift(mom_w)) - 1.0
        df[f"rsi_{rsi_w}"] = rsi(df["adj_close"], rsi_w)
        df[f"vol_{vol_w}"] = df["ret_1d"].rolling(vol_w).std() * np.sqrt(252)

        # Specialist views
        df["spec_momentum"] = df[f"mom_{mom_w}"].fillna(0.0)
        df["spec_meanrev"] = (-df["ret_1d"].rolling(5).mean()).fillna(0.0)

        # Save parquet
        out = proc_dir / (fp.stem + ".parquet")
        df.to_parquet(out, index=False)
        print(f"[features] Saved: {out}")



