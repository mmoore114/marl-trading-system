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
    """
    Ensure we have a 'date' column:
    - If index looks like dates, reset_index -> 'date'
    - Else if a column contains parseable datetimes, rename it to 'date'
    - Otherwise return as-is (caller will raise a clear error later)
    """
    # If index is not a simple RangeIndex, assume it might be dates
    if not isinstance(df.index, pd.RangeIndex):
        tmp = df.reset_index()
        candidates = ["date", "datetime", "timestamp", "index"]
        lower_cols = [c.lower() for c in tmp.columns]
        rename_map = {}
        for cand in candidates:
            if cand in lower_cols:
                rename_map[tmp.columns[lower_cols.index(cand)]] = "date"
                break
        if rename_map:
            tmp = tmp.rename(columns=rename_map)
        # Coerce if present
        if "date" in tmp.columns:
            tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
            if tmp["date"].notna().any():
                return tmp

    # Otherwise, scan columns for something datetime-like
    lower_map = {c.lower(): c for c in df.columns}
    # Known common variants
    for key in ["date", "datetime", "timestamp"]:
        if key in lower_map:
            col = lower_map[key]
            df = df.rename(columns={col: "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if df["date"].notna().any():
                return df

    # Heuristic: find any column that parses to datetime for most rows
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

    return df  # no change


def _normalize_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns: date, open, high, low, close, adj_close, volume, [symbol?]."""
    # Lowercase and unify names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Make/standardize date column
    df = _try_promote_date_column(df)

    # Standardize adjusted close
    if "adj_close" not in df.columns:
        if "adjclose" in df.columns:
            df = df.rename(columns={"adjclose": "adj_close"})
        elif "adjusted_close" in df.columns:
            df = df.rename(columns={"adjusted_close": "adj_close"})
        elif "close" in df.columns:
            df["adj_close"] = df["close"]  # fallback

    # Required columns
    required = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns after normalization: {missing}. "
            f"Detected columns: {list(df.columns)}. "
            f"If your CSV has the date as the first column without a header, "
            f"try saving it with the date as a named column."
        )
    # Coerce date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
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
        # Read with a robust fallback to recover the date from index if needed
        if fp.suffix.lower() == ".csv":
            df = pd.read_csv(fp)
            if "date" not in [c.lower() for c in df.columns]:
                # try reloading with first column as index (common export pattern)
                try:
                    df_alt = pd.read_csv(fp, index_col=0, parse_dates=True)
                    df_alt = df_alt.reset_index().rename(columns={"index": "date"})
                    # Use alt if it really produced a usable date
                    if "date" in df_alt.columns:
                        df = df_alt
                except Exception:
                    pass
        else:
            df = pd.read_parquet(fp)

        df = _normalize_raw(df).sort_values("date")

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


