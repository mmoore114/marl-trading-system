import os
import warnings
import pandas as pd
import numpy as np
import yaml

RAW_DIR = os.path.join("data", "raw")
PROC_DIR = os.path.join("data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

def read_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # Accept common variants
    mapping = {}
    cols = set(df.columns)
    def pick(*options):
        for o in options:
            if o in cols:
                return o
        return None

    open_c = pick("open")
    high_c = pick("high")
    low_c  = pick("low")
    close_c= pick("close","adj close","adj_close","close*")
    vol_c  = pick("volume","vol")

    rename = {}
    if open_c and open_c != "open": rename[open_c] = "open"
    if high_c and high_c != "high": rename[high_c] = "high"
    if low_c  and low_c  != "low":  rename[low_c]  = "low"
    if close_c and close_c != "close": rename[close_c] = "close"
    if vol_c and vol_c != "volume": rename[vol_c] = "volume"
    df = df.rename(columns=rename)

    needed = ["date","open","high","low","close","volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after normalization: {missing}")
    return df[needed]

def _add_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)
    # adj_close proxy = close (yfinance CSV already adjusted for splits/divs on 'Adj Close' normally;
    # here we standardize on 'close' as adjusted)
    df["adj_close"] = df["close"].astype(float)
    return df

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    fcfg = cfg.get("factors", {})
    # Returns
    df["ret_1d"] = df["adj_close"].pct_change()

    if "momentum_21d" in fcfg:
        w = int(fcfg["momentum_21d"].get("window", 21))
        df["mom_21d"] = df["adj_close"].pct_change(w)

    if "momentum_63d" in fcfg:
        w = int(fcfg["momentum_63d"].get("window", 63))
        df["mom_63d"] = df["adj_close"].pct_change(w)

    if "rsi_14" in fcfg:
        w = int(fcfg["rsi_14"].get("window", 14))
        df["rsi_14"] = _rsi(df["adj_close"], w)

    if "vol_21d" in fcfg:
        w = int(fcfg["vol_21d"].get("window", 21))
        df["vol_21d"] = df["ret_1d"].rolling(w).std()

    if "atr_14" in fcfg:
        w = int(fcfg["atr_14"].get("window", 14))
        df["atr_14"] = _atr(df, w)

    if "ma_fast_10" in fcfg:
        w = int(fcfg["ma_fast_10"].get("window", 10))
        df["ma_fast_10"] = df["adj_close"].rolling(w).mean()

    if "ma_slow_50" in fcfg:
        w = int(fcfg["ma_slow_50"].get("window", 50))
        df["ma_slow_50"] = df["adj_close"].rolling(w).mean()

    # Drop warm-up NaNs safely
    df = df.dropna().reset_index(drop=True)
    return df

def main():
    cfg = read_config()
    files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".csv")]
    print(f"[features] raw_dir={RAW_DIR} files={len(files)}")
    for f in sorted(files):
        try:
            print(f"[features] Reading {f}")
            fp = os.path.join(RAW_DIR, f)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                df = pd.read_csv(fp, parse_dates=[0])
            df = _lower_cols(df)
            df = _ensure_ohlcv(df)
            before_cols, before_rows = list(df.columns), len(df)
            df = _add_base(df)
            df = build_features(df, cfg)
            after_cols, after_rows = list(df.columns), len(df)
            print(f"[features] {f} cols(before): {before_cols} rows={before_rows}")
            print(f"[features] {f} cols(after): {after_cols} rows={after_rows}")
            out = os.path.join(PROC_DIR, f.replace(".csv", ".parquet"))
            df.to_parquet(out, index=False)
            print(f"[features] Saved {out}")
        except Exception as e:
            print(f"[features] ERROR {f}: {e}")
    print("[features] Done. Check data\\processed for parquet outputs.")

if __name__ == "__main__":
    main()





