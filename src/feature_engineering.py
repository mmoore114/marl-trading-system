import os
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from ta import add_all_ta_features  # Assuming ta-lib added

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

RAW_DIR = Path(CONFIG["storage"]["local_data_dir"]) / "raw"
PROC_DIR = Path(CONFIG["storage"]["local_data_dir"]) / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        if df.index.name and "date" in df.index.name.lower():
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "date"}, inplace=True)
        else:
            raise ValueError("No 'date' column found")

    needed = ["date", "open", "high", "low", "close", "volume"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    if "adj close" in df.columns:
        df.rename(columns={"adj close": "adj_close"}, inplace=True)
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume", "adj_close"]]

def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_all_ta_features(df, open="open", high="high", low="low", close="adj_close", volume="volume", fillna=True)

    # Custom from config
    factors = CONFIG.get("factors", {})
    df["ret_1d"] = df["adj_close"].pct_change()
    df["mom_21d"] = df["adj_close"] / df["adj_close"].shift(factors.get("momentum_21d", {}).get("window", 21)) - 1
    # ... (add others from config)

    return df.dropna().reset_index(drop=True)

def main():
    files = list(RAW_DIR.glob("*.csv"))
    logger.info(f"Raw files: {len(files)}")

    for fp in files:
        try:
            df = pd.read_csv(fp)
            logger.info(f"Processing {fp.name}: rows={len(df)}")
            df = _normalize_ohlcv(df)
            df = _add_features(df)
            out = PROC_DIR / (fp.stem + ".parquet")
            df.to_parquet(out, index=False)
            logger.info(f"Saved {out}")
        except Exception as e:
            logger.error(f"Error {fp.name}: {e}")

    logger.info("Feature engineering complete.")

if __name__ == "__main__":
    main()
