# src/environment.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from gymnasium import spaces


def _load_cfg(cfg_path: str = "config.yaml") -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _load_processed_parquet(symbol: str, proc_dir: Path) -> pd.DataFrame:
    """
    Strict parquet loader for a single symbol.
    Expects columns: date, ret_1d, spec_momentum, spec_meanrev
    """
    fp = proc_dir / f"{symbol}.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp}. Run: python -m src.feature_engineering")
    df = pd.read_parquet(fp)
    # Normalize and sanity-check
    df.columns = [str(c).lower() for c in df.columns]
    need = {"date", "ret_1d", "spec_momentum", "spec_meanrev"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{fp} missing columns {missing}. Re-run feature_engineering.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df[["date", "ret_1d", "spec_momentum", "spec_meanrev"]].copy()


class MultiStrategyEnv(gym.Env):
    """
    Dict observations (per day, aligned across symbols):
      {
        "momentum": [N],   # spec_momentum per symbol (ordered by self.symbols)
        "meanrev":  [N],   # spec_meanrev per symbol
      }

    Action: weights over N assets (+ optional cash).
    Reward: portfolio return - turnover_cost - vol_penalty - drawdown_penalty.
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg_path: str = "config.yaml", mode: str = "train"):
        super().__init__()
        self.cfg = _load_cfg(cfg_path)
        self.mode = mode

        # Paths
        self.data_dir = Path(self.cfg["storage"]["local_data_dir"])
        self.proc_dir = self.data_dir / "processed"

        # Universe
        self.symbols: List[str] = list(self.cfg["universe"]["tickers"])
        if not self.symbols:
            raise ValueError("No tickers in config.universe.tickers")

        # Load all symbols (parquet only)
        frames: List[pd.DataFrame] = []
        for sym in self.symbols:
            df = _load_processed_parquet(sym, self.proc_dir)
            df["symbol"] = sym
            frames.append(df)
        panel = pd.concat(frames, ignore_index=True)
        panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)

        # Date splits
        sd = pd.Timestamp(self.cfg["data"]["start_date"])
        td = pd.Timestamp(self.cfg["splits"]["train_end"])
        vd = pd.Timestamp(self.cfg["splits"]["val_end"])
        ed = pd.Timestamp(self.cfg["data"]["end_date"])

        if self.mode == "train":
            mask = (panel["date"] >= sd) & (panel["date"] <= td)
        elif self.mode == "val":
            mask = (panel["date"] > td) & (panel["date"] <= vd)
        else:  # test
            mask = (panel["date"] > vd) & (panel["date"] <= ed)

        df_split = panel[mask].copy()

        # Keep only dates where ALL symbols are present (full panel)
        by_day = df_split.groupby("date")["symbol"].nunique()
        full_days = by_day[by_day == len(self.symbols)].index
        self.df = df_split[df_split["date"].isin(full_days)].copy()

        self.dates = sorted(self.df["date"].unique())
        if len(self.dates) < 5:
            raise RuntimeError(
                f"Not enough dates in {self.mode} split: only {len(self.dates)} found."
            )


