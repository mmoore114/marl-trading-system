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
                f"Not enough dates in {self.mode} split ({len(self.dates)}). "
                f"Check your splits in config.yaml or regenerate features."
            )

        self.n_assets = len(self.symbols)
        self.allow_short = bool(self.cfg["actions"]["allow_short"])
        self.cash_node = bool(self.cfg["actions"]["cash_node"])

        # Observation & action spaces
        obs_dim = self.n_assets
        self.observation_space = spaces.Dict(
            {
                "momentum": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
                "meanrev": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            }
        )

        low = -1.0 if self.allow_short else 0.0
        act_dim = self.n_assets + (1 if self.cash_node else 0)
        self.action_space = spaces.Box(low=low, high=1.0, shape=(act_dim,), dtype=np.float32)

        # Pre-index by date for fast step()
        self.panel_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
        for d in self.dates:
            self.panel_by_date[d] = (
                self.df.loc[self.df["date"] == d]
                .set_index("symbol")
                .reindex(self.symbols)
            )

        # Reward knobs
        self.lambda_tc = float(self.cfg["reward"]["lambda_tc_bps"]) / 10000.0
        self.lambda_sigma = float(self.cfg["reward"]["lambda_sigma"])
        self.lambda_dd = float(self.cfg["reward"]["lambda_dd"])
        self.max_w_name = float(self.cfg["actions"]["max_weight_per_name"])

        # Bookkeeping
        self.t = 0
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.prev_w = np.zeros(self.action_space.shape[0], dtype=np.float32)
        if self.cash_node:
            self.prev_w[-1] = 1.0

    # -------- helpers
    def _obs(self, d: pd.Timestamp) -> Dict[str, np.ndarray]:
        panel = self.panel_by_date[d]
        return {
            "momentum": panel["spec_momentum"].to_numpy(np.float32),
            "meanrev": panel["spec_meanrev"].to_numpy(np.float32),
        }

    def _project(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=np.float32).copy()

        cap = self.max_w_name
        if self.cash_node:
            asset_w = w[:-1]
        else:
            asset_w = w

        # Per-name cap & shorting rules
        lo = -cap if self.allow_short else 0.0
        hi = cap
        asset_w = np.clip(asset_w, lo, hi)

        # Normalize + cash node
        if self.cash_node:
            if not self.allow_short:
                asset_w = np.clip(asset_w, 0.0, None)
            s = float(asset_w.sum())
            if s > 1.0:
                asset_w /= s
            cash = 1.0 - float(asset_w.sum())
            w = np.concatenate([asset_w, np.array([cash], dtype=np.float32)])
        else:
            s = float(asset_w.sum())
            if s <= 0:
                w = np.on

