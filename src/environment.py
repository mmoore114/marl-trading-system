# src/environment.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from gymnasium import spaces


class MultiStrategyEnv(gym.Env):
    """
    Dictionary observations:
      {
        "momentum":  [N],
        "meanrev":   [N],
      }

    Action: weights over N assets (+ optional cash).
    Reward: portfolio return - turnover_cost - vol_penalty - drawdown_penalty.
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg_path: str = "config.yaml", mode: str = "train"):
        super().__init__()
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.mode = mode

        self.data_dir = Path(self.cfg["storage"]["local_data_dir"])
        self.proc_dir = self.data_dir / "processed"

        # ---- Load processed data (prefer parquet; fallback to *_processed.csv)
        files = sorted(self.proc_dir.glob("*.parquet"))
        if not files:
            files = sorted(self.proc_dir.glob("*_processed.csv"))
            if not files:
                raise FileNotFoundError(
                    f"No processed files found in {self.proc_dir}. "
                    f"Run: python -m src.feature_engineering"
                )

        frames: List[pd.DataFrame] = []
        symbols: List[str] = []
        for fp in files:
            if fp.suffix.lower() == ".csv":
                sym = fp.stem.replace("_processed", "")
                df = pd.read_csv(fp)
            else:
                sym = fp.stem
                df = pd.read_parquet(fp)

            # Normalize basics
            df.columns = [str(c).lower() for c in df.columns]
            if "date" not in df.columns:
                raise KeyError(f"'date' column missing in {fp}")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            need_cols = {"ret_1d", "spec_momentum", "spec_meanrev"}
            missing = need_cols - set(df.columns)
            if missing:
                raise KeyError(f"Missing columns {missing} in {fp}. Re-run feature_engineering.")

            df = df[["date", "ret_1d", "spec_momentum", "spec_meanrev"]].copy()
            df["symbol"] = sym
            frames.append(df)
            symbols.append(sym)

        self.symbols = sorted(symbols)
        panel = pd.concat(frames, ignore_index=True)
        panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)

        # ---- Split by dates
        sd = pd.Timestamp(self.cfg["data"]["start_date"])
        td = pd.Timestamp(self.cfg["splits"]["train_end"])
        vd = pd.Timestamp(self.cfg["splits"]["val_end"])
        ed = pd.Timestamp(self.cfg["data"]["end_date"])

        if self.mode == "train":
            mask = (panel["date"] >= sd) & (panel["date"] <= td)
        elif self.mode == "val":
            mask = (panel["date"] > td) & (panel["date"] <= vd)
        else:
            mask = (panel["date"] > vd) & (panel["date"] <= ed)

        self.df = panel[mask].copy()

        # Ensure we have a full panel each day (align by intersection)
        by_day = self.df.groupby("date")["symbol"].nunique()
        full_days = by_day[by_day == len(self.symbols)].index
        self.df = self.df[self.df["date"].isin(full_days)].copy()

        self.dates = sorted(self.df["date"].unique())
        if len(self.dates) < 3:
            raise RuntimeError("Not enough dates in the selected split to run the environment.")

        self.n_assets = len(self.symbols)
        self.allow_short = bool(self.cfg["actions"]["allow_short"])
        self.cash_node = bool(self.cfg["actions"]["cash_node"])

        # Observation space
        obs_dim = self.n_assets
        self.observation_space = spaces.Dict(
            {
                "momentum": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
                "meanrev": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            }
        )

        # Action space (weights)
        low = -1.0 if self.allow_short else 0.0
        act_dim = self.n_assets + (1 if self.cash_node else 0)
        self.action_space = spaces.Box(low=low, high=1.0, shape=(act_dim,), dtype=np.float32)

        # Pre-index by date for fast lookup
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

    # ------------- helpers
    def _obs(self, d: pd.Timestamp) -> Dict[str, np.ndarray]:
        panel = self.panel_by_date[d]
        return {
            "momentum": panel["spec_momentum"].to_numpy(np.float32),
            "meanrev": panel["spec_meanrev"].to_numpy(np.float32),
        }

    def _project(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=np.float32).copy()

        # Clip per-name
        name_cap = self.max_w_name
        if self.cash_node:
            asset_w = np.clip(w[:-1], -name_cap if self.allow_short else 0.0, name_cap)
        else:
            asset_w = np.clip(w, -name_cap if self.allow_short else 0.0, name_cap)

        # Long-only projection (if required)
        if not self.allow_short:
            asset_w = np.clip(asset_w, 0.0, None)

        # Normalize + cash node
        if self.cash_node:
            total = float(asset_w.sum())
            if total > 1.0:
                asset_w /= total
            cash = 1.0 - float(asset_w.sum())
            w = np.concatenate([asset_w, np.array([cash], dtype=np.float32)])
        else:
            total = float(asset_w.sum())
            if total > 0:
                w = asset_w / total
            else:
                # fallback to equal weight
                w = np.ones_like(asset_w, dtype=np.float32) / len(asset_w)
        return w

    # ------------- gym api
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.prev_w = np.zeros(self.action_space.shape[0], dtype=np.float32)
        if self.cash_node:
            self.prev_w[-1] = 1.0
        return self._obs(self.dates[self.t]), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        w = self._project(action)

        d = self.dates[self.t]
        panel = self.panel_by_date[d]
        asset_rets = panel["ret_1d"].to_numpy(np.float32)

        if self.cash_node:
            asset_rets = np.concatenate([asset_rets, np.array([0.0], dtype=np.float32)])

        # Turnover & costs
        turnover = float(np.abs(w - self.prev_w).sum())
        tc_cost = self.lambda_tc * turnover

        # Portfolio return (pre-penalty)
        port_ret = float((w * asset_rets).sum())

        # Volatility proxy from cross-sectional dispersion
        vol_pen = self.lambda_sigma * float(np.std(asset_rets))

        # Update equity & drawdown
        self.portfolio_value *= (1.0 + port_ret - tc_cost - vol_pen)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        dd = (self.peak_value - self.portfolio_value) / self.peak_value
        dd_pen = self.lambda_dd * float(dd)

        reward = port_ret - tc_cost - vol_pen - dd_pen
        self.prev_w = w

        self.t += 1
        terminated = self.t >= (len(self.dates) - 1)
        truncated = False
        info = {"turnover": turnover, "tc_cost": tc_cost, "dd": float(dd), "port_ret": port_ret}
        return (self._obs(self.dates[min(self.t, len(self.dates) - 1)]), reward, terminated, truncated, info)
