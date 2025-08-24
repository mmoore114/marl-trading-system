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
    fp = proc_dir / f"{symbol}.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp}. Run: python -m src.feature_engineering")
    df = pd.read_parquet(fp)
    df.columns = [str(c).lower() for c in df.columns]
    need = {"date", "ret_1d", "spec_momentum", "spec_meanrev"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{fp} missing columns {missing}. Re-run feature_engineering.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    # Sanitize
    for c in ["ret_1d", "spec_momentum", "spec_meanrev"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df[["date", "ret_1d", "spec_momentum", "spec_meanrev"]].copy()


class MultiStrategyEnv(gym.Env):
    """
    Dict observations:
      {
        "momentum": [N],
        "meanrev":  [N],
      }
    Action: weights over N assets (+ optional cash).
    Reward: portfolio return - costs - penalties.
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg_path: str = "config.yaml", mode: str = "train"):
        super().__init__()
        self.cfg = _load_cfg(cfg_path)
        self.mode = mode  # requested mode (may change after fallback)

        # ---- placeholder spaces so SB3 can seed immediately
        self.n_assets = 1
        self.allow_short = False
        self.cash_node = False
        self.observation_space = spaces.Dict(
            {
                "momentum": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "meanrev": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # bookkeeping placeholders
        self.t = 0
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.prev_w = np.zeros(self.action_space.shape[0], dtype=np.float32)

        # paths
        self.data_dir = Path(self.cfg["storage"]["local_data_dir"])
        self.proc_dir = self.data_dir / "processed"

        # universe
        self.symbols: List[str] = list(self.cfg["universe"]["tickers"])
        if not self.symbols:
            raise ValueError("No tickers in config.universe.tickers")

        # load panel
        frames: List[pd.DataFrame] = []
        for sym in self.symbols:
            df = _load_processed_parquet(sym, self.proc_dir)
            df["symbol"] = sym
            frames.append(df)
        panel = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)

        # date bounds
        sd = pd.Timestamp(self.cfg["data"]["start_date"])
        td = pd.Timestamp(self.cfg["splits"]["train_end"])
        vd = pd.Timestamp(self.cfg["splits"]["val_end"])
        ed = pd.Timestamp(self.cfg["data"]["end_date"])

        def _mask_for(m: str) -> pd.Series:
            if m == "train":
                return (panel["date"] >= sd) & (panel["date"] <= td)
            if m == "val":
                return (panel["date"] > td) & (panel["date"] <= vd)
            # test
            return (panel["date"] > vd) & (panel["date"] <= ed)

        # try requested mode, then fallbacks
        tried = []
        for candidate in [mode, "val", "train"]:
            if candidate in tried:
                continue
            tried.append(candidate)
            df_split = panel[_mask_for(candidate)].copy()
            # keep only full panel days
            by_day = df_split.groupby("date")["symbol"].nunique()
            full_days = by_day[by_day == len(self.symbols)].index
            df_candidate = df_split[df_split["date"].isin(full_days)].copy()
            dates = sorted(df_candidate["date"].unique())
            if len(dates) >= 5:
                # accept this split
                self.mode = candidate
                self.df = df_candidate
                self.dates = dates
                if candidate != mode:
                    print(f"[env] Requested mode '{mode}' had no data. Falling back to '{candidate}'.")
                break
        else:
            raise RuntimeError(
                "Not enough dates in any split (train/val/test). "
                "Adjust splits in config.yaml or regenerate features to extend the date range."
            )

        # set final spaces now that dimensions are known
        self.n_assets = len(self.symbols)
        self.allow_short = bool(self.cfg["actions"]["allow_short"])
        self.cash_node = bool(self.cfg["actions"]["cash_node"])

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

        # per-date panels, sanitized
        self.panel_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
        for d in self.dates:
            panel_d = (
                self.df.loc[self.df["date"] == d]
                .set_index("symbol")
                .reindex(self.symbols)
            )
            for c in ["ret_1d", "spec_momentum", "spec_meanrev"]:
                panel_d[c] = pd.to_numeric(panel_d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            self.panel_by_date[d] = panel_d

        # reward weights
        self.lambda_tc = float(self.cfg["reward"]["lambda_tc_bps"]) / 10000.0
        self.lambda_sigma = float(self.cfg["reward"]["lambda_sigma"])
        self.lambda_dd = float(self.cfg["reward"]["lambda_dd"])
        self.max_w_name = float(self.cfg["actions"]["max_weight_per_name"])

        # reset state
        self.t = 0
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.prev_w = np.zeros(self.action_space.shape[0], dtype=np.float32)
        if self.cash_node:
            self.prev_w[-1] = 1.0

    # helpers
    @staticmethod
    def _finite(arr: np.ndarray) -> np.ndarray:
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _obs(self, d: pd.Timestamp) -> Dict[str, np.ndarray]:
        panel = self.panel_by_date[d]
        mom = self._finite(panel["spec_momentum"].to_numpy(np.float32))
        mr = self._finite(panel["spec_meanrev"].to_numpy(np.float32))
        return {"momentum": mom, "meanrev": mr}

    def _project(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=np.float32).copy()
        cap = self.max_w_name
        asset_w = w[:-1] if self.cash_node else w
        lo = -cap if self.allow_short else 0.0
        hi = cap
        asset_w = np.clip(asset_w, lo, hi)
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
            w = np.ones_like(asset_w, dtype=np.float32) / len(asset_w) if s <= 0 else asset_w / s
        return w

    # gym api
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        self.t = 0
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.prev_w = np.zeros(self.action_space.shape[0], dtype=np.float32)
        if self.cash_node:
            self.prev_w[-1] = 1.0
        obs = self._obs(self.dates[self.t])
        info: Dict = {"mode": self.mode}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32)
        w = self._project(action)

        d = self.dates[self.t]
        panel = self.panel_by_date[d]
        rets = self._finite(panel["ret_1d"].to_numpy(np.float32))
        if self.cash_node:
            rets = np.concatenate([rets, np.array([0.0], dtype=np.float32)])

        turnover = float(np.abs(w - self.prev_w).sum())
        tc_cost = self.lambda_tc * turnover
        port_ret = float((w * rets).sum())
        vol_pen = self.lambda_sigma * float(np.std(rets))

        pv_next = self.portfolio_value * (1.0 + port_ret - tc_cost - vol_pen)
        if not np.isfinite(pv_next) or pv_next <= 0:
            pv_next = self.portfolio_value
        self.portfolio_value = pv_next
        self.peak_value = max(self.peak_value, self.portfolio_value)
        dd = (self.peak_value - self.portfolio_value) / max(self.peak_value, 1e-12)
        dd_pen = self.lambda_dd * float(dd)

        reward = float(port_ret - tc_cost - vol_pen - dd_pen)
        if not np.isfinite(reward):
            reward = 0.0

        self.prev_w = w
        self.t += 1
        terminated = self.t >= (len(self.dates) - 1)
        truncated = False
        next_obs = self._obs(self.dates[min(self.t, len(self.dates) - 1)])
        info = {"turnover": turnover, "tc_cost": tc_cost, "dd": float(dd), "port_ret": port_ret, "mode": self.mode}
        return next_obs, reward, terminated, truncated, info




