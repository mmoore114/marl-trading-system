import os
import gym
import numpy as np
import pandas as pd
import yaml
from gym import spaces
from datetime import datetime

PROC_DIR = os.path.join("data", "processed")

def read_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def _load_parquet(symbol: str, proc_dir: str) -> pd.DataFrame:
    fp = os.path.join(proc_dir, f"{symbol}.parquet")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Missing {fp}. Run: python -m src.feature_engineering")
    df = pd.read_parquet(fp)
    if "date" not in df.columns:
        raise ValueError(f"{symbol}.parquet missing 'date'")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def _split_dates(cfg):
    start = pd.to_datetime(cfg["data"]["start_date"])
    end   = pd.to_datetime(cfg["data"]["end_date"])
    tr_e  = pd.to_datetime(cfg["splits"]["train_end"])
    va_e  = pd.to_datetime(cfg["splits"]["val_end"])
    return start, tr_e, va_e, end

def _safe_zscore(train_vals: np.ndarray, full_vals: np.ndarray):
    mean = np.nanmean(train_vals, axis=0)
    std  = np.nanstd(train_vals, axis=0)
    std  = np.where(std < 1e-8, 1.0, std)
    def z(x): return (x - mean) / std
    return z(full_vals), mean, std

class MultiStrategyEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, mode: str = "train"):
        super().__init__()
        self.cfg = read_config()
        self.proc_dir = PROC_DIR
        uni = self.cfg["universe"]

        self.tickers = list(dict.fromkeys(uni["tickers"]))  # preserve order, dedupe
        self.benchmark = uni.get("benchmark", "SPY")

        # placeholder spaces for SB3 seeding
        self.n_assets = max(1, len(self.tickers))
        self.n_features = 8  # placeholder until we load real features
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.n_assets, self.n_features), dtype=np.float32)

        # Load data
        self.data = {sym: _load_parquet(sym, self.proc_dir) for sym in self.tickers}
        # Benchmark (ensure present even if not in tickers)
        try:
            self.bench_df = _load_parquet(self.benchmark, self.proc_dir)
        except Exception:
            # fallback: if benchmark missing, synth zero-ret series on market dates
            dates = self._align_dates(list(self.data.values()))
            self.bench_df = pd.DataFrame({"date": dates, "adj_close": 1.0})
            self.bench_df["ret_1d"] = 0.0

        # Align dates across all assets & benchmark
        all_frames = list(self.data.values()) + [self.bench_df]
        self.dates = self._align_dates(all_frames)

        # Reindex all on common dates, forward-fill and drop remaining NaNs
        for k in list(self.data.keys()):
            self.data[k] = self._reindex_one(self.data[k], self.dates)
        self.bench_df = self._reindex_one(self.bench_df, self.dates)

        # Compute factor matrix per asset (use all non-price engineered columns as factors)
        self.price_cols = ["open", "high", "low", "close", "volume", "adj_close"]
        # factors are everything except date + price cols
        sample_df = next(iter(self.data.values()))
        self.factor_cols = [c for c in sample_df.columns if c not in (["date"] + self.price_cols)]
        if not self.factor_cols:
            # fallback to minimal features: 1d return only
            self.factor_cols = ["ret_1d"]

        # Build 3D tensor: (time, assets, factors)
        self.X = self._build_panel(self.factor_cols)
        # Standardize using TRAIN stats only
        start, train_end, val_end, end = _split_dates(self.cfg)
        t_idx = (self.dates <= train_end)
        Xz, self.mu, self.sigma = _safe_zscore(self.X[t_idx].reshape((-1, len(self.factor_cols))),
                                               self.X.reshape((-1, len(self.factor_cols))))
        self.Xz = Xz.reshape(self.X.shape)

        # Daily returns per asset (use adj_close)
        self.R = self._build_returns()  # shape (time, assets)
        # Benchmark returns aligned
        self.bench_ret = self._series_ret(self.bench_df["adj_close"].values)

        # Splits
        self.split_idx = {
            "train": (self.dates <= train_end),
            "val":   (self.dates > train_end) & (self.dates <= val_end),
            "test":  (self.dates > val_end) & (self.dates <= end),
        }
        self.mode = self._resolve_mode(mode)
        self.idx = np.where(self.split_idx[self.mode])[0]
        self._assert_split_or_fallback()

        # Finalize spaces with real shapes
        self.n_assets = len(self.tickers)
        self.n_features = len(self.factor_cols)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.n_assets, self.n_features), dtype=np.float32)

        # state
        self.pos = np.zeros(self.n_assets, dtype=np.float64)
        self.cash = 1.0
        self.t = 1  # start at 1 to have t-1 returns
        self.start_t = int(self.idx.min())
        self.end_t = int(self.idx.max())
        self.t = self.start_t + 1

        # Reward penalties
        rw = self.cfg.get("reward", {})
        self.lambda_tc_bps = float(rw.get("lambda_tc_bps", 1.0))
        self.lambda_sigma  = float(rw.get("lambda_sigma", 0.10))
        self.lambda_dd     = float(rw.get("lambda_dd", 0.05))
        self.turnover_limit= float(rw.get("turnover_limit_bps", 1000.0))

        self._equity = 1.0
        self._peak = 1.0
        self._ret_hist = []

    # --------- helpers

    def _align_dates(self, frames):
        idx = None
        for df in frames:
            d = pd.to_datetime(df["date"])
            idx = d if idx is None else idx.intersection(d)
        return idx.sort_values().unique()

    def _reindex_one(self, df, dates):
        df = df.set_index("date").reindex(dates).ffill().dropna().reset_index().rename(columns={"index": "date"})
        return df

    def _build_panel(self, cols):
        T = len(self.dates)
        A = len(self.tickers)
        F = len(cols)
        X = np.zeros((T, A, F), dtype=np.float64)
        for a, sym in enumerate(self.tickers):
            df = self.data[sym]
            df = df[df["date"].isin(self.dates)]
            arr = df[cols].values.astype(np.float64)
            X[:len(arr), a, :] = arr
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def _series_ret(self, price: np.ndarray) -> np.ndarray:
        p = price.astype(np.float64)
        r = np.zeros_like(p)
        r[1:] = np.where(p[:-1] > 0, p[1:] / p[:-1] - 1.0, 0.0)
        return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_returns(self):
        T = len(self.dates)
        A = len(self.tickers)
        R = np.zeros((T, A), dtype=np.float64)
        for a, sym in enumerate(self.tickers):
            px = self.data[sym]["adj_close"].values
            R[:, a] = self._series_ret(px)
        return np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

    def _resolve_mode(self, mode: str) -> str:
        mode = (mode or "train").lower()
        return "train" if mode not in {"train","val","test"} else mode

    def _assert_split_or_fallback(self):
        # if requested split empty, fallback to val→train with notice
        if len(self.idx) == 0:
            order = ["test","val","train"]
            wanted = self.mode
            for m in order:
                if np.any(self.split_idx[m]):
                    self.mode = m
                    self.idx = np.where(self.split_idx[m])[0]
                    print(f"[env] Requested mode '{wanted}' had no data. Falling back to '{m}'.")
                    return
            raise RuntimeError("No data in any split.")

    # --------- Gym API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos[:] = 0.0
        self.cash = 1.0
        self._equity = 1.0
        self._peak = 1.0
        self._ret_hist = []
        self.t = self.start_t + 1
        return self._obs()

    def _obs(self):
        return self.Xz[self.t, :, :].astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        # Convert action to target weights in [-1,1]; enforce constraints
        if not np.all(np.isfinite(action)):
            action = np.nan_to_num(action, nan=0.0)
        # Long-only option
        if not self.cfg["actions"].get("allow_short", False):
            action = np.clip(action, 0.0, 1.0)
        # Cap per-name
        cap = float(self.cfg["actions"].get("max_weight_per_name", 0.10))
        action = np.clip(action, -cap, cap)
        # Normalize to <= 1 gross, leave remainder as cash if cash_node
        gross = np.sum(np.abs(action))
        cash_node = bool(self.cfg["actions"].get("cash_node", True))
        if gross > 1.0:
            action = action / max(gross, 1e-9)
            gross = 1.0
        cash_w = 1.0 - gross if cash_node else 0.0

        # Turnover cost (bps per 100% turnover)
        turnover = np.sum(np.abs(action - self.pos))
        tc = (self.lambda_tc_bps / 10000.0) * turnover

        # Realized portfolio return next step
        r = self.R[self.t, :]  # at t, use return from t-1→t
        port_ret = float(np.dot(action, r)) + cash_w * 0.0
        bench_r  = float(self.bench_ret[self.t])
        excess   = port_ret - bench_r

        # Track equity and stats
        self._equity *= (1.0 + port_ret)
        self._peak = max(self._peak, self._equity)
        dd = 1.0 - (self._equity / self._peak + 1e-12)

        self._ret_hist.append(port_ret)
        realized_vol = float(np.std(self._ret_hist[-63:]) if len(self._ret_hist) >= 2 else 0.0)

        # Reward (excess return minus penalties)
        reward = excess - tc - self.lambda_sigma * realized_vol - self.lambda_dd * dd

        # Advance time
        self.pos = action
        self.t += 1
        done = bool(self.t >= self.end_t)
        info = {
            "port_ret": port_ret,
            "bench_ret": bench_r,
            "excess_ret": excess,
            "turnover": turnover,
            "equity": self._equity,
            "drawdown": dd,
        }
        return self._obs(), float(reward), done, info





