import logging
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from gymnasium import spaces

# --- Configuration ---
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

PROC_DIR = ROOT / CONFIG["storage"]["local_data_dir"] / "processed"
CACHE_DIR = ROOT / CONFIG["storage"]["local_data_dir"] / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def _cache_and_load_data(tickers: list[str], proc_dir: Path, cache_dir: Path) -> tuple[list[str], pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Checks for cached NumPy arrays. If not present, creates them by loading all data,
    filtering for sufficient history, creating a master date index, and forward-filling.
    """
    ticker_hash = str(hash(tuple(sorted(tickers))))
    dates_cache_fp = cache_dir / f"dates_{ticker_hash}.npy"
    features_cache_fp = cache_dir / f"features_{ticker_hash}.npy"
    prices_cache_fp = cache_dir / f"prices_{ticker_hash}.npy"
    final_tickers_fp = cache_dir / f"tickers_{ticker_hash}.txt"

    if not all([dates_cache_fp.exists(), features_cache_fp.exists(), prices_cache_fp.exists(), final_tickers_fp.exists()]):
        logger.info("Cache not found for this ticker set. Pre-processing and caching data arrays...")
        
        frames = {}
        min_data_points = 252 * 5  # Require at least 5 years of data
        logger.info(f"Screening {len(tickers)} tickers for >= {min_data_points} data points...")

        for ticker in tickers:
            fp = proc_dir / f"{ticker}.parquet"
            if fp.exists():
                df = pd.read_parquet(fp).set_index("date")
                if len(df) >= min_data_points:
                    frames[ticker] = df

        if not frames:
            raise ValueError("No tickers with sufficient data history found.")
        
        logger.info(f"Found {len(frames)} tickers with sufficient history.")

        # Create a master date index from the intersection of all valid frames
        common_dates = None
        for df in frames.values():
            common_dates = df.index if common_dates is None else common_dates.intersection(df.index)

        final_tickers = sorted(frames.keys())
        full_df = pd.concat({ticker: df.loc[common_dates] for ticker, df in frames.items()}, axis=1)
        full_df.ffill(inplace=True)
        full_df.bfill(inplace=True)
        
        price_df = full_df.loc[:, pd.IndexSlice[:, "adj_close"]]
        feature_df = full_df.drop(columns=["open", "high", "low", "close", "adj_close", "volume"], level=1)

        # Use float32 to reduce memory/IO footprint
        np.save(dates_cache_fp, common_dates.to_numpy())
        np.save(features_cache_fp, feature_df.to_numpy(dtype=np.float32))
        np.save(prices_cache_fp, price_df.to_numpy(dtype=np.float32))
        with open(final_tickers_fp, 'w') as f:
            f.write(','.join(final_tickers))
        logger.info(f"Successfully cached data for {len(final_tickers)} tickers to {cache_dir}")

    logger.info("Loading data from cache using memory mapping...")
    all_dates = pd.to_datetime(np.load(dates_cache_fp))
    all_features = np.load(features_cache_fp, mmap_mode='r')
    all_prices = np.load(prices_cache_fp, mmap_mode='r')
    with open(final_tickers_fp, 'r') as f:
        final_tickers = f.read().strip().split(',')
    
    return final_tickers, pd.DatetimeIndex(all_dates), all_features, all_prices

# --- Environment Class ---
class SingleAgentTradingEnv(gym.Env):
    def __init__(self, env_config: dict | None = None):
        super().__init__()
        env_config = env_config or {}

        initial_tickers = sorted([p.stem for p in PROC_DIR.glob("*.parquet")])
        
        self.tickers, self.all_dates, self.all_features, self.all_prices = _cache_and_load_data(initial_tickers, PROC_DIR, CACHE_DIR)
        self.num_assets = len(self.tickers)
        
        self.mode = env_config.get("mode", "train")
        self.lookback_window = env_config.get("lookback_window", 30)
        self.initial_cash = 100_000.0
        self.reward_config = CONFIG["reward"]
        self.tc_bps = self.reward_config["lambda_tc_bps"]
        self.turnover_limit = float(self.reward_config.get("turnover_limit_bps", 0)) / 10000.0 if self.reward_config.get("turnover_limit_bps") else None

        # Action constraints
        actions_cfg = CONFIG.get("actions", {}) or {}
        self.allow_short = bool(actions_cfg.get("allow_short", False))
        self.max_weight_per_name = float(actions_cfg.get("max_weight_per_name", 1.0))
        self.cash_node = bool(actions_cfg.get("cash_node", True))
        
        self._split_data()

        num_features_per_asset = self.all_features.shape[1] // self.num_assets
        market_obs_size = self.lookback_window * (num_features_per_asset * self.num_assets)
        portfolio_obs_size = self.num_assets + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(market_obs_size + portfolio_obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_assets + 1,), dtype=np.float32)
    
    def _split_data(self):
        train_end_date = pd.to_datetime(CONFIG["splits"]["train_end"])
        val_end_date = pd.to_datetime(CONFIG["splits"]["val_end"])
        
        train_mask = self.all_dates <= train_end_date
        val_mask = (self.all_dates > train_end_date) & (self.all_dates <= val_end_date)
        test_mask = self.all_dates > val_end_date
        
        mask = train_mask if self.mode == "train" else (val_mask if self.mode == "validation" else test_mask)
            
        self.dates = self.all_dates[mask]
        
        start_idx = self.all_dates.get_loc(self.dates[0])
        end_idx = self.all_dates.get_loc(self.dates[-1]) + 1
        
        self.features = self.all_features[start_idx:end_idx]
        self.prices = self.all_prices[start_idx:end_idx]
        
        self.max_steps = len(self.dates) - 2 
        if self.max_steps <= self.lookback_window:
            logger.error(f"Partition '{self.mode}' has only {len(self.dates)} dates, which is not enough for a lookback of {self.lookback_window}.")
            raise ValueError("Not enough data in the selected partition for the lookback window.")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        
        self.portfolio_weights = np.zeros(self.num_assets + 1, dtype=np.float32)
        self.portfolio_weights[-1] = 1.0
        
        self.portfolio_value = self.initial_cash
        self.prev_portfolio_value = self.initial_cash
        self.high_water_mark = self.initial_cash
        
        self.terminated = False
        self.truncated = False
        
        obs, info = self._get_obs(), self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        current_prices = self.prices[self.current_step]

        # Map raw action to target weights. For now, long-only via softmax.
        if self.allow_short:
            logger.warning("allow_short=True is not yet supported; falling back to long-only softmax mapping.")
        exps = np.exp(action - np.max(action))  # numerical stability
        target_weights = exps / np.sum(exps)

        prev_weights = self.portfolio_weights.copy()

        # Apply per-name cap and cash policy
        target_weights = self._apply_weight_constraints(target_weights)

        # Enforce turnover limit by scaling move toward target if necessary
        if self.turnover_limit is not None and self.turnover_limit > 0:
            turnover = np.sum(np.abs(target_weights - prev_weights)) / 2.0
            if turnover > self.turnover_limit and turnover > 0:
                k = self.turnover_limit / turnover
                target_weights = prev_weights + k * (target_weights - prev_weights)
                target_weights = self._apply_weight_constraints(target_weights)

        # Transaction cost on executed turnover
        turnover = np.sum(np.abs(target_weights - prev_weights)) / 2.0
        transaction_cost = turnover * self.portfolio_value * (self.tc_bps / 10000)
        self.portfolio_value -= transaction_cost

        # Price evolution on previous asset holdings (excluding cash)
        next_prices = self.prices[self.current_step + 1]
        asset_weights = prev_weights[:-1]
        price_changes = np.divide(next_prices, current_prices, out=np.ones_like(current_prices), where=current_prices!=0)
        portfolio_return = np.dot(asset_weights, price_changes - 1)
        self.portfolio_value *= (1 + portfolio_return)

        # Update to new portfolio weights
        self.portfolio_weights = target_weights.astype(np.float32)
        reward = self._calculate_reward()
        self.current_step += 1
        
        if self.portfolio_value < self.initial_cash * 0.5: self.terminated = True
        if self.current_step >= self.max_steps: self.truncated = True
        
        obs, info = self._get_obs(), self._get_info()
        self.prev_portfolio_value = self.portfolio_value
        return obs, reward, self.terminated, self.truncated, info

    def _get_obs(self) -> np.ndarray:
        market_features_slice = self.features[self.current_step - self.lookback_window : self.current_step].flatten()
        return np.concatenate([market_features_slice, self.portfolio_weights]).astype(np.float32)

    def _get_info(self) -> dict:
        date = self.dates[self.current_step] if self.current_step < len(self.dates) else self.dates[-1]
        return {"date": date, "portfolio_value": self.portfolio_value, "weights": self.portfolio_weights}

    def _calculate_reward(self) -> float:
        ratio = np.clip(self.portfolio_value / self.prev_portfolio_value, 1e-10, None)
        step_return = np.log(ratio) if self.prev_portfolio_value > 0 else 0.0
        self.high_water_mark = max(self.high_water_mark, self.portfolio_value)
        drawdown = (self.portfolio_value - self.high_water_mark) / self.high_water_mark if self.high_water_mark > 0 else 0.0
        drawdown_penalty = self.reward_config["lambda_dd"] * abs(drawdown)
        risk_penalty = self.reward_config["lambda_sigma"] * np.std(self.portfolio_weights[:-1])
        return float(step_return - drawdown_penalty - risk_penalty)

    # --- Internal helpers ---
    def _apply_weight_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Apply long-only, per-name cap, and cash-node policy. Ensures sum=1 and >=0."""
        w = np.clip(weights.astype(np.float64), 0.0, 1.0)
        asset_w = w[:-1].copy()
        cash_w = w[-1]

        # Per-name cap
        cap = min(max(self.max_weight_per_name, 0.0), 1.0)
        if cap < 1.0:
            asset_w = np.minimum(asset_w, cap)

        if self.cash_node:
            s = asset_w.sum()
            if s > 1.0:
                asset_w = asset_w / s
                cash_w = 0.0
            else:
                cash_w = 1.0 - s
            w_out = np.concatenate([asset_w, np.array([cash_w], dtype=np.float64)])
        else:
            # No dedicated cash: renormalize whole vector to sum 1
            w_out = np.concatenate([asset_w, np.array([cash_w], dtype=np.float64)])
            s = w_out.sum()
            if s > 0:
                w_out = w_out / s

        return w_out.astype(np.float32)
