from __future__ import annotations

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def _load_and_prepare_data(tickers: list[str], proc_dir: Path) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """Loads all processed data, finds common dates, and converts to NumPy arrays for performance."""
    frames = {}
    for ticker in tickers:
        fp = proc_dir / f"{ticker}.parquet"
        if not fp.exists():
            raise FileNotFoundError(f"Missing processed file: {fp}. Please run feature_engineering.py")
        frames[ticker] = pd.read_parquet(fp).set_index("date")

    # Find common dates across all tickers
    common_dates = None
    for df in frames.values():
        if common_dates is None:
            common_dates = df.index
        else:
            common_dates = common_dates.intersection(df.index)
    
    # Concatenate into a single DataFrame, then split into features and prices
    full_df = pd.concat({ticker: df.loc[common_dates] for ticker, df in frames.items()}, axis=1)
    
    # Extract features and prices into NumPy arrays for fast slicing
    # Using 'adj_close' for price data to base portfolio value on
    price_df = full_df.loc[:, pd.IndexSlice[:, "adj_close"]]
    feature_df = full_df.drop(columns="adj_close", level=1) # Drop adj_close from features
    
    return common_dates, feature_df.to_numpy(), price_df.to_numpy()

# --- Environment Class ---
class SingleAgentTradingEnv(gym.Env):
    """A single-agent trading environment for portfolio optimization."""
    
    def __init__(self, env_config: dict | None = None):
        super().__init__()
        env_config = env_config or {}

        # --- Load Data ---
        self.tickers = CONFIG["universe"]["tickers"]
        self.num_assets = len(self.tickers)
        self.all_dates, self.all_features, self.all_prices = _load_and_prepare_data(self.tickers, PROC_DIR)

        # --- Environment Configuration ---
        self.mode = env_config.get("mode", "train")
        self.lookback_window = env_config.get("lookback_window", 30)
        self.initial_cash = 100_000.0

        # --- Get Config Parameters ---
        self.reward_config = CONFIG["reward"]
        self.tc_bps = self.reward_config["lambda_tc_bps"]
        
        # --- Data Splitting for Train/Val/Test ---
        self._split_data()

        # --- Define Spaces ---
        num_features_total = self.all_features.shape[1]
        market_obs_size = self.lookback_window * num_features_total
        portfolio_obs_size = self.num_assets + 1
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(market_obs_size + portfolio_obs_size,), 
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_assets + 1,), dtype=np.float32)

    def _split_data(self):
        train_end_date = pd.to_datetime(CONFIG["splits"]["train_end"])
        val_end_date = pd.to_datetime(CONFIG["splits"]["val_end"])
        
        train_mask = self.all_dates <= train_end_date
        val_mask = (self.all_dates > train_end_date) & (self.all_dates <= val_end_date)
        test_mask = self.all_dates > val_end_date
        
        if self.mode == "train":
            mask = train_mask
        elif self.mode == "validation":
            mask = val_mask
        else: # test
            mask = test_mask
            
        self.dates = self.all_dates[mask]
        self.features = self.all_features[mask]
        self.prices = self.all_prices[mask]
        
        self.max_steps = len(self.dates) - 2 
        if self.max_steps <= self.lookback_window:
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
        
        logger.info(f"Environment reset. Starting at step {self.current_step}. Total steps in episode: {self.max_steps - self.lookback_window}")
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        current_prices = self.prices[self.current_step]
        target_weights = np.exp(action) / np.sum(np.exp(action))
        
        prev_weights = self.portfolio_weights
        turnover = np.sum(np.abs(target_weights - prev_weights)) / 2
        transaction_cost = turnover * self.portfolio_value * (self.tc_bps / 10000)
        
        self.portfolio_value -= transaction_cost
        
        next_prices = self.prices[self.current_step + 1]
        asset_weights = prev_weights[:-1]
        
        price_changes = np.divide(next_prices, current_prices, out=np.ones_like(current_prices), where=current_prices!=0)
        portfolio_return = np.dot(asset_weights, price_changes - 1)
        
        self.portfolio_value *= (1 + portfolio_return)
        
        self.portfolio_weights = target_weights
        reward = self._calculate_reward()
        self.current_step += 1
        
        if self.portfolio_value < self.initial_cash * 0.5:
            self.terminated = True
            logger.info(f"--- EPISODE TERMINATED at step {self.current_step} due to portfolio value drop. ---")
            
        if self.current_step >= self.max_steps:
            self.truncated = True
            logger.info(f"--- EPISODE TRUNCATED at step {self.current_step}. Reached max steps. ---")
        
        obs = self._get_obs()
        info = self._get_info()
        
        self.prev_portfolio_value = self.portfolio_value
        
        return obs, reward, self.terminated, self.truncated, info

    def _get_obs(self) -> np.ndarray:
        market_features_slice = self.features[self.current_step - self.lookback_window : self.current_step].flatten()
        
        obs = np.concatenate([
            market_features_slice,
            self.portfolio_weights
        ]).astype(np.float32)
        return obs

    def _get_info(self) -> dict:
        date = self.dates[self.current_step] if self.current_step < len(self.dates) else self.dates[-1]
        return {
            "date": date,
            "portfolio_value": self.portfolio_value,
            "weights": self.portfolio_weights,
        }

    def _calculate_reward(self) -> float:
        ratio = np.clip(self.portfolio_value / self.prev_portfolio_value, 1e-10, None)
        step_return = np.log(ratio) if self.prev_portfolio_value > 0 else 0.0

        self.high_water_mark = max(self.high_water_mark, self.portfolio_value)
        drawdown = (self.portfolio_value - self.high_water_mark) / self.high_water_mark if self.high_water_mark > 0 else 0.0
        drawdown_penalty = self.reward_config["lambda_dd"] * abs(drawdown)

        risk_penalty = self.reward_config["lambda_sigma"] * np.std(self.portfolio_weights[:-1])

        reward = step_return - drawdown_penalty - risk_penalty
        return float(reward)