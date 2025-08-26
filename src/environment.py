import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging

# --- FIX: Robust Path Loading ---
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

PROC_DIR = ROOT / CONFIG["storage"]["local_data_dir"] / "processed"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _load_processed_parquet(sym: str, proc_dir: Path) -> pd.DataFrame:
    fp = proc_dir / f"{sym}.parquet"
    if not fp.exists():
        logger.error(f"Missing {fp}. Run: python src.feature_engineering")
        raise FileNotFoundError(f"Missing {fp}")
    try:
        df = pd.read_parquet(fp)
        df["date"] = pd.to_datetime(df["date"])
        # Use forward-fill for missing values, then backfill for any at the start
        df = df.ffill().bfill()
        return df
    except Exception as e:
        logger.error(f"Error loading {fp}: {e}")
        raise

class TradingEnv(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        self.mode = env_config.get("mode", "train")
        self.proc_dir = PROC_DIR
        self.sym_list = CONFIG["universe"]["tickers"]
        self.frames = {s: _load_processed_parquet(s, self.proc_dir) for s in self.sym_list}
        self.dates = self._get_split_dates()
        
        if len(self.dates) < 2: # Need at least 2 steps to calculate returns
            raise RuntimeError(f"Not enough dates available for mode={self.mode}. Check data and config splits.")

        self.current_step = 0
        self.max_steps = len(self.dates) - 1
        
        # --- State Tracking ---
        self.cash = 100_000.0
        self.portfolio_shares = np.zeros(len(self.sym_list), dtype=np.float32)
        self.portfolio_value = self.cash
        self.prev_portfolio_value = self.cash
        self.high_water_mark = self.cash

        # --- Agents ---
        self.specialists = CONFIG["specialists"]["types"]
        self.agents = self.specialists + ['portfolio_manager']
        self._agent_ids = set(self.agents)

        # --- Obs/Action Spaces (FIXED) ---
        n_features = len(self.frames[self.sym_list[0]].columns) - 1  # Exclude date
        obs_market_dim = len(self.sym_list) * n_features
        obs_recs_dim = len(self.specialists) * len(self.sym_list)
        obs_portfolio_dim = len(self.sym_list) + 1 # Stocks + cash
        
        shared_obs_dim = obs_market_dim + obs_recs_dim
        shared_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(shared_obs_dim,), dtype=np.float32)

        pm_obs_dim = shared_obs_dim + obs_portfolio_dim
        pm_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(pm_obs_dim,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            a: shared_obs_space for a in self.specialists
        })
        self.observation_space['portfolio_manager'] = pm_obs_space
        
        self.action_space = spaces.Dict({
            a: spaces.Box(low=-1.0, high=1.0, shape=(len(self.sym_list),), dtype=np.float32) for a in self.specialists
        })
        self.action_space['portfolio_manager'] = spaces.Box(low=0.0, high=1.0, shape=(len(self.sym_list),), dtype=np.float32)

    def _get_split_dates(self):
        # Align all dates first
        common_dates = None
        for df in self.frames.values():
            dates = pd.DatetimeIndex(df["date"])
            common_dates = dates if common_dates is None else common_dates.intersection(dates)
        
        # Now apply train/val/test splits
        train_end = pd.to_datetime(CONFIG["splits"]["train_end"])
        val_end = pd.to_datetime(CONFIG["splits"]["val_end"])

        if self.mode == "train":
            return common_dates[common_dates <= train_end]
        elif self.mode == "validation":
            return common_dates[(common_dates > train_end) & (common_dates <= val_end)]
        else: # test
            return common_dates[common_dates > val_end]

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.cash = 100_000.0
        self.portfolio_shares = np.zeros(len(self.sym_list), dtype=np.float32)
        self.portfolio_value = self.cash
        self.prev_portfolio_value = self.cash
        self.high_water_mark = self.cash
        
        obs = self._get_obs()
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        # --- 1. Get Recommendations and Portfolio Allocation ---
        recs = {a: actions[a] for a in self.specialists}
        target_weights = actions['portfolio_manager']
        target_weights /= np.sum(target_weights) if np.sum(target_weights) > 0 else 1.0 # Normalize

        # --- 2. Execute Trades ---
        prices_t0 = self._get_prices()
        current_portfolio_value = self.cash + np.dot(self.portfolio_shares, prices_t0)
        
        target_shares = (current_portfolio_value * target_weights) / prices_t0
        trade_shares = target_shares - self.portfolio_shares
        
        # Apply transaction costs
        trade_cost = np.sum(np.abs(trade_shares) * prices_t0) * (CONFIG["reward"]["lambda_tc_bps"] / 10000)
        
        self.cash -= trade_cost
        self.portfolio_shares = target_shares
        
        # --- 3. Update State for Next Step ---
        self.current_step += 1
        
        if self.current_step >= len(self.dates):
            # This is a terminal state, no next prices to get
            terminated = {a: True for a in self.agents}
            terminated["__all__"] = True
            truncated = {"__all__": False}
            obs = self._get_obs(recs)
            rewards = {a: 0.0 for a in self.agents}
            infos = {a: {} for a in self.agents}
            return obs, rewards, terminated, truncated, infos
        
        prices_t1 = self._get_prices() # Prices at the *end* of the step
        
        self.portfolio_value = self.cash + np.dot(self.portfolio_shares, prices_t1)
        
        # --- 4. Calculate Reward ---
        reward = self._calculate_reward()
        rewards = {a: reward for a in self.agents}
        
        # --- 5. Check for Termination ---
        done = self.current_step >= self.max_steps
        terminated = {a: done for a in self.agents}
        terminated["__all__"] = done
        truncated = {"__all__": False}
        
        # --- 6. Prepare Next Observation ---
        obs = self._get_obs(recs)
        infos = {a: {} for a in self.agents}
        
        # Update for next reward calculation
        self.prev_portfolio_value = self.portfolio_value
        
        return obs, rewards, terminated, truncated, infos

    def _get_prices(self):
        date = self.dates[self.current_step]
        prices = np.array([self.frames[s].set_index('date').loc[date]['close'] for s in self.sym_list], dtype=np.float32)
        return prices

    def _get_obs(self, recs=None):
        if recs is None:
            recs = {a: np.zeros(len(self.sym_list), dtype=np.float32) for a in self.specialists}
            
        date = self.dates[self.current_step]
        features = np.concatenate([
            self.frames[s].set_index('date').loc[date].drop(columns=['close']).values for s in self.sym_list]
        ).astype(np.float32)
        
        rec_array = np.concatenate([recs[a] for a in self.specialists]).astype(np.float32)
        shared_obs = np.concatenate([features, rec_array]).astype(np.float32)
        
        obs = {a: shared_obs for a in self.specialists}
        
        portfolio_state = np.append(self.portfolio_shares, self.cash).astype(np.float32)
        obs['portfolio_manager'] = np.concatenate([shared_obs, portfolio_state]).astype(np.float32)
        
        return obs

    def _calculate_reward(self):
        step_return = (self.portfolio_value / self.prev_portfolio_value) - 1 if self.prev_portfolio_value != 0 else 0.0

        self.high_water_mark = max(self.high_water_mark, self.portfolio_value)
        drawdown = (self.portfolio_value - self.high_water_mark) / self.high_water_mark
        
        prices = self._get_prices()
        weights = (self.portfolio_shares * prices) / self.portfolio_value
        risk = np.std(weights)
        
        reward = step_return - CONFIG["reward"]["lambda_sigma"] * risk - CONFIG["reward"]["lambda_dd"] * abs(drawdown)
        return reward