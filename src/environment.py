import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging

CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

PROC_DIR = Path(CONFIG["storage"]["local_data_dir"]) / "processed"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _load_processed_parquet(sym: str, proc_dir: Path) -> pd.DataFrame:
    fp = proc_dir / f"{sym}.parquet"
    if not fp.exists():
        logger.error(f"Missing {fp}. Run: python -m src.feature_engineering")
        raise FileNotFoundError(f"Missing {fp}")
    try:
        df = pd.read_parquet(fp)
        if "date" not in df.columns:
            raise ValueError(f"{sym} parquet missing 'date' column")
        df["date"] = pd.to_datetime(df["date"])
        df = df.fillna(method='ffill').fillna(0)
        return df
    except Exception as e:
        logger.error(f"Error loading {fp}: {e}")
        raise

class TradingEnv(MultiAgentEnv):
    def __init__(self, mode="train"):
        super().__init__()
        self.mode = mode
        self.proc_dir = PROC_DIR
        self.sym_list = CONFIG["universe"]["tickers"]
        self.frames = {s: _load_processed_parquet(s, self.proc_dir) for s in self.sym_list}
        self.dates = self._align_dates(self.frames)
        if len(self.dates) == 0:
            raise RuntimeError(f"No dates available for mode={mode}")

        self.current_step = 0
        self.max_steps = len(self.dates) - 1
        self.portfolio = np.zeros(len(self.sym_list))
        self.cash = 100000

        # Agents
        self.agents = CONFIG["specialists"]["types"] + ['portfolio']
        self.specialists = CONFIG["specialists"]["types"]

        # Obs space
        n_features = len(self.frames[self.sym_list[0]].columns) - 1
        obs_dim = len(self.sym_list) * n_features + len(self.specialists) * len(self.sym_list)
        self.observation_space = {a: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)) for a in self.agents}

        # Action space
        self.action_space = {a: spaces.Box(low=-1, high=1, shape=(len(self.sym_list),)) for a in self.specialists}
        self.action_space['portfolio'] = spaces.Box(low=0, high=1, shape=(len(self.sym_list) + 1,))

    def _align_dates(self, frames: dict) -> pd.DatetimeIndex:
        idx = None
        for df in frames.values():
            d = pd.DatetimeIndex(df["date"])
            idx = d if idx is None else idx.intersection(d)
        return idx

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.portfolio = np.zeros(len(self.sym_list))
        self.cash = 100000
        obs = self._get_obs()
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        recs = {a: actions[a] for a in self.specialists}
        alloc = actions['portfolio']
        alloc /= np.sum(alloc) if np.sum(alloc) > 0 else 1

        prices = np.array([self.frames[s].iloc[self.current_step]['adj_close'] for s in self.sym_list])
        target_value = self._get_portfolio_value(prices) * alloc[:-1]
        for i in range(len(prices)):
            target_shares = target_value[i] / prices[i]
            delta = target_shares - self.portfolio[i]
            self.cash -= np.abs(delta) * prices[i] * (CONFIG["reward"]["lambda_tc_bps"] / 10000)  # bps to decimal
            self.portfolio[i] = target_shares

        self.current_step += 1
        terminated = {a: self.current_step >= self.max_steps for a in self.agents}
        truncated = {a: False for a in self.agents}
        reward = self._calculate_reward(prices)
        rewards = {a: reward for a in self.agents}

        obs = self._get_obs(recs)
        infos = {a: {} for a in self.agents}
        return obs, rewards, terminated, truncated, infos

    def _get_obs(self, recs=None):
        if recs is None:
            recs = {a: np.zeros(len(self.sym_list)) for a in self.specialists}
        features = np.concatenate([self.frames[s].iloc[self.current_step, 1:].values for s in self.sym_list])
        rec_array = np.concatenate([recs[a] for a in self.specialists])
        full_obs = np.concatenate([features, rec_array])
        return {a: full_obs for a in self.agents}

    def _get_portfolio_value(self, prices):
        return self.cash + np.dot(self.portfolio, prices)

    def _calculate_reward(self, prices):
        port_value = self._get_portfolio_value(prices)
        returns = (port_value - 100000) / 100000
        volatility = np.std(self.portfolio * prices) / port_value if port_value > 0 else 0
        dd = (port_value - np.max(self._get_portfolio_value(prices))) / np.max(self._get_portfolio_value(prices)) if port_value > 0 else 0
        return returns - CONFIG["reward"]["lambda_sigma"] * volatility - CONFIG["reward"]["lambda_dd"] * abs(dd)

    def render(self, mode="human"):
        prices = np.array([self.frames[s].iloc[self.current_step]['adj_close'] for s in self.sym_list])
        print(f"Step: {self.current_step}, Value: {self._get_portfolio_value(prices)}")





