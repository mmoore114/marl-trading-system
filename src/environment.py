import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from pathlib import Path

INITIAL_ACCOUNT_BALANCE = 100000

class StockTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(df.columns),), dtype=np.float32
        )
        
    def _get_obs(self):
        return self.df.iloc[self.current_step].values.astype(np.float32)

    def _get_info(self):
        return {'portfolio_value': self.portfolio_value}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        current_price = self.df['Close'].iloc[self.current_step]
        current_holdings = self.shares_held * current_price
        scaled_action = (action[0] + 1) / 2
        desired_value = self.portfolio_value * scaled_action

        if current_price > 0:
            shares_to_trade = (desired_value - current_holdings) / current_price
        else:
            shares_to_trade = 0
        
        self.shares_held += shares_to_trade
        self.current_step += 1

        terminated = self.current_step >= len(self.df) - 1
        
        # Calculate new portfolio value and reward
        next_price = self.df['Close'].iloc[self.current_step] if not terminated else current_price
        cash_on_hand = self.portfolio_value - (shares_to_trade * current_price)
        new_portfolio_value = self.shares_held * next_price + cash_on_hand
        
        # **NEW SAFEGUARD**
        # Check for invalid portfolio value and penalize heavily
        if not np.isfinite(new_portfolio_value) or new_portfolio_value <= 0:
            terminated = True
            reward = -100.0 # Large penalty for going bust or exploding
        else:
            if self.portfolio_value > 0:
                reward = np.log(new_portfolio_value / self.portfolio_value)
            else:
                reward = 0
                
        self.portfolio_value = new_portfolio_value

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, False, info