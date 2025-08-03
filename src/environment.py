import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from pathlib import Path

INITIAL_ACCOUNT_BALANCE = 100000

class StockTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, commission_fee=0.001, trade_limit_percent=0.1):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.commission_fee = commission_fee
        self.trade_limit_percent = trade_limit_percent # NEW: Liquidity limit
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
        current_volume = self.df['Volume'].iloc[self.current_step]
        current_holdings = self.shares_held * current_price
        
        scaled_action = (action[0] + 1) / 2
        desired_value = self.portfolio_value * scaled_action

        if current_price > 0:
            shares_to_trade = (desired_value - current_holdings) / current_price
        else:
            shares_to_trade = 0
            
        # NEW: Enforce the liquidity limit
        # Cap the number of shares traded to a percentage of the day's volume
        volume_limit = current_volume * self.trade_limit_percent
        shares_to_trade = np.clip(shares_to_trade, -volume_limit, volume_limit)
        
        transaction_cost = abs(shares_to_trade * current_price) * self.commission_fee
        
        self.shares_held += shares_to_trade
        self.current_step += 1

        terminated = self.current_step >= len(self.df) - 1
        
        next_price = self.df['Close'].iloc[self.current_step] if not terminated else current_price
        cash_on_hand = self.portfolio_value - (shares_to_trade * current_price) - transaction_cost
        new_portfolio_value = self.shares_held * next_price + cash_on_hand
        
        if not np.isfinite(new_portfolio_value) or new_portfolio_value <= 0:
            terminated = True
            reward = -100.0
        else:
            if self.portfolio_value > 0:
                reward = np.log(new_portfolio_value / self.portfolio_value)
            else:
                reward = 0
                
        self.portfolio_value = new_portfolio_value

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, False, info