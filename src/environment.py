import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from pathlib import Path
import yaml

try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    INITIAL_BALANCE = config['environment_settings']['initial_balance']
except FileNotFoundError:
    print("Error: config.yaml not found.")
    exit()

class MultiStockEnv(gym.Env):
    def __init__(self, data_dict, train_end_date, mode='train'): # NEW: mode parameter
        super().__init__()
        
        self.tickers = list(data_dict.keys())
        self.n_stocks = len(self.tickers)
        
        # Split data into train and test sets
        self.train_df = {ticker: df[df.index <= train_end_date] for ticker, df in data_dict.items()}
        self.test_df = {ticker: df[df.index > train_end_date] for ticker, df in data_dict.items()}
        
        # Set the current dataset based on the mode
        self.df = self.train_df if mode == 'train' else self.test_df
        
        self.max_steps = min(len(df) for df in self.df.values())
        
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_stocks,), dtype=np.float32)
        n_features = self.df[self.tickers[0]].shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_stocks, n_features), dtype=np.float32
        )
        
        self.current_step = 0
        self.portfolio_value = INITIAL_BALANCE

    def _get_obs(self):
        obs = np.array([df.iloc[self.current_step].values for df in self.df.values()])
        return obs.astype(np.float32)

    def _get_info(self):
        return {'portfolio_value': self.portfolio_value}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = INITIAL_BALANCE
        return self._get_obs(), self._get_info()

    def step(self, action):
        target_weights = np.exp(action) / np.sum(np.exp(action))
        
        price_changes = []
        for ticker in self.tickers:
            current_price = self.df[ticker]['Close'].iloc[self.current_step]
            next_price = self.df[ticker]['Close'].iloc[self.current_step + 1]
            price_change = (next_price - current_price) / current_price
            price_changes.append(price_change)
            
        portfolio_return = np.dot(target_weights, price_changes)
        new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
        
        reward = np.log(new_portfolio_value / self.portfolio_value) if self.portfolio_value > 0 else 0
        self.portfolio_value = new_portfolio_value
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps - 2

        return self._get_obs(), reward, terminated, False, self._get_info()
    