import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from pathlib import Path
import yaml

# --- Load Config ---
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    INITIAL_BALANCE = config['environment_settings']['initial_balance']
    COMMISSION_FEE = config['environment_settings']['commission_fee']
    
except FileNotFoundError:
    print("Error: config.yaml not found.")
    exit()


class MultiStockEnv(gym.Env):
    """A multi-stock trading environment for reinforcement learning."""
    
    def __init__(self, data_dict, train_end_date):
        super().__init__()
        
        self.data_dict = data_dict
        self.tickers = list(data_dict.keys())
        self.n_stocks = len(self.tickers)
        
        # Split data into train and test sets
        self.train_df = {ticker: df[df.index <= train_end_date] for ticker, df in self.data_dict.items()}
        self.test_df = {ticker: df[df.index > train_end_date] for ticker, df in self.data_dict.items()}
        
        self.df = self.train_df # Start with training data
        self.max_steps = min(len(df) for df in self.df.values())
        
        # Define action space: weights for each stock + a weight for cash
        # The agent outputs a desired weight for each stock.
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_stocks,), dtype=np.float32)

        # Define observation space: features for all stocks at a given time step
        n_features = self.df[self.tickers[0]].shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_stocks, n_features), dtype=np.float32
        )
        
        self.current_step = 0
        self.portfolio_value = INITIAL_BALANCE
        self.weights = np.array([1.0] + [0.0] * self.n_stocks) # [cash_weight, stock1_weight, ...]

    def _get_obs(self):
        obs = np.array([df.iloc[self.current_step].values for df in self.df.values()])
        return obs.astype(np.float32)

    def _get_info(self):
        return {
            'portfolio_value': self.portfolio_value,
            'weights': self.weights
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = INITIAL_BALANCE
        self.weights = np.array([1.0] + [0.0] * self.n_stocks) # Start with 100% cash
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        # Normalize the agent's raw action to sum to 1 (softmax)
        # This represents the target portfolio allocation
        target_weights = np.exp(action) / np.sum(np.exp(action))

        # We will rebalance the portfolio based on these target weights
        # For simplicity in this version, we'll calculate the daily return based on these weights
        
        # Get price changes for the current step
        price_changes = []
        for ticker in self.tickers:
            current_price = self.df[ticker]['Close'].iloc[self.current_step]
            next_price = self.df[ticker]['Close'].iloc[self.current_step + 1]
            price_change = (next_price - current_price) / current_price
            price_changes.append(price_change)
            
        portfolio_return = np.dot(target_weights, price_changes)
        
        # For now, we assume no transaction costs in this simplified MARL setup
        new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
        
        reward = np.log(new_portfolio_value / self.portfolio_value) if self.portfolio_value > 0 else 0
        self.portfolio_value = new_portfolio_value
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps - 2

        return self._get_obs(), reward, terminated, False, self._get_info()