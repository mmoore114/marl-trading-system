import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import yaml

class MultiStrategyEnv(gym.Env):
    def __init__(self, data_dict, config, mode='train'):
        super().__init__()
        
        self.config = config
        self.mode = mode
        self.tickers = list(data_dict.keys())
        self.n_stocks = len(self.tickers)
        
        # --- Environment Settings ---
        self.initial_balance = self.config['environment_settings']['initial_balance']
        self.risk_aversion = self.config['environment_settings']['risk_aversion']
        
        # --- Data Handling ---
        self._prepare_data(data_dict)
        
        # --- Spaces ---
        n_features = self.df[self.tickers[0]].shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_stocks, n_features), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_stocks,), dtype=np.float32)
        
        # --- State Variables ---
        self.current_step = 0
        self.portfolio_value = self.initial_balance
        self.portfolio_weights = np.array([1.0] + [0.0] * self.n_stocks) # [cash_weight, stock1_weight, ...]
        
    def _prepare_data(self, data_dict):
        """Prepares and splits data based on the mode."""
        train_end = pd.to_datetime(self.config['data_settings']['train_end_date'])
        val_end = pd.to_datetime(self.config['data_settings']['val_end_date'])
        
        if self.mode == 'train':
            self.df = {ticker: df[df.index <= train_end].copy() for ticker, df in data_dict.items()}
        elif self.mode == 'val':
            self.df = {ticker: df[(df.index > train_end) & (df.index <= val_end)].copy() for ticker, df in data_dict.items()}
        else: # test mode
            self.df = {ticker: df[df.index > val_end].copy() for ticker, df in data_dict.items()}

        # Align to common trading days
        common_index = self.df[self.tickers[0]].index
        for ticker in self.tickers[1:]:
            common_index = common_index.intersection(self.df[ticker].index)
        
        for ticker in self.tickers:
            self.df[ticker] = self.df[ticker].reindex(common_index).ffill()
            
        self.max_steps = len(common_index) - 2 # -2 to ensure we can always get next price

    def _get_obs(self):
        obs = np.array([df.iloc[self.current_step].values for df in self.df.values()])
        return obs.astype(np.float32)

    def _get_info(self):
        return {'portfolio_value': self.portfolio_value, 'weights': self.portfolio_weights[1:]}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_balance
        self.portfolio_weights = np.array([1.0] + [0.0] * self.n_stocks)
        return self._get_obs(), self._get_info()

    def step(self, action):
        # Normalize actions to represent portfolio weights summing to 1
        target_weights = np.clip(action, 0, 1)
        target_weights /= np.sum(target_weights) if np.sum(target_weights) > 0 else 1
        
        # --- Transaction Cost Calculation (Corrected) ---
        # Get current weights from portfolio state
        current_weights = self.portfolio_weights[1:] # Exclude cash weight
        
        # Calculate the value of trades
        trades = (target_weights - current_weights) * self.portfolio_value
        transaction_cost = np.sum(np.abs(trades)) * 0.001 # 0.1% commission

        # --- Portfolio Return Calculation ---
        price_changes = []
        volatilities = []
        for ticker in self.tickers:
            current_price = self.df[ticker]['Close'].iloc[self.current_step]
            next_price = self.df[ticker]['Close'].iloc[self.current_step + 1]
            price_changes.append((next_price - current_price) / current_price)
            volatilities.append(self.df[ticker].get('Volatility', 0.01)) # Safely get volatility

        portfolio_return = np.dot(target_weights, price_changes)
        
        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_value -= transaction_cost
        
        # Update portfolio weights for the next step
        new_asset_values = target_weights * self.portfolio_value
        self.portfolio_weights = np.insert(new_asset_values, 0, self.portfolio_value - np.sum(new_asset_values)) / self.portfolio_value

        # --- Reward Calculation ---
        log_return = np.log(self.portfolio_value / (self.portfolio_value - portfolio_return*self.portfolio_value + transaction_cost)) if self.portfolio_value > 0 else 0
        portfolio_volatility = np.dot(target_weights, volatilities)
        reward = log_return - self.risk_aversion * portfolio_volatility
        
        # --- State Update ---
        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, False, self._get_info()