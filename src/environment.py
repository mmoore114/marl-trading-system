import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import yaml
from gymnasium.envs.registration import register

# --- Load Config (Only settings needed by the env) ---
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: config.yaml not found.")
    # Set defaults if config is not found, useful for simple testing
    config = {
        'environment_settings': {'initial_balance': 15000},
        'reward_settings': {'drawdown_penalty': 0.2, 'turnover_penalty': 0.05},
        'data_settings': {'train_end_date': '2022-12-31', 'val_end_date': '2023-12-31'}
    }

class MultiStrategyEnv(gym.Env):
    def __init__(self, data_dict, config, mode='train'):
        super().__init__()
        
        self.config = config
        self.mode = mode
        self.tickers = list(data_dict.keys())
        self.n_stocks = len(self.tickers)
        
        # --- Settings ---
        self.initial_balance = self.config['environment_settings']['initial_balance']
        self.drawdown_penalty = self.config['reward_settings']['drawdown_penalty']
        self.turnover_penalty = self.config['reward_settings']['turnover_penalty']
        
        # --- Data Handling ---
        self._prepare_data(data_dict)
        self._prepare_feature_groups()
        
        # --- Spaces (Dict-based observation) ---
        self.observation_space = spaces.Dict({
            agent_name: spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_stocks, n_features), dtype=np.float32)
            for agent_name, n_features in self.features_per_agent.items()
        })
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_stocks,), dtype=np.float32)
        
        self.reset()

    def _prepare_data(self, data_dict):
        """Prepares and splits data based on the mode."""
        train_end = pd.to_datetime(self.config['data_settings']['train_end_date'])
        val_end = pd.to_datetime(self.config['data_settings']['val_end_date'])
        
        if self.mode == 'train':
            self.df_dict = {ticker: df[df.index <= train_end].copy() for ticker, df in data_dict.items()}
        elif self.mode == 'val':
            self.df_dict = {ticker: df[(df.index > train_end) & (df.index <= val_end)].copy() for ticker, df in data_dict.items()}
        else: # test mode
            self.df_dict = {ticker: df[df.index > val_end].copy() for ticker, df in data_dict.items()}

        common_index = self.df_dict[self.tickers[0]].index
        for ticker in self.tickers[1:]:
            common_index = common_index.intersection(self.df_dict[ticker].index)
        
        for ticker in self.tickers:
            self.df_dict[ticker] = self.df_dict[ticker].reindex(common_index).ffill().bfill()
            
        self.max_steps = len(common_index) - 2

    def _prepare_feature_groups(self):
        """Groups features by specialist agent based on column prefixes."""
        self.feature_groups = {}
        self.features_per_agent = {}
        all_cols = self.df_dict[self.tickers[0]].columns
        
        # Use the agent names from the factor_settings in the config file
        for agent_name in self.config['factor_settings'].keys():
            self.feature_groups[agent_name] = [col for col in all_cols if col.startswith(agent_name)]
            self.features_per_agent[agent_name] = len(self.feature_groups[agent_name])

    def _get_obs(self):
        """Constructs the dictionary observation for the current step."""
        obs = {}
        for agent_name, feature_list in self.feature_groups.items():
            if feature_list: # Ensure the list is not empty
                agent_obs = np.array([self.df_dict[ticker][feature_list].iloc[self.current_step].values for ticker in self.tickers])
                obs[agent_name] = agent_obs.astype(np.float32)
        return obs

    def _get_info(self):
        return {'portfolio_value': self.portfolio_value, 'weights': self.portfolio_weights[1:], 'drawdown': self.current_drawdown}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_balance
        self.max_portfolio_value = self.initial_balance
        self.current_drawdown = 0
        self.portfolio_weights = np.array([1.0] + [0.0] * self.n_stocks)
        return self._get_obs(), self._get_info()

    def step(self, action):
        current_weights = self.portfolio_weights[1:]
        target_weights = np.clip(action, 0, 1)
        target_weights /= np.sum(target_weights) if np.sum(target_weights) > 0 else 1
        
        turnover = np.sum(np.abs(target_weights - current_weights))
        transaction_cost = turnover * self.portfolio_value * 0.001

        price_changes = []
        for ticker in self.tickers:
            current_price = self.df_dict[ticker]['Close'].iloc[self.current_step]
            next_price = self.df_dict[ticker]['Close'].iloc[self.current_step + 1]
            price_changes.append((next_price - current_price) / current_price)
        
        portfolio_return = np.dot(target_weights, price_changes)
        
        previous_portfolio_value = self.portfolio_value
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_value -= transaction_cost
        
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        self.current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        
        log_return = np.log(self.portfolio_value / previous_portfolio_value) if previous_portfolio_value > 0 else 0
        drawdown_penalty = self.current_drawdown * self.drawdown_penalty
        turnover_penalty = turnover * self.turnover_penalty
        reward = log_return - drawdown_penalty - turnover_penalty
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps or self.portfolio_value <= 0

        if self.portfolio_value > 0:
            new_asset_values = target_weights * self.portfolio_value
            self.portfolio_weights = np.insert(new_asset_values, 0, self.portfolio_value - np.sum(new_asset_values)) / self.portfolio_value
        else:
            self.portfolio_weights = np.array([1.0] + [0.0] * self.n_stocks)

        return self._get_obs(), reward, terminated, False, self._get_info()

# --- NEW: Register the environment with gymnasium ---
register(
    id='MultiStrategyEnv-v0',
    entry_point='environment:MultiStrategyEnv',
)