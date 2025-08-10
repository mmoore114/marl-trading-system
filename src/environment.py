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
    RISK_AVERSION = config.get('environment_settings', {}).get('risk_aversion', 0.5)
    LOOKBACK_PERIOD = config.get('environment_settings', {}).get('lookback_period', 20)
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    MIN_DATA_LENGTH = config.get('environment_settings', {}).get('min_data_length', 250)
except FileNotFoundError:
    print("Error: config.yaml not found.")
    exit()

class MultiStrategyEnv(gym.Env):
    def __init__(self, data_dict, train_end_date, val_end_date=None, mode='train'):
        super().__init__()
        
        self.tickers = list(data_dict.keys())
        self.n_stocks = len(self.tickers)
        
        if mode == 'train':
            self.df = {ticker: df[df.index <= pd.to_datetime(train_end_date)].copy() for ticker, df in data_dict.items()}
        elif mode == 'val' and val_end_date:
            self.df = {ticker: df[(df.index > pd.to_datetime(train_end_date)) & (df.index <= pd.to_datetime(val_end_date))].copy() for ticker, df in data_dict.items()}
        else:  # test
            start = pd.to_datetime(val_end_date or train_end_date)
            self.df = {ticker: df[df.index > start].copy() for ticker, df in data_dict.items()}
        
        # Align to common index with ffill for missing data
        if self.df:
            indices = [df.index for df in self.df.values()]
            common_index = indices[0]
            for idx in indices[1:]:
                common_index = common_index.intersection(idx)
            for ticker in self.tickers:
                self.df[ticker] = self.df[ticker].reindex(common_index).ffill().dropna()
        
        for ticker, df in self.df.items():
            if len(df) < MIN_DATA_LENGTH:
                raise ValueError(f"{ticker} has only {len(df)} rows in {mode} mode after alignment.")
        
        lengths = [len(df) for df in self.df.values()]
        if len(set(lengths)) > 1:
            raise ValueError(f"Inconsistent lengths in {mode} data after alignment: {lengths}")
        self.max_steps = lengths[0] - 1 if lengths else 0
        
        self.prev_weights = np.zeros(self.n_stocks)
        
        for ticker in self.tickers:
            df = self.df[ticker]
            if 'Close' not in df.columns:
                raise ValueError(f"'Close' column missing for {ticker}.")
            
            df['Momentum'] = df['Close'].pct_change(LOOKBACK_PERIOD)
            df['Volatility'] = df['Close'].pct_change().rolling(LOOKBACK_PERIOD).std()
            df['RSI'] = self._calculate_rsi(df['Close'])
            df['MACD'], df['MACD_Signal'] = self._calculate_macd(df['Close'])
            df['Golden_Cross'] = (df['Close'].rolling(50).mean() > df['Close'].rolling(200).mean()).astype(float)
            df['Volume_Norm'] = np.log(df['Volume'] + 1)  # Add normalized Volume (log to handle scale)
            df.fillna(0, inplace=True)
        
        self.feature_cols = ['Close', 'Momentum', 'Volatility', 'RSI', 'MACD', 'MACD_Signal', 'Golden_Cross', 'Volume_Norm']  # Added Volume_Norm
        
        for ticker in self.tickers:
            missing_cols = set(self.feature_cols) - set(self.df[ticker].columns)
            if missing_cols:
                raise ValueError(f"Missing columns {missing_cols} for {ticker} after feature calculation.")
            self.df[ticker] = self.df[ticker][self.feature_cols]
        
        n_features = len(self.feature_cols)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_stocks * n_features,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_stocks,), dtype=np.float32)
        
        self.current_step = 0
        self.portfolio_value = INITIAL_BALANCE

    def _calculate_rsi(self, series):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=RSI_PERIOD).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=RSI_PERIOD).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, series):
        ema_fast = series.ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = series.ewm(span=MACD_SLOW, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
        return macd, signal

    def _get_obs(self):
        obs = np.array([df.iloc[self.current_step][self.feature_cols].values for df in self.df.values()])
        return obs.flatten().astype(np.float32)

    def _get_info(self):
        return {'portfolio_value': self.portfolio_value}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = INITIAL_BALANCE
        self.prev_weights = np.zeros(self.n_stocks)
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(action, 0, 1)
        target_weights = action / np.sum(action) if np.sum(action) > 0 else np.ones(self.n_stocks) / self.n_stocks
        print(f"Step action (normalized): {target_weights}, sum: {np.sum(target_weights)}")  # Debug

        # Transaction costs: 0.1% on weight changes
        weight_diff = np.abs(target_weights - self.prev_weights)
        trans_cost = 0.001 * self.portfolio_value * np.sum(weight_diff)
        self.portfolio_value -= trans_cost
        self.prev_weights = target_weights.copy()

        price_changes = []
        volatilities = []
        for ticker in self.tickers:
            current_price = self.df[ticker]['Close'].iloc[self.current_step]
            next_price = self.df[ticker]['Close'].iloc[self.current_step + 1]
            price_change = (next_price - current_price) / current_price
            price_changes.append(price_change)
            volatilities.append(self.df[ticker]['Volatility'].iloc[self.current_step])
        
        portfolio_return = np.dot(target_weights, price_changes)
        avg_vol = np.mean(volatilities)
        new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
        log_return = np.log(new_portfolio_value / self.portfolio_value) if self.portfolio_value > 0 else 0
        reward = log_return - RISK_AVERSION * avg_vol
        self.portfolio_value = new_portfolio_value
        self.current_step += 1
        terminated = self.current_step >= self.max_steps - 1
        return self._get_obs(), reward, terminated, False, self._get_info()