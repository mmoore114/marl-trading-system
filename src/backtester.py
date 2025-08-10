import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt

from environment import MultiStrategyEnv

# Load config
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    TICKERS = config['data_settings']['tickers']
    TRAIN_END_DATE = config['data_settings']['train_end_date']
    VAL_END_DATE = config['data_settings']['val_end_date']
    INITIAL_BALANCE = config['environment_settings']['initial_balance']
    LOOKBACK_PERIOD = config['environment_settings']['lookback_period']  # For factor Momentum
except FileNotFoundError:
    print("Error: config.yaml not found.")
    exit()

# Load data (skip extra rows, name columns)
print("Loading data for all tickers...")
processed_data_dir = Path("data/processed")
data_dict = {ticker: pd.read_csv(processed_data_dir / f"{ticker}_processed.csv", skiprows=3, header=None, names=['Date', 'Close'], index_col='Date', parse_dates=True) for ticker in TICKERS}

# Create test env
print("Creating test environment...")
test_env_lambda = lambda: MultiStrategyEnv(data_dict, TRAIN_END_DATE, VAL_END_DATE, mode='test')
test_env = DummyVecEnv([test_env_lambda])

# Load norm stats
stats_path = Path("models/multi_strategy_vec_normalize_stats.pkl")
if not stats_path.exists():
    raise FileNotFoundError(f"Stats not found at {stats_path}. Run training first.")
test_env = VecNormalize.load(stats_path, test_env)
test_env.training = False
test_env.norm_reward = False

# Load model
print("Loading PPO model...")
model_path = Path("models/ppo_multi_strategy_trader.zip")
if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
model = PPO.load(model_path)

# PPO rollout
print("Evaluating PPO on test set...")
obs = test_env.reset()
done = [False]
portfolio_values = [INITIAL_BALANCE]
while not done[0]:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    portfolio_values.append(info[0]['portfolio_value'])

portfolio_values = np.array(portfolio_values)
returns = np.diff(portfolio_values) / portfolio_values[:-1]

# Metrics
total_return = (portfolio_values[-1] / INITIAL_BALANCE - 1) * 100
sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
downside_returns = returns