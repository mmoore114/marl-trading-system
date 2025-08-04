import pandas as pd
import quantstats as qs
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment import MultiStockEnv, INITIAL_BALANCE

# --- Load Config ---
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    TICKERS = config['data_settings']['tickers']
    TRAIN_END_DATE = config['data_settings']['train_end_date']
except FileNotFoundError:
    print("Error: config.yaml not found.")
    exit()

# --- Paths ---
processed_data_dir = Path("data/processed")
model_path = Path("models/ppo_multi_stock_trader.zip")
stats_path = Path("models/multi_stock_vec_normalize_stats.pkl")

# --- Load All Data ---
print("Loading data for backtest...")
data_dict = {}
for ticker in TICKERS:
    file_path = processed_data_dir / f"{ticker}_processed.csv"
    data_dict[ticker] = pd.read_csv(file_path, index_col=0, parse_dates=True)

# --- Create Test Environment ---
print("Creating test environment...")
env_lambda = lambda: MultiStockEnv(data_dict, TRAIN_END_DATE, mode='test')
eval_env = DummyVecEnv([env_lambda])
eval_env = VecNormalize.load(stats_path, eval_env)
eval_env.training = False
eval_env.norm_reward = False

# --- Load Model ---
print("Loading trained agent...")
model = PPO.load(model_path, env=eval_env)

# --- Run Backtest ---
print("Running backtest on unseen test data...")
obs = eval_env.reset()
account_values = [INITIAL_BALANCE]
terminated = False

while not terminated:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = eval_env.step(action)
    account_values.append(infos[0]['portfolio_value'])
    terminated = dones[0]

# --- Quantitative Analysis with quantstats ---
print("\n--- Quantitative Analysis ---")
# Get the dates from one of the test dataframes
test_dates = data_dict[TICKERS[0]][data_dict[TICKERS[0]].index > TRAIN_END_DATE].index
# Create a pandas Series of portfolio values with dates as index
portfolio_values = pd.Series(account_values, index=test_dates[:len(account_values)])

# Calculate returns from the portfolio values
returns = portfolio_values.pct_change().dropna()

# Generate and save a detailed HTML report
report_path = "multi_agent_performance_report.html"
qs.reports.html(returns, output=report_path, title='Super-Agent Performance')
print(f"Full performance report saved to: {report_path}")

# --- Visualize ---
print("\nPlotting performance...")
portfolio_values.plot(figsize=(15, 7), title="Super-Agent Performance on Unseen Test Data")
plt.ylabel("Portfolio Value (USD)")
plt.grid(True)
plt.show()

print("\nBacktest complete.")
