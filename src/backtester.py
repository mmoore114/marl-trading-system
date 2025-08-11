import pandas as pd
import quantstats as qs
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import your custom environment
from environment import MultiStrategyEnv

# --- 1. Load Configuration ---
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    TICKERS = config['data_settings']['tickers']
    INITIAL_BALANCE = config['environment_settings']['initial_balance']
except (FileNotFoundError, KeyError) as e:
    print(f"Error loading configuration: {e}")
    exit()

# --- 2. Define Paths for the Final Model ---
processed_data_dir = Path("data/processed")
model_path = Path("models/ppo_multi_strategy_final.zip")
stats_path = Path("models/multi_strategy_vec_normalize.pkl")

# --- 3. Load All Data ---
print("Loading data for backtest...")
data_dict = {}
for ticker in TICKERS:
    file_path = processed_data_dir / f"{ticker}_processed.csv"
    data_dict[ticker] = pd.read_csv(file_path, index_col='Date', parse_dates=True)

# --- 4. Create the Test Environment ---
print("Creating test environment...")
# IMPORTANT: Create the environment in 'test' mode
test_env_lambda = lambda: MultiStrategyEnv(data_dict, config, mode='test')
test_env = DummyVecEnv([test_env_lambda])
test_env = VecNormalize.load(stats_path, test_env)
test_env.training = False
test_env.norm_reward = False

# --- 5. Load the Trained Agent ---
print("Loading trained agent...")
model = PPO.load(model_path, env=test_env)

# --- 6. Run the Backtest ---
print("Running backtest on unseen test data...")
obs = test_env.reset()
account_values = [INITIAL_BALANCE]
terminated = False
while not terminated:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = test_env.step(action)
    account_values.append(infos[0]['portfolio_value'])
    terminated = dones[0]

# --- 7. Quantitative Analysis & Visualization ---
print("\n--- Final Performance Analysis ---")
test_start_date = config['data_settings']['val_end_date']
test_dates = data_dict[TICKERS[0]][data_dict[TICKERS[0]].index > test_start_date].index
portfolio_values = pd.Series(account_values, index=test_dates[:len(account_values)])
returns = portfolio_values.pct_change().dropna()

report_path = "final_performance_report.html"
qs.reports.html(returns, output=report_path, title='Final Agent Test Performance')
print(f"Full performance report saved to: {report_path}")

print("\nPlotting performance...")
portfolio_values.plot(figsize=(15, 7), title="Final Agent Performance on Unseen Test Data (2024-Beyond)")
plt.ylabel("Portfolio Value (USD)")
plt.grid(True)
plt.show()

print("\nBacktest complete.")