import pandas as pd
import quantstats as qs
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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

# --- 2. Define Paths ---
processed_data_dir = Path("data/processed")
model_path = Path("models/ppo_multi_strategy_final.zip")
stats_path = Path("models/multi_strategy_vec_normalize.pkl")

# --- 3. Load All Data ---
print("Loading data for backtest...")
data_dict = {}
for ticker in TICKERS:
    file_path = processed_data_dir / f"{ticker}_processed.csv"
    data_dict[ticker] = pd.read_csv(file_path, index_col='Date', parse_dates=True)

# --- 4. Create Test Environment ---
print("Creating test environment...")
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
account_values_agent = [INITIAL_BALANCE]
terminated = False
while not terminated:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = test_env.step(action)
    account_values_agent.append(infos[0]['portfolio_value'])
    terminated = dones[0]

# --- 7. Calculate Benchmark Performance ---
print("Calculating benchmark performance...")
test_start_date = config['data_settings']['val_end_date']
close_prices_df = pd.DataFrame({ticker: df['Close'] for ticker, df in data_dict.items()})
test_prices_df = close_prices_df[close_prices_df.index > test_start_date]
daily_returns = test_prices_df.pct_change().dropna()
benchmark_daily_returns = daily_returns.mean(axis=1)

# --- THE FIX: Give the benchmark Series a name ---
benchmark_daily_returns.name = "Equal-Weight Benchmark"

# Calculate the benchmark equity curve
benchmark_cumulative_returns = (1 + benchmark_daily_returns).cumprod()
benchmark_values = INITIAL_BALANCE * benchmark_cumulative_returns
benchmark_values.iloc[0] = INITIAL_BALANCE

# --- 8. Quantitative Analysis & Visualization ---
print("\n--- Final Performance Analysis ---")
test_dates = test_prices_df.index
portfolio_values_agent = pd.Series(account_values_agent, index=test_dates[:len(account_values_agent)])
agent_returns = portfolio_values_agent.pct_change().dropna()
agent_returns.name = "Agent Strategy" # Also good practice to name the agent's returns

report_path = "final_performance_report.html"
qs.reports.html(agent_returns, benchmark=benchmark_daily_returns, output=report_path, title='Super-Agent vs. Benchmark')
print(f"Full performance report saved to: {report_path}")

print("\nPlotting performance...")
plt.figure(figsize=(15, 7))
plt.plot(portfolio_values_agent, label='Agent Strategy')
plt.plot(benchmark_values, label='Buy and Hold Benchmark', linestyle='--')
plt.title("Agent vs. Buy-and-Hold Benchmark on Test Data")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.legend()
plt.grid(True)
plt.show()

print("\nBacktest complete.")