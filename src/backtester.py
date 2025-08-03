import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import our custom environment
from environment import StockTradingEnv, INITIAL_ACCOUNT_BALANCE

# --- Define Paths ---
processed_data_path = Path("data/processed/AAPL_processed.csv")
model_path = Path("models/ppo_stock_trader.zip")
stats_path = Path("models/vec_normalize_stats.pkl")

# --- 1. Load Data ---
print("Loading data for backtest...")
df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)

# --- 2. Load the Trained Agent and Normalization Stats ---
print("Loading trained agent and normalization stats...")

# Create a dummy environment to load the normalization stats
eval_env = DummyVecEnv([lambda: StockTradingEnv(df)])
# Load the saved statistics
eval_env = VecNormalize.load(stats_path, eval_env)

# Set the environment to evaluation mode (don't update normalization stats)
eval_env.training = False
# Do not normalize rewards during evaluation
eval_env.norm_reward = False

# Load the trained agent and pass the prepared environment
model = PPO.load(model_path, env=eval_env)


# --- 3. Run the Backtest ---
print("Running backtest...")
obs = eval_env.reset()
account_values = [INITIAL_ACCOUNT_BALANCE]

# Loop through the entire dataset
while True:
    # Use deterministic actions for evaluation
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = eval_env.step(action)
    
    # Append the portfolio value from the info dictionary
    # info is a list of dicts for VecEnvs, so we take the first one
    account_values.append(info[0]['portfolio_value'])

    if done:
        break

# --- 4. Visualize the Performance ---
print("Plotting performance...")
plt.figure(figsize=(15, 7))
plt.plot(df.index[:len(account_values)], account_values)
plt.title("Agent Performance on AAPL Stock")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.grid(True)
plt.show()

print("Backtest complete.")