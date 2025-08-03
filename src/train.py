import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import our custom environment
from environment import StockTradingEnv

# Define paths
processed_data_path = Path("data/processed/AAPL_processed.csv")
models_path = Path("models")

# Ensure models directory exists
models_path.mkdir(parents=True, exist_ok=True)

# --- 1. Load Data and Create Environment ---
print("Loading data and creating environment...")
df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)

# Wrap the custom environment in a DummyVecEnv for compatibility
env_lambda = lambda: StockTradingEnv(df)
env = DummyVecEnv([env_lambda])

# Wrap it with the VecNormalize wrapper for observation normalization
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)


# --- 2. Create and Configure the Agent ---
print("Creating PPO agent...")
# 'MlpPolicy' is a standard feed-forward neural network policy.
# verbose=1 will print out training progress.
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.000025)

# --- 3. Train the Agent ---
print("Training agent...")
# total_timesteps is the number of simulation steps the agent will learn from.
model.learn(total_timesteps=20000)

# --- 4. Save the Trained Model and Normalization Stats ---
model_save_path = models_path / "ppo_stock_trader.zip"
stats_path = models_path / "vec_normalize_stats.pkl"

model.save(model_save_path)
env.save(stats_path)

print(f"\nTraining complete. Model saved to {model_save_path}")
print(f"Normalization stats saved to {stats_path}")