import pandas as pd
import yaml
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import our new custom environment
from environment import MultiStockEnv

# --- 1. Load Configuration ---
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    TICKERS = config['data_settings']['tickers']
    TRAIN_END_DATE = config['data_settings']['train_end_date']
    MODEL_SETTINGS = config['model_settings']

except FileNotFoundError:
    print("Error: config.yaml not found.")
    exit()


# --- 2. Load Data for All Tickers ---
print("Loading data for all tickers...")
processed_data_dir = Path("data/processed")
data_dict = {}
for ticker in TICKERS:
    file_path = processed_data_dir / f"{ticker}_processed.csv"
    data_dict[ticker] = pd.read_csv(file_path, index_col=0, parse_dates=True)


# --- 3. Create the Multi-Stock Environment ---
print("Creating multi-stock environment...")
env_lambda = lambda: MultiStockEnv(data_dict, TRAIN_END_DATE)
env = DummyVecEnv([env_lambda])
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)


# --- 4. Create and Train the Agent ---
print("Creating and training the PPO Super-Agent...")
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=MODEL_SETTINGS['learning_rate']
)

model.learn(total_timesteps=MODEL_SETTINGS['total_timesteps'])


# --- 5. Save the Trained Model and Normalization Stats ---
print("Saving model and normalization stats...")
models_path = Path("models")
models_path.mkdir(parents=True, exist_ok=True)

model_save_path = models_path / "ppo_multi_stock_trader.zip"
stats_path = models_path / "multi_stock_vec_normalize_stats.pkl"

model.save(model_save_path)
env.save(stats_path)

print(f"\nTraining complete. Model saved to {model_save_path}")
print(f"Normalization stats saved to {stats_path}")