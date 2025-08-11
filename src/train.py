import pandas as pd
import yaml
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

# Import your custom environment
from environment import MultiStrategyEnv

# --- 1. Load Configuration ---
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    TICKERS = config['data_settings']['tickers']
    MODEL_SETTINGS = config['model_settings']
except (FileNotFoundError, KeyError) as e:
    print(f"Error loading configuration: {e}")
    exit()

# --- 2. Load Data ---
print("Loading data for all tickers...")
processed_data_dir = Path("data/processed")
data_dict = {}
for ticker in TICKERS:
    file_path = processed_data_dir / f"{ticker}_processed.csv"
    data_dict[ticker] = pd.read_csv(file_path, index_col='Date', parse_dates=True)


# --- 3. Create Training and Validation Environments ---
print("Creating environments...")
# The environment for training
train_env_lambda = lambda: MultiStrategyEnv(data_dict, config, mode='train')
train_env = DummyVecEnv([train_env_lambda])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.)

# The environment for validation
# FIX: The validation environment must ALSO be wrapped in VecNormalize
val_env_lambda = lambda: MultiStrategyEnv(data_dict, config, mode='val')
val_env = DummyVecEnv([val_env_lambda])
val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False, clip_obs=10.)


# --- 4. Define Callback and Train the Agent ---
eval_callback = EvalCallback(
    val_env,
    best_model_save_path="./models/best_model/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False,
    n_eval_episodes=1,
    warn=False
)

print("Creating and training the PPO Super-Agent...")
model = PPO(
    "MlpPolicy", 
    train_env, 
    verbose=1, 
    learning_rate=MODEL_SETTINGS.get('learning_rate', 0.0003)
)

model.learn(
    total_timesteps=MODEL_SETTINGS.get('total_timesteps', 50000), 
    callback=eval_callback
)


# --- 5. Save the Final Best Model and Stats ---
print("Saving final best model and normalization stats...")
models_path = Path("models")
final_model_path = models_path / "ppo_multi_strategy_final.zip"
stats_path = models_path / "multi_strategy_vec_normalize.pkl"

# The best model was saved by the callback, so we load it from there
best_model = PPO.load(models_path / "best_model/best_model.zip")
best_model.save(final_model_path)

# Save the normalization stats from the training environment
train_env.save(stats_path)

print(f"\nTraining complete. Best model saved to {final_model_path}")
print(f"Normalization stats saved to {stats_path}")