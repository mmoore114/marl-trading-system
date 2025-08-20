import pandas as pd
import yaml
import numpy as np
import shutil
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from environment import MultiStrategyEnv

# --- 1. Load Configuration ---
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    TICKERS = config['data_settings']['tickers']
    MODEL_SETTINGS = config['model_settings']
    LEARNING_RATES = MODEL_SETTINGS.get('learning_rates', [0.0003]) # Use list from config, or default to one
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

# --- 3. Hyperparameter Tuning Loop ---
best_reward = -np.inf
best_lr = None
best_model_path_during_run = None

for lr in LEARNING_RATES:
    print(f"\n--- Training with Learning Rate: {lr} ---")
    
    # --- Create Environments ---
    train_env_lambda = lambda: MultiStrategyEnv(data_dict, config, mode='train')
    train_env = DummyVecEnv([train_env_lambda])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    val_env_lambda = lambda: MultiStrategyEnv(data_dict, config, mode='val')
    val_env = DummyVecEnv([val_env_lambda])
    val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # --- Define Callback ---
    log_dir = f"./logs/lr_{lr}/"
    save_path = f"./models/best_model_lr_{lr}/"
    
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=save_path,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=1,
        warn=False
    )

    # --- Create and Train Agent ---
    # THE KEY CHANGE: Use "MultiInputPolicy" for our Dict observation space
    model = PPO("MultiInputPolicy", train_env, verbose=0, learning_rate=lr)
    
    model.learn(
        total_timesteps=MODEL_SETTINGS.get('total_timesteps', 50000), 
        callback=eval_callback,
        progress_bar=True
    )
    
    # --- Evaluate and track the best model ---
    eval_results = np.load(f"{log_dir}evaluations.npz")
    mean_reward = np.mean(eval_results['results'])
    
    print(f"Learning Rate {lr} | Mean Validation Reward: {mean_reward:.2f}")

    if mean_reward > best_reward:
        best_reward = mean_reward
        best_lr = lr
        best_model_path_during_run = f"{save_path}best_model.zip"

# --- 4. Save the Overall Best Model ---
print("\n--- Hyperparameter Tuning Complete ---")
print(f"Best performing learning rate: {best_lr} (Reward: {best_reward:.2f})")

if best_model_path_during_run:
    print("Saving final best model and normalization stats...")
    models_path = Path("models")
    final_model_path = models_path / "ppo_specialist_super_agent_final.zip"
    stats_path = models_path / "specialist_super_agent_vec_normalize.pkl"

    # We need the stats that correspond to the best model's training run
    # For simplicity, we'll re-train one last time to get the correct final stats
    print("Re-training final model to save correct stats...")
    final_train_env_lambda = lambda: MultiStrategyEnv(data_dict, config, mode='train')
    final_train_env = DummyVecEnv([final_train_env_lambda])
    final_train_env = VecNormalize(final_train_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    final_model = PPO("MultiInputPolicy", final_train_env, learning_rate=best_lr)
    final_model.learn(total_timesteps=MODEL_SETTINGS.get('total_timesteps', 50000))
    
    final_model.save(final_model_path)
    final_train_env.save(stats_path)
    
    # Clean up intermediate model files
    for lr in LEARNING_RATES:
        shutil.rmtree(f"./models/best_model_lr_{lr}/", ignore_errors=True)

    print(f"\nTraining complete. Best model saved to {final_model_path}")
    print(f"Normalization stats saved to {stats_path}")
else:
    print("No best model was found.")