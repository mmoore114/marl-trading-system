import pandas as pd
import yaml
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np  # Added to fix np.mean/std NameError

from environment import MultiStrategyEnv

# Load config
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    TICKERS = config['data_settings']['tickers']
    TRAIN_END_DATE = config['data_settings']['train_end_date']
    VAL_END_DATE = config['data_settings']['val_end_date']
    MODEL_SETTINGS = config['model_settings']
    INITIAL_BALANCE = config['environment_settings']['initial_balance']  # Added for val eval
except FileNotFoundError:
    print("Error: config.yaml not found.")
    exit()

# Load data (skip extra rows, name columns)
print("Loading data...")
processed_data_dir = Path("data/processed")
data_dict = {ticker: pd.read_csv(processed_data_dir / f"{ticker}_processed.csv", skiprows=3, header=None, names=['Date', 'Close'], index_col='Date', parse_dates=True) for ticker in TICKERS}

# Create train env
print("Creating train environment...")
train_env_lambda = lambda: MultiStrategyEnv(data_dict, TRAIN_END_DATE, VAL_END_DATE, mode='train')
train_env = DummyVecEnv([train_env_lambda])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.)

# Create val env
print("Creating val environment...")
val_env_lambda = lambda: MultiStrategyEnv(data_dict, TRAIN_END_DATE, VAL_END_DATE, mode='val')
val_env = DummyVecEnv([val_env_lambda])
val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False, clip_obs=10.)  # Separate norm for val

# Grid search learning rates
learning_rates = [1e-4, 3e-4, 1e-3]
best_model = None
best_sharpe = -float('inf')

for lr in learning_rates:
    print(f"Training with lr={lr}...")
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        learning_rate=lr
    )
    
    # Eval callback for early stopping
    eval_callback = EvalCallback(
        val_env, 
        best_model_save_path="models/best_model",
        log_path="logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=1,
        warn=False
    )
    
    model.learn(total_timesteps=MODEL_SETTINGS['total_timesteps'], callback=eval_callback)
    
    # Quick val eval for Sharpe (simplified; adapt if needed)
    obs = val_env.reset()
    done = [False]
    returns = []
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = val_env.step(action)
        returns.append(info[0]['portfolio_value'] / INITIAL_BALANCE - 1)  # Cumulative return proxy
    val_sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    
    if val_sharpe > best_sharpe:
        best_sharpe = val_sharpe
        best_model = model

# Save best
models_path = Path("models")
models_path.mkdir(parents=True, exist_ok=True)
best_model.save(models_path / "ppo_multi_strategy_trader.zip")
train_env.save(models_path / "multi_strategy_vec_normalize_stats.pkl")

print(f"\nTraining complete. Best model (lr={best_model.learning_rate}) saved to {models_path / 'ppo_multi_strategy_trader.zip'}")
print(f"Normalization stats saved to {models_path / 'multi_strategy_vec_normalize_stats.pkl'}")
