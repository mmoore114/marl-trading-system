import logging
from pathlib import Path
import random

import numpy as np
import ray
import yaml
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from src.environment import SingleAgentTradingEnv

# --- Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuration ---
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

def main():
    """Main function to run the training script."""
    logger.info("--- Starting Training Run ---")

    ray.init(logging_level=logging.ERROR)

    register_env(
        "SingleAgentTradingEnv-v0",
        lambda env_config: SingleAgentTradingEnv(env_config),
    )

    train_cfg = CONFIG["training"]
    eval_cfg = CONFIG.get("eval", {}) or {}

    # Seed all RNGs for reproducibility
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    policy_cfg = train_cfg["policy_kwargs"]

    config = (
        PPOConfig()
        .environment(
            env="SingleAgentTradingEnv-v0",
            env_config={"mode": "train"},
        )
        .framework("torch")
        .debugging(seed=seed)
        .resources(
            num_gpus=1,
            num_cpus_per_worker=1,
        )
        .rollouts(
            num_rollout_workers=train_cfg["n_envs"],
            batch_mode="complete_episodes",
        )
        .training(
            model={
                "fcnet_hiddens": policy_cfg["net_arch"]["pi"],
                "vf_share_layers": False,
            },
            train_batch_size=8192,
            lr=train_cfg["learning_rate"],
            gamma=train_cfg["gamma"],
            lambda_=train_cfg["gae_lambda"],
            clip_param=train_cfg["clip_range"],
            entropy_coeff=train_cfg["ent_coef"],
            vf_loss_coeff=train_cfg["vf_coef"],
            grad_clip=train_cfg["max_grad_norm"],
        )
        .evaluation(
            evaluation_interval=1,  # evaluate every training iteration
            evaluation_num_workers=1,
            evaluation_duration=int(eval_cfg.get("eval_episodes", 5)),
            evaluation_duration_unit="episodes",
            evaluation_config={
                "env_config": {"mode": "validation"},
                "explore": False,
            },
        )
    )

    logger.info("Building PPO Algorithm...")
    algo = config.build()
    logger.info("Build complete.")

    # Train for configured total timesteps
    total_timesteps = int(train_cfg.get("total_timesteps", 100000))
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    
    timesteps_total = 0
    while timesteps_total < total_timesteps:
        result = algo.train()
        timesteps_total = result["timesteps_total"]

        reward_mean = result.get("episode_reward_mean", float('nan'))
        ep_len_mean = result.get("episode_len_mean", float('nan'))
        
        logger.info(
            f"Iter: {result['training_iteration']}, "
            f"Timesteps: {timesteps_total}, "
            f"Mean Reward: {reward_mean:.4f}, "
            f"Mean Ep Len: {ep_len_mean:.2f}"
        )

    logger.info("--- Training Complete ---")
    
    checkpoint_dir = algo.save()
    logger.info(f"Checkpoint saved in directory: {checkpoint_dir}")

    algo.stop()
    ray.shutdown()
if __name__ == "__main__":
    main()
