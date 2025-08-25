import os
import yaml
import torch
import ray
from ray import air, tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.policy.policy import PolicySpec
from src.environment import TradingEnv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = read_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} device")

    ray.init(num_gpus=1 if device == "cuda" else 0, ignore_reinit_error=True)

    tr_cfg = cfg.get("training", {})
    seed = tr_cfg.get("seed", 42)

    policies = {
        a: PolicySpec() for a in ['technical', 'fundamental', 'sentiment', 'risk', 'portfolio']
    }
    policy_mapping_fn = lambda agent_id: agent_id

    config = {
        "env": TradingEnv,
        "env_config": {"mode": "train"},
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "framework": "torch",
        "num_gpus": 1 if device == "cuda" else 0,
        "num_workers": tr_cfg.get("n_envs", 4),
        "lr": tr_cfg.get("learning_rate", 3e-4),
        "seed": seed,
        "log_level": "INFO",
        "evaluation_interval": 10,
        "evaluation_num_workers": 1,
        "evaluation_config": {"env_config": {"mode": "test"}},
    }

    tuner = tune.Tuner(
        PPOTrainer,
        param_space=config,
        run_config=air.RunConfig(
            stop={"training_iteration": tr_cfg.get("total_timesteps", 150000) // 2048},  # Approx
            verbose=1,
            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=10),
            storage_path="models/ray_checkpoints",
        ),
    )

    results = tuner.fit()
    best_checkpoint = results.get_best_result(metric="episode_reward_mean", mode="max").checkpoint
    logger.info(f"Best checkpoint: {best_checkpoint.path}")

    ray.shutdown()

if __name__ == "__main__":
    main()


