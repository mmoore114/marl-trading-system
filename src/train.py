import os
import yaml
import torch
from pathlib import Path
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from src.environment import TradingEnv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FIX: Robust Path Loading ---
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

def read_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = read_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    ray.init(num_gpus=1 if device == "cuda" else 0, ignore_reinit_error=True)

    tr_cfg = cfg.get("training", {})
    seed = tr_cfg.get("seed", 42)

    specialists = cfg["specialists"]["types"]
    policies = {agent_id: PolicySpec() for agent_id in specialists}
    policies["portfolio_manager"] = PolicySpec()
    
    # FIX: Policy mapping function with correct signature
    def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
        if agent_id in specialists:
            return agent_id
        else:
            return "portfolio_manager"

    config = (
        PPOConfig()
        .environment(TradingEnv, env_config={"mode": "train"})
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .framework("torch")
        .resources(num_gpus=1 if device == "cuda" else 0)
        .learners(num_gpus_per_learner=1 if device == "cuda" else 0)
        .env_runners(num_env_runners=tr_cfg.get("n_envs", 2))
        .training(
            lr=tr_cfg.get("learning_rate", 3e-4),
            gamma=tr_cfg.get("gamma", 0.99),
            lambda_=tr_cfg.get("gae_lambda", 0.95),
            clip_param=tr_cfg.get("clip_range", 0.2),
            vf_clip_param=10.0,
            entropy_coeff=tr_cfg.get("ent_coef", 0.005),
            train_batch_size=tr_cfg.get("batch_size", 4096),
            num_sgd_iter=10
        )
        .debugging(seed=seed, log_level="INFO")
        .evaluation(
            evaluation_interval=10,
            evaluation_num_env_runners=1,
            evaluation_config={"env_config": {"mode": "validation"}}
        )
    )

    stop_condition = {"training_iteration": tr_cfg.get("training_iterations", 100)}
    storage_path = str(ROOT / "models" / "ray_checkpoints")
    
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop=stop_condition,
            verbose=1,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=20, 
                checkpoint_at_end=True),
            storage_path=storage_path,
            name="ppo_trader"
        ),
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="evaluation/episode_reward_mean", mode="max")
    
    if best_result and best_result.checkpoint:
        logger.info(f"Best checkpoint found at: {best_result.checkpoint.path}")
    else:
        logger.warning("Could not determine best checkpoint from results.")

    ray.shutdown()
    logger.info("Training complete.")

if __name__ == "__main__":
    main()