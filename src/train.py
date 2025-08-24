# src/train.py
from __future__ import annotations

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# NOTE: we intentionally do NOT wrap with Monitor or FlattenObservation here.
# - Monitor can be added later for episode stats if needed.
# - PPO with MultiInputPolicy expects Dict observations; do not flatten.

from src.environment import MultiStrategyEnv


def make_env(mode: str):
    def _init():
        e = MultiStrategyEnv(mode=mode)
        return e
    return _init


def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    n_envs = int(cfg["training"]["n_envs"])
    total_timesteps = int(cfg["training"]["total_timesteps"])
    seed = int(cfg["training"]["seed"])
    policy_kwargs = cfg["training"].get("policy_kwargs", {})

    # Vectorized env for faster rollout
    env = make_vec_env(make_env("train"), n_envs=n_envs, seed=seed)

    model = PPO(
        policy="MultiInputPolicy",  # Dict obs
        env=env,
        verbose=1,
        seed=seed,
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save("models/ppo_specialist_super_agent_final.zip")
    print("Training complete. Saved model to models/ppo_specialist_super_agent_final.zip")


if __name__ == "__main__":
    main()

