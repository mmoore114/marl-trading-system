import os
import yaml
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from src.environment import MultiStrategyEnv

def read_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def make_env(mode):
    def _init():
        return MultiStrategyEnv(mode=mode)
    return _init

def main():
    cfg = read_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    tr_cfg = cfg.get("training", {})
    total_timesteps = int(tr_cfg.get("total_timesteps", 150000))
    n_envs = int(tr_cfg.get("n_envs", 4))
    seed = int(tr_cfg.get("seed", 42))

    # Vec env
    env = make_vec_env(make_env("train"), n_envs=n_envs, seed=seed, vec_env_cls=DummyVecEnv)

    policy_kwargs = tr_cfg.get("policy_kwargs", {})
    ppo_kwargs = dict(
        n_steps=int(tr_cfg.get("n_steps", 2048)),
        batch_size=int(tr_cfg.get("batch_size", 4096)),
        gae_lambda=float(tr_cfg.get("gae_lambda", 0.95)),
        gamma=float(tr_cfg.get("gamma", 0.99)),
        ent_coef=float(tr_cfg.get("ent_coef", 0.005)),
        vf_coef=float(tr_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(tr_cfg.get("max_grad_norm", 0.5)),
        clip_range=float(tr_cfg.get("clip_range", 0.2)),
        target_kl=float(tr_cfg.get("target_kl", 0.02)),
        learning_rate=float(tr_cfg.get("learning_rate", 3e-4)),
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        seed=seed,
    )

    model = PPO("MlpPolicy", env, **ppo_kwargs)
    model.learn(total_timesteps=total_timesteps)

    os.makedirs("models", exist_ok=True)
    out = os.path.join("models", "ppo_specialist_super_agent_final.zip")
    model.save(out)
    print(f"Training complete. Saved model to {out}")

if __name__ == "__main__":
    main()


