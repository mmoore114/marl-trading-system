# src/backtester.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO

from src.environment import MultiStrategyEnv


def perf(returns: pd.Series) -> Dict[str, float]:
    returns = returns.fillna(0.0)
    cum = (1 + returns).cumprod()
    n = len(returns)
    if n == 0:
        return dict(CAGR=np.nan, Sharpe=np.nan, Sortino=np.nan, MaxDD=np.nan, Calmar=np.nan)
    cagr = cum.iloc[-1] ** (252 / n) - 1.0
    sharpe = (returns.mean() / (returns.std() + 1e-12)) * np.sqrt(252)
    downside = returns[returns < 0].std() + 1e-12
    sortino = (returns.mean() / downside) * np.sqrt(252)
    dd = ((cum.cummax() - cum) / cum.cummax()).max()
    calmar = np.nan if dd == 0 else cagr / dd
    return dict(CAGR=cagr, Sharpe=sharpe, Sortino=sortino, MaxDD=dd, Calmar=calmar)


def run():
    # Load config for baseline data
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    proc_dir = Path(cfg["storage"]["local_data_dir"]) / "processed"

    # Agent env (test split)
    env = MultiStrategyEnv(mode="test")
    model = PPO.load("models/ppo_specialist_super_agent_final.zip")

    obs, _ = env.reset()
    done = False
    rewards: List[float] = []
    dates: List[pd.Timestamp] = []

    # Roll out through test period
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        dates.append(env.dates[env.t - 1])
        done = bool(terminated or truncated)

    agent_ret = pd.Series(rewards, index=pd.Index(dates, name="date"))

    # Build equal-weight baseline from processed panel on test dates
    frames = []
    for fp in sorted(proc_dir.glob("*.parquet")):
        df = pd.read_parquet(fp, columns=["date", "ret_1d"]).assign(symbol=fp.stem)
        frames.append(df)
    if not frames:
        # Fallback to *_processed.csv if no parquet present
        for fp in sorted(proc_dir.glob("*_processed.csv")):
            df = pd.read_csv(fp, usecols=["date", "ret_1d"]).assign(symbol=fp.stem.replace("_processed", ""))
            frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"].isin(agent_ret.index)]
    ew = panel.groupby("date")["ret_1d"].mean().reindex(agent_ret.index).fillna(0.0)

    # Metrics
    tbl = pd.DataFrame(
        {
            "Metric": ["CAGR", "Sharpe", "Sortino", "MaxDD", "Calmar"],
            "Agent": list(perf(agent_ret).values()),
            "EqualWeight": list(perf(ew).values()),
        }
    )
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    run()
