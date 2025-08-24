# src/backtester.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO

from src.environment import MultiStrategyEnv


def _perf(returns: pd.Series) -> Dict[str, float]:
    r = returns.fillna(0.0)
    if len(r) == 0:
        return dict(CAGR=np.nan, Sharpe=np.nan, Sortino=np.nan, MaxDD=np.nan, Calmar=np.nan)
    equity = (1 + r).cumprod()
    n = len(r)
    cagr = equity.iloc[-1] ** (252 / n) - 1.0
    sharpe = (r.mean() / (r.std() + 1e-12)) * np.sqrt(252)
    downside = r[r < 0].std() + 1e-12
    sortino = (r.mean() / downside) * np.sqrt(252)
    dd = ((equity.cummax() - equity) / equity.cummax()).max()
    calmar = np.nan if dd == 0 else cagr / dd
    return dict(CAGR=cagr, Sharpe=sharpe, Sortino=sortino, MaxDD=dd, Calmar=calmar)


def run():
    # Ensure model exists
    model_path = Path("models/ppo_specialist_super_agent_final.zip")
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run training first.")

    # Create test env (parquet-only loader inside)
    env = MultiStrategyEnv(mode="test")
    model = PPO.load(str(model_path))

    obs, _ = env.reset()
    done = False
    rewards: List[float] = []
    dates: List[pd.Timestamp] = []

    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        dates.append(env.dates[env.t - 1])
        done = bool(terminated or truncated)

    # Agent daily returns (already risk-penalized reward)
    agent_ret = pd.Series(rewards, index=pd.Index(dates, name="date"))

    # Equal-weight baseline using the same dates & universe
    proc_dir = Path(env.cfg["storage"]["local_data_dir"]) / "processed"
    frames = []
    for sym in env.symbols:
        fp = proc_dir / f"{sym}.parquet"
        if not fp.exists():
            raise FileNotFoundError(f"Missing {fp}. Run feature_engineering first.")
        df = pd.read_parquet(fp, columns=["date", "ret_1d"]).assign(symbol=sym)
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    ew = (
        panel[panel["date"].isin(agent_ret.index)]
        .groupby("date")["ret_1d"]
        .mean()
        .reindex(agent_ret.index)
        .fillna(0.0)
    )

    # Metrics table
    agent_m = _perf(agent_ret)
    ew_m = _perf(ew)
    tbl = pd.DataFrame(
        {
            "Metric": ["CAGR", "Sharpe", "Sortino", "MaxDD", "Calmar"],
            "Agent": [agent_m["CAGR"], agent_m["Sharpe"], agent_m["Sortino"], agent_m["MaxDD"], agent_m["Calmar"]],
            "EqualWeight": [ew_m["CAGR"], ew_m["Sharpe"], ew_m["Sortino"], ew_m["MaxDD"], ew_m["Calmar"]],
        }
    )
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    run()

