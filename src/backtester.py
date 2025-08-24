import os
import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO

from src.environment import MultiStrategyEnv, PROC_DIR

def read_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def _load_px(sym: str) -> pd.DataFrame:
    fp = os.path.join(PROC_DIR, f"{sym}.parquet")
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    df = pd.read_parquet(fp)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")

def equity_from_returns(r: np.ndarray, start=1.0) -> np.ndarray:
    eq = np.cumprod(1.0 + np.nan_to_num(r, nan=0.0)) * start
    return eq

def summary_stats(equity: pd.Series):
    ret = equity.pct_change().fillna(0.0)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    ann_vol = ret.std() * np.sqrt(252.0)
    sharpe = (ret.mean() * 252.0) / (ann_vol + 1e-12)
    roll_max = equity.cummax()
    dd = 1.0 - (equity / (roll_max + 1e-12))
    mdd = dd.max()
    return total_return, ann_vol, sharpe, mdd

def run():
    cfg = read_config()
    model_path = os.path.join("models", "ppo_specialist_super_agent_final.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Run training first.")

    env = MultiStrategyEnv(mode="test")
    model = PPO.load(model_path)

    obs = env.reset()
    agent_rets = []
    dates = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        agent_rets.append(info["port_ret"])
        dates.append(pd.to_datetime(env.dates[env.t]))

    # Agent equity
    agent_eq = equity_from_returns(np.array(agent_rets), start=1.0)

    # Bench equity (SPY)
    bench_px = _load_px(cfg["universe"]["benchmark"])["adj_close"]
    bench_px = bench_px[bench_px.index.isin(pd.Index(np.arange(len(bench_px))))]  # keep as-is
    # align by dates used in env
    bench_df = _load_px(cfg["universe"]["benchmark"])
    bench_df = bench_df[bench_df["date"].isin(env.dates)].reset_index(drop=True)
    bench_eq = bench_df["adj_close"] / bench_df["adj_close"].iloc[0]

    # Equal-weight baseline on universe
    # Build returns aligned to env.dates
    uni = cfg["universe"]["tickers"]
    mats = []
    for sym in uni:
        df = _load_px(sym)
        df = df[df["date"].isin(env.dates)].reset_index(drop=True)
        r = df["adj_close"].pct_change().fillna(0.0).values
        mats.append(r)
    mats = np.array(mats)  # shape (A, T)
    ew_ret = mats.mean(axis=0)
    ew_eq = equity_from_returns(ew_ret, start=1.0)

    # Build report frame
    out_dates = pd.Series(env.dates[1:1+len(agent_eq)], name="date")
    rpt = pd.DataFrame({
        "date": out_dates,
        "agent": agent_eq,
        "equal_weight": ew_eq[:len(agent_eq)],
        "benchmark": bench_eq.values[:len(agent_eq)],
    })
    os.makedirs("reports", exist_ok=True)
    csv_path = os.path.join("reports", "backtest_equity_curve.csv")
    rpt.to_csv(csv_path, index=False)
    print(f"[backtest] Saved equity curve -> {csv_path}")

    # Summary
    a_tr, a_vol, a_sh, a_mdd = summary_stats(rpt["agent"])
    print("[backtest] Summary")
    print(f"  Total Return: {a_tr*100:.2f}%")
    print(f"  Ann. Vol:      {a_vol*100:.2f}%")
    print(f"  Sharpe~:      {a_sh:.2f}")
    print(f"  Max Drawdown:  {a_mdd*100:.2f}%")

    with open(os.path.join("reports","summary.txt"), "w") as f:
        f.write("Agent (test split)\n")
        f.write(f"Total Return: {a_tr*100:.2f}%\n")
        f.write(f"Ann. Vol:     {a_vol*100:.2f}%\n")
        f.write(f"Sharpe~:      {a_sh:.2f}\n")
        f.write(f"Max DD:       {a_mdd*100:.2f}%\n")
    print("[backtest] Saved summary -> reports\\summary.txt")

if __name__ == "__main__":
    run()

