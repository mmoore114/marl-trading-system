# src/backtester.py
from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from src.environment import MultiStrategyEnv


def run():
    models_dir = Path("models")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "ppo_specialist_super_agent_final.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run training first.")

    # Create test env (single env is fine for evaluation)
    env = MultiStrategyEnv(mode="test")

    # Load trained policy
    model = PPO.load(model_path, device="cpu")  # swap to 'cuda' later on GCP if desired

    obs, _info = env.reset()
    done = False
    equity = [float(env.portfolio_value)]
    t_idx = 0

    # We’ll keep the actual trading dates from env
    dates = [env.dates[0]]

    while not done:
        # Deterministic policy during evaluation
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Track equity curve & date
        equity.append(float(env.portfolio_value))
        t_idx = min(env.t, len(env.dates) - 1)
        dates.append(env.dates[t_idx])

        done = bool(terminated or truncated)

    # Build results frame
    equity = np.array(equity, dtype=np.float64)
    res = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "equity": equity,
        }
    ).drop_duplicates(subset=["date"]).set_index("date").sort_index()

    res["ret"] = res["equity"].pct_change().fillna(0.0)

    # Save CSVs
    out_csv = reports_dir / "backtest_equity_curve.csv"
    res.to_csv(out_csv, index=True)
    print(f"[backtest] Saved equity curve -> {out_csv}")

    # Basic summary stats
    total_return = float(res["equity"].iloc[-1] / max(res["equity"].iloc[0], 1e-12) - 1.0)
    ann_factor = 252  # daily bars
    vol = float(res["ret"].std() * np.sqrt(ann_factor))
    sharpe = float((res["ret"].mean() * ann_factor) / vol) if vol > 0 else 0.0
    max_dd = 0.0
    if len(res) > 0:
        running_max = res["equity"].cummax()
        dd = (running_max - res["equity"]) / running_max.replace(0, np.nan)
        max_dd = float(dd.max(skipna=True))

    print("[backtest] Summary")
    print(f"  Total Return: {total_return: .2%}")
    print(f"  Ann. Vol:     {vol: .2%}")
    print(f"  Sharpe~:      {sharpe: .2f}")
    print(f"  Max Drawdown: {max_dd: .2%}")

    # Optional: QuantStats HTML report
    try:
        import quantstats as qs

        # Quantstats expects a return series indexed by date
        ret_ser = res["ret"].copy()
        ret_ser.index = pd.to_datetime(ret_ser.index)

        html_path = reports_dir / "quantstats_report.html"
        # silent=True to avoid opening a browser; title shows ticker universe size
        qs.reports.html(
            ret_ser,
            output=html_path.as_posix(),
            title=f"MARL PPO Backtest (N={env.n_assets})",
            compounded=True,
            download_filename=html_path.name,
            benchmark=None,  # could add SPY ret series later
            rf=0.0,
            grayscale=True,
            figfmt="svg",
            template="template.html" if (reports_dir / "template.html").exists() else None,
        )
        print(f"[backtest] Saved QuantStats report -> {html_path}")
    except Exception as e:
        print(f"[backtest] QuantStats report skipped: {e}")

    # Also drop a simple TXT summary
    summary_txt = reports_dir / "summary.txt"
    with open(summary_txt, "w") as f:
        f.write("MARL PPO Backtest Summary\n")
        f.write("=========================\n")
        f.write(f"Bars:          {len(res)}\n")
        f.write(f"Start:         {res.index[0].date() if len(res) else 'n/a'}\n")
        f.write(f"End:           {res.index[-1].date() if len(res) else 'n/a'}\n")
        f.write(f"Total Return:  {total_return:.4%}\n")
        f.write(f"Ann. Vol:      {vol:.4%}\n")
        f.write(f"Sharpe~:       {sharpe:.2f}\n")
        f.write(f"Max Drawdown:  {max_dd:.4%}\n")
    print(f"[backtest] Saved summary -> {summary_txt}")


if __name__ == "__main__":
    run()

