from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import logging
from stable_baselines3 import PPO
from ray.rllib.policy.policy import PolicySpec  # For MARL if needed

try:
    import quantstats as qs
except ImportError:
    qs = None

from src.environment import TradingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _ensure_reports_dir() -> Path:
    d = Path("reports")
    d.mkdir(parents=True, exist_ok=True)
    return d

def _load_model(model_path: Path) -> PPO:
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    logger.info(f"Loading model: {model_path}")
    return PPO.load(str(model_path), device="cpu")

def main():
    parser = argparse.ArgumentParser(description="Run backtest with trained PPO model.")
    parser.add_argument("--model", default="models/ppo_specialist_super_agent_final.zip", help="SB3 .zip path.")
    parser.add_argument("--max-steps", type=int, default=250000, help="Step cap.")
    parser.add_argument("--progress-every", type=int, default=500, help="Log every N steps.")
    parser.add_argument("--write-every", type=int, default=250, help="Save CSV every N steps.")
    args = parser.parse_args()

    reports_dir = _ensure_reports_dir()

    model_path = Path(args.model)
    model = _load_model(model_path)
    env = TradingEnv(mode="test")

    obs = env.reset()[0]  # obs dict
    rewards = []
    equities = []
    steps = 0

    tmp_eq_path = reports_dir / "backtest_equity_curve.tmp.csv"
    final_eq_path = reports_dir / "backtest_equity_curve.csv"
    summary_path = reports_dir / "summary.txt"

    logger.info("Running backtest...")

    try:
        while True:
            actions = {a: model.predict(obs[a], deterministic=True)[0] for a in env.agents}
            obs, reward, terminated, truncated, info = env.step(actions)
            rewards.append(list(reward.values())[0])  # Assume shared reward

            eq = env._get_portfolio_value(np.array([env.frames[s].iloc[env.current_step]['Close'] for s in env.sym_list]))
            if eq is not None:
                equities.append(eq)

            steps += 1

            if steps % args.progress_every == 0:
                logger.info(f"Progress: {steps} steps")

            if steps % args.write_every == 0 and equities:
                pd.Series(equities, name="equity").to_csv(tmp_eq_path, index=False)

            if any(terminated.values()):
                logger.info(f"Episode finished after {steps} steps.")
                break

            if steps >= args.max_steps:
                logger.info(f"Reached max_steps={args.max_steps}.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted. Saving outputs...")

    # Equity series
    if equities:
        equity_series = pd.Series(equities, name="equity")
    else:
        rets = pd.Series(rewards, name="ret").fillna(0.0)
        equity_series = (1.0 + rets).cumprod().rename("equity")

    equity_series.to_csv(final_eq_path, header=True, index=False)
    logger.info(f"Saved equity curve: {final_eq_path}")

    if tmp_eq_path.exists():
        tmp_eq_path.unlink()

    # Summary
    try:
        rets = equity_series.pct_change().fillna(0.0)
        total_return = (1 + rets).prod() - 1
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = (rets.mean() * 252) / (ann_vol + 1e-12)
        dd = (equity_series / equity_series.cummax() - 1.0).min()

        summary_txt = f"""
[backtest] Summary
  Steps: {steps}
  Total Return: {total_return * 100:.2f}%
  Ann. Vol: {ann_vol * 100:.2f}%
  Sharpe: {sharpe:.2f}
  Max Drawdown: {abs(dd) * 100:.2f}%
        """
        print(summary_txt)
        summary_path.write_text(summary_txt)
        logger.info(f"Saved summary: {summary_path}")
    except Exception as e:
        logger.error(f"Summary failed: {e}")

    # QuantStats
    if qs is not None:
        try:
            rets = equity_series.pct_change().fillna(0.0)
            rets.name = "strategy"
            report_path = reports_dir / "quantstats_report.html"
            qs.reports.html(rets, output=str(report_path))
            logger.info(f"Saved QuantStats report: {report_path}")
        except Exception as e:
            logger.error(f"QuantStats failed: {e}")
    else:
        logger.info("quantstats not installed; skipped HTML report.")

if __name__ == "__main__":
    main()


