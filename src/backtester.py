import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from src.environment import SingleAgentTradingEnv

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Run backtest with a trained RLlib agent.")
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Path to the Ray RLlib checkpoint directory.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_dir)
    if not checkpoint_path.is_dir():
        logger.error(f"Checkpoint directory not found: {checkpoint_path}")
        return

    # Register the custom environment so Ray knows how to create it
    register_env(
        "SingleAgentTradingEnv-v0",
        lambda env_config: SingleAgentTradingEnv(env_config),
    )

    logger.info(f"Loading agent from checkpoint: {checkpoint_path}")
    agent = Algorithm.from_checkpoint(checkpoint_path)

    logger.info("Initializing test environment...")
    env = SingleAgentTradingEnv(env_config={"mode": "test"})
    
    obs, info = env.reset()
    terminated = truncated = False
    
    portfolio_values = [env.portfolio_value]
    dates = [info["date"]]
    steps = 0

    logger.info("--- Starting Backtest ---")
    while not terminated and not truncated:
        action = agent.compute_single_action(observation=obs, explore=False)
        obs, reward, terminated, truncated, info = env.step(action)
        
        portfolio_values.append(env.portfolio_value)
        dates.append(info["date"])
        steps += 1
        
        if steps % 100 == 0:
            logger.info(f"Step: {steps}, Date: {info['date'].strftime('%Y-%m-%d')}, Portfolio Value: ${env.portfolio_value:,.2f}")

    logger.info(f"--- Backtest Complete ---")
    logger.info(f"Finished after {steps} steps. Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
    
    # --- Reporting ---
    equity_curve = pd.Series(portfolio_values, index=pd.to_datetime(dates), name="equity")
    returns = equity_curve.pct_change().dropna()
    
    if returns.empty:
        logger.warning("No returns were generated, cannot create reports.")
        return

    equity_curve.to_csv(REPORTS_DIR / "backtest_equity_curve.csv")
    logger.info(f"Saved equity curve to: {REPORTS_DIR / 'backtest_equity_curve.csv'}")

    # --- FIX: Calculate metrics robustly ---
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    annualized_return = qs.stats.annualized_return(returns) if QUANTSTATS_AVAILABLE else (1 + total_return) ** (252 / len(equity_curve)) - 1
    annualized_vol = qs.stats.volatility(returns) if QUANTSTATS_AVAILABLE else returns.std() * np.sqrt(252)
    sharpe_ratio = qs.stats.sharpe(returns) if QUANTSTATS_AVAILABLE else annualized_return / annualized_vol
    max_drawdown = qs.stats.max_drawdown(returns) if QUANTSTATS_AVAILABLE else (equity_curve / equity_curve.cummax() - 1).min()

    summary_txt = f"""
    --- Backtest Summary ---
    Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}
    
    Final Portfolio Value: ${equity_curve.iloc[-1]:,.2f}
    Total Return: {total_return:.2%}
    Annualized Return: {annualized_return:.2%}
    Annualized Volatility: {annualized_vol:.2%}
    Sharpe Ratio: {sharpe_ratio:.2f}
    Max Drawdown: {max_drawdown:.2%}
    """
    print(summary_txt)
    (REPORTS_DIR / "summary.txt").write_text(summary_txt)
    logger.info(f"Saved summary to: {REPORTS_DIR / 'summary.txt'}")

    if QUANTSTATS_AVAILABLE:
        report_path = str(REPORTS_DIR / "quantstats_report.html")
        qs.reports.html(returns, output=report_path, title="Single Agent Baseline Backtest")
        logger.info(f"Saved QuantStats HTML report to: {report_path}")
    else:
        logger.warning("QuantStats not installed. Skipping HTML report generation.")


if __name__ == "__main__":
    main()

