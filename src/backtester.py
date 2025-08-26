import argparse
import yaml
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import torch

from ray.rllib.algorithms.algorithm import Algorithm
from src.environment import TradingEnv

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
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_path}")
        return

    logger.info(f"Loading agent from checkpoint: {checkpoint_path}")
    agent = Algorithm.from_checkpoint(checkpoint_path)

    logger.info("Initializing test environment...")
    env = TradingEnv(env_config={"mode": "test"})
    
    modules = {module_id: agent.get_module(module_id) for module_id in env.action_space.keys()}

    obs, info = env.reset()
    terminated = {"__all__": False}
    
    portfolio_values = [env.portfolio_value]
    steps = 0

    logger.info("--- Starting Backtest ---")
    while not terminated["__all__"]:
        actions = {}
        for agent_id, agent_obs in obs.items():
            obs_tensor = torch.from_numpy(np.expand_dims(agent_obs, axis=0)).float()
            
            output_dict = modules[agent_id].forward_inference({"obs": obs_tensor})
            raw_action = output_dict['action_dist_inputs'].cpu().numpy()[0]
            
            # --- FIX: Extract only the 'mean' part of the action distribution ---
            num_actions = env.action_space[agent_id].shape[0]
            actions[agent_id] = raw_action[:num_actions]

        obs, rewards, terminated, truncated, info = env.step(actions)
        
        portfolio_values.append(env.portfolio_value)
        steps += 1
        if steps % 252 == 0:
            logger.info(f"Step: {steps}, Portfolio Value: ${env.portfolio_value:,.2f}")

    logger.info(f"--- Backtest Complete ---")
    logger.info(f"Finished after {steps} steps. Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
    
    equity_curve = pd.Series(portfolio_values, index=pd.to_datetime(env.dates[:len(portfolio_values)]))
    returns = equity_curve.pct_change().dropna()
    
    equity_curve.to_csv(REPORTS_DIR / "backtest_equity_curve.csv", header=['equity'])
    logger.info(f"Saved equity curve to: {REPORTS_DIR / 'backtest_equity_curve.csv'}")

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(equity_curve)) - 1 if len(equity_curve) > 0 else 0
    annualized_vol = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
    max_drawdown = (equity_curve / equity_curve.cummax() - 1).min()

    summary_txt = f"""
    --- Backtest Summary ---
    Period: {env.dates[0].strftime('%Y-%m-%d')} to {env.dates[-1].strftime('%Y-%m-%d')}
    
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
        qs.reports.html(returns, output=report_path, title="MARL Agent Backtest")
        logger.info(f"Saved QuantStats HTML report to: {report_path}")
    else:
        logger.warning("QuantStats not installed. Skipping HTML report generation.")


if __name__ == "__main__":
    main()

