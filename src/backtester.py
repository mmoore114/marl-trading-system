import argparse
import yaml
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import torch

# --- FIX: Import the correct Ray RLlib Algorithm class ---
from ray.rllib.algorithms.algorithm import Algorithm

from src.environment import TradingEnv

# --- FIX: Use QuantStats if available ---
try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FIX: Use robust pathing ---
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
    # --- FIX: Load agent using Algorithm.from_checkpoint ---
    agent = Algorithm.from_checkpoint(checkpoint_path)

    logger.info("Initializing test environment...")
    # --- FIX: Pass env_config correctly ---
    env = TradingEnv(env_config={"mode": "test"})
    
    # --- FIX: Get all RL Modules for each policy ---
    modules = {policy_id: agent.get_policy(policy_id).model for policy_id in env.action_space.keys()}

    obs, info = env.reset()
    terminated = {"__all__": False}
    
    portfolio_values = [env.portfolio_value]
    steps = 0

    logger.info("--- Starting Backtest ---")
    while not terminated["__all__"]:
        actions = {}
        # --- FIX: New action computation loop using RL Modules ---
        for agent_id, agent_obs in obs.items():
            # Convert observation to a batch of 1 and a torch tensor
            obs_tensor = torch.from_numpy(np.expand_dims(agent_obs, axis=0)).float()
            
            # Use the new forward_inference method
            action_dist_inputs, _ = modules[agent_id].forward_inference({"obs": obs_tensor})
            
            # Convert action back to numpy and remove the batch dimension
            actions[agent_id] = action_dist_inputs.cpu().numpy()[0]

        obs, rewards, terminated, truncated, info = env.step(actions)
        
        portfolio_values.append(env.portfolio_value)
        steps += 1
        if steps % 252 == 0: # Log roughly once per trading year
            logger.info(f"Step: {steps}, Portfolio Value: ${env.portfolio_value:,.2f}")

    logger.info(f"--- Backtest Complete ---")
    logger.info(f"Finished after {steps} steps. Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
    
    # --- Reporting ---
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

