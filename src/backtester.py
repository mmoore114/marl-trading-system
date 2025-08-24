"""
Backtester (Gymnasium/Gym compatible)

- Loads the trained model
- Runs on MultiStrategyEnv(mode="test")
- Normalizes env.reset() / env.step() returns for Gymnasium vs Gym
- Saves equity curve CSV and a QuantStats HTML report
"""

from __future__ import annotations

import os
import glob
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# SB3
from stable_baselines3 import PPO

# Reporting
import quantstats as qs

# Your trading env
from src.environment import MultiStrategyEnv


# ---------- Helpers to normalize Gym vs Gymnasium API ----------

def reset_compat(env) -> np.ndarray:
    """
    Gymnasium: obs, info = env.reset()
    Gym:       obs = env.reset()
    """
    out = env.reset()
    if isinstance(out, tuple) and len(out) >= 1:
        obs = out[0]
    else:
        obs = out
    return obs


def step_compat(env, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    """
    Returns: obs, reward, done, info

    Gymnasium: obs, reward, terminated, truncated, info
               done := terminated or truncated
    Gym:       obs, reward, done, info
    """
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, float(reward), done, info
    elif isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward), bool(done), info
    else:
        raise RuntimeError(
            f"Unexpected step() return: type={type(out)} len={len(out) if isinstance(out, tuple) else 'n/a'}"
        )


# ---------- Model loading ----------

def _find_latest_model(models_dir: Path) -> Path | None:
    zips = sorted(models_dir.glob("*.zip"))
    return zips[-1] if zips else None


def _load_model(models_dir: Path) -> PPO:
    preferred = models_dir / "ppo_specialist_super_agent_final.zip"
    ckpt = preferred if preferred.exists() else _find_latest_model(models_dir)
    if ckpt is None:
        raise FileNotFoundError(
            f"No model checkpoint found in {models_dir}. "
            f"Expected {preferred.name} or any *.zip file."
        )
    print(f"[backtest] Loading model -> {ckpt}")
    return PPO.load(str(ckpt), device="cpu")


# ---------- Backtest loop ----------

def run() -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    env = MultiStrategyEnv(mode="test")
    model = _load_model(Path("models"))

    obs = reset_compat(env)
    done = False

    dates: list[pd.Timestamp] = []
    equity: list[float] = []

    # Start NAV at 1.0 and compound daily PnL (assuming reward ~ daily return)
    nav = 1.0

    step_counter = 0
    while not done:
        # SB3 expects observation as np.ndarray (or dict), not tuple
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = step_compat(env, action)

        # Try to fetch a timestamp to index the equity curve
        dt = (
            info.get("date")
            if isinstance(info, dict)
            else None
        )
        # Fallbacks if env supplies other date keys or properties
        if dt is None and isinstance(info, dict):
            dt = info.get("timestamp") or info.get("t") or info.get("bar_time")
        if dt is None and hasattr(env, "current_date"):
            dt = getattr(env, "current_date")
        if dt is None and hasattr(env, "dates"):
            # If env exposes full date index and a pointer
            try:
                cur_idx = getattr(env, "t", None)
                if cur_idx is not None and 0 <= cur_idx < len(env.dates):
                    dt = env.dates[cur_idx]
            except Exception:
                pass

        # Normalize dt into pandas Timestamp or use step index
        try:
            dt_ts = pd.Timestamp(dt)
        except Exception:
            dt_ts = pd.NaT

        # Assume reward is daily portfolio return (e.g., +0.004 = +0.4%)
        if not (reward is None or (isinstance(reward, float) and (math.isnan(reward) or math.isinf(reward)))):
            nav *= (1.0 + float(reward))

        dates.append(dt_ts if not pd.isna(dt_ts) else pd.NaT)
        equity.append(nav)

        step_counter += 1

    # Build equity curve DataFrame
    eq = pd.DataFrame({"date": dates, "equity": equity})
    # If dates are NaT (unknown), fall back to integer index but keep a date column
    if eq["date"].isna().all():
        eq["date"] = pd.RangeIndex(start=0, stop=len(eq), step=1)

    # Drop potential duplicates and ensure sorted by date/index
    try:
        eq = eq.drop_duplicates(subset=["date"]).sort_values("date")
    except Exception:
        eq = eq.reset_index(drop=True)

    csv_path = reports_dir / "backtest_equity_curve.csv"
    eq.to_csv(csv_path, index=False)
    print(f"[backtest] Saved equity curve -> {csv_path}")

    # Summary stats (simple)
    returns = eq["equity"].pct_change().fillna(0.0)
    total_return = (eq["equity"].iloc[-1] - 1.0) * 100.0 if len(eq) > 0 else 0.0
    ann_vol = returns.std() * np.sqrt(252) * 100.0 if len(eq) > 1 else 0.0
    sharpe = (
        (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        if returns.std() > 0
        else np.nan
    )
    max_dd = (
        (eq["equity"] / eq["equity"].cummax() - 1.0).min() * 100.0
        if len(eq) > 0
        else 0.0
    )

    print("[backtest] Summary")
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Ann. Vol:      {ann_vol:.2f}%")
    print(f"  Sharpe~:       {sharpe:.2f}" if not np.isnan(sharpe) else "  Sharpe~:       n/a")
    print(f"  Max Drawdown:  {abs(max_dd):.2f}%")

    # Save text summary
    (reports_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"Total Return: {total_return:.2f}%",
                f"Ann. Vol:     {ann_vol:.2f}%",
                f"Sharpe~:      {sharpe:.2f}" if not np.isnan(sharpe) else "Sharpe~: n/a",
                f"Max Drawdown: {abs(max_dd):.2f}%",
            ]
        ),
        encoding="utf-8",
    )

    # QuantStats HTML report (uses equity curve to compute returns)
    try:
        # Convert equity curve to price series (index=Datetime, values=equity)
        s = eq.set_index("date")["equity"].copy()
        # If index is not datetime-like, QS can still work, but let's try to coerce:
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            pass

        # Derive returns from equity
        rets = s.pct_change().fillna(0.0)
        rets.name = "strategy"

        report_path = reports_dir / "quantstats_report.html"
        qs.reports.html(rets, output=str(report_path), title="Backtest Report")
        print(f"[backtest] Saved QuantStats report -> {report_path}")
    except Exception as e:
        print(f"[backtest] QuantStats report skipped: {e}")


if __name__ == "__main__":
    run()
