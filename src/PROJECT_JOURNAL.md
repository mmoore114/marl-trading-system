# Project Journal: MARL Trading System

## Project Objective
To develop a sophisticated, multi-agent reinforcement learning (MARL) system to manage a portfolio of equities, based on technical and alternative data factors.

---
## Current Status (As of August 16, 2025)

The project has successfully evolved from a single-agent prototype to a robust, factor-driven, multi-agent research platform. The system is end-to-end functional.

### Key Architectural Components:
* **Data Pipeline**: Downloads raw EOD data for 15 stocks using `yfinance`.
* **Factor Engine**: A config-driven script (`feature_engineering.py`) that calculates a suite of technical factors, including **RSI, MACD, EMAs, Bollinger Bands, ATR, OBV**, and the **Hurst Exponent**.
* **Regime Detection**: The factor engine now includes a **Hidden Markov Model (HMM)** to train and predict market regimes for each stock, adding regime probabilities as features.
* **MARL Environment**: A custom `MultiStrategyEnv` that simulates a realistic portfolio with a **train/validation/test split**, and an **advanced risk-aware reward function** that penalizes drawdown and portfolio turnover.
* **Training Pipeline**: A `train.py` script capable of **hyperparameter tuning** (e.g., testing multiple learning rates) and using an `EvalCallback` to automatically save the best-performing model on the validation set.
* **Backtester**: A `backtester.py` script that evaluates the final agent on unseen test data and generates a `quantstats` report comparing its performance to an **equal-weight buy-and-hold benchmark**.

### Key Findings & Conclusion:
* The system is technically robust and the training process is stable.
* The current agent, trained on technical factors alone, has learned a conservative, risk-managed strategy but **does not consistently outperform its benchmark**.
* Multiple experiments in tuning the reward function and learning rate have confirmed that we have likely reached the performance limit of the current feature set.
* **Conclusion**: The project is now **data-constrained**. The next major phase of development requires an upgrade to a professional data source to integrate higher-quality fundamental and news sentiment data.

### Next Steps:
1.  **Upgrade to EODHD Paid Data Plan.**
2.  Refactor the data pipeline scripts to use the new EODHD API.
3.  Integrate fundamental and news sentiment data into the factor engine.
4.  Re-train and evaluate the agent with this new, richer dataset.

---
## (Previous Journal Entries)

### Folder & File Structure
... (The rest of your journal file remains the same) ...