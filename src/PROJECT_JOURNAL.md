# Project Journal: MARL Trading System (Single‑Agent Baseline)

Date: 2025‑08‑31

Purpose
- Provide a complete, shareable description of the project: goals, architecture, data and modeling pipelines, environment mechanics, training and evaluation, current status, and roadmap to a hierarchical multi‑agent system.

Objective
- Build a professional‑grade trading research platform for a large U.S. equities universe, ultimately evolving into a hierarchical Multi‑Agent RL (MARL) system. The current milestone is a robust single‑agent PPO baseline that validates data, features, environment, reward, and backtesting at scale.

High‑Level Architecture
- Data ingestion: EODHD for historical OHLCV adjusted data; optional sentiment pipeline (NewsAPI + FinBERT) prepared but not yet integrated.
- Feature engineering: technical factors with consistent normalization; saved as Parquet.
- Caching: consolidated cross‑sectional arrays cached as NumPy memory‑mapped files for scale.
- Environment: custom `SingleAgentTradingEnv` with realistic constraints and costs.
- Training: RLlib PPO with reproducible seeds and periodic validation.
- Evaluation: deterministic backtest on test split; QuantStats summaries.
- Reports and artifacts: equity curve CSV/HTML, logs, and Ray checkpoints.

Repository Map (key items)
- `config.yaml`: single source of truth for universe, dates/splits, factors, reward/costs, actions/constraints, PPO settings, logging.
- `src/data_pipeline.py`: downloads EODHD data (threaded, with retries), cleans and saves raw CSVs.
- `src/feature_engineering.py`: computes factors per ticker, normalizes, writes Parquet (float32).
- `src/environment.py`: memory‑mapped environment; observation assembly; action mapping; constraints; reward; costs; split handling.
- `src/train.py`: PPO training loop with seeds, validation evaluation, and checkpointing.
- `src/backtester.py`: loads a checkpoint, runs test split, writes summary/QuantStats.
- `src/sentiment_pipeline.py`: NewsAPI fetch + FinBERT scoring; outputs daily sentiment (not yet merged into features).
- `src/fetch_data.py`: shim that proxies to `data_pipeline.run()` (legacy yfinance removed).
- `get_tickers.py`: utility to fetch S&P 500 symbols to `sp500_tickers.csv`.
- `requirements.txt`: pinned stack for reproducibility.
- `reports/`: backtest outputs (CSV, HTML, logs).

Configuration Contract (config.yaml)
- `universe`
  - `tickers`: either a list or a CSV path (e.g., `sp500_tickers.csv`).
  - `benchmark`: symbol (e.g., `SPY`), appended to the universe if not present.
- `data`
  - `start_date`, `end_date`, `price_frequency` (currently daily only).
- `eodhd`
  - `api_key_env`: environment variable name (default `EODHD_API_KEY`).
- `splits`
  - `train_end`, `val_end` (test is after `val_end`).
- `factors`
  - Example: `momentum_21d`, `rsi_14`, `vol_21d`, `atr_14`, `ma_fast_10`, `ma_slow_50`.
- `reward`
  - `lambda_tc_bps`: transaction cost basis points.
  - `lambda_sigma`: risk penalty weight.
  - `lambda_dd`: drawdown penalty weight.
  - `turnover_limit_bps`: hard turnover cap per step.
- `actions`
  - `allow_short`: currently not enabled; long‑only mapping is enforced.
  - `max_weight_per_name`: per‑asset cap (e.g., 0.10).
  - `cash_node`: last action dimension is cash.
- `training`
  - `algo`: `PPO` (RLlib).
  - `total_timesteps`, `n_envs`, `seed`, PPO hyperparams, `policy_kwargs` net sizes.
  - `multi_agent`: placeholder for future MARL; not used yet.
- `eval`
  - `eval_episodes`, `eval_freq_steps` (semantic intent), `metrics`.
- `storage`
  - `local_data_dir`: `data` (with `raw`, `processed`, `cache`).
- `logging`
  - `use_mlflow`, `mlflow_uri`, `use_tensorboard` (hooks ready; optional integration).

Data Ingestion (EODHD)
- `src/data_pipeline.py`
  - Reads `config.yaml`, resolves universe (list or CSV), and date range.
  - Pulls `ticker.US` series from EODHD with retry/backoff (tenacity), basic adjustments applied.
  - Writes `data/raw/{TICKER}.csv` with standardized columns: `date, Open, High, Low, Close, Adj Close, Volume`.
  - Rate limit controlled by `eodhd.rate_limit_per_sec` (external compliance recommended by user code).

Feature Engineering and Normalization
- `src/feature_engineering.py`
  - Converts columns to lowercase standardized names.
  - Uses `adj_close` and derived `returns` for return‑based signals.
  - Implements factors from config:
    - Momentum: pct change over window.
    - RSI: `ta.momentum.rsi`.
    - Volatility: rolling std of returns (annualized).
    - ATR: `ta.volatility.average_true_range` on unadjusted OHLC.
    - EMA fast/slow and ratio feature.
  - Normalization for learning stability:
    - RSI scaled to [0,1].
    - Others: 252‑day rolling z‑score (min_periods=20) then `tanh` clip to tame tails.
  - Outputs `data/processed/{TICKER}.parquet` (float32) after dropping NaN windows.

Memory‑Mapped Cache (for Scale)
- `src/environment.py` builds a consolidated panel:
  - Screens tickers for sufficient history (≥ 5 years by default).
  - Intersects dates across tickers; forward/back fills sparse points.
  - Splits into prices (`adj_close`) and feature tensor (all non‑OHLCV columns).
  - Saves to `data/cache/` as NumPy arrays with float32 dtypes and a text file of final tickers.
  - Loads with `mmap_mode='r'` for low RAM usage.

Trading Environment (`SingleAgentTradingEnv`)
- Observation
  - Market features: last `lookback_window` slices of all features across all assets, flattened.
  - Portfolio state: current weights (assets + cash).
- Action → weights mapping
  - Long‑only softmax mapping over assets + cash (numerically stable).
  - If `actions.allow_short: true`, current implementation warns and remains long‑only.
- Constraints and costs
  - Per‑asset cap via `actions.max_weight_per_name` and renormalization.
  - Cash node (`actions.cash_node`): ensures weights sum to 1 with cash as residual.
  - Turnover cap via `reward.turnover_limit_bps`: scales the move toward target to stay within cap.
  - Transaction costs via `reward.lambda_tc_bps` applied to actual turnover.
- Reward
  - Log return of portfolio value step‑to‑step.
  - Penalties: drawdown (lambda_dd) and weight dispersion proxy (lambda_sigma).
  - Hooks exist to replace dispersion with more principled risk later (e.g., volatility or exposure control).
- Splits
  - Train/validation/test split by dates in `config.yaml`.

Training (`src/train.py`)
- RLlib PPO with Torch backend.
- Config‑driven hyperparameters and network sizes (`policy_kwargs`).
- Reproducibility: sets Python/NumPy/Torch seeds and RLlib seed.
- Validation evaluation every training iteration (episodes from `eval.eval_episodes`) on `mode=validation` with `explore=False`.
- Trains until `training.total_timesteps`; saves checkpoint and shuts down Ray.

Backtesting (`src/backtester.py`)
- Loads a checkpoint into the env in `mode=test`.
- Steps deterministically with `compute_single_action(..., explore=False)`.
- Outputs
  - `reports/backtest_equity_curve.csv`
  - `reports/summary.txt` with Total/Annualized Return, Volatility, Sharpe, Max Drawdown
  - `reports/quantstats_report.html` if QuantStats is installed

Sentiment Pipeline (Prepared, Optional)
- `src/sentiment_pipeline.py`
  - Universe from `config.yaml`.
  - Fetches daily articles via NewsAPI with retry and rate limiting.
  - Scores with FinBERT; writes `data/sentiment/{TICKER}_sentiment.csv` of daily scores.
  - Not yet merged into `feature_engineering.py`; integration planned (left‑join on date, normalization, leakage‑safe alignment).

How to Run (End‑to‑End)
1) Setup
   - `pip install -r requirements.txt`
   - Add `.env` with `EODHD_API_KEY=...` (rotate if exposed; never commit).
   - Optionally generate `sp500_tickers.csv` via `python get_tickers.py` and point `config.yaml:universe.tickers` to it.
2) Data
   - `python -m src.data_pipeline`
   - `python -m src.feature_engineering`
3) Train
   - `python -m src.train`
4) Backtest
   - `python -m src.backtester --checkpoint_dir <checkpoint_dir>`

Reproducibility & Logging
- Pinned versions in `requirements.txt` (notably RLlib, NumPy, PyArrow).
- Seeds applied across stacks; training writes Ray checkpoints.
- TensorBoard supported by RLlib; MLflow hooks available via config (optional wiring).

Security & Keys
- `.env` is git‑ignored. Avoid sharing keys in notebooks/logs. Rotate if exposure is suspected.

Current Status (as of 2025‑08‑31)
- End‑to‑end single‑agent pipeline is functional at S&P‑500 scale.
- Environment updated with action constraints, turnover cap, and float32 memory‑mapped cache.
- Training honors config timesteps; evaluation and seeding added.
- Features normalized for learning stability; fetcher unified under EODHD.

Known Limitations
- Shorting not yet supported in the action mapping; planned with borrow costs and proper projection.
- Slippage and execution delay not modeled; to be added for realism.
- Multi‑agent configuration fields are placeholders; single‑agent PPO only.
- Sentiment features are not yet merged into the main feature set.

Roadmap
- Quant research loop
  - Continue feature and reward shaping iterations; stress test turnover and cost sensitivity.
  - Add benchmark alignment and sector/industry context features.
- Realism
  - Add slippage, borrow costs, and T+1 fills; exposure and leverage constraints.
- Sentiment integration
  - Join daily sentiment with features; evaluate impact on validation metrics.
- Multi‑Agent (Phase 2)
  - Introduce specialist policies (technical, sentiment, risk) and a super‑agent allocator via RLlib multi‑agent API.
  - Stage training: pretrain specialists → joint finetune; consider centralized critic.
- MLOps
  - Optional MLflow integration; simple CLI/Makefile for pipeline runs; experiment tracking and artifact logging.
- Testing
  - Unit tests for factor correctness, cache shapes/dtypes, env invariants (weights sum to 1, turnover/costs), and tiny smoke‑train.

Contact & Contributions
- Issues/PRs welcome. Please avoid committing data, models, or secrets.
