MARL Trading System — Single-Agent Baseline
===========================================

Overview
- End-to-end pipeline for equities MARL research: data → features → env → train → backtest.
- Current implementation uses a single PPO agent with a scalable, memory-mapped environment.

Quickstart
- Install deps: `pip install -r requirements.txt`
- Set API key: create `.env` with `EODHD_API_KEY=...`
- Configure: edit `config.yaml` (universe, dates, factors, training)

Run Pipeline
- Download raw data: `python -m src.data_pipeline`
- Generate features: `python -m src.feature_engineering`
- Train agent: `python -m src.train`
- Backtest: `python -m src.backtester --checkpoint_dir <ray_checkpoint_dir>`

Key Files
- `config.yaml`: project configuration (universe, factors, reward, training)
- `src/environment.py`: `SingleAgentTradingEnv` with action constraints and turnover cap
- `src/train.py`: RLlib PPO training with seeding and validation evaluation
- `src/backtester.py`: test-period evaluation and QuantStats reporting
- `src/data_pipeline.py`: EODHD downloader honoring `config.yaml`
- `src/feature_engineering.py`: factor computation + normalization (float32)

Notes
- `src/fetch_data.py` is deprecated and proxies to `src.data_pipeline.run()`
- Multi-agent settings in `config.yaml` are placeholders; code runs single-agent PPO
