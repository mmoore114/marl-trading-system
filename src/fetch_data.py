"""Deprecated standalone fetcher.

This module now proxies to the unified EODHD-based pipeline in `src/data_pipeline.py`.
Keep this file as a thin wrapper for backwards compatibility with any notebooks/scripts
that import `src.fetch_data`.
"""

from src.data_pipeline import run as run_data_pipeline


def main():
    run_data_pipeline()


if __name__ == "__main__":
    main()

