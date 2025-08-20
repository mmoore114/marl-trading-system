import pandas as pd
import numpy as np
import yaml
import joblib
from ta.momentum import rsi, roc
from ta.trend import macd, macd_signal, ema_indicator
from ta.volatility import bollinger_pband
from ta.volume import on_balance_volume
from hurst import compute_Hc
from hmmlearn.hmm import GaussianHMM
from pathlib import Path

def calculate_factors(df, factor_config):
    """
    Calculates a wide range of factors based on the config file.
    The factors are prefixed with their strategy group name.
    """
    # --- Momentum Agent Factors ---
    if factor_config.get('Momentum', {}):
        mom_conf = factor_config['Momentum']
        if mom_conf.get('RSI_14', {}).get('enabled'):
            df['Momentum_RSI_14'] = rsi(df['Close'], window=mom_conf['RSI_14'].get('window', 14))
        if mom_conf.get('MACD', {}).get('enabled'):
            params = mom_conf['MACD']
            df['Momentum_MACD'] = macd(df['Close'], window_fast=params.get('fast', 12), window_slow=params.get('slow', 26))
        if mom_conf.get('EMA_Cross_50_200', {}).get('enabled'):
            params = mom_conf['EMA_Cross_50_200']
            ema_fast = ema_indicator(df['Close'], window=params.get('fast', 50))
            ema_slow = ema_indicator(df['Close'], window=params.get('slow', 200))
            df['Momentum_EMA_Cross'] = ema_fast - ema_slow
        if mom_conf.get('ROC_21', {}).get('enabled'):
            df['Momentum_ROC_21'] = roc(df['Close'], window=mom_conf['ROC_21'].get('window', 21))

    # --- Mean-Reversion Agent Factors ---
    if factor_config.get('MeanReversion', {}):
        mr_conf = factor_config['MeanReversion']
        if mr_conf.get('BBands_Percent_20', {}).get('enabled'):
            params = mr_conf['BBands_Percent_20']
            df['MeanReversion_BBP'] = bollinger_pband(df['Close'], window=params.get('window', 20), window_dev=params.get('std', 2))
        if mr_conf.get('RSI_Reversion_14', {}).get('enabled'):
            # This can be the same calculation as momentum RSI, but its interpretation is different.
            df['MeanReversion_RSI'] = rsi(df['Close'], window=mr_conf['RSI_Reversion_14'].get('window', 14))

    # --- Shared Factors ---
    if factor_config.get('Shared', {}):
        shared_conf = factor_config['Shared']
        if shared_conf.get('Volatility_21', {}).get('enabled'):
            df['Shared_Volatility'] = df['Close'].pct_change().rolling(window=shared_conf['Volatility_21'].get('window', 21)).std()
        if shared_conf.get('Hurst_252', {}).get('enabled'):
            window = shared_conf['Hurst_252'].get('window', 252)
            df['Shared_Hurst'] = df['Close'].rolling(window=window).apply(lambda x: compute_Hc(x)[0], raw=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        TICKERS = config['data_settings']['tickers']
        TRAIN_END_DATE = config['data_settings']['train_end_date']
        FACTOR_CONFIG = config['factor_settings']
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading configuration: {e}")
        exit()

    raw_data_dir = Path("data/raw")
    processed_data_dir = Path("data/processed")
    models_dir = Path("models")
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        print(f"--- Processing factors for {ticker} ---")
        raw_data_path = raw_data_dir / f"{ticker}.csv"
        
        if not raw_data_path.exists():
            print(f"  Raw data for {ticker} not found, skipping.")
            continue
        
        column_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = pd.read_csv(raw_data_path, header=None, names=column_names, index_col='Date', parse_dates=True, skiprows=3)
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        df_factors = calculate_factors(df.copy(), FACTOR_CONFIG)
        
        # HMM Regime Detection Logic remains the same
        print(f"  Training HMM for {ticker}...")
        hmm_train_df = df[df.index <= TRAIN_END_DATE].copy()
        hmm_train_df['Log_Returns'] = np.log(hmm_train_df['Close'] / hmm_train_df['Close'].shift(1))
        hmm_train_df['Volatility'] = hmm_train_df['Log_Returns'].rolling(window=21).std()
        hmm_train_df.dropna(inplace=True)
        hmm_features = hmm_train_df[['Log_Returns', 'Volatility']].values
        
        hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, random_state=42)
        hmm.fit(hmm_features)
        hmm_model_path = models_dir / f"hmm_model_{ticker}.pkl"
        joblib.dump(hmm, hmm_model_path)
        
        print(f"  Predicting regime probabilities for {ticker}...")
        full_hmm_features_df = df_factors[['Close']].copy()
        full_hmm_features_df['Log_Returns'] = np.log(full_hmm_features_df['Close'] / full_hmm_features_df['Close'].shift(1))
        full_hmm_features_df['Volatility'] = full_hmm_features_df['Log_Returns'].rolling(window=21).std()
        full_hmm_features_df.dropna(inplace=True)
        full_hmm_features = full_hmm_features_df[['Log_Returns', 'Volatility']].values
        
        regime_probs = hmm.predict_proba(full_hmm_features)
        
        probs_df = pd.DataFrame(regime_probs, index=full_hmm_features_df.index, columns=[f'Shared_Regime_{i}_Prob' for i in range(hmm.n_components)])
        
        processed_df = df_factors.join(probs_df, how='inner')

        output_file = processed_data_dir / f"{ticker}_processed.csv"
        processed_df.to_csv(output_file)
        print(f"  Successfully processed and saved final features to {output_file}")

    print("\nFactor and Regime engineering complete for all tickers.")