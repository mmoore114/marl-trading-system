import pandas as pd
import numpy as np
import yaml
import joblib # For saving the HMM model
from ta.momentum import rsi
from ta.trend import macd, macd_signal, ema_indicator
from ta.volatility import bollinger_hband, bollinger_lband, bollinger_pband, average_true_range
from ta.volume import on_balance_volume
from hurst import compute_Hc
from hmmlearn.hmm import GaussianHMM # HMM library
from pathlib import Path

def calculate_factors(df, factor_config):
    """
    Calculates technical analysis factors based on a configuration dictionary.
    """
    if factor_config.get('RSI', {}).get('enabled', False):
        params = factor_config['RSI']
        df['RSI'] = rsi(df['Close'], window=params.get('window', 14))

    # ... (the rest of your existing factor calculations remain the same) ...
    if factor_config.get('MACD', {}).get('enabled', False):
        params = factor_config['MACD']
        df['MACD'] = macd(df['Close'], window_fast=params.get('fast', 12), window_slow=params.get('slow', 26))
        df['MACD_signal'] = macd_signal(df['Close'], window_fast=params.get('fast', 12), window_slow=params.get('slow', 26), window_sign=params.get('signal', 9))

    if factor_config.get('EMA', {}).get('enabled', False):
        params = factor_config['EMA']
        for window in params.get('windows', []):
            df[f'EMA_{window}'] = ema_indicator(df['Close'], window=window)

    if factor_config.get('BBands', {}).get('enabled', False):
        params = factor_config['BBands']
        df['BB_upper'] = bollinger_hband(df['Close'], window=params.get('window', 20), window_dev=params.get('std', 2))
        df['BB_lower'] = bollinger_lband(df['Close'], window=params.get('window', 20), window_dev=params.get('std', 2))
        df['BB_percent'] = bollinger_pband(df['Close'], window=params.get('window', 20), window_dev=params.get('std', 2))

    if factor_config.get('ATR', {}).get('enabled', False):
        params = factor_config['ATR']
        df['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], window=params.get('window', 14))

    if factor_config.get('OBV', {}).get('enabled', False):
        df['OBV'] = on_balance_volume(df['Close'], df['Volume'])
        
    if factor_config.get('Hurst', {}).get('enabled', False):
        params = factor_config['Hurst']
        window = params.get('window', 252)
        df['Hurst'] = df['Close'].rolling(window=window).apply(lambda x: compute_Hc(x)[0], raw=True)

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

        # --- HMM REGIME DETECTION (NEW) ---
        print(f"  Training HMM for {ticker}...")
        
        # 1. Prepare HMM features on the training set only
        hmm_train_df = df[df.index <= TRAIN_END_DATE].copy()
        hmm_train_df['Log_Returns'] = np.log(hmm_train_df['Close'] / hmm_train_df['Close'].shift(1))
        hmm_train_df['Volatility'] = hmm_train_df['Log_Returns'].rolling(window=21).std()
        hmm_train_df.dropna(inplace=True)
        hmm_features = hmm_train_df[['Log_Returns', 'Volatility']].values
        
        # 2. Train and save the HMM
        hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, random_state=42)
        hmm.fit(hmm_features)
        hmm_model_path = models_dir / f"hmm_model_{ticker}.pkl"
        joblib.dump(hmm, hmm_model_path)
        print(f"  HMM model saved to {hmm_model_path}")
        
        # --- CALCULATE ALL FACTORS ---
        df_factors = calculate_factors(df.copy(), FACTOR_CONFIG)
        
        # --- PREDICT REGIME PROBABILITIES (NEW) ---
        print(f"  Predicting regime probabilities for {ticker}...")
        
        # 1. Prepare HMM features for the entire dataset
        full_hmm_features_df = df_factors[['Close']].copy()
        full_hmm_features_df['Log_Returns'] = np.log(full_hmm_features_df['Close'] / full_hmm_features_df['Close'].shift(1))
        full_hmm_features_df['Volatility'] = full_hmm_features_df['Log_Returns'].rolling(window=21).std()
        full_hmm_features_df.dropna(inplace=True)
        full_hmm_features = full_hmm_features_df[['Log_Returns', 'Volatility']].values
        
        # 2. Predict probabilities using the trained model
        regime_probs = hmm.predict_proba(full_hmm_features)
        
        # 3. Add probabilities to our main dataframe
        # Create a temporary dataframe for the probabilities with the correct index
        probs_df = pd.DataFrame(regime_probs, 
                                index=full_hmm_features_df.index, 
                                columns=[f'Regime_{i}_Prob' for i in range(hmm.n_components)])
        
        # Merge the probabilities back into our main factor dataframe
        processed_df = df_factors.join(probs_df, how='inner')

        # --- SAVE FINAL DATA ---
        output_file = processed_data_dir / f"{ticker}_processed.csv"
        processed_df.to_csv(output_file)
        print(f"  Successfully processed and saved final features to {output_file}")

    print("\nFactor and Regime engineering complete for all tickers.")