import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from pathlib import Path

INITIAL_ACCOUNT_BALANCE = 100000

class StockTradingEnv(gym.Env):
    """A stock trading environment for reinforcement learning"""
    metadata = {'render_modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.current_step = 0

        # Define the action space: Continuous value for portfolio weight (0 to 1)
        # 0 = 100% cash, 1 = 100% in stock
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Define the observation space: The features for a given day
        # The shape is the number of feature columns we have
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(df.columns),), dtype=np.float32
        )
        
    def _get_obs(self):
        # Return the features of the current time step as the observation
        return self.df.iloc[self.current_step].values.astype(np.float32)

    def _get_info(self):
        # Return auxiliary information (e.g., portfolio value)
        return {'portfolio_value': self.portfolio_value}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.portfolio_value = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # Get the current price
        current_price = self.df['Close'].iloc[self.current_step]
        
        # Calculate the value of our current holdings
        current_holdings = self.shares_held * current_price
        
        # Calculate the desired value of our holdings based on the agent's action
        # The action is the desired portfolio weight
        scaled_action = (action[0] + 1) / 2  # Scales [-1, 1] to [0, 1]
        desired_value = self.portfolio_value * scaled_action

        # Calculate how many shares we need to buy or sell
        # Positive value means buy, negative value means sell
        shares_to_trade = (desired_value - current_holdings) / current_price
        
        # Update shares held
        self.shares_held += shares_to_trade
        
        # Move to the next time step
        self.current_step += 1
        
        # Calculate the new portfolio value
        next_price = self.df['Close'].iloc[self.current_step]
        new_portfolio_value = self.shares_held * next_price + (self.portfolio_value - (shares_to_trade * current_price))
        
        # Calculate reward as the change in portfolio value
        reward = new_portfolio_value - self.portfolio_value
        self.portfolio_value = new_portfolio_value
        
        # Check if the episode is done
        terminated = self.current_step >= len(self.df) - 1
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, False, info


# --- Testing Block ---
if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    # Load the processed data
    data_path = Path("data/processed/AAPL_processed.csv")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Create the environment
    env = StockTradingEnv(df)

    # Check the environment to ensure it's compatible with Stable Baselines3
    # This will raise an error if the environment is not correctly structured.
    check_env(env)

    print("Environment check passed!")