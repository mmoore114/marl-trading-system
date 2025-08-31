# get_tickers.py
import pandas as pd
import logging
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_sp500_tickers():
    """Fetches the list of S&P 500 tickers from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        
        # --- FIX: Add a User-Agent header to mimic a browser ---
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Use requests to get the HTML content with the header
        response = requests.get(url, headers=headers)
        response.raise_for_status() # This will raise an error for bad status codes

        # Pass the HTML text to pandas to parse the table
        table = pd.read_html(response.text, header=0)[0]
        # --- END FIX ---

        tickers = table['Symbol'].tolist()
        tickers = [ticker.replace('-', '.') for ticker in tickers]
        
        logger.info(f"Successfully fetched {len(tickers)} S&P 500 tickers.")
        return tickers
    except Exception as e:
        logger.error(f"Failed to fetch tickers: {e}")
        return []

if __name__ == "__main__":
    sp500_tickers = get_sp500_tickers()
    if sp500_tickers:
        pd.DataFrame(sp500_tickers, columns=["ticker"]).to_csv("sp500_tickers.csv", index=False)
        logger.info("Saved tickers to sp500_tickers.csv")