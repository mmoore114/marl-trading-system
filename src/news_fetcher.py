import os
from dotenv import load_dotenv
from newsapi import NewsApiClient
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_key():
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        source = config['data'].get('source', 'newsapi')
        if source != 'newsapi':
            logger.info(f"Source is {source} - skipping NewsAPI test.")
            return
    except FileNotFoundError:
        logger.error("config.yaml missing.")
        return

    load_dotenv()
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        logger.error("NEWSAPI_KEY missing.")
        return

    logger.info("Loaded API key.")

    try:
        newsapi = NewsApiClient(api_key=api_key)
        logger.info("Fetching test headlines...")
        top_headlines = newsapi.get_top_headlines(q='stock market', category='business', language='en', country='us')

        if top_headlines['status'] != 'ok':
            logger.error(f"API error: {top_headlines.get('message')}")
            return

        logger.info("\n--- Top 5 Headlines ---")
        for i, article in enumerate(top_headlines['articles'][:5]):
            logger.info(f"{i+1}. {article['title']}")

        logger.info("NewsAPI working!")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    test_api_key()