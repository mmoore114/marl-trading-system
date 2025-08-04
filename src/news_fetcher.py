import os
from dotenv import load_dotenv
from newsapi import NewsApiClient

def test_api_key():
    """
    Loads the NewsAPI key from the .env file and fetches a few headlines
    to confirm that it is working correctly.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Get the API key
    api_key = os.getenv("NEWSAPI_KEY")

    if not api_key:
        print("Error: NEWSAPI_KEY not found in .env file.")
        return

    print("Successfully loaded API key.")

    try:
        # Initialize the client
        newsapi = NewsApiClient(api_key=api_key)

        # Make a test call to the API for a major company
        print("Fetching test headlines for 'Apple'...")
        top_headlines = newsapi.get_top_headlines(q='Apple',
                                                  category='business',
                                                  language='en',
                                                  country='us')

        # Print the titles of the first 5 articles
        print("\n--- Top 5 Headlines ---")
        for i, article in enumerate(top_headlines['articles'][:5]):
            print(f"{i+1}. {article['title']}")
        
        print("\nNewsAPI key is working correctly!")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please double-check your API key in the .env file.")

if __name__ == "__main__":
    test_api_key()