# Configuration settings
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
BINANCE_CONFIG = {
    'api_key': os.getenv('BINANCE_API_KEY'),
    'secret_key': os.getenv('BINANCE_SECRET_KEY')
}

REDDIT_CONFIG = {
    'client_id': os.getenv('REDDIT_CLIENT_ID'),
    'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
    'user_agent': os.getenv('REDDIT_USER_AGENT')
}

# Model Configuration
MODEL_CONFIG = {
    'target_symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
    'subreddits': ['CryptoCurrency', 'stocks', 'investing'],
    'prediction_horizon': 1,  # hours
    'confidence_threshold': 0.65
}