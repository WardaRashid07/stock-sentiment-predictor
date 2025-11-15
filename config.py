# Configuration settings
import os

# API Configuration - Support both .env and Streamlit secrets
def get_config_value(key, default=None):
    """Get configuration value from environment or Streamlit secrets"""
    try:
        # Try Streamlit secrets first (for production)
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except:
        pass
    
    # Fall back to environment variables
    return os.getenv(key, default)

BINANCE_CONFIG = {
    'api_key': get_config_value('BINANCE_API_KEY', 'demo_key'),
    'secret_key': get_config_value('BINANCE_SECRET_KEY', 'demo_secret')
}

REDDIT_CONFIG = {
    'client_id': get_config_value('REDDIT_CLIENT_ID', 'demo_client_id'),
    'client_secret': get_config_value('REDDIT_CLIENT_SECRET', 'demo_client_secret'),
    'user_agent': get_config_value('REDDIT_USER_AGENT', 'StockSentimentBot/1.0')
}

# Model Configuration
MODEL_CONFIG = {
    'target_symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
    'subreddits': ['CryptoCurrency', 'stocks', 'investing'],
    'prediction_horizon': 1,
    'confidence_threshold': 0.65
}
