import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
from config import BINANCE_CONFIG, MODEL_CONFIG
import time


class DataCollector:
    def __init__(self):
        try:
            self.client = Client(
                BINANCE_CONFIG['api_key'],
                BINANCE_CONFIG['secret_key']
            )
            self.connection_working = True
        except Exception as e:
            print(f"❌ Binance connection failed: {e}")
            self.connection_working = False

    def get_current_prices(self):
        """Get current prices for all target symbols"""
        prices = {}
        if not self.connection_working:
            # Return demo prices if connection failed
            demo_prices = {'BTCUSDT': 45000, 'ETHUSDT': 2500, 'ADAUSDT': 0.45}
            return demo_prices

        for symbol in MODEL_CONFIG['target_symbols']:
            try:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                prices[symbol] = float(ticker['price'])
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")
                # Return demo price if API fails
                if symbol == 'BTCUSDT':
                    prices[symbol] = 45000
                elif symbol == 'ETHUSDT':
                    prices[symbol] = 2500
                else:
                    prices[symbol] = 0.45
        return prices

    def get_historical_data(self, symbol, hours=24):
        """Get historical price data"""
        if not self.connection_working:
            return self._generate_demo_data(symbol, hours)

        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1HOUR,
                limit=hours
            )

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])

            # Convert to proper types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)

            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            print(f"Error getting historical data for {symbol}: {e}")
            return self._generate_demo_data(symbol, hours)

    def _generate_demo_data(self, symbol, hours=24):
        """Generate demo data when API is not available"""
        base_price = 45000 if symbol == 'BTCUSDT' else 2500 if symbol == 'ETHUSDT' else 0.45

        timestamps = pd.date_range(end=datetime.now(), periods=hours, freq='H')
        data = []

        current_price = base_price
        for timestamp in timestamps:
            # Simulate price movement
            change = np.random.normal(0, 0.01)
            current_price = current_price * (1 + change)

            # Generate OHLC data
            open_price = current_price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, current_price) * (1 + abs(np.random.normal(0, 0.002)))
            low_price = min(open_price, current_price) * (1 - abs(np.random.normal(0, 0.002)))
            volume = np.random.uniform(1000, 5000)

            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': current_price,
                'volume': volume
            })

        return pd.DataFrame(data)

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for features"""
        if df.empty:
            return df

        # Price movements
        df['price_change_1h'] = df['close'].pct_change()
        df['price_change_24h'] = df['close'].pct_change(periods=min(24, len(df) - 1))

        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=min(6, len(df))).mean()
        df['volume_change'] = df['volume'].pct_change()

        # Volatility
        df['volatility'] = df['close'].rolling(window=min(6, len(df))).std()

        # Moving averages
        df['ma_6h'] = df['close'].rolling(window=min(6, len(df))).mean()
        df['ma_12h'] = df['close'].rolling(window=min(12, len(df))).mean()

        return df.fillna(method='bfill').fillna(0)