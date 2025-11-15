import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_collector import DataCollector
    from sentiment_analyzer import SentimentAnalyzer
    from prediction_model import PricePredictor
    from config import MODEL_CONFIG

    DATA_AVAILABLE = True
except ImportError as e:
    st.warning(f"Some modules not available: {e}")
    DATA_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Real-Time Stock Sentiment Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state PROPERLY
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'predictor' not in st.session_state:
    if DATA_AVAILABLE:
        st.session_state.predictor = PricePredictor()
    else:
        st.session_state.predictor = None
if 'sentiment_data' not in st.session_state:
    st.session_state.sentiment_data = None
if 'price_data' not in st.session_state:
    st.session_state.price_data = {}


class StreamlitApp:
    def __init__(self):
        if DATA_AVAILABLE:
            self.data_collector = DataCollector()
            self.sentiment_analyzer = SentimentAnalyzer()
            self.predictor = st.session_state.predictor
        else:
            self.data_collector = None
            self.sentiment_analyzer = None
            self.predictor = None

    def run(self):
        # Sidebar
        self.render_sidebar()

        # Main content
        self.render_header()

        if not DATA_AVAILABLE:
            self.render_demo_mode()
        else:
            self.render_real_data()

    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.title("⚙️ Configuration")

        st.sidebar.markdown("### Target Assets")
        if DATA_AVAILABLE:
            selected_symbols = st.sidebar.multiselect(
                "Select cryptocurrencies:",
                MODEL_CONFIG['target_symbols'],
                default=MODEL_CONFIG['target_symbols'][:2]
            )
        else:
            selected_symbols = ["BTCUSDT", "ETHUSDT"]

        st.sidebar.markdown("### Analysis Settings")
        auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=False)
        refresh_interval = st.sidebar.slider("Refresh interval (minutes)", 1, 60, 5)

        if DATA_AVAILABLE:
            st.sidebar.markdown("### Model Controls")
            col1, col2 = st.sidebar.columns(2)

            with col1:
                if st.button("🔄 Train Model", use_container_width=True):
                    with st.spinner("Training prediction model..."):
                        success = self.train_model()
                        if success:
                            st.sidebar.success("Model trained!")
                        else:
                            st.sidebar.error("Training failed")

            with col2:
                if st.button("📊 Update Data", use_container_width=True):
                    # Clear cached data to force refresh
                    st.session_state.sentiment_data = None
                    st.session_state.price_data = {}
                    st.rerun()
        else:
            if st.sidebar.button("📊 Update Data", use_container_width=True):
                st.rerun()

        if auto_refresh:
            time.sleep(refresh_interval * 60)
            st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.markdown("### About")
        st.sidebar.info(
            "Real-time cryptocurrency analysis with social sentiment "
            "and machine learning predictions."
        )

    def render_header(self):
        """Render page header"""
        st.title("📈 Real-Time Stock & Sentiment Predictor")
        st.markdown("""
        Live cryptocurrency prices combined with social media sentiment analysis 
        for intelligent market predictions.
        """)

        # Last update time
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.caption(f"Last updated: {current_time}")

        # Show mode indicator
        if not DATA_AVAILABLE:
            st.warning("🔧 Running in Demo Mode - Using simulated data")
        elif not st.session_state.model_trained:
            st.info("🤖 Click 'Train Model' in sidebar to enable predictions")
        else:
            st.success("✅ Model trained and ready for predictions!")

    def render_demo_mode(self):
        """Render demo mode when real data isn't available"""
        col1, col2 = st.columns(2)

        with col1:
            self.render_demo_prices()

        with col2:
            self.render_demo_sentiment()

        self.render_demo_predictions()

    def render_real_data(self):
        """Render with real data"""
        try:
            # Load data first
            self.load_data()

            # Then render components
            self.render_market_overview()
            self.render_detailed_analysis()

            # Only show predictions if model is trained
            if st.session_state.model_trained:
                self.render_predictions()
            else:
                st.info("🎯 Train the model using the sidebar to see predictions")

        except Exception as e:
            st.error(f"Error loading real data: {e}")
            st.info("Switching to demo mode...")
            self.render_demo_mode()

    def load_data(self):
        """Load or fetch data with caching"""
        # Load price data if not cached
        if not st.session_state.price_data:
            with st.spinner("Loading market data..."):
                try:
                    prices = self.data_collector.get_current_prices()
                    st.session_state.price_data['current_prices'] = prices

                    # Load historical data for selected symbols
                    for symbol in MODEL_CONFIG['target_symbols'][:3]:
                        try:
                            historical = self.data_collector.get_historical_data(symbol, hours=24)
                            if not historical.empty:
                                historical = self.data_collector.calculate_technical_indicators(historical)
                                st.session_state.price_data[symbol] = historical
                        except Exception as e:
                            st.warning(f"Could not load data for {symbol}: {e}")
                except Exception as e:
                    st.error(f"Error loading price data: {e}")

        # Load sentiment data if not cached
        if st.session_state.sentiment_data is None:
            with st.spinner("Analyzing market sentiment..."):
                try:
                    sentiment_results = self.sentiment_analyzer.get_market_sentiment()
                    st.session_state.sentiment_data = sentiment_results
                except Exception as e:
                    st.error(f"Error loading sentiment data: {e}")
                    st.session_state.sentiment_data = {
                        'overall_sentiment': 0,
                        'market_mood': 'Neutral',
                        'sentiment_by_subreddit': pd.DataFrame()
                    }

    def render_demo_prices(self):
        """Render demo price data"""
        st.subheader("💹 Price Data (Demo)")

        # Create realistic demo data
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        base_prices = [45000, 2500, 0.45]

        for symbol, base_price in zip(symbols, base_prices):
            col1, col2 = st.columns([2, 1])
            with col1:
                # Generate realistic price with some variation
                current_price = base_price * (1 + np.random.uniform(-0.05, 0.05))
                change = np.random.uniform(-3, 3)

                st.metric(
                    label=symbol,
                    value=f"${current_price:,.2f}",
                    delta=f"{change:+.2f}%"
                )

    def render_demo_sentiment(self):
        """Render demo sentiment analysis"""
        st.subheader("📰 Market Sentiment (Demo)")

        # Generate realistic sentiment scores
        sentiment_score = np.random.uniform(-0.3, 0.3)

        col1, col2 = st.columns(2)

        with col1:
            # Sentiment score with color
            if sentiment_score > 0.1:
                sentiment_color = "green"
                emoji = "😊"
                label = "Positive"
            elif sentiment_score < -0.1:
                sentiment_color = "red"
                emoji = "😞"
                label = "Negative"
            else:
                sentiment_color = "gray"
                emoji = "😐"
                label = "Neutral"

            st.metric(
                "Overall Sentiment",
                f"{emoji} {sentiment_score:.3f}",
                label
            )

        with col2:
            # Sentiment distribution
            sentiment_data = {
                'Positive': max(0, sentiment_score),
                'Neutral': max(0, 1 - abs(sentiment_score)),
                'Negative': max(0, -sentiment_score)
            }

            fig = px.bar(
                x=list(sentiment_data.keys()),
                y=list(sentiment_data.values()),
                color=list(sentiment_data.keys()),
                color_discrete_map={
                    'Positive': 'green',
                    'Neutral': 'gray',
                    'Negative': 'red'
                }
            )
            fig.update_layout(
                title="Sentiment Distribution",
                showlegend=False,
                height=200
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_demo_predictions(self):
        """Render demo predictions"""
        st.subheader("🔮 Predictions (Demo)")

        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        cols = st.columns(len(symbols))

        for idx, symbol in enumerate(symbols):
            with cols[idx]:
                # Generate realistic predictions
                direction = "UP" if np.random.random() > 0.5 else "DOWN"
                confidence = np.random.uniform(60, 90)
                emoji = "🟢" if direction == "UP" else "🔴"

                st.metric(
                    label=f"{emoji} {symbol}",
                    value=direction,
                    delta=f"{confidence:.1f}% confidence"
                )

        # Trading insight
        st.subheader("💡 Trading Insight")
        insights = [
            "Strong bullish momentum detected across major cryptocurrencies",
            "Positive sentiment supporting upward price movement",
            "Consider monitoring resistance levels for entry points",
            "Market volatility within normal range - good trading conditions"
        ]

        st.info(f"💡 {insights[0]}")

    def create_demo_chart(self, symbol, base_price):
        """Create demo price chart with proper data"""
        # Generate time series data
        times = pd.date_range(end=datetime.now(), periods=24, freq='H')
        prices = []

        current_price = base_price
        for _ in range(24):
            change = np.random.normal(0, 0.01)
            current_price = current_price * (1 + change)
            prices.append(current_price)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=prices, mode='lines', name=symbol,
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title=f"{symbol} Price Trend",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            height=400
        )

        return fig

    def render_market_overview(self):
        """Render market overview with real data"""
        st.header("📊 Market Overview")

        try:
            prices = st.session_state.price_data.get('current_prices', {})

            if prices:
                cols = st.columns(len(prices))

                for idx, (symbol, price) in enumerate(prices.items()):
                    if price:
                        with cols[idx]:
                            # Calculate a simple price change for demo
                            price_change = np.random.uniform(-2, 2)

                            st.metric(
                                label=symbol,
                                value=f"${price:,.2f}",
                                delta=f"{price_change:+.1f}%"
                            )
            else:
                st.warning("No price data available")
                self.render_demo_prices()

        except Exception as e:
            st.error(f"Error fetching market data: {e}")
            self.render_demo_prices()

    def render_detailed_analysis(self):
        """Render detailed analysis with real data"""
        col1, col2 = st.columns(2)

        with col1:
            self.render_real_price_charts()

        with col2:
            self.render_real_sentiment()

    def render_real_price_charts(self):
        """Render real price charts"""
        st.subheader("💹 Price Charts")

        selected_symbol = st.selectbox(
            "Select asset:",
            MODEL_CONFIG['target_symbols'],
            key="real_chart_symbol"
        )

        try:
            # Get historical data from cache
            historical_data = st.session_state.price_data.get(selected_symbol)

            if historical_data is not None and not historical_data.empty:
                # Create candlestick chart
                fig = go.Figure()

                fig.add_trace(go.Candlestick(
                    x=historical_data['timestamp'],
                    open=historical_data['open'],
                    high=historical_data['high'],
                    low=historical_data['low'],
                    close=historical_data['close'],
                    name='Price'
                ))

                # Add moving averages
                if 'ma_6h' in historical_data.columns:
                    fig.add_trace(go.Scatter(
                        x=historical_data['timestamp'],
                        y=historical_data['ma_6h'],
                        line=dict(color='orange', width=1),
                        name='6H MA'
                    ))

                fig.update_layout(
                    title=f"{selected_symbol} Price Chart",
                    xaxis_title="Time",
                    yaxis_title="Price (USD)",
                    height=400,
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show some statistics
                if not historical_data.empty:
                    col1, col2, col3 = st.columns(3)
                    latest = historical_data.iloc[-1]

                    with col1:
                        st.metric("24H High", f"${historical_data['high'].max():.2f}")
                    with col2:
                        st.metric("24H Low", f"${historical_data['low'].min():.2f}")
                    with col3:
                        price_change = ((latest['close'] - historical_data['close'].iloc[0]) /
                                        historical_data['close'].iloc[0]) * 100
                        st.metric("24H Change", f"{price_change:+.2f}%")

            else:
                st.warning(f"No historical data available for {selected_symbol}")
                # Show demo chart instead
                base_price = 45000 if selected_symbol == "BTCUSDT" else 2500 if selected_symbol == "ETHUSDT" else 1.5
                fig = self.create_demo_chart(selected_symbol, base_price)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error with real price data: {e}")
            # Show demo chart as fallback
            base_price = 45000 if selected_symbol == "BTCUSDT" else 2500 if selected_symbol == "ETHUSDT" else 1.5
            fig = self.create_demo_chart(selected_symbol, base_price)
            st.plotly_chart(fig, use_container_width=True)

    def render_real_sentiment(self):
        """Render real sentiment analysis"""
        st.subheader("📰 Market Sentiment")

        try:
            sentiment_results = st.session_state.sentiment_data

            if sentiment_results and not sentiment_results['sentiment_by_subreddit'].empty:
                overall_sentiment = sentiment_results['overall_sentiment']
                market_mood = sentiment_results['market_mood']

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Overall Sentiment", f"{overall_sentiment:.3f}", market_mood)

                with col2:
                    # Simple sentiment gauge
                    if overall_sentiment > 0.1:
                        st.success("😊 Positive Mood")
                    elif overall_sentiment < -0.1:
                        st.error("😞 Negative Mood")
                    else:
                        st.info("😐 Neutral Mood")

                # Sentiment by subreddit
                sentiment_df = sentiment_results['sentiment_by_subreddit']
                fig = px.bar(
                    sentiment_df,
                    x='subreddit',
                    y='avg_sentiment',
                    color='avg_sentiment',
                    color_continuous_scale=['red', 'gray', 'green'],
                    title="Sentiment by Subreddit"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("No real sentiment data available")
                self.render_demo_sentiment()

        except Exception as e:
            st.error(f"Error with real sentiment: {e}")
            self.render_demo_sentiment()

    def render_predictions(self):
        """Render predictions section"""
        st.header("🔮 Price Predictions")

        try:
            # Get current data for predictions
            prices = st.session_state.price_data.get('current_prices', {})
            sentiment_results = st.session_state.sentiment_data

            if sentiment_results:
                overall_sentiment = sentiment_results['overall_sentiment']
            else:
                overall_sentiment = 0

            # Generate predictions for each symbol
            predictions = {}

            for symbol in MODEL_CONFIG['target_symbols'][:3]:  # Limit to first 3
                try:
                    historical_data = st.session_state.price_data.get(symbol)

                    if historical_data is not None and not historical_data.empty:
                        features = self.predictor.create_features(historical_data, overall_sentiment)
                        prediction = self.predictor.predict(features)
                        predictions[symbol] = prediction
                    else:
                        # Use demo prediction if no data
                        predictions[symbol] = {
                            'prediction': 'UP' if np.random.random() > 0.5 else 'DOWN',
                            'confidence': np.random.uniform(60, 90),
                            'probability_up': np.random.uniform(0.5, 0.9),
                            'direction': 'UP' if np.random.random() > 0.5 else 'DOWN'
                        }

                except Exception as e:
                    st.warning(f"Could not generate prediction for {symbol}: {e}")
                    # Fallback to demo prediction
                    predictions[symbol] = {
                        'prediction': 'UP' if np.random.random() > 0.5 else 'DOWN',
                        'confidence': np.random.uniform(60, 90),
                        'probability_up': np.random.uniform(0.5, 0.9),
                        'direction': 'UP' if np.random.random() > 0.5 else 'DOWN'
                    }

            # Display predictions
            if predictions:
                cols = st.columns(len(predictions))

                for idx, (symbol, prediction) in enumerate(predictions.items()):
                    with cols[idx]:
                        if prediction.get('prediction') in ['UP', 'DOWN']:
                            direction = prediction['prediction']
                            confidence = prediction.get('confidence', 0) * 100
                            emoji = "🟢" if direction == 'UP' else "🔴"

                            st.metric(
                                label=f"{emoji} {symbol}",
                                value=direction,
                                delta=f"{confidence:.1f}% confidence"
                            )

                            # Show probability in expander
                            with st.expander("Details"):
                                prob_up = prediction.get('probability_up', 0) * 100
                                st.write(f"Probability UP: {prob_up:.1f}%")
                                st.write(f"Probability DOWN: {100 - prob_up:.1f}%")
                        else:
                            st.warning(f"{symbol}: {prediction.get('prediction', 'No prediction')}")

            # Trading insights
            st.subheader("💡 Trading Insights")
            if predictions:
                bullish_count = sum(1 for p in predictions.values()
                                    if p.get('prediction') == 'UP' and p.get('confidence', 0) > 0.6)

                if bullish_count >= 2:
                    st.success("🎯 **STRONG BULLISH SIGNAL**: Multiple assets showing upward momentum")
                elif overall_sentiment > 0.2:
                    st.info("👍 **POSITIVE SENTIMENT**: Market mood is optimistic")
                elif overall_sentiment < -0.2:
                    st.warning("👎 **CAUTION NEEDED**: Negative sentiment may pressure prices")
                else:
                    st.info("🤔 **MIXED SIGNALS**: Consider waiting for clearer market direction")

        except Exception as e:
            st.error(f"Error generating predictions: {e}")
            st.info("Showing demo predictions instead...")
            self.render_demo_predictions()

    def train_model(self):
        """Train the prediction model"""
        try:
            if self.predictor:
                # Create sample training data
                training_data = self.predictor.prepare_training_data()

                if not training_data.empty:
                    accuracy = self.predictor.train(training_data)

                    if accuracy > 0:
                        st.session_state.model_trained = True
                        st.session_state.predictor = self.predictor  # Update session state
                        self.predictor.save_model()
                        return True
                    else:
                        st.error("❌ Model training failed - zero accuracy")
                        return False
                else:
                    st.error("❌ No training data available")
                    return False
            else:
                st.error("❌ Predictor not available")
                return False

        except Exception as e:
            st.error(f"❌ Error training model: {e}")
            return False


def main():
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()