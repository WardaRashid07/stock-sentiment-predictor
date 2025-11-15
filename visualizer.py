import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ResultVisualizer:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_price_trend(self, price_data, symbol, save=True):
        """Plot price trend with technical indicators"""
        if price_data.empty:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Price chart
        ax1.plot(price_data['timestamp'], price_data['close'],
                 label='Close Price', linewidth=2, color='blue')
        ax1.set_title(f'{symbol} Price Trend', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Volume chart
        ax2.bar(price_data['timestamp'], price_data['volume'],
                alpha=0.7, color='orange', label='Volume')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Volume')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = f"{self.output_dir}/{symbol}_price_trend.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Price chart saved: {filename}")

        plt.show()

    def plot_sentiment_analysis(self, sentiment_results, save=True):
        """Plot sentiment analysis results"""
        if sentiment_results['sentiment_by_subreddit'].empty:
            return

        df = sentiment_results['sentiment_by_subreddit']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Sentiment by subreddit
        colors = ['red' if x < 0 else 'green' if x > 0 else 'gray' for x in df['avg_sentiment']]
        bars = ax1.bar(df['subreddit'], df['avg_sentiment'], color=colors, alpha=0.7)
        ax1.set_title('Average Sentiment by Subreddit', fontweight='bold')
        ax1.set_ylabel('Sentiment Score')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom')

        # Post count by subreddit
        ax2.bar(df['subreddit'], df['post_count'], color='skyblue', alpha=0.7)
        ax2.set_title('Posts Analyzed by Subreddit', fontweight='bold')
        ax2.set_ylabel('Number of Posts')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save:
            filename = f"{self.output_dir}/sentiment_analysis.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📈 Sentiment analysis saved: {filename}")

        plt.show()

    def plot_prediction_summary(self, predictions, save=True):
        """Plot prediction results summary"""
        symbols = list(predictions.keys())
        directions = []
        confidences = []

        for symbol, pred in predictions.items():
            directions.append(1 if pred['prediction'] == 'UP' else -1)
            confidences.append(pred['confidence'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Prediction directions
        colors = ['green' if d == 1 else 'red' for d in directions]
        ax1.bar(symbols, directions, color=colors, alpha=0.7)
        ax1.set_title('Prediction Directions', fontweight='bold')
        ax1.set_ylabel('Direction (UP=1, DOWN=-1)')
        ax1.set_ylim(-1.5, 1.5)

        # Confidence levels
        ax2.bar(symbols, confidences, color='orange', alpha=0.7)
        ax2.set_title('Prediction Confidence', fontweight='bold')
        ax2.set_ylabel('Confidence Level')
        ax2.set_ylim(0, 1)

        # Add confidence values on bars
        for i, conf in enumerate(confidences):
            ax2.text(i, conf + 0.02, f'{conf:.1%}', ha='center', va='bottom')

        plt.tight_layout()

        if save:
            filename = f"{self.output_dir}/prediction_summary.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"🔮 Prediction summary saved: {filename}")

        plt.show()

    def generate_report(self, market_data, sentiment_results, predictions, model_accuracy):
        """Generate a comprehensive text report"""
        report = []
        report.append("=" * 60)
        report.append("         REAL-TIME STOCK SENTIMENT PREDICTOR")
        report.append("=" * 60)
        report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Market Data Section
        report.append("📊 CURRENT MARKET DATA")
        report.append("-" * 30)
        for symbol, price in market_data['current_prices'].items():
            if price:
                report.append(f"{symbol}: ${price:,.2f}")

        # Sentiment Analysis Section
        report.append("")
        report.append("📰 MARKET SENTIMENT ANALYSIS")
        report.append("-" * 35)
        report.append(f"Overall Market Mood: {sentiment_results['market_mood']}")
        report.append(f"Average Sentiment Score: {sentiment_results['overall_sentiment']:.3f}")
        report.append("")
        report.append("By Subreddit:")
        for _, row in sentiment_results['sentiment_by_subreddit'].iterrows():
            report.append(f"  r/{row['subreddit']}: {row['avg_sentiment']:.3f} ({row['post_count']} posts)")

        # Predictions Section
        report.append("")
        report.append("🔮 PRICE PREDICTIONS")
        report.append("-" * 20)
        report.append(f"Model Accuracy: {model_accuracy:.1%}")
        report.append("")

        for symbol, prediction in predictions.items():
            if prediction['prediction'] not in ['MODEL_NOT_TRAINED', 'INSUFFICIENT_DATA']:
                arrow = "↑" if prediction['prediction'] == 'UP' else "↓"
                report.append(f"{symbol}: {prediction['prediction']} {arrow}")
                report.append(f"   Confidence: {prediction['confidence']:.1%}")
                report.append(f"   Probability UP: {prediction['probability_up']:.1%}")
            else:
                report.append(f"{symbol}: {prediction['prediction']}")

        # Trading Recommendations
        report.append("")
        report.append("💡 TRADING INSIGHTS")
        report.append("-" * 20)

        bullish_signals = sum(1 for p in predictions.values()
                              if p.get('prediction') == 'UP' and p.get('confidence', 0) > 0.6)

        if bullish_signals >= 2:
            report.append("🎯 STRONG BULLISH SIGNAL detected across multiple assets")
        elif sentiment_results['overall_sentiment'] > 0.2:
            report.append("👍 Positive market sentiment supporting upward movement")
        elif sentiment_results['overall_sentiment'] < -0.2:
            report.append("👎 Negative sentiment may pressure prices downward")
        else:
            report.append("🤔 Mixed signals - consider waiting for clearer trend")

        report.append("")
        report.append("=" * 60)

        # Save report to file
        report_text = "\n".join(report)
        filename = f"{self.output_dir}/market_analysis_report.txt"
        with open(filename, 'w') as f:
            f.write(report_text)

        print(f"📄 Report saved: {filename}")
        return report_text