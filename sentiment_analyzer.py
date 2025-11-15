import praw
from textblob import TextBlob
import pandas as pd
from config import REDDIT_CONFIG, MODEL_CONFIG
import requests
import time


class SentimentAnalyzer:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=REDDIT_CONFIG['client_id'],
            client_secret=REDDIT_CONFIG['client_secret'],
            user_agent=REDDIT_CONFIG['user_agent']
        )
        self.timeout = 10  # 10 second timeout

    def analyze_text_sentiment(self, text):
        """Analyze sentiment of a single text"""
        try:
            analysis = TextBlob(str(text))
            return analysis.sentiment.polarity
        except:
            return 0.0

    def get_subreddit_sentiment(self, subreddit_name, limit=10):
        """Get sentiment analysis for a subreddit with timeout"""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts_data = []

            # Use hot posts with limit to avoid timeout
            for post in subreddit.hot(limit=min(limit, 15)):
                try:
                    title_sentiment = self.analyze_text_sentiment(post.title)

                    # Get a few comments quickly
                    comment_sentiments = []
                    try:
                        post.comments.replace_more(limit=0)  # No nested comments
                        for comment in post.comments[:5]:  # First 5 comments only
                            if hasattr(comment, 'body'):
                                comment_sentiments.append(self.analyze_text_sentiment(comment.body))
                    except:
                        pass

                    avg_comment_sentiment = sum(comment_sentiments) / len(
                        comment_sentiments) if comment_sentiments else 0
                    overall_sentiment = (title_sentiment + avg_comment_sentiment) / 2

                    posts_data.append({
                        'subreddit': subreddit_name,
                        'title': post.title[:100],
                        'score': post.score,
                        'title_sentiment': title_sentiment,
                        'comment_sentiment': avg_comment_sentiment,
                        'overall_sentiment': overall_sentiment,
                        'sentiment_label': self._classify_sentiment(overall_sentiment)
                    })

                except Exception as e:
                    continue  # Skip problematic posts

            return pd.DataFrame(posts_data)

        except Exception as e:
            print(f"Error analyzing {subreddit_name}: {e}")
            return pd.DataFrame()

    def _classify_sentiment(self, score):
        """Classify sentiment score into category"""
        if score > 0.1:
            return 'Positive'
        elif score < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    def get_market_sentiment(self):
        """Get overall market sentiment across all subreddits with timeout handling"""
        all_sentiments = []

        for subreddit in MODEL_CONFIG['subreddits'][:2]:  # Limit to 2 subreddits for speed
            try:
                df = self.get_subreddit_sentiment(subreddit, limit=8)
                if not df.empty and len(df) > 2:  # Need at least 3 posts
                    subreddit_avg = df['overall_sentiment'].mean()
                    all_sentiments.append({
                        'subreddit': subreddit,
                        'avg_sentiment': subreddit_avg,
                        'post_count': len(df)
                    })
            except Exception as e:
                print(f"Skipping {subreddit}: {e}")
                continue

        sentiment_df = pd.DataFrame(all_sentiments)

        if not sentiment_df.empty:
            overall_avg = sentiment_df['avg_sentiment'].mean()
            return {
                'overall_sentiment': overall_avg,
                'sentiment_by_subreddit': sentiment_df,
                'market_mood': self._classify_sentiment(overall_avg)
            }
        else:
            # Return demo data if no real data
            return {
                'overall_sentiment': 0.05,
                'market_mood': 'Neutral',
                'sentiment_by_subreddit': pd.DataFrame({
                    'subreddit': ['CryptoCurrency', 'stocks'],
                    'avg_sentiment': [0.06, 0.04],
                    'post_count': [5, 5]
                })
            }