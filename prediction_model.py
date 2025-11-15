import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class PricePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.accuracy = 0.0
        self.feature_names = [
            'current_price', 'price_change_1h', 'price_change_24h', 'volume_change',
            'volatility', 'ma_6h', 'ma_12h', 'sentiment_score', 'volume_ratio',
            'rsi', 'macd', 'bollinger_position', 'price_momentum', 'volume_momentum'
        ]

    def create_features(self, price_data, sentiment_score):
        """Create features for prediction from price data and sentiment"""
        if price_data is None or price_data.empty or len(price_data) < 10:
            # Return default features if no data
            return self._get_default_features(sentiment_score)

        try:
            # Use the latest data point
            latest = price_data.iloc[-1]

            # Calculate additional technical indicators
            price_data = self._calculate_advanced_indicators(price_data)
            latest_advanced = price_data.iloc[-1]

            features = {
                'current_price': latest.get('close', 45000),
                'price_change_1h': latest.get('price_change_1h', 0),
                'price_change_24h': latest.get('price_change_24h', 0),
                'volume_change': latest.get('volume_change', 0),
                'volatility': latest.get('volatility', 200),
                'ma_6h': latest.get('ma_6h', 44800),
                'ma_12h': latest.get('ma_12h', 44600),
                'sentiment_score': sentiment_score,
                'volume_ratio': latest.get('volume_ratio', 1.0),
                'rsi': latest_advanced.get('rsi', 50),
                'macd': latest_advanced.get('macd', 0),
                'bollinger_position': latest_advanced.get('bollinger_position', 0.5),
                'price_momentum': latest_advanced.get('price_momentum', 0),
                'volume_momentum': latest_advanced.get('volume_momentum', 0)
            }

            return features

        except Exception as e:
            print(f"Error creating features: {e}")
            return self._get_default_features(sentiment_score)

    def _get_default_features(self, sentiment_score):
        """Get default features when real data is unavailable"""
        return {
            'current_price': 45000,
            'price_change_1h': 0.01,
            'price_change_24h': 0.05,
            'volume_change': 0.1,
            'volatility': 200,
            'ma_6h': 44800,
            'ma_12h': 44600,
            'sentiment_score': sentiment_score,
            'volume_ratio': 1.2,
            'rsi': 55,
            'macd': 2.5,
            'bollinger_position': 0.6,
            'price_momentum': 0.02,
            'volume_momentum': 0.15
        }

    def _calculate_advanced_indicators(self, df):
        """Calculate advanced technical indicators"""
        if len(df) < 14:  # Need enough data for indicators
            return df

        try:
            # RSI (Relative Strength Index)
            df = self._calculate_rsi(df)

            # MACD (Moving Average Convergence Divergence)
            df = self._calculate_macd(df)

            # Bollinger Bands
            df = self._calculate_bollinger_bands(df)

            # Momentum indicators
            df['price_momentum'] = df['close'].pct_change(periods=3).fillna(0)
            df['volume_momentum'] = df['volume'].pct_change(periods=3).fillna(0)

            return df.fillna(method='bfill').fillna(method='ffill')
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return df

    def _calculate_rsi(self, df, period=14):
        """Calculate RSI indicator"""
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            return df.fillna(50)  # Default to neutral RSI
        except:
            df['rsi'] = 50
            return df

    def _calculate_macd(self, df):
        """Calculate MACD indicator"""
        try:
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            return df.fillna(0)
        except:
            df['macd'] = 0
            return df

    def _calculate_bollinger_bands(self, df, period=20):
        """Calculate Bollinger Bands"""
        try:
            df['bb_middle'] = df['close'].rolling(window=period).mean()
            bb_std = df['close'].rolling(window=period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bollinger_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            return df.fillna(0.5)  # Default to middle
        except:
            df['bollinger_position'] = 0.5
            return df

    def prepare_training_data(self, historical_data=None, sentiment_history=None):
        """Prepare comprehensive training data with realistic patterns"""
        print("📊 Generating training data with realistic market patterns...")

        samples = []
        np.random.seed(42)  # For reproducible results

        # Generate 500 training samples with realistic market behavior
        for i in range(500):
            # Base price around typical crypto ranges
            base_price = np.random.uniform(30000, 60000)

            # Create correlated features (real markets have correlations)
            sentiment = np.random.normal(0, 0.3)  # Sentiment usually centered around 0

            # Price changes correlated with sentiment
            price_change_1h = sentiment * 0.1 + np.random.normal(0, 0.02)
            price_change_24h = sentiment * 0.2 + np.random.normal(0, 0.05)

            # Volume often increases with large price moves
            volume_change = abs(price_change_1h) * 10 + np.random.normal(0, 0.3)

            # Volatility clusters (high vol often follows high vol)
            volatility = np.random.gamma(2, 50)  # Gamma distribution for volatility

            # Moving averages (should be close to current price)
            ma_6h = base_price * (1 + np.random.normal(0, 0.01))
            ma_12h = base_price * (1 + np.random.normal(0, 0.015))

            # Technical indicators with realistic ranges
            rsi = np.random.uniform(20, 80)  # RSI typically between 20-80
            macd = np.random.normal(0, 10)
            bollinger_position = np.random.uniform(0, 1)

            features = {
                'current_price': base_price,
                'price_change_1h': price_change_1h,
                'price_change_24h': price_change_24h,
                'volume_change': volume_change,
                'volatility': volatility,
                'ma_6h': ma_6h,
                'ma_12h': ma_12h,
                'sentiment_score': sentiment,
                'volume_ratio': np.random.uniform(0.3, 2.0),
                'rsi': rsi,
                'macd': macd,
                'bollinger_position': bollinger_position,
                'price_momentum': np.random.normal(0, 0.02),
                'volume_momentum': np.random.normal(0, 0.3)
            }

            # Realistic target generation based on multiple factors
            price_up_probability = self._calculate_price_up_probability(features)
            target = 1 if np.random.random() < price_up_probability else 0

            samples.append({**features, 'target': target})

        df = pd.DataFrame(samples)
        print(f"✅ Generated {len(df)} training samples")
        return df

    def _calculate_price_up_probability(self, features):
        """Calculate realistic probability of price going up"""
        prob = 0.5  # Base probability

        # Sentiment effect
        prob += features['sentiment_score'] * 0.3

        # Momentum effect
        prob += features['price_change_1h'] * 2
        prob += features['price_momentum'] * 1.5

        # RSI effect (mean reversion)
        rsi = features['rsi']
        if rsi < 30:  # Oversold - likely to bounce
            prob += 0.2
        elif rsi > 70:  # Overbought - likely to pull back
            prob -= 0.2

        # Bollinger bands effect (mean reversion)
        bb_pos = features['bollinger_position']
        if bb_pos < 0.2:  # Near lower band - likely to bounce
            prob += 0.15
        elif bb_pos > 0.8:  # Near upper band - likely to pull back
            prob -= 0.15

        # Ensure probability stays between 0 and 1
        return max(0.1, min(0.9, prob))

    def train(self, training_data):
        """Train the prediction model with comprehensive evaluation"""
        if training_data.empty:
            print("❌ No training data available")
            return 0.0

        try:
            # Prepare features and target
            X = training_data[self.feature_names]  # Use only known features
            y = training_data['target']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            print("🤖 Training Random Forest model...")
            self.model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)

            # Calculate accuracy
            self.accuracy = accuracy_score(y_test, y_pred)

            # Cross-validation for more robust accuracy estimate
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            cv_accuracy = cv_scores.mean()

            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("📈 Model Training Results:")
            print(f"   Test Accuracy: {self.accuracy:.2%}")
            print(f"   Cross-Validation Accuracy: {cv_accuracy:.2%}")
            print(f"   Best Features: {', '.join(feature_importance['feature'].head(3).tolist())}")

            self.is_trained = True
            return self.accuracy

        except Exception as e:
            print(f"❌ Error during training: {e}")
            return 0.0

    def predict(self, features):
        """Make prediction for current market conditions"""
        if not self.is_trained:
            return {
                'prediction': 'MODEL_NOT_TRAINED',
                'confidence': 0.0,
                'direction': 'UNKNOWN',
                'probability_up': 0.5,
                'probability_down': 0.5
            }

        if features is None:
            return {
                'prediction': 'INSUFFICIENT_DATA',
                'confidence': 0.0,
                'direction': 'UNKNOWN',
                'probability_up': 0.5,
                'probability_down': 0.5
            }

        try:
            # Convert features to DataFrame with correct column order
            features_df = pd.DataFrame([features])[self.feature_names]

            # Scale features
            features_scaled = self.scaler.transform(features_df)

            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]

            confidence = max(probabilities)
            direction = 'UP' if prediction == 1 else 'DOWN'
            probability_up = probabilities[1]
            probability_down = probabilities[0]

            return {
                'prediction': direction,
                'confidence': confidence,
                'direction': direction,
                'probability_up': probability_up,
                'probability_down': probability_down,
                'features_used': len(features)
            }

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            # Fallback to simple prediction based on sentiment
            sentiment = features.get('sentiment_score', 0)
            if sentiment > 0.1:
                direction = 'UP'
                confidence = 0.6 + min(0.3, sentiment)
            elif sentiment < -0.1:
                direction = 'DOWN'
                confidence = 0.6 + min(0.3, -sentiment)
            else:
                direction = 'NEUTRAL'
                confidence = 0.5

            return {
                'prediction': direction,
                'confidence': confidence,
                'direction': direction,
                'probability_up': 0.5 if direction == 'NEUTRAL' else (
                    confidence if direction == 'UP' else 1 - confidence),
                'probability_down': 0.5 if direction == 'NEUTRAL' else (
                    1 - confidence if direction == 'UP' else confidence),
                'features_used': len(features),
                'fallback': True
            }

    def evaluate_model(self, test_data):
        """Comprehensive model evaluation with multiple metrics"""
        if not self.is_trained or test_data.empty:
            return "Model not trained or no test data available"

        try:
            X_test = test_data[self.feature_names]
            y_test = test_data['target']

            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            evaluation_results = {
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'feature_importance': feature_importance,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

            return evaluation_results

        except Exception as e:
            return f"Evaluation error: {e}"

    def save_model(self, filename='trained_predictor.joblib'):
        """Save the trained model and scaler"""
        if self.is_trained:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'accuracy': self.accuracy,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, filename)
            print(f"💾 Model saved as {filename}")

    def load_model(self, filename='trained_predictor.joblib'):
        """Load a pre-trained model and scaler"""
        try:
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.accuracy = model_data['accuracy']
            self.is_trained = model_data['is_trained']
            self.feature_names = model_data['feature_names']
            print(f"📂 Model loaded from {filename}")
            print(f"   Previous Accuracy: {self.accuracy:.2%}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")


# Model accuracy checker and tester
class ModelTester:
    def __init__(self, predictor):
        self.predictor = predictor

    def run_comprehensive_test(self):
        """Run comprehensive model testing"""
        print("🧪 Running comprehensive model tests...")

        # Generate test data
        test_data = self.predictor.prepare_training_data()

        if test_data.empty:
            return "No test data generated"

        # Split into train and test
        X = test_data[self.predictor.feature_names]
        y = test_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        train_data = pd.concat([X_train, y_train], axis=1)
        test_data_eval = pd.concat([X_test, y_test], axis=1)

        # Train model
        accuracy = self.predictor.train(train_data)

        # Comprehensive evaluation
        evaluation = self.predictor.evaluate_model(test_data_eval)

        if isinstance(evaluation, dict):
            print("📊 COMPREHENSIVE MODEL EVALUATION:")
            print(f"✅ Test Accuracy: {evaluation['accuracy']:.2%}")
            print(f"📈 Classification Report:\n{evaluation['classification_report']}")
            print(f"🎯 Top 5 Features:")
            print(evaluation['feature_importance'].head(5))

            return evaluation
        else:
            return evaluation

    def test_prediction_consistency(self, num_tests=10):
        """Test prediction consistency"""
        print(f"🔍 Testing prediction consistency ({num_tests} tests)...")

        test_features = {
            'current_price': 45000,
            'price_change_1h': 0.01,
            'price_change_24h': 0.05,
            'volume_change': 0.1,
            'volatility': 200,
            'ma_6h': 44800,
            'ma_12h': 44600,
            'sentiment_score': 0.2,
            'volume_ratio': 1.2,
            'rsi': 55,
            'macd': 2.5,
            'bollinger_position': 0.6,
            'price_momentum': 0.02,
            'volume_momentum': 0.15
        }

        predictions = []
        for i in range(num_tests):
            pred = self.predictor.predict(test_features)
            predictions.append(pred)
            print(f"   Test {i + 1}: {pred['prediction']} (Confidence: {pred['confidence']:.1%})")

        # Check consistency
        directions = [p['direction'] for p in predictions if p['direction'] in ['UP', 'DOWN']]
        if directions:
            consistent_direction = len(set(directions)) == 1
            avg_confidence = np.mean([p['confidence'] for p in predictions])

            print(f"   Consistent predictions: {consistent_direction}")
            print(f"   Average confidence: {avg_confidence:.2%}")
            print(f"   Most common direction: {max(set(directions), key=directions.count)}")

            return {
                'consistent': consistent_direction,
                'avg_confidence': avg_confidence,
                'common_direction': max(set(directions), key=directions.count)
            }

        return "No valid predictions generated"


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing Price Predictor...")

    # Create predictor
    predictor = PricePredictor()

    # Test model training
    print("\n1. Testing model training...")
    training_data = predictor.prepare_training_data()
    accuracy = predictor.train(training_data)

    if accuracy > 0:
        print(f"✅ Model trained successfully with {accuracy:.2%} accuracy")

        # Test predictions
        print("\n2. Testing predictions...")
        test_features = {
            'current_price': 45000,
            'price_change_1h': 0.01,
            'price_change_24h': 0.05,
            'volume_change': 0.1,
            'volatility': 200,
            'ma_6h': 44800,
            'ma_12h': 44600,
            'sentiment_score': 0.2,
            'volume_ratio': 1.2,
            'rsi': 55,
            'macd': 2.5,
            'bollinger_position': 0.6,
            'price_momentum': 0.02,
            'volume_momentum': 0.15
        }

        prediction = predictor.predict(test_features)
        print(f"📊 Prediction: {prediction}")

        # Run comprehensive tests
        print("\n3. Running comprehensive tests...")
        tester = ModelTester(predictor)
        comprehensive_results = tester.run_comprehensive_test()

        # Test consistency
        print("\n4. Testing prediction consistency...")
        consistency_results = tester.test_prediction_consistency()

        # Save model
        predictor.save_model()

    else:
        print("❌ Model training failed")