import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from prediction_model import PricePredictor
import plotly.graph_objects as go
import plotly.express as px


class ModelEvaluator:
    def __init__(self, predictor):
        self.predictor = predictor

    def comprehensive_evaluation(self):
        """Run comprehensive model evaluation with visualizations"""
        print("📊 Running Comprehensive Model Evaluation...")

        # Generate test data
        test_data = self.predictor.prepare_training_data()

        if test_data.empty:
            return "No test data available"

        # Split data
        from sklearn.model_selection import train_test_split
        X = test_data[self.predictor.feature_names]
        y = test_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        train_data = pd.concat([X_train, y_train], axis=1)
        test_data_eval = pd.concat([X_test, y_test], axis=1)

        # Train model if not already trained
        if not self.predictor.is_trained:
            accuracy = self.predictor.train(train_data)
            print(f"✅ Model trained with accuracy: {accuracy:.2%}")

        # Get evaluation results
        evaluation = self.predictor.evaluate_model(test_data_eval)

        if isinstance(evaluation, dict):
            self._create_evaluation_report(evaluation, test_data_eval)
            return evaluation
        else:
            return evaluation

    def _create_evaluation_report(self, evaluation, test_data):
        """Create comprehensive evaluation report with visualizations"""

        # 1. Basic Metrics
        print("\n" + "=" * 60)
        print("🎯 MODEL EVALUATION REPORT")
        print("=" * 60)
        print(f"📈 Accuracy: {evaluation['accuracy']:.2%}")
        print(f"📊 Classification Report:")
        print(evaluation['classification_report'])

        # 2. Feature Importance Visualization
        self._plot_feature_importance(evaluation['feature_importance'])

        # 3. Confusion Matrix
        self._plot_confusion_matrix(evaluation['confusion_matrix'])

        print("\n✅ Evaluation complete!")

    def _plot_feature_importance(self, feature_importance):
        """Plot feature importance"""
        fig = px.bar(
            feature_importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Most Important Features',
            color='importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False, height=400)
        fig.show()

    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale='Blues',
            title='Confusion Matrix',
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        fig.update_xaxes(side="top")
        fig.show()


# Run evaluation
if __name__ == "__main__":
    print("🚀 Starting Model Evaluation...")

    # Create and train predictor
    predictor = PricePredictor()
    training_data = predictor.prepare_training_data()
    accuracy = predictor.train(training_data)

    if accuracy > 0.5:  # Only evaluate if model is somewhat accurate
        # Run comprehensive evaluation
        evaluator = ModelEvaluator(predictor)
        results = evaluator.comprehensive_evaluation()

        print(f"\n🎯 Final Model Performance:")
        print(f"   Training Accuracy: {accuracy:.2%}")
        if isinstance(results, dict):
            print(f"   Test Accuracy: {results['accuracy']:.2%}")
    else:
        print("❌ Model accuracy too low for proper evaluation")