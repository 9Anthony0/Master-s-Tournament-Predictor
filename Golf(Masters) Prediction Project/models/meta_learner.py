"""
Masters Tournament Predictor — Meta-Learner (Stacking Ensemble)
================================================================
The meta-learner sits on top of the 5 base models and learns
which model to trust most for different types of players.
It takes the base model predictions as input features and
produces a final, refined prediction.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_STATE, N_FOLDS


class MetaLearner:
    """
    Stacking ensemble meta-learner.

    Takes predictions from 5 base models as input features
    and learns the optimal weighting/combination to produce
    the best final prediction.

    Uses Ridge regression as the meta-model (simple, effective,
    low risk of overfitting the already-processed predictions).
    """

    def __init__(self, use_xgb_meta=False):
        """
        Args:
            use_xgb_meta: If True, use XGBoost as meta-model instead of Ridge.
                         Ridge is default (simpler, less overfitting risk).
        """
        if use_xgb_meta:
            self.meta_model = XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=RANDOM_STATE,
            )
        else:
            self.meta_model = Ridge(alpha=1.0)

        self.model_names = []
        self.model_weights = None
        self.is_fitted = False

    def fit(self, oof_predictions, y, verbose=True):
        """
        Train the meta-learner on out-of-fold predictions.

        Args:
            oof_predictions: dict of {model_name: np.array of predictions}
            y: true target values
        """
        self.model_names = list(oof_predictions.keys())

        # Stack predictions into a matrix [n_samples, n_models]
        X_meta = np.column_stack([oof_predictions[name] for name in self.model_names])

        if verbose:
            print(f"  Training meta-learner on {X_meta.shape[1]} base model predictions...")

        self.meta_model.fit(X_meta, y)
        self.is_fitted = True

        # Extract learned weights (for interpretability)
        if hasattr(self.meta_model, 'coef_'):
            self.model_weights = dict(zip(self.model_names, self.meta_model.coef_))
            if verbose:
                print("  Meta-learner weights:")
                for name, w in self.model_weights.items():
                    print(f"    {name:15s}: {w:+.4f}")

        # Cross-validation score
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            self.meta_model, X_meta, y,
            cv=kf, scoring="neg_mean_absolute_error"
        )
        if verbose:
            print(f"  Meta-learner CV MAE: {-cv_scores.mean():.2f} (±{cv_scores.std():.2f})")

    def predict(self, base_predictions):
        """
        Generate final predictions using the meta-learner.

        Args:
            base_predictions: dict of {model_name: np.array of predictions}

        Returns:
            np.array of final predicted finishing positions
        """
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted first.")

        X_meta = np.column_stack([base_predictions[name] for name in self.model_names])
        predictions = self.meta_model.predict(X_meta)
        return np.clip(predictions, 1, 90)

    def get_model_contributions(self):
        """Get the learned contribution/weight of each base model."""
        if self.model_weights is None:
            return {name: 1.0 / len(self.model_names) for name in self.model_names}
        return self.model_weights

    def predict_with_uncertainty(self, base_predictions):
        """
        Predict with uncertainty estimate.
        Uses disagreement between base models as uncertainty proxy.
        """
        predictions = self.predict(base_predictions)

        # Uncertainty = std deviation across base model predictions
        all_preds = np.column_stack([base_predictions[name] for name in self.model_names])
        uncertainty = np.std(all_preds, axis=1)

        return predictions, uncertainty
