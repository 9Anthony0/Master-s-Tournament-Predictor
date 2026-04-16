"""
Masters Tournament Predictor — Base ML Models
===============================================
Five diverse machine learning models that each capture
different patterns in the data.

1. XGBoost       — gradient boosting, non-linear interactions
2. LightGBM      — alternative gradient boosting, different splits
3. Random Forest  — bagging ensemble, robust to overfitting
4. Ridge Regression — linear model, regularized
5. Neural Network  — MLP, abstract representations
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_predict, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, RF_PARAMS,
    RIDGE_PARAMS, MLP_PARAMS, RANDOM_STATE, N_FOLDS,
)


class BaseModelSuite:
    """
    Suite of 5 base ML models for predicting Masters finishing positions.
    Each model sees the same features but captures different patterns.
    """

    def __init__(self):
        self.models = {
            "xgboost": XGBRegressor(**XGBOOST_PARAMS),
            "lightgbm": LGBMRegressor(**LIGHTGBM_PARAMS),
            "random_forest": RandomForestRegressor(**RF_PARAMS),
            "ridge": Ridge(**RIDGE_PARAMS),
            "neural_net": MLPRegressor(**MLP_PARAMS),
        }
        self.model_names = list(self.models.keys())
        self.is_fitted = False

    def fit(self, X, y, verbose=True):
        """Train all 5 base models."""
        for name, model in self.models.items():
            if verbose:
                print(f"  Training {name}...")
            model.fit(X, y)
        self.is_fitted = True
        if verbose:
            print("  All base models trained.")

    def predict(self, X):
        """Get predictions from all 5 models. Returns dict of arrays."""
        if not self.is_fitted:
            raise ValueError("Models must be fitted first.")

        predictions = {}
        for name, model in self.models.items():
            preds = model.predict(X)
            # Clip to valid range [1, 90]
            preds = np.clip(preds, 1, 90)
            predictions[name] = preds
        return predictions

    def get_oof_predictions(self, X, y, verbose=True):
        """
        Generate out-of-fold predictions using k-fold CV.
        These are used to train the meta-learner without data leakage.
        """
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        oof_predictions = {}

        for name, model in self.models.items():
            if verbose:
                print(f"  Generating OOF predictions for {name}...")

            oof_preds = cross_val_predict(model, X, y, cv=kf)
            oof_preds = np.clip(oof_preds, 1, 90)
            oof_predictions[name] = oof_preds

        return oof_predictions

    def get_feature_importance(self, feature_names):
        """Get feature importances from tree-based models."""
        importances = {}

        # XGBoost
        xgb_imp = self.models["xgboost"].feature_importances_
        importances["xgboost"] = dict(zip(feature_names, xgb_imp))

        # LightGBM
        lgb_imp = self.models["lightgbm"].feature_importances_
        lgb_imp = lgb_imp / lgb_imp.sum()  # Normalize
        importances["lightgbm"] = dict(zip(feature_names, lgb_imp))

        # Random Forest
        rf_imp = self.models["random_forest"].feature_importances_
        importances["random_forest"] = dict(zip(feature_names, rf_imp))

        # Average across tree models
        avg_imp = {}
        for feat in feature_names:
            avg_imp[feat] = np.mean([
                importances["xgboost"].get(feat, 0),
                importances["lightgbm"].get(feat, 0),
                importances["random_forest"].get(feat, 0),
            ])
        importances["average"] = avg_imp

        return importances
