"""
Masters Tournament Predictor — End-to-End Pipeline
====================================================
Orchestrates the full prediction workflow:
1. Generate/load data
2. Engineer features (5 tiers)
3. Train 5 base models
4. Generate OOF predictions for meta-learner
5. Train meta-learner (stacking ensemble)
6. Predict for target tournament
7. Run 10,000 Monte Carlo simulations
8. Output rankings with probabilities
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR,
    TRAINING_YEARS, VALIDATION_YEAR, PREDICTION_YEAR,
    N_SIMULATIONS, RANDOM_STATE,
)
from data.generate_data import (
    generate_historical_data, generate_prediction_data,
    generate_prediction_data_2027, get_feature_columns,
)
from data.real_data import get_data_quality_report
from features.engineering import FeatureEngineer
from models.base_models import BaseModelSuite
from models.meta_learner import MetaLearner
from models.monte_carlo import MonteCarloSimulator


class MastersPipeline:
    """
    Full prediction pipeline for the Masters Tournament.
    """

    def __init__(self, target_year=2026, verbose=True):
        self.verbose = verbose
        self.target_year = target_year
        self.feature_engineer = FeatureEngineer()
        self.base_models = BaseModelSuite()
        self.meta_learner = MetaLearner()
        self.simulator = MonteCarloSimulator()

        # Data storage
        self.train_data = None
        self.pred_data = None
        self.X_train = None
        self.y_train = None
        self.X_pred = None

        # Results
        self.base_predictions = None
        self.meta_predictions = None
        self.meta_uncertainties = None
        self.simulation_results = None
        self.feature_importances = None

    def log(self, msg):
        if self.verbose:
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode('ascii', 'replace').decode())

    def generate_data(self):
        """Step 1: Generate training and prediction data."""
        self.log("\n" + "="*60)
        self.log("STEP 1: DATA GENERATION")
        self.log("="*60)

        self.log("  Generating historical training data (2015-2025)...")
        self.train_data = generate_historical_data()
        self.log(f"  -> {len(self.train_data)} training records")
        self.log(f"  -> {self.train_data['player_name'].nunique()} unique players")
        self.log(f"  -> Years: {sorted(self.train_data['year'].unique())}")

        self.log(f"\n  Generating {self.target_year} prediction data...")
        if self.target_year == 2027:
            self.pred_data = generate_prediction_data_2027()
        else:
            self.pred_data = generate_prediction_data()
        self.log(f"  -> {len(self.pred_data)} players in prediction field")

        # Show real data quality
        report = get_data_quality_report(self.pred_data)
        self.log(f"  -> Real data coverage: {report['real_data_pct']}% ({report['real_data_players']}/{report['total_players']} players)")

        # Save raw data
        os.makedirs(DATA_DIR, exist_ok=True)
        self.train_data.to_csv(os.path.join(DATA_DIR, "historical_stats.csv"), index=False)
        self.pred_data.to_csv(os.path.join(DATA_DIR, f"prediction_{self.target_year}.csv"), index=False)

    def engineer_features(self):
        """Step 2: Apply feature engineering across all 5 tiers."""
        self.log("\n" + "="*60)
        self.log("STEP 2: FEATURE ENGINEERING (5 Tiers)")
        self.log("="*60)

        # Extract target variable
        self.y_train = self.train_data["finish_position"].values

        # Fit on training data, transform both
        self.X_train = self.feature_engineer.fit_transform(self.train_data)
        self.X_pred = self.feature_engineer.transform(self.pred_data)

        self.log(f"  Feature matrix shape: {self.X_train.shape}")
        self.log(f"  Features per tier:")
        tier_map = self.feature_engineer.get_feature_tier_map()
        tier_counts = {}
        for feat, tier in tier_map.items():
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        for tier, count in sorted(tier_counts.items()):
            self.log(f"    {tier}: {count} features")

    def train_base_models(self):
        """Step 3: Train all 5 base ML models."""
        self.log("\n" + "="*60)
        self.log("STEP 3: TRAINING 5 BASE MODELS")
        self.log("="*60)

        self.base_models.fit(self.X_train.values, self.y_train, verbose=self.verbose)

        # Get feature importances
        self.feature_importances = self.base_models.get_feature_importance(
            self.feature_engineer.feature_names
        )

        # Show top features
        avg_imp = self.feature_importances["average"]
        top_features = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)[:10]
        self.log("\n  Top 10 most important features (avg across tree models):")
        for feat, imp in top_features:
            tier = self.feature_engineer.get_feature_tier_map().get(feat, "Unknown")
            self.log(f"    {feat:30s} {imp:.4f}  [{tier}]")

    def train_meta_learner(self):
        """Step 4: Generate OOF predictions and train meta-learner."""
        self.log("\n" + "="*60)
        self.log("STEP 4: TRAINING META-LEARNER (Stacking Ensemble)")
        self.log("="*60)

        # Generate out-of-fold predictions
        self.log("  Generating out-of-fold predictions for each base model...")
        oof_predictions = self.base_models.get_oof_predictions(
            self.X_train.values, self.y_train, verbose=self.verbose
        )

        # Train meta-learner on OOF predictions
        self.meta_learner.fit(oof_predictions, self.y_train, verbose=self.verbose)

    def predict(self):
        """Step 5: Generate predictions for the 2026 Masters field."""
        self.log("\n" + "="*60)
        self.log("STEP 5: GENERATING PREDICTIONS")
        self.log("="*60)

        # Get base model predictions
        self.base_predictions = self.base_models.predict(self.X_pred.values)

        self.log("  Base model prediction ranges:")
        for name, preds in self.base_predictions.items():
            self.log(f"    {name:15s}: [{preds.min():.1f}, {preds.max():.1f}] (mean={preds.mean():.1f})")

        # Get meta-learner final predictions
        self.meta_predictions, self.meta_uncertainties = self.meta_learner.predict_with_uncertainty(
            self.base_predictions
        )

        self.log(f"\n  Meta-learner prediction range: [{self.meta_predictions.min():.1f}, {self.meta_predictions.max():.1f}]")
        self.log(f"  Uncertainty range: [{self.meta_uncertainties.min():.2f}, {self.meta_uncertainties.max():.2f}]")

        # Show top 10 predicted finishers
        top_idx = np.argsort(self.meta_predictions)[:10]
        self.log("\n  Top 10 predicted finishers:")
        for rank, idx in enumerate(top_idx, 1):
            name = self.pred_data.iloc[idx]["player_name"]
            pred = self.meta_predictions[idx]
            unc = self.meta_uncertainties[idx]
            self.log(f"    {rank:2d}. {name:25s} Predicted: {pred:5.1f} (±{unc:.2f})")

    def run_monte_carlo(self):
        """Step 6: Run 10,000 Monte Carlo simulations."""
        self.log("\n" + "="*60)
        self.log("STEP 6: MONTE CARLO SIMULATION (10,000 tournaments)")
        self.log("="*60)

        player_names = self.pred_data["player_name"].values

        self.simulation_results = self.simulator.simulate(
            player_names=player_names,
            predicted_positions=self.meta_predictions,
            uncertainties=self.meta_uncertainties,
            verbose=self.verbose,
        )

        # Display top results
        self.log(f"\n  [TROPHY] FINAL PREDICTIONS -- {self.target_year} Masters Tournament")
        self.log("  " + "-" * 78)
        self.log(f"  {'Rank':>4s}  {'Player':<25s}  {'Win%':>6s}  {'Top5%':>6s}  {'Top10%':>7s}  {'Exp.Fin':>7s}")
        self.log("  " + "-" * 78)

        for i, row in self.simulation_results.head(20).iterrows():
            self.log(
                f"  {i+1:4d}  {row['player_name']:<25s}  "
                f"{row['win_probability']:5.1f}%  {row['top5_probability']:5.1f}%  "
                f"{row['top10_probability']:6.1f}%  {row['expected_finish']:7.1f}"
            )

    def save_results(self):
        """Save all results to files."""
        self.log("\n" + "="*60)
        self.log("SAVING RESULTS")
        self.log("="*60)

        # Save simulation results
        results_path = os.path.join(DATA_DIR, "simulation_results.csv")
        self.simulation_results.to_csv(results_path, index=False)
        self.log(f"  -> Simulation results: {results_path}")

        # Save feature importances (convert numpy types to native Python)
        imp_path = os.path.join(DATA_DIR, "feature_importances.json")
        serializable_imp = {}
        for model_name, feat_dict in self.feature_importances.items():
            serializable_imp[model_name] = {k: float(v) for k, v in feat_dict.items()}
        with open(imp_path, "w") as f:
            json.dump(serializable_imp, f, indent=2)
        self.log(f"  -> Feature importances: {imp_path}")

        # Save model weights
        weights = self.meta_learner.get_model_contributions()
        weights_path = os.path.join(DATA_DIR, "model_weights.json")
        with open(weights_path, "w") as f:
            json.dump({k: float(v) for k, v in weights.items()}, f, indent=2)
        self.log(f"  -> Model weights: {weights_path}")

        # Save pipeline state for Streamlit
        state = {
            "simulation_results": self.simulation_results,
            "feature_importances": self.feature_importances,
            "model_weights": self.meta_learner.get_model_contributions(),
            "pred_data": self.pred_data,
            "train_data": self.train_data,
            "base_predictions": self.base_predictions,
            "meta_predictions": self.meta_predictions,
            "meta_uncertainties": self.meta_uncertainties,
            "feature_names": self.feature_engineer.feature_names,
            "tier_map": self.feature_engineer.get_feature_tier_map(),
            "player_names": list(self.pred_data["player_name"].values),
            "simulator": self.simulator,
            "prediction_year": self.target_year,
        }
        state_path = os.path.join(DATA_DIR, "pipeline_state.pkl")
        with open(state_path, "wb") as f:
            pickle.dump(state, f)
        self.log(f"  -> Pipeline state: {state_path}")

    def run(self):
        """Execute the full pipeline."""
        start = datetime.now()
        self.log("\n" + "="*60)
        self.log("  MASTERS TOURNAMENT PREDICTOR")
        self.log("  Machine Learning Pipeline v1.0")
        self.log("="*60)
        self.log(f"\n  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")

        self.generate_data()
        self.engineer_features()
        self.train_base_models()
        self.train_meta_learner()
        self.predict()
        self.run_monte_carlo()
        self.save_results()

        elapsed = (datetime.now() - start).total_seconds()
        self.log(f"\n  Pipeline completed in {elapsed:.1f} seconds")
        self.log("  Results saved to data/ directory")
        self.log("\n  Run 'streamlit run app.py' to launch the dashboard!")

        return self.simulation_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Masters Tournament Predictor Pipeline")
    parser.add_argument("--year", type=int, default=2027, choices=[2026, 2027],
                        help="Prediction target year (default: 2027)")
    args = parser.parse_args()

    pipeline = MastersPipeline(target_year=args.year, verbose=True)
    results = pipeline.run()
