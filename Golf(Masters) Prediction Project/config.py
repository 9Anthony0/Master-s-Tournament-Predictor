"""
Masters Tournament Predictor — Configuration
=============================================
Central configuration for the entire prediction system.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "trained_models")

# ── Tournament Settings ───────────────────────────────────────────────
CURRENT_YEAR = 2026
TRAINING_YEARS = list(range(2015, 2025))   # Train on 2015-2024
VALIDATION_YEAR = 2025                      # Validate on 2025
PREDICTION_YEAR = 2026                      # Predict 2026

# ── Augusta National Course Info ──────────────────────────────────────
AUGUSTA_PAR = 72
AMEN_CORNER_HOLES = [11, 12, 13]
AMEN_CORNER_PARS = {11: 4, 12: 3, 13: 5}  # Total par = 12
TOTAL_HOLES = 18
TOTAL_ROUNDS = 4

# ── Feature Engineering ───────────────────────────────────────────────
# Weights for recent form calculation (most recent = highest weight)
FORM_WEIGHTS_5 = [0.35, 0.25, 0.20, 0.12, 0.08]
FORM_WEIGHTS_10 = [0.20, 0.15, 0.13, 0.11, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05]

# Amen Corner composite weights (Hole 11 hardest, 12 treacherous, 13 scoring opp)
AMEN_CORNER_WEIGHTS = {11: 0.40, 12: 0.35, 13: 0.25}

# Past champion recency decay factor
CHAMPION_DECAY_RATE = 0.85  # Per year

# ── Model Settings ────────────────────────────────────────────────────
RANDOM_STATE = 42
N_FOLDS = 5  # For cross-validation
TEST_SIZE = 0.2

# XGBoost defaults
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
}

# LightGBM defaults
LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

# Random Forest defaults
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
}

# Ridge Regression
RIDGE_PARAMS = {
    "alpha": 1.0,
}

# Neural Network (MLP)
MLP_PARAMS = {
    "hidden_layer_sizes": (128, 64, 32),
    "activation": "relu",
    "solver": "adam",
    "max_iter": 500,
    "learning_rate": "adaptive",
    "learning_rate_init": 0.001,
    "random_state": RANDOM_STATE,
}

# ── Monte Carlo Settings ─────────────────────────────────────────────
N_SIMULATIONS = 10_000
PERFORMANCE_STD_BASE = 2.5  # Base std dev for finishing position noise

# ── Streamlit Theme ──────────────────────────────────────────────────
MASTERS_GREEN = "#006747"
MASTERS_GREEN_LIGHT = "#00875A"
MASTERS_GREEN_DARK = "#004D35"
AUGUSTA_GOLD = "#FFD700"
AUGUSTA_YELLOW = "#FEDB00"
FAIRWAY_GREEN = "#228B22"
SAND_BUNKER = "#C2B280"
AZALEA_PINK = "#FF69B4"
WHITE = "#FFFFFF"
DARK_BG = "#0E1117"
CARD_BG = "#1A1F2E"
TEXT_PRIMARY = "#E8E8E8"
TEXT_SECONDARY = "#8B8B8B"
