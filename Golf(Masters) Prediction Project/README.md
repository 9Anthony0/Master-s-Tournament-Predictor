# Masters Tournament Predictor (Inspired by https://www.instagram.com/reel/DXGbTvGCCO3/?igsh=YTlmbjJpc2pzM2Vo)

A machine learning prediction system for the Masters Tournament at Augusta National Golf Club.
Uses **5 base ML models**, a **stacking ensemble meta-learner**, and **10,000 Monte Carlo simulations**
to predict win probabilities for every player in the field.

## Architecture

```
Data Layer → Feature Engineering (5 Tiers) → 5 Base Models → Meta-Learner → Monte Carlo → Dashboard
```

### 5 Feature Tiers
1. **Strokes Gained** — SG: Off the Tee, Approach, Around Green, Putting, Total
2. **Augusta Experience** — Masters appearances, cuts made, rookie status
3. **Past Winner Status** — Champion flag, number of wins, recency decay
4. **Amen Corner Composite** — Scoring on holes 11 (White Dogwood), 12 (Golden Bell), 13 (Azalea)
5. **Recent Form & Rankings** — OWGR, recent tournament finishes, momentum, top-10 rate

### 5 Base ML Models
| Model | Purpose |
|-------|---------|
| XGBoost | Non-linear feature interactions via gradient boosting |
| LightGBM | Alternative gradient boosting with different splitting strategy |
| Random Forest | Bagging ensemble — robust against overfitting |
| Ridge Regression | Regularized linear model for linear relationships |
| Neural Network (MLP) | Multi-layer perceptron for abstract pattern learning |

### Meta-Learner
A **stacking ensemble** that takes the 5 base model predictions as input and learns which model
to trust most. Uses Ridge Regression as the meta-model to avoid overfitting.

### Monte Carlo Simulation
Runs **10,000 tournament simulations** where each player's performance is drawn from a normal
distribution centered on their predicted finish. Outputs:
- Win probability (% finishing 1st)
- Top 5, 10, 20 probabilities
- Expected and median finish positions

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the ML pipeline (generates data, trains models, runs simulations)
python pipeline.py

# Launch the interactive dashboard
python -m streamlit run app.py
```

## Dashboard Features
- **Prediction Leaderboard** — Sortable rankings with win probabilities
- **Monte Carlo Analysis** — Simulation distributions and head-to-head comparisons
- **Player Deep-Dive** — Radar chart of 5 feature tiers, detailed stats
- **Model Insights** — Feature importances, meta-learner weights, base model comparisons
- **Simulation Convergence** — Watch probabilities stabilize over N simulations

## Project Structure
```
├── config.py                    # Configuration constants & hyperparameters
├── pipeline.py                  # End-to-end ML pipeline orchestration
├── app.py                       # Streamlit dashboard
├── requirements.txt
├── data/
│   ├── player_database.py       # Real player profiles (~90 players)
│   ├── generate_data.py         # Training data generation
│   └── [generated CSVs]
├── features/
│   └── engineering.py           # 5-tier feature engineering
└── models/
    ├── base_models.py           # 5 base ML models
    ├── meta_learner.py          # Stacking ensemble
    └── monte_carlo.py           # 10,000 simulation engine
```

## Data Sources
- Player profiles based on publicly available information (OWGR rankings, Masters history)
- Historical Masters winners from public records
- Playing style indicators calibrated to known player characteristics
