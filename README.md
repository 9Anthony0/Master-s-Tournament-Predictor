# 2027 Masters Tournament Predictor

A machine learning and simulation engine that estimates every player's chance of winning the Masters at Augusta National. Five base models feed a stacking meta-learner, and 10,000 Monte Carlo tournaments turn predicted finishes into win, top-5, top-10 and top-20 probabilities.

🚀 **[Try the live dashboard](https://petrescu-masters.streamlit.app)**

📝 **Write-up:** [Predicting the 2027 Masters: Building a Machine Learning & Simulation Engine from the Ground Up](https://9numbers0.substack.com/p/predicting-the-2027-masters-building)

## How it works

```
Player data → 5-tier feature engineering → 5 base models → stacking meta-learner → 10,000 Monte Carlo runs → Streamlit dashboard
```

### Feature tiers
| Tier | What it captures |
|------|------------------|
| Strokes Gained | Off the tee, approach, around the green, putting, total |
| Augusta Experience | Masters appearances, cuts made, rookie penalty |
| Past Winner Status | Champion flag, number of wins, recency decay |
| Amen Corner Composite | Scoring on holes 11, 12 and 13 |
| Recent Form & Rankings | OWGR, recent finishes, momentum, top-10 rate |

### Models
| Model | Role |
|-------|------|
| XGBoost | Non-linear feature interactions (gradient boosting) |
| LightGBM | Gradient boosting with a different split strategy |
| Random Forest | Bagging for stability |
| Ridge Regression | Regularized linear trends |
| Neural Network (MLP) | Abstract pattern learning |
| **Meta-learner (Ridge)** | Learns how much to trust each base model |

### Simulation
Each of 10,000 simulated tournaments draws every player's finish from a normal distribution centered on the ensemble prediction. The output is win probability, top-5/10/20 probability, and expected and median finish.

## Dashboard
- **Prediction leaderboard** sorted by simulated win probability
- **Monte Carlo analysis** with finish distributions and head-to-head comparisons
- **Player deep-dive** radar charts across the five feature tiers
- **Model insights**: feature importances and meta-learner weights
- **Simulation convergence**: watch probabilities stabilize as runs accumulate

## Run it locally

```bash
pip install -r requirements.txt
cd "Golf(Masters) Prediction Project"

# Optional: regenerate data, retrain models and rerun simulations
python pipeline.py

# Launch the dashboard (uses the saved results in data/)
python -m streamlit run app.py
```

## Project structure
```
Golf(Masters) Prediction Project/
├── app.py                  # Streamlit dashboard
├── pipeline.py             # End-to-end pipeline: data → features → models → simulation
├── config.py               # Constants and hyperparameters
├── data/                   # Player database, generated data, saved results
├── features/engineering.py # 5-tier feature engineering
└── models/                 # Base models, meta-learner, Monte Carlo engine
```

## Tech
Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · Plotly · Streamlit

## Notes
Player profiles are built from public information (OWGR, Masters history). Inspired by a 2026 Masters predictor I saw shared online. This version extends it to 2027 with a stacked ensemble.

---
Built by [Anthony Petrescu](https://www.linkedin.com/in/anthony-petrescu/) · Purdue Finance + Computer Science '27
