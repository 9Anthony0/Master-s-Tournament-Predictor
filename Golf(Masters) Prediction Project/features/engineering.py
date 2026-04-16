"""
Masters Tournament Predictor — Feature Engineering
====================================================
Constructs all 5 tiers of features from raw data.
Each tier captures a different dimension of player ability
relevant to Augusta National performance.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatureEngineer:
    """
    Builds and transforms all 5 feature tiers.

    Tier 1: Strokes Gained (core performance metrics)
    Tier 2: Augusta Experience (course familiarity)
    Tier 3: Past Winner Status (champion advantage)
    Tier 4: Amen Corner Composite (Augusta-specific)
    Tier 5: Recent Form & Rankings (current momentum)
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []

    def get_tier1_features(self):
        """Strokes Gained metrics — core performance indicators."""
        return [
            "sg_off_tee", "sg_approach", "sg_around_green",
            "sg_putting", "sg_tee_to_green", "sg_total",
        ]

    def get_tier2_features(self):
        """Augusta Experience — course familiarity indicators."""
        return [
            "masters_appearances", "years_since_first",
            "cuts_made_pct", "is_rookie",
        ]

    def get_tier3_features(self):
        """Past Winner Status — champion advantage."""
        return [
            "is_past_champion", "num_masters_wins",
            "years_since_last_win", "champion_recency_score",
        ]

    def get_tier4_features(self):
        """Amen Corner Composite — Augusta-specific scoring."""
        return [
            "hole11_avg", "hole12_avg", "hole13_avg",
            "amen_corner_avg", "amen_corner_vs_par",
            "amen_corner_composite",
        ]

    def get_tier5_features(self):
        """Recent Form & Rankings — current momentum."""
        return [
            "owgr_normalized", "recent_form_5", "recent_form_10",
            "momentum_score", "top10_rate_season",
        ]

    def get_all_features(self):
        """All feature names across all tiers."""
        return (
            self.get_tier1_features() +
            self.get_tier2_features() +
            self.get_tier3_features() +
            self.get_tier4_features() +
            self.get_tier5_features()
        )

    def add_interaction_features(self, df):
        """Create interaction features that capture synergies."""
        df = df.copy()

        # Power × Course Experience — power players who know Augusta dominate
        df["power_x_experience"] = df["sg_off_tee"] * np.log1p(df["masters_appearances"])

        # Iron Play × Amen Corner — precise players handle Amen Corner better
        df["iron_x_amen"] = df["sg_approach"] * (-df["amen_corner_vs_par"])

        # Champion × Form — past winners in good form are dangerous
        df["champion_x_form"] = df["champion_recency_score"] * (1.0 / (1.0 + df["recent_form_5"] / 50))

        # Total SG × Experience — skilled + experienced is the best combo
        df["sg_x_experience"] = df["sg_total"] * np.log1p(df["masters_appearances"])

        # Putting × Short Game — finishing ability
        df["finishing_ability"] = df["sg_putting"] + df["sg_around_green"]

        return df

    def get_interaction_features(self):
        """Names of interaction features."""
        return [
            "power_x_experience", "iron_x_amen", "champion_x_form",
            "sg_x_experience", "finishing_ability",
        ]

    def fit_transform(self, df):
        """Fit scaler on training data and transform."""
        df = self.add_interaction_features(df)
        self.feature_names = self.get_all_features() + self.get_interaction_features()

        X = df[self.feature_names].copy()

        # Handle any infinities or NaNs
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True

        return pd.DataFrame(X_scaled, columns=self.feature_names, index=df.index)

    def transform(self, df):
        """Transform new data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted first. Call fit_transform().")

        df = self.add_interaction_features(df)
        X = df[self.feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=df.index)

    def get_feature_tier_map(self):
        """Return a dict mapping feature name → tier label."""
        tier_map = {}
        for f in self.get_tier1_features():
            tier_map[f] = "Tier 1: Strokes Gained"
        for f in self.get_tier2_features():
            tier_map[f] = "Tier 2: Augusta Experience"
        for f in self.get_tier3_features():
            tier_map[f] = "Tier 3: Past Winner"
        for f in self.get_tier4_features():
            tier_map[f] = "Tier 4: Amen Corner"
        for f in self.get_tier5_features():
            tier_map[f] = "Tier 5: Form & Rankings"
        for f in self.get_interaction_features():
            tier_map[f] = "Interaction Features"
        return tier_map
