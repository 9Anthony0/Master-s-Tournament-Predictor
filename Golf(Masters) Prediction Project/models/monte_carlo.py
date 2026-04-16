"""
Masters Tournament Predictor — Monte Carlo Simulation Engine
=============================================================
Runs 10,000 tournament simulations to convert point predictions
into win probabilities. Each simulation:
1. Draws a random performance from each player's distribution
2. Ranks all players
3. Records finishing positions

After all simulations, aggregates results into:
- Win probability (% of sims finishing 1st)
- Top 5 probability
- Top 10 probability
- Expected (mean) finish
- Median finish
"""

import numpy as np
import pandas as pd
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_SIMULATIONS, PERFORMANCE_STD_BASE, RANDOM_STATE


class MonteCarloSimulator:
    """
    Monte Carlo tournament simulator.

    Models each player's performance as a distribution centered on
    their predicted finishing position, with variance proportional to
    their historical consistency.
    """

    def __init__(self, n_simulations=N_SIMULATIONS, seed=RANDOM_STATE):
        self.n_simulations = n_simulations
        self.rng = np.random.RandomState(seed)
        self.simulation_results = None
        self.player_names = None

    def simulate(self, player_names, predicted_positions, uncertainties=None, verbose=True):
        """
        Run Monte Carlo tournament simulations.

        Args:
            player_names: list of player names
            predicted_positions: numpy array of predicted finishing positions
            uncertainties: numpy array of uncertainty estimates (optional)
                          If None, uses base std dev for all players.

        Returns:
            DataFrame with simulation summary statistics
        """
        n_players = len(player_names)
        self.player_names = player_names

        # Set up uncertainty (std dev of each player's performance distribution)
        if uncertainties is None:
            uncertainties = np.full(n_players, PERFORMANCE_STD_BASE)
        else:
            # Scale uncertainties to reasonable range
            uncertainties = np.clip(uncertainties * 1.5, 1.5, 8.0)

        if verbose:
            print(f"\n  Running {self.n_simulations:,} Monte Carlo simulations...")
            print(f"  Field size: {n_players} players")

        # Pre-allocate results matrix [n_simulations, n_players]
        finish_matrix = np.zeros((self.n_simulations, n_players), dtype=np.int32)

        for sim in range(self.n_simulations):
            # Draw random performances for each player
            performances = np.array([
                self.rng.normal(predicted_positions[i], uncertainties[i])
                for i in range(n_players)
            ])

            # Rank players (lower performance value = better finish)
            ranks = stats.rankdata(performances, method='ordinal').astype(np.int32)
            finish_matrix[sim] = ranks

            if verbose and (sim + 1) % 2500 == 0:
                print(f"    Completed {sim + 1:,} / {self.n_simulations:,} simulations")

        self.simulation_results = finish_matrix

        # ── Aggregate results ──────────────────────────────────────
        results = []
        for i, name in enumerate(player_names):
            finishes = finish_matrix[:, i]

            win_count = np.sum(finishes == 1)
            top5_count = np.sum(finishes <= 5)
            top10_count = np.sum(finishes <= 10)
            top20_count = np.sum(finishes <= 20)

            results.append({
                "player_name": name,
                "predicted_position": round(predicted_positions[i], 1),
                "win_probability": round(100.0 * win_count / self.n_simulations, 2),
                "top5_probability": round(100.0 * top5_count / self.n_simulations, 2),
                "top10_probability": round(100.0 * top10_count / self.n_simulations, 2),
                "top20_probability": round(100.0 * top20_count / self.n_simulations, 2),
                "expected_finish": round(np.mean(finishes), 1),
                "median_finish": int(np.median(finishes)),
                "best_sim_finish": int(np.min(finishes)),
                "worst_sim_finish": int(np.max(finishes)),
                "finish_std": round(np.std(finishes), 2),
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("win_probability", ascending=False).reset_index(drop=True)

        if verbose:
            print(f"\n  Simulation complete!")
            print(f"  Total win probability distributed: {results_df['win_probability'].sum():.1f}%")

        return results_df

    def get_player_distribution(self, player_name):
        """
        Get the full distribution of finishing positions for a player.
        Useful for visualization (histograms, etc.)
        """
        if self.simulation_results is None:
            raise ValueError("Must run simulate() first.")

        idx = list(self.player_names).index(player_name)
        return self.simulation_results[:, idx]

    def get_convergence_data(self, player_name, checkpoints=50):
        """
        Track how a player's win probability converges as simulations increase.
        Returns data points for convergence visualization.
        """
        if self.simulation_results is None:
            raise ValueError("Must run simulate() first.")

        idx = list(self.player_names).index(player_name)
        finishes = self.simulation_results[:, idx]

        step = max(1, self.n_simulations // checkpoints)
        data = []
        for n in range(step, self.n_simulations + 1, step):
            subset = finishes[:n]
            win_pct = 100.0 * np.sum(subset == 1) / n
            top5_pct = 100.0 * np.sum(subset <= 5) / n
            data.append({
                "n_simulations": n,
                "win_probability": win_pct,
                "top5_probability": top5_pct,
            })

        return pd.DataFrame(data)

    def head_to_head(self, player1, player2):
        """
        Calculate head-to-head statistics between two players.
        """
        if self.simulation_results is None:
            raise ValueError("Must run simulate() first.")

        idx1 = list(self.player_names).index(player1)
        idx2 = list(self.player_names).index(player2)

        f1 = self.simulation_results[:, idx1]
        f2 = self.simulation_results[:, idx2]

        p1_wins = np.sum(f1 < f2)
        p2_wins = np.sum(f2 < f1)
        ties = np.sum(f1 == f2)

        return {
            "player1": player1,
            "player2": player2,
            "p1_win_pct": round(100 * p1_wins / self.n_simulations, 1),
            "p2_win_pct": round(100 * p2_wins / self.n_simulations, 1),
            "tie_pct": round(100 * ties / self.n_simulations, 1),
        }
