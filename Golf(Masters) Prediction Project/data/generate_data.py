"""
Masters Tournament Predictor — Data Generator
===============================================
Generates training data from player profiles, enriched with
REAL statistics where available. Creates historical tournament
performances and forward-looking prediction datasets.

Data hierarchy:
  1. Real stats (from real_data.py) — used when available
  2. Calibrated estimates (from player style ratings) — fallback
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAINING_YEARS, VALIDATION_YEAR, PREDICTION_YEAR,
    AUGUSTA_PAR, AMEN_CORNER_PARS, RANDOM_STATE,
)
from data.player_database import get_player_profiles, get_masters_winners
from data.real_data import get_real_stats, get_masters_2026_results, get_data_quality_report


np.random.seed(RANDOM_STATE)


def _style_to_sg(style_rating, noise_std=0.3):
    """Convert a 1-10 style rating to a strokes gained value.
    PGA Tour SG mean=0, std~1.0-1.5. Rating 5.5 ≈ average (SG=0)."""
    base_sg = (style_rating - 5.5) * 0.45  # Maps 1→-2.0, 10→+2.0
    return base_sg + np.random.normal(0, noise_std)


def _rank_to_skill(owgr, noise_std=0.15):
    """Convert OWGR to a skill multiplier. #1 → ~1.3, #50 → ~1.0, #200+ → ~0.6"""
    if owgr <= 0:
        owgr = 100
    raw = 1.35 - 0.45 * np.log10(owgr)
    return max(0.3, raw + np.random.normal(0, noise_std))


def _generate_amen_corner_scores(player, year, skill_mult):
    """Generate Amen Corner hole scores for a player-year."""
    # Hole 11 (par 4) — hardest, avg ~4.30
    h11_base = 4.30 - (player["style_accuracy"] - 5.5) * 0.08
    h11_base -= (player["style_iron_play"] - 5.5) * 0.06
    h11 = max(3, h11_base + np.random.normal(0, 0.25))

    # Hole 12 (par 3) — treacherous, avg ~3.15
    h12_base = 3.15 - (player["style_iron_play"] - 5.5) * 0.07
    h12_base -= (player["style_course_mgmt"] - 5.5) * 0.05
    h12 = max(2, h12_base + np.random.normal(0, 0.30))

    # Hole 13 (par 5) — scoring opportunity, avg ~4.65
    h13_base = 4.65 - (player["style_power"] - 5.5) * 0.08
    h13_base -= (player["style_short_game"] - 5.5) * 0.05
    h13 = max(3, h13_base + np.random.normal(0, 0.25))

    return {"hole11_avg": h11, "hole12_avg": h12, "hole13_avg": h13}


def _generate_recent_form(player, year, n=10):
    """Generate recent tournament finishes (n tournaments before Masters)."""
    base_finish = min(80, max(1, player["owgr"] * 0.6))
    form = player.get("recent_form", 5.0)
    consistency = player.get("consistency", 5.0)

    finishes = []
    for i in range(n):
        noise_std = max(3, 20 - consistency * 1.5)
        finish = base_finish - (form - 5.0) * 3.0 + np.random.normal(0, noise_std)
        finish = max(1, min(80, finish))
        finishes.append(round(finish))

    return finishes


def generate_historical_data():
    """
    Generate historical Masters data (2015-2025) for all players.
    Returns a DataFrame with one row per player-year observation.
    """
    players = get_player_profiles()
    winners = get_masters_winners()
    all_years = TRAINING_YEARS + [VALIDATION_YEAR]
    records = []

    for year in all_years:
        # Determine which players were in the field that year
        for player in players:
            # Skip if player wasn't active that year
            first_possible_year = 2026 - player["masters_appearances"]
            if year < first_possible_year:
                continue

            # Very old / very low ranked players may not have played every year
            if player["owgr"] > 100 and np.random.random() > 0.5:
                continue

            skill_mult = _rank_to_skill(player["owgr"])

            # ── Tier 1: Strokes Gained ──────────────────────────────
            # Add year-to-year variation
            year_noise = np.random.normal(0, 0.15)
            sg_ott = _style_to_sg(player["style_power"]) * skill_mult + year_noise
            sg_app = _style_to_sg(player["style_iron_play"]) * skill_mult + year_noise
            sg_arg = _style_to_sg(player["style_short_game"]) * skill_mult + year_noise
            sg_putt = _style_to_sg(player["style_putting"]) * skill_mult + year_noise
            sg_t2g = sg_ott + sg_app + sg_arg
            sg_total = sg_t2g + sg_putt

            # ── Tier 2: Augusta Experience (as of that year) ────────
            years_playing = year - first_possible_year + 1
            appearances = min(years_playing, player["masters_appearances"])
            cuts_made = max(0, int(appearances * (0.5 + skill_mult * 0.3) + np.random.normal(0, 1)))
            cuts_made = min(cuts_made, appearances)
            cuts_made_pct = cuts_made / max(1, appearances)
            is_rookie = 1 if years_playing <= 1 else 0

            # ── Tier 3: Past Winner Status (as of that year) ────────
            wins_before = [y for y in player.get("years_won", []) if y < year]
            is_past_champion = 1 if len(wins_before) > 0 else 0
            num_wins = len(wins_before)
            years_since_last_win = (year - max(wins_before)) if wins_before else 99
            champion_recency = sum(0.85 ** (year - y) for y in wins_before)

            # ── Tier 4: Amen Corner ─────────────────────────────────
            amen = _generate_amen_corner_scores(player, year, skill_mult)
            amen_avg = (amen["hole11_avg"] + amen["hole12_avg"] + amen["hole13_avg"]) / 3
            amen_vs_par = (
                (amen["hole11_avg"] - AMEN_CORNER_PARS[11]) +
                (amen["hole12_avg"] - AMEN_CORNER_PARS[12]) +
                (amen["hole13_avg"] - AMEN_CORNER_PARS[13])
            )
            amen_composite = (
                amen["hole11_avg"] * 0.40 +
                amen["hole12_avg"] * 0.35 +
                amen["hole13_avg"] * 0.25
            )

            # ── Tier 5: Recent Form & Rankings ──────────────────────
            recent_finishes = _generate_recent_form(player, year)
            w5 = [0.35, 0.25, 0.20, 0.12, 0.08]
            w10 = [0.20, 0.15, 0.13, 0.11, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05]
            recent_form_5 = sum(f * w for f, w in zip(recent_finishes[:5], w5))
            recent_form_10 = sum(f * w for f, w in zip(recent_finishes, w10))

            # Momentum: are recent results improving?
            if len(recent_finishes) >= 5:
                first_half_avg = np.mean(recent_finishes[:5])
                second_half_avg = np.mean(recent_finishes[5:]) if len(recent_finishes) > 5 else first_half_avg
                momentum = (second_half_avg - first_half_avg) / max(1, second_half_avg)
            else:
                momentum = 0

            top10_count = sum(1 for f in recent_finishes if f <= 10)
            top10_rate = top10_count / len(recent_finishes)

            owgr_normalized = 1.0 / (1.0 + np.log10(max(1, player["owgr"])))

            # ── Target: Finishing Position ──────────────────────────
            # Calculate expected finishing position based on player quality
            base_pos = 1.0 + 50.0 * (1.0 - skill_mult)
            base_pos += (10.0 - player.get("recent_form", 5.0)) * 3.0
            base_pos -= is_past_champion * 5.0
            base_pos -= (appearances > 5) * 3.0
            base_pos += is_rookie * 8.0

            # If this player won this year, force position 1
            if winners.get(year) == player["name"]:
                finish_position = 1
            else:
                finish_position = max(2, int(base_pos + np.random.normal(0, 12)))
                finish_position = min(finish_position, 80)

            # Did they make the cut?
            made_cut = 1 if finish_position <= 50 else 0

            record = {
                "year": year,
                "player_name": player["name"],
                "owgr": player["owgr"],
                "age": player["age"] - (2026 - year),
                # Tier 1
                "sg_off_tee": round(sg_ott, 3),
                "sg_approach": round(sg_app, 3),
                "sg_around_green": round(sg_arg, 3),
                "sg_putting": round(sg_putt, 3),
                "sg_tee_to_green": round(sg_t2g, 3),
                "sg_total": round(sg_total, 3),
                # Tier 2
                "masters_appearances": appearances,
                "years_since_first": years_playing,
                "cuts_made_pct": round(cuts_made_pct, 3),
                "is_rookie": is_rookie,
                # Tier 3
                "is_past_champion": is_past_champion,
                "num_masters_wins": num_wins,
                "years_since_last_win": years_since_last_win,
                "champion_recency_score": round(champion_recency, 4),
                # Tier 4
                "hole11_avg": round(amen["hole11_avg"], 3),
                "hole12_avg": round(amen["hole12_avg"], 3),
                "hole13_avg": round(amen["hole13_avg"], 3),
                "amen_corner_avg": round(amen_avg, 3),
                "amen_corner_vs_par": round(amen_vs_par, 3),
                "amen_corner_composite": round(amen_composite, 3),
                # Tier 5
                "owgr_normalized": round(owgr_normalized, 4),
                "recent_form_5": round(recent_form_5, 2),
                "recent_form_10": round(recent_form_10, 2),
                "momentum_score": round(momentum, 4),
                "top10_rate_season": round(top10_rate, 3),
                # Target
                "finish_position": finish_position,
                "made_cut": made_cut,
            }
            records.append(record)

    df = pd.DataFrame(records)
    return df


def generate_prediction_data():
    """
    Generate feature data for the prediction year (2026).
    Same features but no target variable.
    """
    players = get_player_profiles()
    winners = get_masters_winners()
    year = PREDICTION_YEAR
    records = []

    for player in players:
        skill_mult = _rank_to_skill(player["owgr"])
        year_noise = np.random.normal(0, 0.10)

        sg_ott = _style_to_sg(player["style_power"], noise_std=0.2) * skill_mult + year_noise
        sg_app = _style_to_sg(player["style_iron_play"], noise_std=0.2) * skill_mult + year_noise
        sg_arg = _style_to_sg(player["style_short_game"], noise_std=0.2) * skill_mult + year_noise
        sg_putt = _style_to_sg(player["style_putting"], noise_std=0.2) * skill_mult + year_noise
        sg_t2g = sg_ott + sg_app + sg_arg
        sg_total = sg_t2g + sg_putt

        appearances = player["masters_appearances"]
        years_playing = player["masters_appearances"]
        cuts_pct = 0.5 + skill_mult * 0.3
        is_rookie = 1 if appearances <= 1 else 0

        wins_before = [y for y in player.get("years_won", []) if y < year]
        is_past_champion = 1 if len(wins_before) > 0 else 0
        num_wins = len(wins_before)
        years_since_last_win = (year - max(wins_before)) if wins_before else 99
        champion_recency = sum(0.85 ** (year - y) for y in wins_before)

        amen = _generate_amen_corner_scores(player, year, skill_mult)
        amen_avg = (amen["hole11_avg"] + amen["hole12_avg"] + amen["hole13_avg"]) / 3
        amen_vs_par = (
            (amen["hole11_avg"] - AMEN_CORNER_PARS[11]) +
            (amen["hole12_avg"] - AMEN_CORNER_PARS[12]) +
            (amen["hole13_avg"] - AMEN_CORNER_PARS[13])
        )
        amen_composite = (
            amen["hole11_avg"] * 0.40 +
            amen["hole12_avg"] * 0.35 +
            amen["hole13_avg"] * 0.25
        )

        recent_finishes = _generate_recent_form(player, year)
        w5 = [0.35, 0.25, 0.20, 0.12, 0.08]
        w10 = [0.20, 0.15, 0.13, 0.11, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05]
        recent_form_5 = sum(f * w for f, w in zip(recent_finishes[:5], w5))
        recent_form_10 = sum(f * w for f, w in zip(recent_finishes, w10))

        if len(recent_finishes) >= 5:
            first_half_avg = np.mean(recent_finishes[:5])
            second_half_avg = np.mean(recent_finishes[5:]) if len(recent_finishes) > 5 else first_half_avg
            momentum = (second_half_avg - first_half_avg) / max(1, second_half_avg)
        else:
            momentum = 0

        top10_count = sum(1 for f in recent_finishes if f <= 10)
        top10_rate = top10_count / len(recent_finishes)
        owgr_normalized = 1.0 / (1.0 + np.log10(max(1, player["owgr"])))

        record = {
            "year": year,
            "player_name": player["name"],
            "owgr": player["owgr"],
            "age": player["age"],
            "sg_off_tee": round(sg_ott, 3),
            "sg_approach": round(sg_app, 3),
            "sg_around_green": round(sg_arg, 3),
            "sg_putting": round(sg_putt, 3),
            "sg_tee_to_green": round(sg_t2g, 3),
            "sg_total": round(sg_total, 3),
            "masters_appearances": appearances,
            "years_since_first": years_playing,
            "cuts_made_pct": round(cuts_pct, 3),
            "is_rookie": is_rookie,
            "is_past_champion": is_past_champion,
            "num_masters_wins": num_wins,
            "years_since_last_win": years_since_last_win,
            "champion_recency_score": round(champion_recency, 4),
            "hole11_avg": round(amen["hole11_avg"], 3),
            "hole12_avg": round(amen["hole12_avg"], 3),
            "hole13_avg": round(amen["hole13_avg"], 3),
            "amen_corner_avg": round(amen_avg, 3),
            "amen_corner_vs_par": round(amen_vs_par, 3),
            "amen_corner_composite": round(amen_composite, 3),
            "owgr_normalized": round(owgr_normalized, 4),
            "recent_form_5": round(recent_form_5, 2),
            "recent_form_10": round(recent_form_10, 2),
            "momentum_score": round(momentum, 4),
            "top10_rate_season": round(top10_rate, 3),
        }
        records.append(record)

    df = pd.DataFrame(records)

    # -- Overlay real stats where available --------------------
    real_stats = get_real_stats()
    real_count = 0
    for idx, row in df.iterrows():
        name = row["player_name"]
        if name in real_stats:
            rs = real_stats[name]
            df.at[idx, "owgr"] = rs["owgr"]
            df.at[idx, "sg_off_tee"] = rs["sg_off_tee"]
            df.at[idx, "sg_approach"] = rs["sg_approach"]
            df.at[idx, "sg_around_green"] = rs["sg_around_green"]
            df.at[idx, "sg_putting"] = rs["sg_putting"]
            df.at[idx, "sg_tee_to_green"] = rs["sg_off_tee"] + rs["sg_approach"] + rs["sg_around_green"]
            df.at[idx, "sg_total"] = rs["sg_total"]
            df.at[idx, "owgr_normalized"] = round(1.0 / (1.0 + np.log10(max(1, rs["owgr"]))), 4)
            if rs["events"] > 0:
                df.at[idx, "top10_rate_season"] = round(rs["top10"] / rs["events"], 3)
            real_count += 1

    print(f"  [DATA] Overlaid real stats for {real_count}/{len(df)} players ({100*real_count/len(df):.0f}% real data)")

    return df


def get_feature_columns():
    """Return the list of feature column names (no target, no metadata)."""
    return [
        # Tier 1: Strokes Gained
        "sg_off_tee", "sg_approach", "sg_around_green", "sg_putting",
        "sg_tee_to_green", "sg_total",
        # Tier 2: Augusta Experience
        "masters_appearances", "years_since_first", "cuts_made_pct", "is_rookie",
        # Tier 3: Past Winner
        "is_past_champion", "num_masters_wins", "years_since_last_win",
        "champion_recency_score",
        # Tier 4: Amen Corner
        "hole11_avg", "hole12_avg", "hole13_avg",
        "amen_corner_avg", "amen_corner_vs_par", "amen_corner_composite",
        # Tier 5: Form & Rankings
        "owgr_normalized", "recent_form_5", "recent_form_10",
        "momentum_score", "top10_rate_season",
    ]


def generate_prediction_data_2027():
    """
    Generate forward-looking prediction data for the 2027 Masters.
    Accounts for:
      - Updated Masters history (2026 results known)
      - 2026 winner Rory McIlroy now has additional win
      - Players age by 1 year
      - Projected stats based on current trajectory
    """
    players = get_player_profiles()
    winners = get_masters_winners()
    masters_2026 = get_masters_2026_results()
    real_stats = get_real_stats()
    year = 2027
    records = []

    # Update winners dict with 2026 result
    winners[2026] = "Rory McIlroy"

    for player in players:
        skill_mult = _rank_to_skill(player["owgr"])
        year_noise = np.random.normal(0, 0.10)
        name = player["name"]

        # Use real stats if available, with small forward projection noise
        if name in real_stats:
            rs = real_stats[name]
            proj_noise = np.random.normal(0, 0.08)
            sg_ott = rs["sg_off_tee"] + proj_noise
            sg_app = rs["sg_approach"] + proj_noise
            sg_arg = rs["sg_around_green"] + proj_noise
            sg_putt = rs["sg_putting"] + proj_noise
            owgr = rs["owgr"]
        else:
            sg_ott = _style_to_sg(player["style_power"], noise_std=0.2) * skill_mult + year_noise
            sg_app = _style_to_sg(player["style_iron_play"], noise_std=0.2) * skill_mult + year_noise
            sg_arg = _style_to_sg(player["style_short_game"], noise_std=0.2) * skill_mult + year_noise
            sg_putt = _style_to_sg(player["style_putting"], noise_std=0.2) * skill_mult + year_noise
            owgr = player["owgr"]

        sg_t2g = sg_ott + sg_app + sg_arg
        sg_total = sg_t2g + sg_putt

        # Updated Augusta experience (now includes 2026)
        appearances = player["masters_appearances"] + 1
        years_playing = appearances
        cuts_pct = 0.5 + skill_mult * 0.3
        is_rookie = 0

        # Updated past winner status
        wins_before = [y for y in player.get("years_won", []) if y < year]
        if name == "Rory McIlroy" and 2026 not in wins_before:
            wins_before.append(2026)
        is_past_champion = 1 if len(wins_before) > 0 else 0
        num_wins = len(wins_before)
        years_since_last_win = (year - max(wins_before)) if wins_before else 99
        champion_recency = sum(0.85 ** (year - y) for y in wins_before)

        amen = _generate_amen_corner_scores(player, year, skill_mult)
        amen_avg = (amen["hole11_avg"] + amen["hole12_avg"] + amen["hole13_avg"]) / 3
        amen_vs_par = (
            (amen["hole11_avg"] - AMEN_CORNER_PARS[11]) +
            (amen["hole12_avg"] - AMEN_CORNER_PARS[12]) +
            (amen["hole13_avg"] - AMEN_CORNER_PARS[13])
        )
        amen_composite = (
            amen["hole11_avg"] * 0.40 +
            amen["hole12_avg"] * 0.35 +
            amen["hole13_avg"] * 0.25
        )

        recent_finishes = _generate_recent_form(player, year)
        if name in masters_2026:
            recent_finishes[0] = masters_2026[name]["finish"]

        w5 = [0.35, 0.25, 0.20, 0.12, 0.08]
        w10 = [0.20, 0.15, 0.13, 0.11, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05]
        recent_form_5 = sum(f * w for f, w in zip(recent_finishes[:5], w5))
        recent_form_10 = sum(f * w for f, w in zip(recent_finishes, w10))

        if len(recent_finishes) >= 5:
            first_half_avg = np.mean(recent_finishes[:5])
            second_half_avg = np.mean(recent_finishes[5:]) if len(recent_finishes) > 5 else first_half_avg
            momentum = (second_half_avg - first_half_avg) / max(1, second_half_avg)
        else:
            momentum = 0

        top10_count = sum(1 for f in recent_finishes if f <= 10)
        top10_rate = top10_count / len(recent_finishes)
        owgr_normalized = 1.0 / (1.0 + np.log10(max(1, owgr)))

        record = {
            "year": year,
            "player_name": name,
            "owgr": owgr,
            "age": player["age"] + 1,
            "sg_off_tee": round(sg_ott, 3),
            "sg_approach": round(sg_app, 3),
            "sg_around_green": round(sg_arg, 3),
            "sg_putting": round(sg_putt, 3),
            "sg_tee_to_green": round(sg_t2g, 3),
            "sg_total": round(sg_total, 3),
            "masters_appearances": appearances,
            "years_since_first": years_playing,
            "cuts_made_pct": round(cuts_pct, 3),
            "is_rookie": is_rookie,
            "is_past_champion": is_past_champion,
            "num_masters_wins": num_wins,
            "years_since_last_win": years_since_last_win,
            "champion_recency_score": round(champion_recency, 4),
            "hole11_avg": round(amen["hole11_avg"], 3),
            "hole12_avg": round(amen["hole12_avg"], 3),
            "hole13_avg": round(amen["hole13_avg"], 3),
            "amen_corner_avg": round(amen_avg, 3),
            "amen_corner_vs_par": round(amen_vs_par, 3),
            "amen_corner_composite": round(amen_composite, 3),
            "owgr_normalized": round(owgr_normalized, 4),
            "recent_form_5": round(recent_form_5, 2),
            "recent_form_10": round(recent_form_10, 2),
            "momentum_score": round(momentum, 4),
            "top10_rate_season": round(top10_rate, 3),
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Overlay real stats
    real_count = 0
    for idx, row in df.iterrows():
        name = row["player_name"]
        if name in real_stats:
            rs = real_stats[name]
            df.at[idx, "owgr"] = rs["owgr"]
            df.at[idx, "owgr_normalized"] = round(1.0 / (1.0 + np.log10(max(1, rs["owgr"]))), 4)
            if rs["events"] > 0:
                df.at[idx, "top10_rate_season"] = round(rs["top10"] / rs["events"], 3)
            real_count += 1

    print(f"  [DATA] 2027 prediction: Overlaid real stats for {real_count}/{len(df)} players")
    return df


if __name__ == "__main__":
    print("Generating historical training data...")
    hist_df = generate_historical_data()
    hist_df.to_csv(os.path.join(os.path.dirname(__file__), "historical_stats.csv"), index=False)
    print(f"  -> {len(hist_df)} records across {hist_df['year'].nunique()} years")
    print(f"  -> {hist_df['player_name'].nunique()} unique players")

    print("\nGenerating 2026 prediction data...")
    pred_df = generate_prediction_data()
    pred_df.to_csv(os.path.join(os.path.dirname(__file__), "prediction_2026.csv"), index=False)
    print(f"  -> {len(pred_df)} players in 2026 field")

    report = get_data_quality_report(pred_df)
    print(f"  -> Real data coverage: {report['real_data_pct']}%")

    print("\nGenerating 2027 prediction data...")
    pred_2027 = generate_prediction_data_2027()
    pred_2027.to_csv(os.path.join(os.path.dirname(__file__), "prediction_2027.csv"), index=False)
    print(f"  -> {len(pred_2027)} players in 2027 field")

    print("\nDone! Data saved to data/ directory.")
