"""
Masters Tournament Predictor — Streamlit Dashboard
====================================================
Premium Augusta National-themed interactive dashboard.
Displays predictions, Monte Carlo distributions,
player analysis, and model insights.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Masters Tournament Predictor",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════
# CUSTOM CSS — Augusta National Theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* ── Global ─────────────────────────────────────── */
    .stApp {
        background: linear-gradient(175deg, #0a0f0d 0%, #0d1a14 40%, #0e1117 100%);
    }

    .main .block-container {
        padding-top: 1rem;
        max-width: 1400px;
    }

    /* ── Hero Header ────────────────────────────────── */
    .hero-container {
        background: linear-gradient(135deg, #004D35 0%, #006747 30%, #00875A 60%, #004D35 100%);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 215, 0, 0.25);
        box-shadow: 0 20px 60px rgba(0, 103, 71, 0.3), 0 0 40px rgba(0, 103, 71, 0.15);
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(ellipse at 30% 20%, rgba(255, 215, 0, 0.08) 0%, transparent 60%);
        pointer-events: none;
    }

    .hero-container::after {
        content: '';
        position: absolute;
        bottom: 0;
        right: 0;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255, 215, 0, 0.05) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        color: #FFD700;
        text-shadow: 0 2px 20px rgba(255, 215, 0, 0.3);
        margin: 0;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem;
        color: rgba(255, 255, 255, 0.85);
        margin-top: 0.5rem;
        font-weight: 300;
        letter-spacing: 0.3px;
        position: relative;
        z-index: 1;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255, 215, 0, 0.15);
        border: 1px solid rgba(255, 215, 0, 0.35);
        color: #FFD700;
        padding: 0.3rem 0.9rem;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        margin-top: 1rem;
        letter-spacing: 0.5px;
        position: relative;
        z-index: 1;
    }

    /* ── Cards ──────────────────────────────────────── */
    .stat-card {
        background: linear-gradient(145deg, rgba(26, 31, 46, 0.95), rgba(20, 25, 38, 0.95));
        border: 1px solid rgba(255, 215, 0, 0.12);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(20px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }

    .stat-card:hover {
        border-color: rgba(255, 215, 0, 0.3);
        box-shadow: 0 8px 40px rgba(0, 103, 71, 0.2);
        transform: translateY(-2px);
    }

    .stat-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFD700;
        line-height: 1;
    }

    .stat-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: rgba(232, 232, 232, 0.6);
        margin-top: 0.4rem;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Section Headers ───────────────────────────── */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #E8E8E8;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(255, 215, 0, 0.2);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Leaderboard Table ─────────────────────────── */
    .leaderboard-row {
        display: flex;
        align-items: center;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
    }

    .leaderboard-row:hover {
        background: rgba(0, 103, 71, 0.15);
    }

    .rank-badge {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 1rem;
        flex-shrink: 0;
    }

    .rank-1 { background: linear-gradient(135deg, #FFD700, #FFA500); color: #1a1f2e; }
    .rank-2 { background: linear-gradient(135deg, #C0C0C0, #A8A8A8); color: #1a1f2e; }
    .rank-3 { background: linear-gradient(135deg, #CD7F32, #B8860B); color: #1a1f2e; }
    .rank-other { background: rgba(255, 255, 255, 0.08); color: #E8E8E8; }

    /* ── Progress Bars ─────────────────────────────── */
    .prob-bar-container {
        width: 100%;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 6px;
        height: 24px;
        overflow: hidden;
        position: relative;
    }

    .prob-bar {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #006747, #00875A, #00A86B);
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        align-items: center;
        padding-left: 8px;
    }

    .prob-bar-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
        white-space: nowrap;
    }

    /* ── Sidebar ───────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1a14, #0e1117);
        border-right: 1px solid rgba(255, 215, 0, 0.1);
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #FFD700 !important;
        font-family: 'Outfit', sans-serif;
    }

    /* ── Tabs ──────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: rgba(232, 232, 232, 0.6);
        border-radius: 10px;
        padding: 10px 20px;
        background: rgba(26, 31, 46, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [aria-selected="true"] {
        color: #FFD700 !important;
        background: rgba(0, 103, 71, 0.3) !important;
        border-color: rgba(255, 215, 0, 0.3) !important;
    }

    /* ── Selectbox ─────────────────────────────────── */
    .stSelectbox label {
        color: rgba(232, 232, 232, 0.8) !important;
        font-family: 'Inter', sans-serif;
    }

    /* ── Metrics ───────────────────────────────────── */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(26, 31, 46, 0.95), rgba(20, 25, 38, 0.95));
        border: 1px solid rgba(255, 215, 0, 0.12);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    [data-testid="stMetric"] label {
        color: rgba(232, 232, 232, 0.6) !important;
        font-family: 'Inter', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-size: 0.75rem !important;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #FFD700 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
    }

    /* ── Hide default Streamlit elements ────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_pipeline_state():
    """Load the saved pipeline state."""
    state_path = os.path.join(DATA_DIR, "pipeline_state.pkl")
    if not os.path.exists(state_path):
        st.error("Pipeline state not found. Please run `python pipeline.py` first.")
        st.stop()
    with open(state_path, "rb") as f:
        return pickle.load(f)


state = load_pipeline_state()
results = state["simulation_results"]
pred_data = state["pred_data"]
feature_importances = state["feature_importances"]
model_weights = state["model_weights"]
simulator = state["simulator"]
player_names = state["player_names"]
tier_map = state["tier_map"]
feature_names = state["feature_names"]
base_predictions = state["base_predictions"]
prediction_year = state.get("prediction_year", 2026)


# ══════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#E8E8E8"),
    margin=dict(l=40, r=40, t=50, b=40),
)


def masters_colors(n):
    """Generate Augusta-themed color palette."""
    colors = [
        "#FFD700", "#00A86B", "#006747", "#00875A",
        "#FEDB00", "#228B22", "#C2B280", "#FF69B4",
        "#4ECDC4", "#FF6B6B", "#45B7D1", "#96CEB4",
    ]
    return colors[:n] if n <= len(colors) else colors * (n // len(colors) + 1)


# ══════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">The Masters Tournament Predictor</div>
    <div class="hero-subtitle">
        {prediction_year} Predictions  |  Machine Learning Ensemble with Monte Carlo Simulation  |  Augusta National Golf Club
    </div>
    <div class="hero-badge">5 ML Models  &middot;  5 Feature Tiers  &middot;  10,000 Simulations  &middot;  Real PGA Tour Data</div>
</div>
""", unsafe_allow_html=True)

# ── Top Stats Row ──────────────────────────────────────────
top_player = results.iloc[0]
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Predicted Winner", top_player["player_name"])
with col2:
    st.metric("Win Probability", f"{top_player['win_probability']:.1f}%")
with col3:
    st.metric("Top 5 Probability", f"{top_player['top5_probability']:.1f}%")
with col4:
    st.metric("Field Size", f"{len(results)} Players")


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Prediction Leaderboard",
    "Monte Carlo Analysis",
    "Player Deep-Dive",
    "Model Insights",
    "Simulation Convergence",
])


# ──────────────────────────────────────────────────────────────
# TAB 1: PREDICTION LEADERBOARD
# ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Prediction Leaderboard</div>', unsafe_allow_html=True)

    # How many players to show
    n_show = st.slider("Number of players to display", 10, len(results), 25, key="lb_slider")

    display_df = results.head(n_show).copy()
    display_df.insert(0, "Rank", range(1, n_show + 1))
    display_df = display_df.rename(columns={
        "player_name": "Player",
        "win_probability": "Win %",
        "top5_probability": "Top 5 %",
        "top10_probability": "Top 10 %",
        "top20_probability": "Top 20 %",
        "expected_finish": "Exp. Finish",
        "median_finish": "Med. Finish",
        "predicted_position": "Model Pred.",
    })

    # Win Probability Bar Chart
    fig_bar = go.Figure()
    show_n = min(20, n_show)
    bar_data = display_df.head(show_n)

    fig_bar.add_trace(go.Bar(
        y=bar_data["Player"],
        x=bar_data["Win %"],
        orientation='h',
        marker=dict(
            color=[f"rgba(0, {min(255, 120 + i*6)}, {min(255, 71 + i*4)}, 0.85)" for i in range(show_n)],
            line=dict(color="rgba(255, 215, 0, 0.3)", width=1),
        ),
        text=[f"{v:.1f}%" for v in bar_data["Win %"]],
        textposition="outside",
        textfont=dict(color="#FFD700", size=12, family="Outfit"),
    ))

    fig_bar.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Win Probability by Player", font=dict(size=18, family="Outfit", color="#FFD700")),
        height=max(400, show_n * 35),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        xaxis=dict(title="Win Probability (%)", range=[0, max(bar_data["Win %"]) * 1.3]),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Data Table
    st.markdown("#### Detailed Predictions")
    st.dataframe(
        display_df[["Rank", "Player", "Win %", "Top 5 %", "Top 10 %", "Top 20 %", "Exp. Finish", "Med. Finish"]],
        use_container_width=True,
        hide_index=True,
        height=min(600, n_show * 38 + 40),
    )


# ──────────────────────────────────────────────────────────────
# TAB 2: MONTE CARLO ANALYSIS
# ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Monte Carlo Simulation Analysis</div>', unsafe_allow_html=True)

    mc_col1, mc_col2 = st.columns([1, 1])

    with mc_col1:
        selected_player = st.selectbox(
            "Select a Player",
            results["player_name"].tolist(),
            key="mc_player",
        )

    with mc_col2:
        player_row = results[results["player_name"] == selected_player].iloc[0]
        st.markdown(f"""
        <div class="stat-card">
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <div class="stat-label">Win Probability</div>
                    <div class="stat-value">{player_row['win_probability']:.1f}%</div>
                </div>
                <div>
                    <div class="stat-label">Top 5</div>
                    <div class="stat-value">{player_row['top5_probability']:.1f}%</div>
                </div>
                <div>
                    <div class="stat-label">Top 10</div>
                    <div class="stat-value">{player_row['top10_probability']:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Distribution Histogram
    dist = simulator.get_player_distribution(selected_player)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=dist,
        nbinsx=50,
        marker=dict(
            color="rgba(0, 167, 107, 0.6)",
            line=dict(color="rgba(255, 215, 0, 0.4)", width=1),
        ),
        name="Simulated Finishes",
    ))

    # Add median line
    median_finish = int(np.median(dist))
    fig_hist.add_vline(
        x=median_finish, line_dash="dash",
        line_color="#FFD700", line_width=2,
        annotation_text=f"Median: {median_finish}",
        annotation_font_color="#FFD700",
    )

    fig_hist.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"Simulated Finish Distribution — {selected_player}",
            font=dict(size=18, family="Outfit", color="#FFD700"),
        ),
        xaxis_title="Finishing Position",
        yaxis_title="Frequency (out of 10,000 simulations)",
        height=450,
        bargap=0.05,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Probability Breakdown
    st.markdown("#### Probability Breakdown")
    prob_cols = st.columns(5)
    probs = [
        ("Win (1st)", player_row["win_probability"]),
        ("Top 5", player_row["top5_probability"]),
        ("Top 10", player_row["top10_probability"]),
        ("Top 20", player_row["top20_probability"]),
        ("Expected Finish", player_row["expected_finish"]),
    ]
    for col, (label, val) in zip(prob_cols, probs):
        with col:
            if "Finish" in label:
                st.metric(label, f"{val:.0f}")
            else:
                st.metric(label, f"{val:.1f}%")

    # Head-to-Head Section
    st.markdown("#### Head-to-Head Comparison")
    h2h_col1, h2h_col2 = st.columns(2)
    with h2h_col1:
        p1 = st.selectbox("Player 1", results["player_name"].tolist()[:20], index=0, key="h2h_p1")
    with h2h_col2:
        p2 = st.selectbox("Player 2", results["player_name"].tolist()[:20], index=1, key="h2h_p2")

    if p1 != p2:
        h2h = simulator.head_to_head(p1, p2)

        h2h_fig = go.Figure()
        h2h_fig.add_trace(go.Bar(
            x=[h2h["p1_win_pct"]],
            y=["Result"],
            orientation='h',
            name=p1,
            marker_color="#006747",
            text=f"{h2h['p1_win_pct']}%",
            textposition="inside",
            textfont=dict(size=14, family="Outfit", color="white"),
        ))
        h2h_fig.add_trace(go.Bar(
            x=[h2h["p2_win_pct"]],
            y=["Result"],
            orientation='h',
            name=p2,
            marker_color="#FFD700",
            text=f"{h2h['p2_win_pct']}%",
            textposition="inside",
            textfont=dict(size=14, family="Outfit", color="#1a1f2e"),
        ))

        h2h_fig.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text=f"Head-to-Head: {p1} vs {p2}", font=dict(size=16, family="Outfit", color="#FFD700")),
            barmode='stack',
            height=180,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.5),
            yaxis=dict(visible=False),
        )
        st.plotly_chart(h2h_fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB 3: PLAYER DEEP-DIVE
# ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Player Deep-Dive Analysis</div>', unsafe_allow_html=True)

    dd_player = st.selectbox("Select Player for Analysis", results["player_name"].tolist(), key="dd_player")

    player_pred = pred_data[pred_data["player_name"] == dd_player].iloc[0]
    player_result = results[results["player_name"] == dd_player].iloc[0]

    # Radar Chart — 5 Feature Tiers
    st.markdown("#### Feature Profile (5 Tiers)")

    # Get normalized values for each tier
    tier_scores = {}
    raw_features = {
        "Strokes Gained": ["sg_total"],
        "Augusta Experience": ["masters_appearances"],
        "Past Winner": ["champion_recency_score"],
        "Amen Corner": ["amen_corner_vs_par"],
        "Form & Rankings": ["owgr_normalized"],
    }

    # Normalize each to 0-10 scale
    for tier_name, cols in raw_features.items():
        val = player_pred[cols[0]]
        all_vals = pred_data[cols[0]]
        if tier_name == "Amen Corner":
            # Lower is better for amen corner vs par
            normalized = 10 * (1 - (val - all_vals.min()) / max(0.01, all_vals.max() - all_vals.min()))
        else:
            normalized = 10 * (val - all_vals.min()) / max(0.01, all_vals.max() - all_vals.min())
        tier_scores[tier_name] = max(0, min(10, normalized))

    categories = list(tier_scores.keys())
    values = list(tier_scores.values())
    values.append(values[0])  # Close the polygon
    categories.append(categories[0])

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 167, 107, 0.2)',
        line=dict(color='#FFD700', width=2),
        marker=dict(size=8, color='#FFD700'),
        name=dd_player,
    ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True, range=[0, 10],
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(size=10, color='rgba(255,255,255,0.5)'),
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(size=12, color='#E8E8E8', family='Inter'),
            ),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color="#E8E8E8"),
        title=dict(
            text=f"Feature Profile — {dd_player}",
            font=dict(size=18, family="Outfit", color="#FFD700"),
        ),
        height=500,
        showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Detailed Stats
    st.markdown("#### Key Statistics")
    stat_col1, stat_col2, stat_col3 = st.columns(3)

    with stat_col1:
        st.markdown("**Strokes Gained**")
        st.metric("Off the Tee", f"{player_pred['sg_off_tee']:.2f}")
        st.metric("Approach", f"{player_pred['sg_approach']:.2f}")
        st.metric("Around Green", f"{player_pred['sg_around_green']:.2f}")
        st.metric("Putting", f"{player_pred['sg_putting']:.2f}")

    with stat_col2:
        st.markdown("**Augusta Factors**")
        st.metric("Masters Appearances", int(player_pred['masters_appearances']))
        st.metric("Past Champion", "Yes" if player_pred['is_past_champion'] else "No")
        st.metric("Masters Wins", int(player_pred['num_masters_wins']))
        st.metric("Amen Corner vs Par", f"{player_pred['amen_corner_vs_par']:+.2f}")

    with stat_col3:
        st.markdown("**Current Form**")
        st.metric("World Ranking", int(player_pred['owgr']))
        st.metric("Recent Form (5)", f"{player_pred['recent_form_5']:.1f}")
        st.metric("Top 10 Rate", f"{player_pred['top10_rate_season']:.0%}")
        st.metric("Momentum", f"{player_pred['momentum_score']:+.3f}")

    # Compare to another player
    st.markdown("---")
    st.markdown("#### Compare Two Players")
    cmp_col1, cmp_col2 = st.columns(2)
    with cmp_col1:
        cmp_p1 = st.selectbox("Player A", results["player_name"].tolist(), index=0, key="cmp1")
    with cmp_col2:
        cmp_p2 = st.selectbox("Player B", results["player_name"].tolist(), index=1, key="cmp2")

    if cmp_p1 != cmp_p2:
        p1_data = pred_data[pred_data["player_name"] == cmp_p1].iloc[0]
        p2_data = pred_data[pred_data["player_name"] == cmp_p2].iloc[0]

        compare_features = ["sg_total", "sg_off_tee", "sg_approach", "sg_putting",
                          "masters_appearances", "amen_corner_vs_par", "owgr_normalized"]
        feature_labels = ["SG Total", "SG Off Tee", "SG Approach", "SG Putting",
                         "Masters Apps", "Amen Corner", "OWGR Score"]

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name=cmp_p1,
            y=feature_labels,
            x=[p1_data[f] for f in compare_features],
            orientation='h',
            marker_color='#006747',
        ))
        fig_compare.add_trace(go.Bar(
            name=cmp_p2,
            y=feature_labels,
            x=[p2_data[f] for f in compare_features],
            orientation='h',
            marker_color='#FFD700',
        ))

        fig_compare.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text=f"{cmp_p1} vs {cmp_p2}", font=dict(size=16, family="Outfit", color="#FFD700")),
            barmode='group',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        st.plotly_chart(fig_compare, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB 4: MODEL INSIGHTS
# ──────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Model Architecture & Insights</div>', unsafe_allow_html=True)

    # Model Weights
    st.markdown("#### Meta-Learner Model Weights")
    st.markdown("The meta-learner learned these weights for each base model:")

    model_names_display = {
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "random_forest": "Random Forest",
        "ridge": "Ridge Regression",
        "neural_net": "Neural Network",
    }

    weights_df = pd.DataFrame([
        {"Model": model_names_display.get(k, k), "Weight": float(v)}
        for k, v in model_weights.items()
    ]).sort_values("Weight", ascending=True)

    fig_weights = go.Figure()
    fig_weights.add_trace(go.Bar(
        y=weights_df["Model"],
        x=weights_df["Weight"],
        orientation='h',
        marker=dict(
            color=['#FF6B6B' if w < 0 else '#00A86B' for w in weights_df["Weight"]],
            line=dict(color="rgba(255, 215, 0, 0.3)", width=1),
        ),
        text=[f"{w:.3f}" for w in weights_df["Weight"]],
        textposition="outside",
        textfont=dict(color="#FFD700", family="Outfit"),
    ))

    fig_weights.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Meta-Learner Weights (Higher = More Trusted)", font=dict(size=16, family="Outfit", color="#FFD700")),
        height=300,
        xaxis_title="Weight",
    )
    st.plotly_chart(fig_weights, use_container_width=True)

    # Feature Importances
    st.markdown("#### Feature Importance (Averaged Across Tree Models)")

    avg_imp = feature_importances.get("average", {})
    imp_df = pd.DataFrame([
        {"Feature": k, "Importance": float(v), "Tier": tier_map.get(k, "Unknown")}
        for k, v in avg_imp.items()
    ]).sort_values("Importance", ascending=False).head(20)

    tier_colors = {
        "Tier 1: Strokes Gained": "#006747",
        "Tier 2: Augusta Experience": "#FFD700",
        "Tier 3: Past Winner": "#FF6B6B",
        "Tier 4: Amen Corner": "#4ECDC4",
        "Tier 5: Form & Rankings": "#FF69B4",
        "Interaction Features": "#C2B280",
    }

    fig_imp = go.Figure()
    fig_imp.add_trace(go.Bar(
        y=imp_df["Feature"],
        x=imp_df["Importance"],
        orientation='h',
        marker=dict(
            color=[tier_colors.get(t, "#666") for t in imp_df["Tier"]],
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        text=[f"{v:.4f}" for v in imp_df["Importance"]],
        textposition="outside",
        textfont=dict(color="#E8E8E8", size=10),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<br>%{customdata}<extra></extra>",
        customdata=imp_df["Tier"],
    ))

    fig_imp.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Top 20 Most Important Features", font=dict(size=16, family="Outfit", color="#FFD700")),
        height=600,
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        xaxis_title="Importance Score",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Legend for tier colors
    st.markdown("**Feature Tier Legend:**")
    legend_cols = st.columns(6)
    for i, (tier, color) in enumerate(tier_colors.items()):
        with legend_cols[i % 6]:
            st.markdown(f'<span style="color:{color}; font-weight:600;">●</span> {tier.split(": ")[-1] if ": " in tier else tier}', unsafe_allow_html=True)

    # Base Model Comparison
    st.markdown("#### Base Model Predictions (Top 10)")
    st.markdown("How each base model ranked the top contenders:")

    top_players = results.head(10)["player_name"].tolist()
    bp_data = []
    for name in top_players:
        idx = list(pred_data["player_name"]).index(name)
        row = {"Player": name}
        for model_name, preds in base_predictions.items():
            row[model_names_display.get(model_name, model_name)] = round(preds[idx], 1)
        bp_data.append(row)

    bp_df = pd.DataFrame(bp_data)
    st.dataframe(bp_df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
# TAB 5: SIMULATION CONVERGENCE
# ──────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Simulation Convergence Analysis</div>', unsafe_allow_html=True)
    st.markdown("Watch how win probabilities stabilize as more simulations are run.")

    conv_players = st.multiselect(
        "Select players to track",
        results["player_name"].tolist()[:20],
        default=results["player_name"].tolist()[:5],
        key="conv_players",
    )

    if conv_players:
        fig_conv = go.Figure()
        colors = masters_colors(len(conv_players))

        for i, player in enumerate(conv_players):
            conv_data = simulator.get_convergence_data(player, checkpoints=100)
            fig_conv.add_trace(go.Scatter(
                x=conv_data["n_simulations"],
                y=conv_data["win_probability"],
                mode="lines",
                name=player,
                line=dict(color=colors[i], width=2),
                hovertemplate=f"<b>{player}</b><br>" +
                             "Sims: %{x:,}<br>" +
                             "Win %: %{y:.2f}%<extra></extra>",
            ))

        fig_conv.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Win Probability Convergence", font=dict(size=18, family="Outfit", color="#FFD700")),
            xaxis_title="Number of Simulations",
            yaxis_title="Win Probability (%)",
            height=500,
            legend=dict(
                bgcolor="rgba(26, 31, 46, 0.8)",
                bordercolor="rgba(255, 215, 0, 0.2)",
                borderwidth=1,
                font=dict(size=11),
            ),
        )
        st.plotly_chart(fig_conv, use_container_width=True)

        # Top 5 convergence
        st.markdown("#### Top 5 Probability Convergence")
        fig_conv5 = go.Figure()
        for i, player in enumerate(conv_players):
            conv_data = simulator.get_convergence_data(player, checkpoints=100)
            fig_conv5.add_trace(go.Scatter(
                x=conv_data["n_simulations"],
                y=conv_data["top5_probability"],
                mode="lines",
                name=player,
                line=dict(color=colors[i], width=2),
            ))

        fig_conv5.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Top 5 Probability Convergence", font=dict(size=18, family="Outfit", color="#FFD700")),
            xaxis_title="Number of Simulations",
            yaxis_title="Top 5 Probability (%)",
            height=500,
            legend=dict(
                bgcolor="rgba(26, 31, 46, 0.8)",
                bordercolor="rgba(255, 215, 0, 0.2)",
                borderwidth=1,
            ),
        )
        st.plotly_chart(fig_conv5, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 2.5rem;">⛳</span>
        <div style="font-family: 'Outfit', sans-serif; font-size: 1.3rem; font-weight: 700; color: #FFD700; margin-top: 0.5rem;">
            Masters Predictor
        </div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.8rem; color: rgba(232,232,232,0.5); margin-top: 0.2rem;">
            Augusta National Golf Club
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Model Architecture")
    st.markdown("""
    **5 Base Models:**
    - XGBoost
    - LightGBM
    - Random Forest
    - Ridge Regression
    - Neural Network (MLP)

    **Meta-Learner:**
    - Stacking Ensemble
    - Ridge Regression meta-model

    **Simulation:**
    - 10,000 Monte Carlo runs
    - Normal distribution per player
    """)

    st.markdown("---")
    st.markdown("### Feature Tiers")
    st.markdown("""
    1. **Strokes Gained** — SG:OTT, APP, ARG, PUTT
    2. **Augusta Experience** — Appearances, cuts made
    3. **Past Winner** — Champion status & recency
    4. **Amen Corner** — Holes 11, 12, 13 composite
    5. **Form & Rankings** — OWGR, recent finishes
    """)

    st.markdown("---")
    st.markdown(f"<div style='text-align:center; color:rgba(232,232,232,0.3); font-size:0.75rem;'>v1.0 | {len(results)} players | {N_SIMULATIONS:,} simulations</div>", unsafe_allow_html=True)
