import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.graph_objects as go
from data_generator import generate_dataset
from sensor_processor import SENSOR_COLUMNS

from predict import DisasterPredictor
from utils import (
    severity_distribution_chart,
    disaster_distribution_chart,
    sensor_heatmap_chart,
    prediction_gauge_chart,
    radar_chart,
    disaster_by_severity_chart,
    sensor_distribution_chart,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Modal Disaster Prediction",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Theme ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --danger: #FF5722;
    --danger-dim: #E64A19;
    --blue: #2196F3;
    --purple: #9C27B0;
    --orange: #FF9800;
    --cyan: #00BCD4;
    --bg-dark: #0a0a0f;
    --bg-card: rgba(15, 15, 25, 0.85);
    --bg-glass: rgba(255, 255, 255, 0.04);
    --border-glass: rgba(255, 255, 255, 0.08);
    --text-primary: #ffffff;
    --text-secondary: #a0a0b8;
}

#MainMenu, footer, header {visibility: hidden;}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #1a0a0a 100%);
    border-right: 1px solid rgba(255, 87, 34, 0.15);
}
section[data-testid="stSidebar"] .stRadio label {
    color: var(--text-secondary);
    font-size: 0.92rem;
    padding: 6px 12px;
    border-radius: 8px;
    transition: all 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255, 87, 34, 0.1);
    color: var(--danger);
}
section[data-testid="stSidebar"] .stRadio [aria-checked="true"] {
    color: var(--danger) !important;
    font-weight: 600;
}

.stApp {
    background: var(--bg-dark);
    color: var(--text-primary);
    position: relative;
    z-index: 1;
}

.stApp > header {
    visibility: visible !important;
}

section[data-testid="stSidebar"] {
    z-index: 2 !important;
}

.stButton > button {
    z-index: 10 !important;
}

.particles {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none !important;
    z-index: -1;
    overflow: hidden;
}
.particle {
    position: absolute;
    width: 3px; height: 3px;
    background: var(--danger);
    border-radius: 50%;
    opacity: 0.25;
    animation: float-particle linear infinite;
}
@keyframes float-particle {
    0% { transform: translateY(100vh) translateX(0); opacity: 0; }
    10% { opacity: 0.3; }
    90% { opacity: 0.3; }
    100% { transform: translateY(-10vh) translateX(40px); opacity: 0; }
}

.hero-section {
    position: relative;
    background: linear-gradient(135deg, #1a0a0a 0%, #0a0a1a 40%, #0a1a0a 100%);
    border-radius: 16px;
    padding: 48px 40px;
    margin-bottom: 32px;
    overflow: hidden;
    border: 1px solid rgba(255, 87, 34, 0.15);
}
.hero-section::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 30% 50%, rgba(255, 87, 34, 0.08) 0%, transparent 60%);
    animation: hero-glow 8s ease-in-out infinite alternate;
}
@keyframes hero-glow {
    0% { transform: translate(0, 0); }
    100% { transform: translate(5%, -3%); }
}
.hero-section h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #FF5722 60%, #FF9800 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
    position: relative;
    z-index: 1;
}
.hero-section p {
    color: var(--text-secondary);
    font-size: 1.1rem;
    position: relative;
    z-index: 1;
}

.glass-card {
    background: var(--bg-glass);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass);
    border-radius: 14px;
    padding: 24px;
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(255, 87, 34, 0.3);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(255, 87, 34, 0.1);
}

.severity-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.severity-low { background: rgba(0, 188, 212, 0.15); color: var(--cyan); border: 1px solid rgba(0, 188, 212, 0.3); }
.severity-moderate { background: rgba(255, 152, 0, 0.15); color: var(--orange); border: 1px solid rgba(255, 152, 0, 0.3); }
.severity-high { background: rgba(255, 87, 34, 0.15); color: var(--danger); border: 1px solid rgba(255, 87, 34, 0.3); }
.severity-extreme { background: rgba(156, 39, 176, 0.15); color: var(--purple); border: 1px solid rgba(156, 39, 176, 0.3); }

.pulse-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(1.4); }
}

.disaster-card {
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.disaster-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
}
.disaster-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
}
.disaster-card .icon { font-size: 2.4rem; margin-bottom: 10px; }
.disaster-card .label { font-size: 0.9rem; color: var(--text-secondary); margin-top: 4px; }
.disaster-card .title { font-size: 1rem; font-weight: 600; margin-top: 6px; }

.workflow-step {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 16px 20px;
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    transition: all 0.3s;
}
.workflow-step:hover {
    border-color: rgba(255, 87, 34, 0.3);
    background: rgba(255, 87, 34, 0.05);
}
.workflow-step .step-num {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--danger), var(--danger-dim));
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
    flex-shrink: 0;
}
.workflow-step .step-text { font-size: 0.88rem; color: var(--text-secondary); }
.workflow-step .step-title { font-weight: 600; color: var(--text-primary); font-size: 0.95rem; }

.metric-card {
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 22px;
    text-align: center;
    transition: all 0.3s;
}
.metric-card:hover {
    border-color: rgba(255, 87, 34, 0.25);
    box-shadow: 0 4px 20px rgba(255, 87, 34, 0.08);
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--danger), var(--orange));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card .metric-label {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-top: 4px;
}

.prediction-result {
    background: var(--bg-glass);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}
.prediction-icon {
    font-size: 5rem;
    margin-bottom: 12px;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}
.prediction-type {
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 8px;
}

.confidence-bar {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
    margin-top: 6px;
}
.confidence-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s ease;
}

.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid rgba(255, 87, 34, 0.2);
}
.section-title span {
    background: linear-gradient(135deg, #ffffff, var(--danger));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.about-item {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 10px;
}
.about-item h4 { color: var(--danger); margin-bottom: 6px; }
.about-item p { color: var(--text-secondary); font-size: 0.9rem; }

.tech-badge {
    display: inline-block;
    padding: 6px 14px;
    margin: 4px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
    background: rgba(255, 87, 34, 0.1);
    border: 1px solid rgba(255, 87, 34, 0.2);
    color: var(--danger);
}
</style>

<div class="particles">
    <div class="particle" style="left:5%; animation-duration:14s; animation-delay:0s; width:2px; height:2px;"></div>
    <div class="particle" style="left:15%; animation-duration:18s; animation-delay:2s;"></div>
    <div class="particle" style="left:25%; animation-duration:12s; animation-delay:4s; width:4px; height:4px; opacity:0.15;"></div>
    <div class="particle" style="left:35%; animation-duration:16s; animation-delay:1s;"></div>
    <div class="particle" style="left:45%; animation-duration:20s; animation-delay:3s; width:2px; height:2px;"></div>
    <div class="particle" style="left:55%; animation-duration:13s; animation-delay:5s;"></div>
    <div class="particle" style="left:65%; animation-duration:17s; animation-delay:0.5s; width:4px; height:4px; opacity:0.12;"></div>
    <div class="particle" style="left:75%; animation-duration:15s; animation-delay:2.5s;"></div>
    <div class="particle" style="left:85%; animation-duration:19s; animation-delay:4.5s; width:2px; height:2px;"></div>
    <div class="particle" style="left:95%; animation-duration:11s; animation-delay:1.5s;"></div>
    <div class="particle" style="left:10%; animation-duration:22s; animation-delay:6s;"></div>
    <div class="particle" style="left:30%; animation-duration:14s; animation-delay:3.5s; width:2px; height:2px;"></div>
    <div class="particle" style="left:50%; animation-duration:18s; animation-delay:7s; opacity:0.18;"></div>
    <div class="particle" style="left:70%; animation-duration:16s; animation-delay:1.2s;"></div>
    <div class="particle" style="left:90%; animation-duration:21s; animation-delay:5.5s; width:2px; height:2px;"></div>
</div>
""", unsafe_allow_html=True)

# ─── Data & Model Loading ────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
DATA_PATH = os.path.join(DATA_DIR, "disaster_data.csv")


@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_DIR, exist_ok=True)
        df = generate_dataset(n_samples=2000)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    predictor = DisasterPredictor(model_dir=MODEL_DIR)
    if not predictor.is_trained:
        df = load_data()
        predictor.train(df)
    return predictor


df = load_data()
predictor = load_model()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:10px 0 20px 0;">
        <div style="font-size:2.4rem;">🌪️</div>
        <div style="font-size:1.1rem; font-weight:700; color:#FF5722; margin-top:4px;">
            Disaster AI
        </div>
        <div style="font-size:0.75rem; color:#a0a0b8; margin-top:2px;">
            Multi-Modal Prediction System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Data Analysis", "🔮 Predict", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown(f"""
    <div style="padding:10px; text-align:center;">
        <div style="font-size:0.75rem; color:#a0a0b8;">
            Dataset Size
        </div>
        <div style="font-size:1.1rem; font-weight:700; color:#FF5722; margin-top:4px;">
            {len(df):,} records
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding:6px; text-align:center;">
        <div style="font-size:0.75rem; color:#a0a0b8;">
            Model Status
        </div>
        <div style="margin-top:6px;">
            <span class="pulse-dot" style="background:#00BCD4;"></span>
            <span style="color:#00BCD4; font-size:0.82rem; font-weight:600;">Trained & Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":

    st.markdown("""
    <div class="hero-section">
        <h1>🌪️ Multi-Modal Natural Disaster Prediction</h1>
        <p>Harnessing the power of AI to analyze text reports and sensor data for real-time disaster prediction and early warning.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics ────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Reports</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">6</div>
            <div class="metric-label">Disaster Types</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">4</div>
            <div class="metric-label">Severity Levels</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">8</div>
            <div class="metric-label">Sensor Features</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 5-Step Workflow ────────────────────────────────────────────────────
    st.markdown('<div class="section-title"><span>⚡ How It Works — 5-Step Pipeline</span></div>', unsafe_allow_html=True)

    workflow_steps = [
        ("1", "Text Input", "Submit a disaster report or eyewitness description"),
        ("2", "Sensor Data", "Feed real-time environmental sensor readings"),
        ("3", "Multi-Modal AI", "NLP + tabular model fusion for analysis"),
        ("4", "Analysis", "Cross-reference text patterns with sensor anomalies"),
        ("5", "Alert", "Generate severity classification & early warning"),
    ]
    wcols = st.columns(5)
    for i, (num, title, desc) in enumerate(workflow_steps):
        with wcols[i]:
            st.markdown(f"""
            <div class="workflow-step" style="flex-direction:column; text-align:center; min-height:140px; justify-content:center;">
                <div class="step-num">{num}</div>
                <div class="step-title" style="margin-top:10px;">{title}</div>
                <div class="step-text" style="margin-top:6px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 6 Disaster Category Cards ──────────────────────────────────────────
    st.markdown('<div class="section-title"><span>🌍 Disaster Categories</span></div>', unsafe_allow_html=True)

    disaster_cards = [
        ("🌋", "Earthquake / Volcano", "Seismic activity & eruptions", "#FF5722", "rgba(255,87,34,0.15)"),
        ("🌊", "Flood / Water", "Rising water levels & overflow", "#2196F3", "rgba(33,150,243,0.15)"),
        ("🌀", "Hurricane / Cyclone", "Extreme wind & atmospheric", "#9C27B0", "rgba(156,39,176,0.15)"),
        ("🔥", "Wildfire / Fire", "Uncontrolled vegetation fires", "#FF9800", "rgba(255,152,0,0.15)"),
        ("🌪️", "Tornado / Wind", "Violent rotating wind columns", "#00BCD4", "rgba(0,188,212,0.15)"),
        ("🌊", "Tsunami / Wave", "Large ocean wave surges", "#2196F3", "rgba(33,150,243,0.15)"),
    ]

    dcols = st.columns(6)
    for i, (icon, title, desc, color, bg) in enumerate(disaster_cards):
        with dcols[i]:
            st.markdown(f"""
            <div class="disaster-card">
                <div style="position:absolute; top:0; left:0; width:100%; height:3px; background:{color};"></div>
                <div class="icon">{icon}</div>
                <div class="title" style="color:{color};">{title}</div>
                <div class="label">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dataset Overview ───────────────────────────────────────────────────
    st.markdown('<div class="section-title"><span>📋 Dataset Overview</span></div>', unsafe_allow_html=True)

    overview_col1, overview_col2 = st.columns([2, 1])

    with overview_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Sample Records**")
        st.dataframe(
            df.head(10).style.set_properties(**{
                "background-color": "rgba(255,87,34,0.05)",
                "color": "#ffffff",
                "border": "1px solid rgba(255,255,255,0.06)",
            }).set_table_styles([
                {"selector": "th", "props": [
                    ("background-color", "#1a1a2e"),
                    ("color", "#FF5722"),
                    ("font-weight", "600"),
                    ("border", "1px solid rgba(255,255,255,0.08)"),
                ]},
            ]),
            use_container_width=True,
            height=340,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with overview_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Quick Stats**")

        disaster_counts = df["disaster_type"].value_counts()
        for dtype, count in disaster_counts.items():
            pct = count / len(df) * 100
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; padding:5px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#a0a0b8; font-size:0.85rem;">{dtype.title()}</span>
                <span style="color:#FF5722; font-weight:600; font-size:0.85rem;">{count:,} <span style="color:#555; font-weight:400;">({pct:.1f}%)</span></span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA ANALYSIS PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Analysis":

    st.markdown('<div class="section-title"><span>📊 Data Analysis Dashboard</span></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔬 Sensor Analysis", "🔗 Cross Analysis"])

    # ── Tab 1: Distributions ───────────────────────────────────────────────
    with tab1:
        dcol1, dcol2 = st.columns(2)

        with dcol1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Disaster Type Distribution**")
            fig = disaster_distribution_chart(df)
            st.plotly_chart(fig, use_container_width=True, key="dist_bar")
            st.markdown('</div>', unsafe_allow_html=True)

        with dcol2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Severity Level Distribution**")
            fig = severity_distribution_chart(df)
            st.plotly_chart(fig, use_container_width=True, key="dist_pie")
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Tab 2: Sensor Analysis ─────────────────────────────────────────────
    with tab2:
        scol1, scol2 = st.columns([1, 2])

        with scol1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Select Sensor**")
            selected_sensor = st.selectbox(
                "Sensor feature",
                SENSOR_COLUMNS,
                format_func=lambda x: x.replace("_", " ").title(),
                label_visibility="collapsed",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="glass-card" style="margin-top:16px;">', unsafe_allow_html=True)
            st.markdown(f"**{selected_sensor.replace('_', ' ').title()} Stats**")
            stats = df[selected_sensor].describe()
            st.markdown(f"""
            <div style="font-size:0.85rem; color:#a0a0b8; line-height:1.8;">
                <div>Mean: <span style="color:#FF5722; font-weight:600;">{stats['mean']:.2f}</span></div>
                <div>Std: <span style="color:#FF9800; font-weight:600;">{stats['std']:.2f}</span></div>
                <div>Min: <span style="color:#00BCD4; font-weight:600;">{stats['min']:.2f}</span></div>
                <div>Max: <span style="color:#9C27B0; font-weight:600;">{stats['max']:.2f}</span></div>
                <div>Median: <span style="color:#2196F3; font-weight:600;">{stats['50%']:.2f}</span></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with scol2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Sensor Correlation Heatmap**")
            fig = sensor_heatmap_chart(df)
            st.plotly_chart(fig, use_container_width=True, key="heatmap")
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Tab 3: Cross Analysis ──────────────────────────────────────────────
    with tab3:
        ccol1, ccol2 = st.columns(2)

        with ccol1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Disaster Type × Severity**")
            fig = disaster_by_severity_chart(df)
            st.plotly_chart(fig, use_container_width=True, key="cross_stacked")
            st.markdown('</div>', unsafe_allow_html=True)

        with ccol2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Sensor Distribution by Disaster Type**")
            fig = sensor_distribution_chart(df, "seismic_activity")
            st.plotly_chart(fig, use_container_width=True, key="cross_loc")
            st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":

    st.markdown('<div class="section-title"><span>🔮 Multi-Modal Disaster Prediction</span></div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1])

    # ── Left Column: Inputs ────────────────────────────────────────────────
    with left_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**📝 Disaster Report Text**")
        report_text = st.text_area(
            "Describe the disaster situation",
            value=(
                "Strong earthquake detected in the region. Buildings are shaking violently. "
                "Cracks appearing on walls. People are panicking and running to open areas. "
                "Seismic sensors show high magnitude readings. Aftershocks are expected."
            ),
            height=160,
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**📡 Sensor Readings**")

        scol1, scol2 = st.columns(2)

        with scol1:
            seismic_activity = st.slider(
                "Seismic Activity",
                0.0, 10.0, 7.5, 0.1,
                help="Earthquake magnitude reading (Richter scale)",
            )
            ground_vibration = st.slider(
                "Ground Vibration",
                0.0, 100.0, 65.0, 0.5,
                help="Vibration level in Hz",
            )
            temperature = st.slider(
                "Temperature (°C)",
                -20.0, 60.0, 35.0, 0.5,
                help="Ambient temperature",
            )
            humidity = st.slider(
                "Humidity (%)",
                0.0, 100.0, 45.0, 1.0,
                help="Relative humidity",
            )

        with scol2:
            wind_speed = st.slider(
                "Wind Speed (km/h)",
                0.0, 300.0, 25.0, 1.0,
                help="Wind velocity",
            )
            pressure = st.slider(
                "Pressure (hPa)",
                950.0, 1050.0, 1013.0, 0.5,
                help="Atmospheric pressure",
            )
            rainfall = st.slider(
                "Rainfall (mm/h)",
                0.0, 200.0, 5.0, 1.0,
                help="Precipitation rate",
            )
            water_level = st.slider(
                "Water Level (m)",
                0.0, 20.0, 1.2, 0.1,
                help="Water level measurement",
            )

        st.markdown('</div>', unsafe_allow_html=True)

        predict_btn = st.button(
            "🚀 Run Prediction",
            use_container_width=True,
            type="primary",
        )

    # ── Right Column: Results ──────────────────────────────────────────────
    with right_col:
        if predict_btn:
            sensor_data = {
                "seismic_activity": seismic_activity,
                "ground_vibration": ground_vibration,
                "temperature_c": temperature,
                "humidity_pct": humidity,
                "wind_speed_kmh": wind_speed,
                "air_pressure_hpa": pressure,
                "rainfall_mm": rainfall,
                "water_level_m": water_level,
                "visibility_km": 5.0,
            }

            with st.spinner("🔄 Analyzing multi-modal data..."):
                result = predictor.predict(report_text, sensor_data)

            predicted_type = result["disaster_type"]
            severity = result["severity"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]

            icon_map = {
                "earthquake": "🌋",
                "flood": "🌊",
                "hurricane": "🌀",
                "wildfire": "🔥",
                "tornado": "🌪️",
                "tsunami": "🌊",
            }
            severity_class = {
                "low": "severity-low",
                "moderate": "severity-moderate",
                "high": "severity-high",
                "extreme": "severity-extreme",
            }
            severity_colors = {
                "low": "#00BCD4",
                "moderate": "#FF9800",
                "high": "#FF5722",
                "extreme": "#9C27B0",
            }

            icon = icon_map.get(predicted_type, "⚠️")
            badge_cls = severity_class.get(severity, "severity-moderate")
            sev_color = severity_colors.get(severity, "#FF9800")

            st.markdown(f"""
            <div class="prediction-result">
                <div class="prediction-icon">{icon}</div>
                <div class="prediction-type" style="color:{sev_color};">{predicted_type.title()}</div>
                <div style="margin-top:10px;">
                    <span class="severity-badge {badge_cls}">{severity.upper()}</span>
                </div>
                <div style="margin-top:16px; color:#a0a0b8; font-size:0.9rem;">
                    Model Confidence
                </div>
                <div style="font-size:2rem; font-weight:800; color:{sev_color}; margin-top:4px;">
                    {confidence * 100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Confidence Percentages ─────────────────────────────────────
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**🎯 Class Probabilities**")

            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            for ptype, pval in sorted_probs:
                picon = icon_map.get(ptype, "⚠️")
                bar_color = sev_color if ptype == predicted_type else "#a0a0b8"
                st.markdown(f"""
                <div style="margin:8px 0;">
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
                        <span style="color:#a0a0b8;">{picon} {ptype.title()}</span>
                        <span style="color:{bar_color}; font-weight:600;">{pval * 100:.1f}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:{pval * 100:.1f}%; background:linear-gradient(90deg, {bar_color}, {bar_color}aa);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Charts Row ─────────────────────────────────────────────────
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("**📊 Confidence Gauge**")
                fig = prediction_gauge_chart(confidence, title=f"{predicted_type.title()} Confidence")
                st.plotly_chart(fig, use_container_width=True, key="pred_gauge")
                st.markdown('</div>', unsafe_allow_html=True)

            with chart_col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("**📊 Probability Distribution**")
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                ptypes = [p[0].title() for p in sorted_probs]
                pvals = [p[1] * 100 for p in sorted_probs]
                fig = go.Figure(data=[go.Bar(
                    x=ptypes, y=pvals,
                    marker=dict(color=["#FF5722" if p[0] == predicted_type else "#a0a0b8" for p in sorted_probs]),
                    text=[f"{v:.1f}%" for v in pvals], textposition="auto",
                    textfont=dict(color="#fff"),
                )])
                fig.update_layout(
                    font=dict(family="Inter, sans-serif", color="#e0e0e0"),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(color="#8892a4"),
                    yaxis=dict(title="Probability (%)", color="#8892a4", gridcolor="rgba(255,255,255,0.05)"),
                    margin=dict(l=20, r=20, t=20, b=20), height=300,
                )
                st.plotly_chart(fig, use_container_width=True, key="pred_bar")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**🕸️ Multi-Sensor Radar Profile**")
            from sensor_processor import SENSOR_COLUMNS
            sensor_stats = {col: {"mean": float(df[col].mean()), "std": float(df[col].std())} for col in SENSOR_COLUMNS}
            fig = radar_chart(sensor_data, sensor_stats)
            st.plotly_chart(fig, use_container_width=True, key="pred_radar")
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="prediction-result" style="min-height:400px; display:flex; flex-direction:column; justify-content:center; align-items:center;">
                <div style="font-size:5rem; opacity:0.3;">🔮</div>
                <div style="font-size:1.1rem; color:#a0a0b8; margin-top:16px;">
                    Enter disaster report and sensor data,<br>then click <strong style="color:#FF5722;">Run Prediction</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ABOUT PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":

    st.markdown('<div class="section-title"><span>ℹ️ About This System</span></div>', unsafe_allow_html=True)

    # ── Problem Statement ──────────────────────────────────────────────────
    st.markdown("""
    <div class="glass-card" style="border-left:3px solid #FF5722;">
        <h3 style="color:#FF5722; margin-bottom:10px;">🎯 Problem Statement</h3>
        <p style="color:#a0a0b8; line-height:1.7; font-size:0.95rem;">
            Natural disasters cause devastating loss of life and property every year. Current early warning systems
            often rely on single data modalities, limiting their accuracy and lead time. There is a critical need for
            an intelligent system that can fuse <strong style="color:#ffffff;">unstructured text reports</strong> (eyewitness accounts,
            news feeds, social media) with <strong style="color:#ffffff;">structured sensor data</strong> (seismic, atmospheric, hydrological)
            to provide faster, more accurate disaster classification and severity assessment. This project builds a
            <strong style="color:#FF5722;">multi-modal AI system</strong> that bridges this gap using natural language processing and
            ensemble machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Objectives ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title"><span>🎯 Objectives</span></div>', unsafe_allow_html=True)

    objectives = [
        ("Multi-Modal Fusion", "Combine NLP-based text analysis with tabular sensor data processing to create a unified disaster classification system."),
        ("Disaster Classification", "Accurately classify incoming reports into 6 disaster types: Earthquake, Flood, Hurricane, Wildfire, Tornado, and Tsunami."),
        ("Severity Assessment", "Determine disaster severity across 4 levels (Low, Moderate, High, Extreme) to prioritize emergency response."),
        ("Real-Time Prediction", "Enable real-time inference from live sensor streams and incoming text reports for timely warnings."),
        ("Sensor Correlation", "Identify correlations between environmental sensor readings and disaster types to improve prediction accuracy."),
        ("Interactive Visualization", "Provide an intuitive dashboard with charts, gauges, and heatmaps to communicate risk assessments effectively."),
        ("Scalable Architecture", "Design a modular system with separate data generation, sensor processing, prediction, and visualization components for easy extension."),
    ]

    for i, (title, desc) in enumerate(objectives):
        st.markdown(f"""
        <div class="about-item">
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:28px; height:28px; border-radius:50%; background:linear-gradient(135deg, #FF5722, #E64A19); color:#fff; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:0.8rem; flex-shrink:0;">{i + 1}</div>
                <h4 style="color:#FF5722; margin:0; font-size:0.95rem;">{title}</h4>
            </div>
            <p style="margin-top:8px; color:#a0a0b8; font-size:0.88rem; line-height:1.6;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Technologies ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title"><span>🛠️ Technologies Used</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div style="display:flex; flex-wrap:wrap; gap:4px;">
            <span class="tech-badge">Python 3.10+</span>
            <span class="tech-badge">Streamlit</span>
            <span class="tech-badge">Pandas</span>
            <span class="tech-badge">NumPy</span>
            <span class="tech-badge">Scikit-learn</span>
            <span class="tech-badge">Plotly</span>
            <span class="tech-badge">NLTK / TextBlob</span>
            <span class="tech-badge">TF-IDF Vectorizer</span>
            <span class="tech-badge">Random Forest</span>
            <span class="tech-badge">Gradient Boosting</span>
            <span class="tech-badge">Ensemble Voting</span>
            <span class="tech-badge">Pickle Serialization</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── How It Works ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title"><span>⚙️ How It Works</span></div>', unsafe_allow_html=True)

    how_steps = [
        ("1. Data Ingestion", "The system ingests disaster reports (text) and corresponding sensor readings from CSV datasets. The data generator creates synthetic but realistic disaster data for training."),
        ("2. Text Processing (NLP)", "Text reports are processed using TF-IDF vectorization to extract meaningful features. The NLP pipeline converts unstructured text into numerical feature vectors that capture keywords, patterns, and sentiment related to disasters."),
        ("3. Sensor Feature Engineering", "Raw sensor data (seismic activity, ground vibration, temperature, humidity, wind speed, pressure, rainfall, water level) is normalized and engineered into meaningful features. Sensor correlations are computed to identify cross-sensor patterns."),
        ("4. Multi-Modal Fusion", "Text features and sensor features are concatenated into a unified feature vector. This multi-modal representation captures both the contextual information from text and the quantitative information from sensors."),
        ("5. Ensemble Classification", "An ensemble model combining Random Forest and Gradient Boosting classifiers is trained on the fused features. The model predicts both the disaster type (6 classes) and severity level (4 classes) simultaneously."),
        ("6. Real-Time Inference", "During prediction, user inputs (text report + sensor sliders) are processed through the same pipeline. The trained model outputs class probabilities, confidence scores, and severity assessments displayed through interactive visualizations."),
    ]

    for title, desc in how_steps:
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom:12px; border-left:3px solid rgba(255,87,34,0.4);">
            <h4 style="color:#FF5722; margin-bottom:6px; font-size:0.95rem;">{title}</h4>
            <p style="color:#a0a0b8; font-size:0.88rem; line-height:1.6; margin:0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Architecture ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title"><span>🏗️ System Architecture</span></div>', unsafe_allow_html=True)

    arch_cols = st.columns(4)

    arch_items = [
        ("📄", "data_generator.py", "Generates synthetic disaster dataset with text reports, sensor readings, and labels for 6 disaster types."),
        ("📡", "sensor_processor.py", "Defines sensor columns, normalizes readings, and computes sensor features & correlations."),
        ("🤖", "predict.py", "NLP + ensemble model training and inference. Handles text vectorization, model fusion, and prediction output."),
        ("📊", "utils.py", "Plotly chart factories: pie charts, bar charts, heatmaps, gauges, radar charts, and distribution plots."),
    ]

    for i, (icon, fname, desc) in enumerate(arch_items):
        with arch_cols[i]:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; min-height:200px;">
                <div style="font-size:2rem;">{icon}</div>
                <div style="font-size:0.9rem; font-weight:600; color:#FF5722; margin-top:8px;">{fname}</div>
                <div style="font-size:0.8rem; color:#a0a0b8; margin-top:8px; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
