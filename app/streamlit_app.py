import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from services.youtube_client import build_channel_videos_dataframe, YouTubeAPIError
from models.performance_predictor import PerformancePredictor
from models.virality_classifier import ViralityClassifier

# =========================
# Schema normalization
# =========================

REQUIRED_COLUMNS = {
    "niche": "unknown",
    "country": "US",
    "topic": "general",
    "subscriber_count": 0,
    "upload_hour": 18,
    "day_of_week": 5,
    "title_length": 20,
    "thumbnail_text_length": 15,
    "video_duration_sec": 600,
    "sentiment_score": 0.0,
    "like_ratio_24h": 0.02,
    "comment_ratio_24h": 0.005,
    "engagement_velocity_24h": 10.0,
    "retention_avg_pct": 40.0,
    "retention_p25_pct": 30.0,
    "retention_p50_pct": 45.0,
    "retention_p75_pct": 60.0,
    "virality_label": 0,
    "views_24h": 0.0,
    "views_7d": 0.0,
}

EMBEDDING_DIM = 16  # topic_emb_0 ... topic_emb_15


def normalize_client_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = [c.strip() for c in df.columns]

    for col, default in REQUIRED_COLUMNS.items():
        if col not in df.columns:
            df[col] = default

    for i in range(EMBEDDING_DIM):
        emb_col = f"topic_emb_{i}"
        if emb_col not in df.columns:
            df[emb_col] = 0.0

    if "title" in df.columns:
        df["title_length"] = df["title"].astype(str).str.len()

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    else:
        df["published_at"] = pd.date_range(
            end=pd.Timestamp.utcnow(), periods=len(df), freq="H"
        )

    numeric_cols = [
        "subscriber_count", "upload_hour", "day_of_week", "video_duration_sec",
        "title_length", "thumbnail_text_length", "sentiment_score",
        "like_ratio_24h", "comment_ratio_24h", "engagement_velocity_24h",
        "retention_avg_pct", "retention_p25_pct", "retention_p50_pct",
        "retention_p75_pct", "views_24h", "views_7d",
        *[f"topic_emb_{i}" for i in range(EMBEDDING_DIM)],
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
            REQUIRED_COLUMNS.get(col, 0.0)
        )

    df["virality_label"] = (df["virality_label"].astype(float) > 0.5).astype(int)
    return df


# =========================
# Config and basic setup
# =========================

MODEL_PATH_PERF = "models/artifacts/performance_predictor"
MODEL_PATH_VIRAL = "models/artifacts/virality_classifier"

st.set_page_config(
    page_title="YouTube Growth Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional styling with subtle colors
st.markdown(
    """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global styles */
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Main container - Professional dark blue gradient */
        .stApp {
            background: linear-gradient(135deg, #1a1f3a 0%, #252b48 100%);
        }
        
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #1a1f3a 0%, #252b48 100%);
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #151929 0%, #1a1f3a 100%);
            border-right: 1px solid #2d3748;
        }
        
        /* Main content area */
        .main .block-container {
            max-width: 1200px;
            padding: 2rem 3rem;
        }
        
        /* Headers - Subtle professional colors */
        h1 {
            color: #e8eaf0;
            font-weight: 600;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        h2 {
            color: #d4d7e0;
            font-weight: 600;
            font-size: 1.5rem;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        
        h3 {
            color: #c4c9d4;
            font-weight: 600;
            font-size: 1.25rem;
            margin-bottom: 1rem;
        }
        
        /* Paragraph text */
        p, li {
            color: #9ca3af;
            font-size: 1rem;
            line-height: 1.6;
        }
        
        /* Metric containers */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 600;
            color: #e8eaf0;
        }
        
        [data-testid="stMetricLabel"] {
            color: #9ca3af;
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        /* Cards/containers */
        .stMarkdown, [data-testid="stVerticalBlock"] {
            background: rgba(30, 35, 52, 0.4);
            border-radius: 12px;
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            background: rgba(30, 35, 52, 0.6);
            border: 2px dashed #4a5568;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: #5b7a9f;
            background: rgba(40, 45, 62, 0.6);
        }
        
        /* Buttons - Professional blue tone */
        .stButton > button {
            background: linear-gradient(135deg, #4a6fa5 0%, #5b7a9f 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 500;
            font-size: 1rem;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 20px rgba(74, 111, 165, 0.3);
            background: linear-gradient(135deg, #5b7a9f 0%, #6b8aaf 100%);
        }
        
        /* Radio buttons (navigation) */
        .stRadio > div {
            display: flex;
            justify-content: center;
            gap: 1rem;
            background: rgba(30, 35, 52, 0.6);
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }
        
        .stRadio > div > label {
            background: transparent;
            color: #9ca3af;
            padding: 0.5rem 1.5rem;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        
        .stRadio > div > label:hover {
            color: #d4d7e0;
            background: rgba(74, 111, 165, 0.15);
        }
        
        .stRadio > div > label[data-baseweb="radio"] > div:first-child {
            display: none;
        }
        
        /* Selectbox and inputs */
        .stSelectbox, .stTextInput, .stSlider {
            margin-bottom: 1rem;
        }
        
        .stSelectbox > div > div, .stTextInput > div > div > input {
            background: #1e2334;
            border: 1px solid #2d3748;
            border-radius: 8px;
            color: #d4d7e0;
            padding: 0.75rem;
        }
        
        .stSelectbox > div > div:hover, .stTextInput > div > div > input:hover {
            border-color: #4a6fa5;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background: rgba(30, 35, 52, 0.5);
            padding: 0.5rem;
            border-radius: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #9ca3af;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #4a6fa5 0%, #5b7a9f 100%);
            color: white;
        }
        
        /* Success/Info/Warning boxes */
        .stSuccess, .stInfo, .stWarning {
            background: rgba(30, 35, 52, 0.8);
            border-radius: 12px;
            border-left: 4px solid;
            padding: 1rem;
        }
        
        .stSuccess {
            border-left-color: #5b9279;
        }
        
        .stInfo {
            border-left-color: #5b7a9f;
        }
        
        .stWarning {
            border-left-color: #c7956d;
        }
        
        /* Plotly charts */
        .js-plotly-plot {
            border-radius: 12px;
            overflow: hidden;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #1e2334 0%, #252b48 100%);
            border-radius: 8px;
            color: #d4d7e0;
            font-weight: 600;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1a1f3a;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #4a6fa5;
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #5b7a9f;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    perf_model = PerformancePredictor.load(MODEL_PATH_PERF)
    viral_model = ViralityClassifier.load(MODEL_PATH_VIRAL)
    return perf_model, viral_model


def build_prediction_input(
    niche, country, subs, topic, hour, duration, title, thumb_text, sentiment
):
    title_len = len(title)
    thumb_len = len(thumb_text)

    data = {
        "niche": niche,
        "country": country,
        "subscriber_count": subs,
        "upload_hour": hour,
        "day_of_week": 5,
        "title": title,
        "title_length": title_len,
        "thumbnail_text_length": thumb_len,
        "video_duration_sec": duration,
        "topic": topic,
        "sentiment_score": sentiment,
        "like_ratio_24h": 0.03,
        "comment_ratio_24h": 0.008,
        "engagement_velocity_24h": 20.0,
        "retention_avg_pct": 45.0,
        "retention_p25_pct": 35.0,
        "retention_p50_pct": 50.0,
        "retention_p75_pct": 65.0,
    }

    for i in range(EMBEDDING_DIM):
        data[f"topic_emb_{i}"] = 0.0

    return pd.DataFrame([data])


perf_model, viral_model = load_models()


# =========================
# Session state: uploaded data
# =========================

if "df_data" not in st.session_state:
    st.session_state.df_data = None

# =========================
# Main header
# =========================

st.markdown(
    '<h1 style="text-align:center; margin-bottom: 0;">YouTube Growth Intelligence Platform</h1>',
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align:center; font-size:1.1rem; color: #9ca3af; margin-bottom: 2rem;'>Advanced analytics and predictive modeling for data-driven content strategy and channel optimization.</p>",
    unsafe_allow_html=True,
)

# =========================
# Navigation
# =========================

nav_options = ["Home", "Video Analyzer", "Advanced Analytics", "CSV Generator", "Pricing"]
selected_page = st.radio(
    "Navigation",
    nav_options,
    horizontal=True,
    label_visibility="collapsed",
)

st.write("")

# =========================
# Global upload section
# =========================

if selected_page in ["Home", "Advanced Analytics"]:
    st.markdown("### Upload Channel Data")
    uploaded_file = st.file_uploader(
        "Upload your channel CSV file for analysis",
        type=["csv"],
        key="global_uploader",
    )

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.session_state.df_data = normalize_client_df(raw_df)
        st.success(f"Successfully loaded {len(st.session_state.df_data):,} rows. Data normalized and ready for analysis.")
        if selected_page == "Home":
            st.info("CSV uploaded successfully. Navigate to Advanced Analytics to explore detailed insights.")

    st.write("")

# =========================
# Home page
# =========================

if selected_page == "Home":
    
    # What this platform does
    st.markdown("### Platform Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4a6fa5 0%, #5b7a9f 100%); padding: 1.5rem; border-radius: 12px; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;'>
            <div style='color: white; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;'>Performance Prediction</div>
            <div style='color: rgba(255,255,255,0.85); font-size: 0.9rem; margin-top: 0.5rem;'>Forecast views and engagement metrics for content planning</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #5b7a9f 0%, #6b8aaf 100%); padding: 1.5rem; border-radius: 12px; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;'>
            <div style='color: white; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;'>Data Analysis</div>
            <div style='color: rgba(255,255,255,0.85); font-size: 0.9rem; margin-top: 0.5rem;'>Identify growth opportunities and optimization strategies</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #6b8aaf 0%, #7b9abf 100%); padding: 1.5rem; border-radius: 12px; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;'>
            <div style='color: white; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;'>Strategic Insights</div>
            <div style='color: rgba(255,255,255,0.85); font-size: 0.9rem; margin-top: 0.5rem;'>Optimize timing, formats, and content topics</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    # How it works
    st.markdown("### Workflow")
    
    st.markdown("""
    <div style='background: rgba(30, 35, 52, 0.6); padding: 2rem; border-radius: 16px; border: 1px solid #2d3748;'>
        <div style='display: flex; align-items: center; justify-content: space-around; flex-wrap: wrap; gap: 2rem;'>
            <div style='text-align: center;'>
                <div style='width: 80px; height: 80px; background: linear-gradient(135deg, #4a6fa5 0%, #5b7a9f 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-size: 1.5rem; color: white; font-weight: 600;'>1</div>
                <div style='color: #d4d7e0; font-weight: 600;'>Upload CSV</div>
                <div style='color: #9ca3af; font-size: 0.875rem; margin-top: 0.5rem;'>Import channel data</div>
            </div>
            <div style='color: #4a5568; font-size: 2rem;'>→</div>
            <div style='text-align: center;'>
                <div style='width: 80px; height: 80px; background: linear-gradient(135deg, #5b7a9f 0%, #6b8aaf 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-size: 1.5rem; color: white; font-weight: 600;'>2</div>
                <div style='color: #d4d7e0; font-weight: 600;'>ML Analysis</div>
                <div style='color: #9ca3af; font-size: 0.875rem; margin-top: 0.5rem;'>AI-powered processing</div>
            </div>
            <div style='color: #4a5568; font-size: 2rem;'>→</div>
            <div style='text-align: center;'>
                <div style='width: 80px; height: 80px; background: linear-gradient(135deg, #6b8aaf 0%, #7b9abf 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-size: 1.5rem; color: white; font-weight: 600;'>3</div>
                <div style='color: #d4d7e0; font-weight: 600;'>Insights</div>
                <div style='color: #9ca3af; font-size: 0.875rem; margin-top: 0.5rem;'>Actionable recommendations</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Video Analyzer page
# =========================

elif selected_page == "Video Analyzer":
    st.markdown("### Video Performance Predictor")
    st.markdown("<p style='color: #9ca3af; margin-bottom: 2rem;'>Configure video parameters to generate AI-powered performance predictions</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Channel & Content Settings")
        niche = st.selectbox("Channel niche", ["gaming", "tech", "vlog", "education", "beauty"], index=0)
        subs = st.slider("Subscribers", 100, 1_000_000, 50_000, format="%d")
        st.markdown(f"<p style='color: #9ca3af; font-size: 0.875rem;'>Current: {subs:,} subscribers</p>", unsafe_allow_html=True)
        
        topic = st.selectbox("Video topic", ["tutorial", "review", "vlog", "challenge", "news"], index=0)
        hour = st.slider("Upload hour (24h format)", 0, 23, 19)
        st.markdown(f"<p style='color: #9ca3af; font-size: 0.875rem;'>Scheduled for: {hour}:00</p>", unsafe_allow_html=True)
        
        duration = st.slider("Video duration (seconds)", 60, 3600, 720)
        duration_min = duration // 60
        duration_sec = duration % 60
        st.markdown(f"<p style='color: #9ca3af; font-size: 0.875rem;'>Duration: {duration_min}m {duration_sec}s</p>", unsafe_allow_html=True)

    with col2:
        st.markdown("#### Details & Metadata")
        country = st.selectbox("Target country", ["US", "IN", "GB", "CA"], index=0)
        title = st.text_input("Video title", "My Amazing Video!")
        thumb_text = st.text_input("Thumbnail text", "Click now!")
        sentiment = st.slider("Title sentiment score", -1.0, 1.0, 0.3, step=0.1)
        
        if sentiment > 0.5:
            sentiment_label = "Very Positive"
            sentiment_color = "#5b9279"
        elif sentiment > 0:
            sentiment_label = "Positive"
            sentiment_color = "#5b7a9f"
        elif sentiment > -0.5:
            sentiment_label = "Neutral"
            sentiment_color = "#c7956d"
        else:
            sentiment_label = "Negative"
            sentiment_color = "#c77d6d"
        
        st.markdown(f"<p style='color: {sentiment_color}; font-weight: 600;'>{sentiment_label}</p>", unsafe_allow_html=True)

    st.write("")
    
    if st.button("Generate Prediction"):
        with st.spinner("Analyzing video configuration..."):
            input_df = build_prediction_input(
                niche, country, subs, topic, hour, duration, title, thumb_text, sentiment
            )
            perf_preds = perf_model.predict(input_df)
            viral_prob = viral_model.predict_proba(input_df)[0]

        st.success("Analysis complete")
        st.write("")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #4a6fa5 0%, #5b7a9f 100%); padding: 1.5rem; border-radius: 12px; text-align: center;'>
                <div style='color: white; font-size: 1.75rem; font-weight: 600;'>{perf_preds['views_24h_pred'][0]:,.0f}</div>
                <div style='color: rgba(255,255,255,0.85); font-size: 0.875rem; margin-top: 0.5rem;'>Views in 24h</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #5b7a9f 0%, #6b8aaf 100%); padding: 1.5rem; border-radius: 12px; text-align: center;'>
                <div style='color: white; font-size: 1.75rem; font-weight: 600;'>{perf_preds['views_7d_pred'][0]:,.0f}</div>
                <div style='color: rgba(255,255,255,0.85); font-size: 0.875rem; margin-top: 0.5rem;'>Views in 7 days</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #6b8aaf 0%, #7b9abf 100%); padding: 1.5rem; border-radius: 12px; text-align: center;'>
                <div style='color: white; font-size: 1.75rem; font-weight: 600;'>{viral_prob:.1%}</div>
                <div style='color: rgba(255,255,255,0.85); font-size: 0.875rem; margin-top: 0.5rem;'>Virality probability</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_d:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #7b9abf 0%, #8baacf 100%); padding: 1.5rem; border-radius: 12px; text-align: center;'>
                <div style='color: white; font-size: 1.75rem; font-weight: 600;'>{perf_preds['engagement_velocity_24h_pred'][0]:.0f}</div>
                <div style='color: rgba(255,255,255,0.85); font-size: 0.875rem; margin-top: 0.5rem;'>Engagement per hour</div>
            </div>
            """, unsafe_allow_html=True)

# =========================
# Advanced Analytics page
# =========================

elif selected_page == "Advanced Analytics":
    df_data = st.session_state.df_data
    if df_data is None:
        st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem;'>
            <h3 style='color: #d4d7e0; margin-bottom: 1rem;'>No Data Available</h3>
            <p style='color: #9ca3af;'>Upload your channel CSV on the Home or Advanced Analytics tab to access detailed analytics and insights.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### Channel Analytics Dashboard")

        tab_overview, tab_strategy, tab_explain = st.tabs(
            ["Creator Overview", "Strategy Advisor", "Explainability"]
        )

        with tab_overview:
            st.markdown("#### Channel Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            avg_views = df_data["views_24h"].mean()
            viral_rate = df_data["virality_label"].mean()
            avg_retention = df_data["retention_avg_pct"].mean()
            
            with col1:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #4a6fa5 0%, #5b7a9f 100%); padding: 2rem; border-radius: 12px; text-align: center;'>
                    <div style='color: rgba(255,255,255,0.85); font-size: 0.875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Average 24h Views</div>
                    <div style='color: white; font-size: 2.5rem; font-weight: 600;'>{avg_views:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #5b7a9f 0%, #6b8aaf 100%); padding: 2rem; border-radius: 12px; text-align: center;'>
                    <div style='color: rgba(255,255,255,0.85); font-size: 0.875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Virality Rate</div>
                    <div style='color: white; font-size: 2.5rem; font-weight: 600;'>{viral_rate:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #6b8aaf 0%, #7b9abf 100%); padding: 2rem; border-radius: 12px; text-align: center;'>
                    <div style='color: rgba(255,255,255,0.85); font-size: 0.875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Avg Retention</div>
                    <div style='color: white; font-size: 2.5rem; font-weight: 600;'>{avg_retention:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.write("")
            st.markdown("#### Views Over Time")
            
            # Create a beautiful line chart with Plotly
            views_over_time = df_data.groupby("published_at")["views_24h"].mean().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=views_over_time["published_at"],
                y=views_over_time["views_24h"],
                mode='lines',
                name='Average Views',
                line=dict(color='#5b7a9f', width=3),
                fill='tozeroy',
                fillcolor='rgba(91, 122, 159, 0.2)'
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(30, 35, 52, 0.8)',
                paper_bgcolor='rgba(30, 35, 52, 0.8)',
                font=dict(color='#d4d7e0'),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    title="Date"
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    title="Average Views"
                ),
                hovermode='x unified',
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)

        with tab_strategy:
            st.markdown("#### Data-Driven Growth Strategies")
            
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### Optimal Upload Timing")
                best_hours = df_data.groupby("upload_hour")["virality_label"].mean().nlargest(3)
                
                for idx, (h, prob) in enumerate(best_hours.items()):
                    gradient = ["#4a6fa5", "#5b7a9f", "#6b8aaf"][idx]
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {gradient} 0%, rgba(74, 111, 165, 0.3) 100%); 
                                padding: 1rem 1.5rem; border-radius: 10px; margin-bottom: 0.75rem;
                                border-left: 4px solid {gradient};'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <span style='color: white; font-size: 1.25rem; font-weight: 600;'>{h:02d}:00</span>
                                <span style='color: rgba(255,255,255,0.8); font-size: 0.875rem; margin-left: 0.5rem;'>Upload Hour</span>
                            </div>
                            <div style='background: rgba(255,255,255,0.2); padding: 0.25rem 0.75rem; border-radius: 20px;'>
                                <span style='color: white; font-weight: 600;'>{prob:.1%}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown("##### High-Potential Topics")
                topic_viral = df_data.groupby("topic")["virality_label"].mean().nlargest(3)
                
                for idx, (t, prob) in enumerate(topic_viral.items()):
                    gradient = ["#5b7a9f", "#6b8aaf", "#7b9abf"][idx]
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {gradient} 0%, rgba(91, 122, 159, 0.3) 100%); 
                                padding: 1rem 1.5rem; border-radius: 10px; margin-bottom: 0.75rem;
                                border-left: 4px solid {gradient};'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <span style='color: white; font-size: 1.25rem; font-weight: 600; text-transform: capitalize;'>{t}</span>
                            </div>
                            <div style='background: rgba(255,255,255,0.2); padding: 0.25rem 0.75rem; border-radius: 20px;'>
                                <span style='color: white; font-weight: 600;'>{prob:.1%}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with tab_explain:
            st.markdown("#### Understanding Virality Drivers")
            st.markdown("<p style='color: #9ca3af; margin-bottom: 2rem;'>Discover which factors most strongly influence whether your videos go viral</p>", unsafe_allow_html=True)
            
            importance_df = viral_model.feature_importance(top_k=10)
            
            # Create horizontal bar chart
            fig = go.Figure()
            
            colors = [f'rgba({74 + i*10}, {111 - i*5}, {165 - i*10}, 0.8)' for i in range(len(importance_df))]
            
            fig.add_trace(go.Bar(
                y=importance_df["feature"],
                x=importance_df["importance"],
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
                ),
                text=importance_df["importance"].round(1),
                textposition='auto',
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(30, 35, 52, 0.8)',
                paper_bgcolor='rgba(30, 35, 52, 0.8)',
                font=dict(color='#d4d7e0', size=12),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    title="Importance Score"
                ),
                yaxis=dict(
                    showgrid=False,
                    title=""
                ),
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

            top_feature = importance_df.iloc[0]["feature"]
            st.success(f"**{top_feature}** is currently the strongest virality driver in your data.")

elif selected_page == "CSV Generator":
    st.subheader("Generate CSV from YouTube Channel")

    st.markdown(
        "Enter a YouTube channel URL or handle to fetch recent uploads and convert them into a CSV compatible with Advance Analytics."
    )

    channel_input = st.text_input("Channel URL or @handle", placeholder="https://www.youtube.com/@creatorname")

    col_btn1, col_btn2 = st.columns(2)
    df_generated = st.session_state.get("generated_df")

    if col_btn1.button("Fetch channel data", type="primary"):
        if not channel_input.strip():
            st.warning("Please enter a channel URL or handle.")
        else:
            try:
                with st.spinner("Fetching uploads from YouTube API..."):
                    df_raw = build_channel_videos_dataframe(channel_input.strip())
                    if df_raw.empty:
                        st.warning("No videos found for this channel.")
                    else:
                        # Reuse your existing normalizer
                        df_norm = normalize_client_df(df_raw)
                        st.session_state.generated_df = df_norm
                        st.session_state.df_data = df_norm   # for Advance Analytics
                        st.success(f"Fetched and normalized {len(df_norm):,} videos.")
            except YouTubeAPIError as e:
                st.error(f"API error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    if df_generated is not None:
        st.markdown("Preview of generated dataset (first 10 rows):")
        st.dataframe(df_generated.head(10))

        csv_bytes = df_generated.to_csv(index=False).encode("utf-8")
        col_btn2.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="channel_videos_normalized.csv",
            mime="text/csv",
        )

        if st.button("Open in Advance Analytics"):
            # user will click Advance Analytics tab; data is already in session
            st.info("Switch to the Advance Analytics tab to explore channel insights.")


elif selected_page == "Pricing":
    st.markdown("### Choose Your Plan")
    st.markdown("<p style='text-align: center; color: #9ca3af; margin-bottom: 3rem;'>Select the perfect plan for your YouTube growth journey</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style='background: rgba(30, 35, 52, 0.6); padding: 2rem; border-radius: 16px; border: 1px solid #2d3748; height: 450px;'>
            <div style='text-align: center; margin-bottom: 1.5rem;'>
                <h3 style='color: #d4d7e0; margin-bottom: 0.5rem;'>Starter</h3>
                <div style='color: #5b9279; font-size: 2rem; font-weight: 600;'>Free</div>
            </div>
            <div style='color: #9ca3af; line-height: 2;'>
                <div style='margin-bottom: 0.5rem;'>✓ For individual creators</div>
                <div style='margin-bottom: 0.5rem;'>✓ 1 channel</div>
                <div style='margin-bottom: 0.5rem;'>✓ Limited analytics</div>
                <div style='margin-bottom: 0.5rem;'>✓ Community support</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4a6fa5 0%, #5b7a9f 100%); padding: 2rem; border-radius: 16px; border: 2px solid #4a6fa5; height: 450px; transform: scale(1.05); box-shadow: 0 20px 40px rgba(74, 111, 165, 0.3);'>
            <div style='text-align: center; margin-bottom: 1.5rem;'>
                <div style='background: rgba(255,255,255,0.2); display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-bottom: 1rem;'>MOST POPULAR</div>
                <h3 style='color: white; margin-bottom: 0.5rem;'>Pro</h3>
                <div style='color: white; font-size: 2rem; font-weight: 600;'>$29/mo</div>
            </div>
            <div style='color: rgba(255,255,255,0.9); line-height: 2;'>
                <div style='margin-bottom: 0.5rem;'>✓ For growing creators</div>
                <div style='margin-bottom: 0.5rem;'>✓ Up to 5 channels</div>
                <div style='margin-bottom: 0.5rem;'>✓ Full analytics & strategy</div>
                <div style='margin-bottom: 0.5rem;'>✓ Priority support</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='background: rgba(30, 35, 52, 0.6); padding: 2rem; border-radius: 16px; border: 1px solid #2d3748; height: 450px;'>
            <div style='text-align: center; margin-bottom: 1.5rem;'>
                <h3 style='color: #d4d7e0; margin-bottom: 0.5rem;'>Agency</h3>
                <div style='color: #6b8aaf; font-size: 2rem; font-weight: 600;'>Custom</div>
            </div>
            <div style='color: #9ca3af; line-height: 2;'>
                <div style='margin-bottom: 0.5rem;'>✓ For agencies & brands</div>
                <div style='margin-bottom: 0.5rem;'>✓ Unlimited channels</div>
                <div style='margin-bottom: 0.5rem;'>✓ Team workspaces & API</div>
                <div style='margin-bottom: 0.5rem;'>✓ Dedicated success manager</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#718096; font-size: 0.875rem; padding: 1rem;'>Models are pre-trained on internal datasets. Your CSV is used for analytics and predictions only, not for training.</div>",
    unsafe_allow_html=True,
)