# =========================================================================================
# 🛒 RETAILNEXUS INTELLIGENCE TERMINAL (ENTERPRISE E-COMMERCE EDITION)
# Version: 11.2.0 | Build: Production/Max-Scale (Intelligent Keyword Engine)
# Description: Advanced Content-Based Filtering Dashboard for E-Commerce Recommendations.
# Features TF-IDF vectorization, Dual-Pass Search (Substring + Fuzzy), and Vector Analytics.
# Theme: RetailNexus (Deep Charcoal, Electric Cyan, Commerce Gold)
# =========================================================================================

import streamlit as st
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import base64
import json
import difflib
from datetime import datetime
import uuid

# Scikit-learn is required to calculate dot products/cosine similarity on the fly
try:
    from sklearn.metrics.pairwise import linear_kernel
except ImportError:
    pass

# =========================================================================================
# 1. PAGE CONFIGURATION & SECURE INITIALIZATION
# =========================================================================================
st.set_page_config(
    page_title="RetailNexus | Product Recommendation Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================================
# 2. MACHINE LEARNING ASSET INGESTION (SINGLE BUNDLE)
# =========================================================================================
@st.cache_resource
def load_ml_infrastructure():
    """
    Safely loads the bundled TF-IDF arrays, DataFrames, and vectorizers.
    Uses pd.read_pickle to natively handle any Pandas StringDtype memory issues.
    """
    df = None
    tfidf = None
    tfidf_matrix = None
    
    try:
        bundle = pd.read_pickle("recommendation_system.pkl")
        df = bundle.get('df')
        tfidf = bundle.get('tfidf')
        tfidf_matrix = bundle.get('tfidf_matrix')
    except Exception as e1:
        try:
            with open("recommendation_system.pkl", "rb") as f:
                bundle = pickle.load(f)
                df = bundle.get('df')
                tfidf = bundle.get('tfidf')
                tfidf_matrix = bundle.get('tfidf_matrix')
        except Exception as e2:
            st.sidebar.error(f"🔴 BUNDLE LOAD ERROR: {str(e2)}\n\n(Ensure `recommendation_system.pkl` exists)")
            
    return df, tfidf, tfidf_matrix

df, tfidf_vectorizer, tfidf_matrix = load_ml_infrastructure()

# Extracting a clean list of product names for fuzzy & substring matching
if df is not None and 'Name' in df.columns:
    ALL_PRODUCTS = df['Name'].dropna().astype(str).tolist()
else:
    ALL_PRODUCTS = []

# =========================================================================================
# 3. ENTERPRISE CSS INJECTION (MASSIVE STYLESHEET FOR RETAILNEXUS THEME)
# =========================================================================================
st.markdown(
"""<style>
@import url('https://fonts.googleapis.com/css2?family=Syncopate:wght@400;700&family=Inter:wght@300;400;500;700&family=Space+Mono:wght@400;700&display=swap');

/* ── GLOBAL COLOR PALETTE & CSS VARIABLES ── */
:root {
    --retail-900:    #09090b;
    --retail-800:    #18181b;
    --retail-700:    #27272a;
    --cyan-core:     #06b6d4;
    --cyan-dim:      rgba(6, 182, 212, 0.2);
    --gold-core:     #f59e0b;
    --gold-dim:      rgba(245, 158, 11, 0.2);
    --white-main:    #f8fafc;
    --slate-light:   #94a3b8;
    --slate-dark:    #475569;
    --glass-bg:      rgba(24, 24, 27, 0.6);
    --glass-border:  rgba(6, 182, 212, 0.15);
    --glow-cyan:     0 0 35px rgba(6, 182, 212, 0.25);
    --glow-gold:     0 0 35px rgba(245, 158, 11, 0.25);
}

/* ── BASE APPLICATION STYLING & TYPOGRAPHY ── */
.stApp {
    background: var(--retail-900);
    font-family: 'Inter', sans-serif;
    color: var(--slate-light);
    overflow-x: hidden;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Syncopate', sans-serif;
    color: var(--white-main);
    letter-spacing: 1px;
}

/* ── DYNAMIC BACKGROUND ANIMATIONS ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: 
        radial-gradient(circle at 15% 20%, rgba(6, 182, 212, 0.05) 0%, transparent 40%),
        radial-gradient(circle at 85% 80%, rgba(245, 158, 11, 0.03) 0%, transparent 40%),
        radial-gradient(circle at 50% 50%, rgba(9, 9, 11, 0.8) 0%, transparent 80%);
    pointer-events: none;
    z-index: 0;
    animation: retailPulse 15s ease-in-out infinite alternate;
}

@keyframes retailPulse {
    0%   { opacity: 0.5; filter: hue-rotate(0deg); }
    100% { opacity: 1.0; filter: hue-rotate(10deg); }
}

/* ── DATA GRID OVERLAY ── */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image: 
        radial-gradient(rgba(6, 182, 212, 0.04) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

/* ── MAIN CONTAINER SPACING ── */
.main .block-container {
    position: relative;
    z-index: 1;
    padding-top: 30px;
    padding-bottom: 90px;
    max-width: 1550px;
}

/* ── HERO SECTION & HEADERS ── */
.hero {
    text-align: center;
    padding: 80px 20px 60px;
    animation: slideDown 0.9s cubic-bezier(0.22,1,0.36,1) both;
}

@keyframes slideDown {
    from { opacity: 0; transform: translateY(-50px); }
    to   { opacity: 1; transform: translateY(0); }
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 15px;
    background: rgba(6, 182, 212, 0.05);
    border: 1px solid rgba(6, 182, 212, 0.3);
    border-radius: 50px;
    padding: 10px 30px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: var(--cyan-core);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 25px;
    box-shadow: var(--glow-cyan);
}

.hero-badge-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--gold-core);
    box-shadow: 0 0 12px var(--gold-core);
    animation: recordingTick 1.5s ease-in-out infinite;
}

@keyframes recordingTick {
    0%, 100% { transform: scale(1); opacity: 0.6; }
    50%      { transform: scale(1.6); opacity: 1; box-shadow: 0 0 20px var(--gold-core); }
}

.hero-title {
    font-family: 'Syncopate', sans-serif;
    font-size: clamp(40px, 6vw, 85px);
    font-weight: 700;
    letter-spacing: 2px;
    line-height: 1.1;
    margin-bottom: 18px;
    text-transform: uppercase;
}

.hero-title em {
    font-style: normal;
    color: var(--cyan-core);
    text-shadow: 0 0 35px rgba(6, 182, 212, 0.4);
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 15px;
    font-weight: 400;
    color: var(--slate-light);
    letter-spacing: 4px;
    text-transform: uppercase;
}

/* ── GLASS PANELS & UI CARDS ── */
.glass-panel {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 45px;
    margin-bottom: 35px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(12px);
    transition: all 0.4s ease;
    animation: fadeUp 0.8s ease both;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}

.glass-panel:hover {
    border-color: rgba(6, 182, 212, 0.4);
    box-shadow: var(--glow-cyan);
    transform: translateY(-2px);
}

.panel-heading {
    font-family: 'Syncopate', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--white-main);
    letter-spacing: 2px;
    margin-bottom: 35px;
    border-bottom: 1px solid rgba(6, 182, 212, 0.2);
    padding-bottom: 15px;
    text-transform: uppercase;
}

/* ── COMPONENT OVERRIDES (STREAMLIT NATIVE) ── */
div[data-testid="stTextInput"] label { display: none !important; }

/* REVISED TEXT INPUT STYLING */
div[data-testid="stTextInput"] > div > div > input {
    background: rgba(24, 24, 27, 0.9) !important;
    border: 1px solid rgba(6, 182, 212, 0.4) !important;
    color: var(--white-main) !important;
    border-radius: 8px !important;
    padding: 18px 20px !important;
    font-size: 16px !important;
    line-height: 1.5 !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.5) !important;
}

div[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--cyan-core) !important;
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.3), inset 0 2px 10px rgba(0,0,0,0.5) !important;
}

/* REVISED BUTTON STYLING */
div.stButton > button {
    width: 100% !important;
    background: transparent !important;
    color: var(--cyan-core) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: 1px solid var(--cyan-core) !important;
    border-radius: 8px !important;
    padding: 18px 20px !important;
    line-height: 1.5 !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    background-color: rgba(6, 182, 212, 0.05) !important;
    margin-top: 0px !important; 
    box-shadow: 0 5px 15px rgba(6, 182, 212, 0.1) !important;
}

div.stButton > button:hover {
    background-color: rgba(6, 182, 212, 0.15) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 25px rgba(6, 182, 212, 0.3) !important;
}

/* ── PREDICTION RESULT BOX (PRODUCT CARD STYLE) ── */
.product-card {
    background: var(--retail-800) !important;
    border: 1px solid var(--glass-border) !important;
    padding: 30px !important;
    border-radius: 12px !important;
    position: relative !important;
    overflow: hidden !important;
    margin-bottom: 20px !important;
    transition: all 0.3s ease !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
}

.product-card:hover {
    border-color: var(--cyan-core) !important;
    box-shadow: var(--glow-cyan) !important;
    transform: translateX(5px) !important;
}

.product-rank {
    position: absolute;
    top: -10px;
    right: 15px;
    font-family: 'Syncopate', sans-serif;
    font-size: 60px;
    font-weight: 700;
    color: rgba(255,255,255,0.03);
    z-index: 0;
}

.product-title {
    font-family: 'Inter', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--white-main);
    letter-spacing: 0.5px;
    margin-bottom: 10px;
    position: relative;
    z-index: 1;
}

.product-meta {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: var(--gold-core);
    margin-bottom: 15px;
    position: relative;
    z-index: 1;
}

.product-desc {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: var(--slate-light);
    line-height: 1.6;
    margin-bottom: 20px;
    position: relative;
    z-index: 1;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.product-sim {
    display: inline-block;
    background: rgba(6, 182, 212, 0.1);
    border: 1px solid rgba(6, 182, 212, 0.4);
    color: var(--cyan-core);
    padding: 8px 20px;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    letter-spacing: 2px;
    position: relative;
    z-index: 1;
}

/* ── TABS NAVIGATION STYLING ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--retail-800) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(6, 182, 212, 0.2) !important;
    padding: 8px !important;
    gap: 12px !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--slate-dark) !important;
    border-radius: 6px !important;
    padding: 18px 30px !important;
    transition: all 0.3s ease !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(6, 182, 212, 0.1) !important;
    color: var(--cyan-core) !important;
    border: 1px solid rgba(6, 182, 212, 0.4) !important;
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.1) !important;
}

/* ── SIDEBAR STYLING & TELEMETRY ── */
section[data-testid="stSidebar"] {
    background: var(--retail-900) !important;
    border-right: 1px solid rgba(6, 182, 212, 0.15) !important;
}

.sb-logo-text {
    font-family: 'Syncopate', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: var(--white-main);
    letter-spacing: 2px;
}

.sb-title {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    color: var(--slate-light);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding-bottom: 10px;
    margin-top: 35px;
}

.telemetry-card {
    background: rgba(24, 24, 27, 0.5) !important;
    border: 1px solid rgba(6, 182, 212, 0.15) !important;
    padding: 22px !important;
    border-radius: 8px !important;
    text-align: center !important;
    margin-bottom: 18px !important;
    transition: all 0.3s ease;
}

.telemetry-card:hover {
    background: rgba(39, 39, 42, 0.9) !important;
    border-color: rgba(6, 182, 212, 0.4) !important;
    transform: translateY(-2px);
}

.telemetry-val {
    font-family: 'Space Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: var(--cyan-core);
}

.telemetry-lbl {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    color: var(--slate-dark);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================================================
# 4. SESSION STATE MANAGEMENT
# =========================================================================================
if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"RET-IDX-{str(uuid.uuid4())[:8].upper()}"
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""
if "matched_product" not in st.session_state:
    st.session_state["matched_product"] = None
if "match_confidence" not in st.session_state:
    st.session_state["match_confidence"] = 0.0
if "match_type" not in st.session_state:
    st.session_state["match_type"] = ""
if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = None
if "compute_latency" not in st.session_state:
    st.session_state["compute_latency"] = 0.0
if "timestamp" not in st.session_state:
    st.session_state["timestamp"] = None

# =========================================================================================
# 5. ENTERPRISE SIDEBAR LOGIC (SYSTEM TELEMETRY)
# =========================================================================================
with st.sidebar:
    st.markdown(
"""<div style='text-align:center; padding:25px 0 35px;'>
<div class="sb-logo-text">RETAILNEXUS</div>
<div style="font-family:'Space Mono'; font-size:10px; color:rgba(6,182,212,0.8); letter-spacing:4px; margin-top:8px;">RECOMMENDATION ENGINE</div>
<div style="font-family:'Space Mono'; font-size:9px; color:rgba(255,255,255,0.3); margin-top:12px;">ID: {}</div>
</div>""".format(st.session_state["session_id"]),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-title">⚙️ Architecture Specs</div>', unsafe_allow_html=True)
    st.markdown(
"""<div style="background:rgba(24,24,27,0.6); padding:20px; border-radius:8px; border:1px solid rgba(6,182,212,0.2); font-family:Inter; font-size:13px; color:rgba(248,250,252,0.8); line-height:1.9;">
<b>Algorithm:</b> Content-Based Filtering<br>
<b>Search:</b> Dual-Pass (Keyword + Fuzzy)<br>
<b>Distance Metric:</b> Cosine Similarity<br>
<b>Data Bundle:</b> Multi-Asset .pkl<br>
</div>""", unsafe_allow_html=True
    )

    st.markdown('<div class="sb-title">📊 Catalog Telemetry</div>', unsafe_allow_html=True)
    
    total_products = len(df) if df is not None else 0
    vocab_size = len(tfidf_vectorizer.vocabulary_) if tfidf_vectorizer and hasattr(tfidf_vectorizer, 'vocabulary_') else "N/A"
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f'<div class="telemetry-card"><div class="telemetry-val" style="color:var(--white-main);">{total_products}</div><div class="telemetry-lbl">Products</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="telemetry-card"><div class="telemetry-val" style="color:var(--gold-core);">1.0</div><div class="telemetry-lbl">Max Sim</div></div>', unsafe_allow_html=True)
    with col_s2:
        st.markdown(f'<div class="telemetry-card"><div class="telemetry-val" style="color:var(--cyan-core);">{vocab_size}</div><div class="telemetry-lbl">NLP Tokens</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="telemetry-card"><div class="telemetry-val">{st.session_state["compute_latency"]}s</div><div class="telemetry-lbl">Latency</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.session_state["recommendations"] is None:
        st.markdown(
"""<div style="padding:15px; border-left:4px solid var(--slate-dark); background:rgba(255,255,255,0.05); border-radius:4px; font-family:Inter; font-size:12px; color:var(--slate-light);">
<b>SYSTEM STANDBY</b><br>Awaiting query input for vector matching.
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(
f"""<div style="padding:15px; border-left:4px solid var(--cyan-core); background:rgba(6,182,212,0.05); border-radius:4px; font-family:Inter; font-size:12px; color:var(--cyan-core);">
<b>COMPUTE COMPLETE</b><br>Cosine Dot-Product Latency: {st.session_state['compute_latency']}s
</div>""", unsafe_allow_html=True)

# =========================================================================================
# 6. HERO HEADER SECTION
# =========================================================================================
st.markdown(
"""<div class="hero">
<div class="hero-badge">
<div class="hero-badge-dot"></div>
TF-IDF VECTORIZATION | COSINE SIMILARITY MATCHING
</div>
<div class="hero-title">OMNI-CHANNEL <em>DISCOVERY</em></div>
<div class="hero-sub">Enterprise Machine Learning Dashboard For E-Commerce Intelligence</div>
</div>""",
    unsafe_allow_html=True,
)

# =========================================================================================
# 7. MAIN APPLICATION TABS (6-TAB MONOLITHIC ARCHITECTURE)
# =========================================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🛒 PRODUCT DISCOVERY", 
    "📊 VECTOR ANALYTICS", 
    "🧠 NLP ARCHITECTURE", 
    "📈 CONVERSION FORECASTING",
    "🎲 BEHAVIORAL VARIANCE",
    "📋 EXPORT DOSSIER"
])

# =========================================================================================
# TAB 1 - DISCOVERY ENGINE (DUAL-PASS MATCHING & RESULTS)
# =========================================================================================
with tab1:
    
    st.markdown('<div class="glass-panel"><div class="panel-heading" style="margin-bottom:15px; border:none; padding-bottom:0;">🔍 Query Catalog Data Lake</div>', unsafe_allow_html=True)
    
    col_input, col_btn = st.columns([4, 1], vertical_alignment="bottom")
    
    with col_input:
        user_query = st.text_input("Search Catalog", placeholder="Enter a generic term ('Laptop') or specific product ('Dell XPS').")
        
    with col_btn:
        search_clicked = st.button("RUN SIMILARITY SEARCH", use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

    if search_clicked and user_query:
        if df is None or tfidf_matrix is None:
            st.error("SYSTEM HALT: Core `recommendation_system.pkl` missing or corrupted. Cannot execute vector search.")
        elif not ALL_PRODUCTS:
            st.error("DATA ERROR: Could not extract 'Name' column from the DataFrame. Check your data structure.")
        else:
            with st.spinner("Running Intelligent Keyword Mapping and computing Cosine similarities..."):
                start_time = time.time()
                time.sleep(0.8) # UI polish
                
                # --- NEW DUAL-PASS MATCHING LOGIC ---
                matched_product = None
                match_ratio = 0.0
                match_type = ""
                
                # Pass 1: Substring/Keyword Scan (Case Insensitive)
                # This catches searches like "Laptop" or "Shoe" and anchors to the cleanest product name containing that word.
                substring_matches = [p for p in ALL_PRODUCTS if user_query.lower() in str(p).lower()]
                
                if substring_matches:
                    # Sort by length so the most generic/shortest matching name is picked as the anchor
                    substring_matches.sort(key=len)
                    matched_product = substring_matches[0]
                    match_ratio = 100.0
                    match_type = "KEYWORD"
                else:
                    # Pass 2: Typo Resolution / Fuzzy Matching (Levenshtein Distance)
                    closest_matches = difflib.get_close_matches(user_query, ALL_PRODUCTS, n=1, cutoff=0.45)
                    if closest_matches:
                        matched_product = closest_matches[0]
                        match_ratio = difflib.SequenceMatcher(None, user_query.lower(), matched_product.lower()).ratio() * 100
                        match_type = "FUZZY"
                
                if not matched_product:
                    st.warning(f"CATALOG MISS: No products or categories found matching '{user_query}'. Please try a different search term.")
                    st.session_state["recommendations"] = None
                else:
                    st.session_state["matched_product"] = matched_product
                    st.session_state["match_confidence"] = match_ratio
                    st.session_state["match_type"] = match_type
                    st.session_state["user_input"] = user_query
                    
                    # --- COSINE SIMILARITY LOGIC ---
                    try:
                        # 1. Find the exact DataFrame index
                        matched_row_series = df[df['Name'] == matched_product]
                        if matched_row_series.empty:
                            st.error("Database alignment error: Matched name not found in index.")
                        else:
                            actual_index = matched_row_series.index[0]
                            row_position = df.index.get_loc(actual_index)
                            
                            # 2. Compute dot product
                            sim_scores = linear_kernel(tfidf_matrix[row_position], tfidf_matrix).flatten()
                            
                            # 3. Sort
                            sim_scores_enum = list(enumerate(sim_scores))
                            sim_scores_sorted = sorted(sim_scores_enum, key=lambda x: x[1], reverse=True)
                            
                            # 4. Get top 10 (skipping index 0 which is the product itself)
                            top_10_positions = [i[0] for i in sim_scores_sorted[1:11]]
                            top_10_scores = [i[1] for i in sim_scores_sorted[1:11]]
                            
                            # 5. Extract to new DataFrame
                            recommendations_df = df.iloc[top_10_positions].copy()
                            recommendations_df['Similarity_Score'] = top_10_scores
                            
                            st.session_state["recommendations"] = recommendations_df
                            
                            end_time = time.time()
                            st.session_state["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
                            st.session_state["compute_latency"] = round(end_time - start_time, 3)
                        
                    except Exception as e:
                        st.error(f"Computation Error during Matrix Multiplication: {e}")

    # --- RENDER RESULTS ---
    if st.session_state["recommendations"] is not None:
        matched = st.session_state["matched_product"]
        conf = st.session_state["match_confidence"]
        m_type = st.session_state.get("match_type", "")
        query = st.session_state["user_input"]
        recs = st.session_state["recommendations"]
        
        # Intelligent Search Alert Banner
        if m_type == "KEYWORD" and query.lower() != matched.lower():
            st.markdown(
f"""<div style="background:rgba(6, 182, 212, 0.1); border-left:4px solid var(--cyan-core); padding:15px 20px; margin-bottom:30px; border-radius:4px;">
<span style="color:var(--cyan-core); font-family:'Space Mono'; font-size:12px; letter-spacing:2px; text-transform:uppercase;">🔍 KEYWORD ANCHOR ACTIVATED</span><br>
<span style="color:var(--white-main); font-family:'Inter'; font-size:14px;">Broad query <b>"{query}"</b> was automatically anchored to the foundational product: <b>"{matched}"</b> to compute matrix distances.</span>
</div>""", unsafe_allow_html=True)
        elif m_type == "FUZZY" and conf < 99.0:
            st.markdown(
f"""<div style="background:rgba(245,158,11,0.1); border-left:4px solid var(--gold-core); padding:15px 20px; margin-bottom:30px; border-radius:4px;">
<span style="color:var(--gold-core); font-family:'Space Mono'; font-size:12px; letter-spacing:2px; text-transform:uppercase;">⚠️ FUZZY MATCH ACTIVATED</span><br>
<span style="color:var(--white-main); font-family:'Inter'; font-size:14px;">Typo detected. Input was auto-corrected to <b>"{matched}"</b> with a {conf:.1f}% confidence threshold.</span>
</div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="panel-heading" style="border:none;">🛒 Top Recommendations based on: <span style="color:var(--cyan-core);">{matched}</span></div>', unsafe_allow_html=True)
        
        # Render custom UI cards
        for i, row in recs.iterrows():
            title = str(row.get('Name', 'Unknown Product'))
            score = float(row.get('Similarity_Score', 0.0)) * 100
            brand = str(row.get('Brand', 'No Brand'))
            category = str(row.get('category', 'Uncategorized'))
            price = str(row.get('Selling Price', 'N/A'))
            rating = str(row.get('Ratings', 'N/A'))
            reviews = str(row.get('No_of_ratings', '0'))
            
            details = str(row.get('Details', 'No details available.'))
            if len(details) > 200: details = details[:197] + "..."
            
            st.markdown(
f"""<div class="product-card">
<div class="product-rank">{(recs.index.get_loc(i) + 1):02d}</div>
<div class="product-title">{title}</div>
<div class="product-meta">BRAND: {brand} | CATEGORY: {category}</div>
<div class="product-meta" style="color:var(--white-main); margin-bottom:15px;">
    PRICE: <span style="color:var(--gold-core); font-weight:700;">{price}</span> | 
    RATING: ⭐ {rating} ({reviews} reviews)
</div>
<div class="product-desc">{details}</div>
<div><div class="product-sim">COSINE SIMILARITY: {score:.1f}%</div></div>
</div>""", unsafe_allow_html=True)

# =========================================================================================
# TAB 2 - VECTOR ANALYTICS & PRODUCT DISTRIBUTION
# =========================================================================================
with tab2:
    if st.session_state["recommendations"] is None:
        st.markdown(
"""<div style='text-align:center; padding:150px 20px; font-family:"Syncopate",serif; font-size:20px; letter-spacing:4px; color:rgba(6,182,212,0.4); text-transform:uppercase;'>
⚠️ Execute Discovery Search To Unlock Vector Analytics
</div>""",
            unsafe_allow_html=True,
        )
    else:
        recs = st.session_state["recommendations"]
        
        col_a1, col_a2 = st.columns(2)

        with col_a1:
            st.markdown('<div class="panel-heading" style="border:none;">📉 Cosine Similarity Decay Curve</div>', unsafe_allow_html=True)
            
            # Truncate titles for the X-axis to keep charts clean
            raw_titles = recs.get('Name', [f"Rec {i}" for i in range(1, 11)]).astype(str).tolist()
            short_titles = [t[:15]+"..." if len(t)>15 else t for t in raw_titles]
            scores = (recs.get('Similarity_Score', [0]*10) * 100).tolist()
            
            fig_decay = go.Figure()
            fig_decay.add_trace(go.Scatter(
                x=short_titles, y=scores, mode='lines+markers',
                line=dict(color='#06b6d4', width=3, shape='spline'),
                marker=dict(size=8, color='#f8fafc', line=dict(width=2, color='#06b6d4')),
                fill='tozeroy', fillcolor='rgba(6, 182, 212, 0.1)'
            ))
            
            fig_decay.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(24,24,27,0.5)",
                font=dict(family="Inter", color="#f8fafc"),
                xaxis=dict(title="", tickangle=45, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Similarity Score (%)", range=[min(scores)-5, max(scores)+5], gridcolor="rgba(255,255,255,0.05)"),
                height=450, margin=dict(l=20, r=20, t=20, b=100)
            )
            st.plotly_chart(fig_decay, use_container_width=True)

        with col_a2:
            st.markdown('<div class="panel-heading" style="border:none;">📊 Recommended Slate Pricing Analysis</div>', unsafe_allow_html=True)
            
            # Attempt to safely extract numerical pricing
            try:
                prices = recs['Selling Price'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float).fillna(0).tolist()
            except:
                prices = [0]*len(recs)

            fig_prices = px.bar(
                x=short_titles, y=prices,
                color=prices, color_continuous_scale='Teal',
                labels={'x': '', 'y': 'Selling Price'}
            )
            
            fig_prices.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(24,24,27,0.5)",
                font=dict(family="Inter", color="#f8fafc"),
                xaxis=dict(tickangle=45, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                showlegend=False, coloraxis_showscale=False,
                height=450, margin=dict(l=20, r=20, t=20, b=100)
            )
            st.plotly_chart(fig_prices, use_container_width=True)

# =========================================================================================
# TAB 3 - NLP ARCHITECTURE (TF-IDF EXPLAINED FOR E-COMMERCE)
# =========================================================================================
with tab3:
    st.markdown('<div class="panel-heading" style="border:none;">🧠 Natural Language Processing Pipeline</div>', unsafe_allow_html=True)
    
    st.info("💡 **Data Science Insight:** This engine uses a combined text feature (`Name`, `Brand`, `Category`, `Details`) to perform **Content-Based Filtering**. It mathematically analyzes product descriptions to find semantic overlap, meaning it recommends products that are objectively similar in design, spec, or category, rather than relying purely on user purchase history.")
    
    col_i1, col_i2 = st.columns(2)
    
    insights = [
        ("🧮 Term Frequency (TF)", "Measures how frequently a word appears in a specific product's combined text. If the word 'Bluetooth' appears 5 times in a mouse's details, its TF score increases, signaling its importance to that specific item."),
        ("📉 Inverse Document Frequency (IDF)", "Words like 'product', 'quality', or 'new' appear everywhere in an e-commerce catalog and provide no unique value. IDF penalizes these common marketing words, assigning massive mathematical weight to rare, identifying specs (like 'Snapdragon' or 'Gore-Tex')."),
        ("🌌 Vector Space Conversion", "The TF-IDF vectorizer converts every single product into a high-dimensional mathematical vector. If there are 20,000 unique descriptive words in your catalog, every product is plotted as a specific coordinate in a 20,000-dimensional graph."),
        ("📐 Cosine Similarity", "To find recommendations, we calculate the Cosine Angle between two product vectors. An angle of 0 degrees = 1.0 (100% identical text). The products with the smallest mathematical angles to your search query are returned as recommendations.")
    ]
    
    for i, (title, desc) in enumerate(insights):
        target = col_i1 if i % 2 == 0 else col_i2
        with target:
            st.markdown(
f"""<div class="glass-panel" style="padding:30px;">
<h4 style="color:var(--cyan-core); margin-bottom:15px; font-family:'Syncopate'; font-size:18px; letter-spacing:1px;">{title}</h4>
<p style="color:var(--slate-light); font-size:14px; line-height:1.8;">{desc}</p>
</div>""", unsafe_allow_html=True)

# =========================================================================================
# TAB 4 - CONVERSION FORECASTING (SIMULATED PROBABILITY)
# =========================================================================================
with tab4:
    if st.session_state["recommendations"] is None:
        st.markdown(
"""<div style='text-align:center; padding:150px 20px; font-family:"Syncopate",serif; font-size:20px; letter-spacing:4px; color:rgba(6,182,212,0.4); text-transform:uppercase;'>
⚠️ Execute Discovery Search To Access Forecaster
</div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="panel-heading" style="border:none;">📈 Simulated Cart Addition Probability based on Slate</div>', unsafe_allow_html=True)
        
        recs = st.session_state["recommendations"]
        raw_titles = recs.head(3).get('Name', [f"Rec {i}" for i in range(1, 4)]).astype(str).tolist()
        top_3 = [t[:25]+"..." if len(t)>25 else t for t in raw_titles]
        
        # Simulate interaction over time (seconds viewing page) vs Probability to Add to Cart
        seconds = np.arange(0, 120, 10)
        
        fig_prob = go.Figure()
        
        colors = ['#06b6d4', '#f59e0b', '#8b5cf6']
        for i, title in enumerate(top_3):
            # Simulate a logistic growth curve for purchase intent
            # Higher similarity = faster growth to a higher plateau
            sim_score = float(recs.iloc[i].get('Similarity_Score', 0.5))
            plateau = min(95.0, sim_score * 120) 
            growth_rate = 0.05 + (i * 0.01)
            
            prob = plateau / (1 + np.exp(-growth_rate * (seconds - 40)))
            
            fig_prob.add_trace(go.Scatter(
                x=seconds, y=prob, mode='lines+markers', 
                line=dict(color=colors[i], width=3), name=title
            ))
            
        fig_prob.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(24,24,27,0.5)",
            font=dict(family="Inter", color="#f8fafc"),
            xaxis=dict(title="Seconds Viewing Product Details", gridcolor="rgba(255,255,255,0.05)", dtick=20),
            yaxis=dict(title="Simulated 'Add to Cart' Probability (%)", range=[0,105], gridcolor="rgba(255,255,255,0.05)"),
            hovermode="x unified",
            height=500, margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_prob, use_container_width=True)

# =========================================================================================
# TAB 5 - BEHAVIORAL VARIANCE (MONTE CARLO SIMULATION)
# =========================================================================================
with tab5:
    if st.session_state["recommendations"] is None:
        st.markdown(
"""<div style='text-align:center; padding:150px 20px; font-family:"Syncopate",serif; font-size:20px; letter-spacing:4px; color:rgba(6,182,212,0.4); text-transform:uppercase;'>
⚠️ Execute Discovery Search To Access Variance Systems
</div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="panel-heading" style="border:none;">🎲 Click-Through Rate (CTR) Volatility</div>', unsafe_allow_html=True)
        
        st.info("Simulating 1,000 hypothetical customer interactions. Even highly similar recommendations face behavioral variance (users ignoring the suggestion due to price, brand loyalty, or browsing fatigue). This Monte Carlo simulation maps the probable Click-Through Rate (CTR) for your #1 recommendation.")
        
        top_score = float(st.session_state["recommendations"].iloc[0].get('Similarity_Score', 0.5))
        base_ctr = top_score * 100 
        np.random.seed(42)
        
        error_variance = 12.5 
        simulated_cohort = np.random.normal(base_ctr, error_variance, 1000)
        simulated_cohort = np.clip(simulated_cohort, 0, 100) 
        
        fig_mc = go.Figure()
        
        fig_mc.add_trace(go.Histogram(
            x=simulated_cohort,
            nbinsx=40,
            marker_color='rgba(245, 158, 11, 0.7)',
            marker_line_color='rgba(245, 158, 11, 1.0)',
            marker_line_width=2,
            opacity=0.8
        ))
        
        fig_mc.add_vline(
            x=base_ctr, line=dict(color="#f8fafc", width=3, dash="dash"),
            annotation_text=f"Expected Base CTR: {base_ctr:.1f}%", annotation_font_color="#f8fafc"
        )
        
        fig_mc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(24,24,27,0.5)",
            font=dict(family="Inter", color="#f8fafc"),
            xaxis=dict(title="Simulated Click-Through Rate (%)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Frequency (Out of 1,000 Customers)", gridcolor="rgba(255,255,255,0.05)"),
            height=500, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_mc, use_container_width=True)

# =========================================================================================
# TAB 6 - EXPORT DOSSIER
# =========================================================================================
with tab6:
    if st.session_state["recommendations"] is None:
        st.markdown(
"""<div style='text-align:center; padding:150px 20px; font-family:"Syncopate",serif; font-size:20px; letter-spacing:4px; color:rgba(6,182,212,0.4); text-transform:uppercase;'>
⚠️ Execute Discovery Search To Generate Official Dossier
</div>""",
            unsafe_allow_html=True,
        )
    else:
        matched = st.session_state["matched_product"]
        ts = st.session_state["timestamp"]
        sess_id = st.session_state["session_id"]
        recs = st.session_state["recommendations"]
        
        st.markdown(
f"""<div class="glass-panel" style="background:rgba(6, 182, 212, 0.05); border-color:rgba(6, 182, 212, 0.3); padding:60px;">
<div style="font-family:'Space Mono'; font-size:14px; color:var(--cyan-core); margin-bottom:15px; letter-spacing:3px;">✅ RETAIL SLATE GENERATED: {ts}</div>
<div style="font-family:'Syncopate'; font-weight:700; font-size:35px; color:white; margin-bottom:10px; letter-spacing:1px;">SOURCE: {matched}</div>
<div style="font-family:'Inter'; font-size:16px; color:var(--slate-light);">Vector Transaction ID: <span style="color:var(--cyan-core); font-family:'Space Mono';">{sess_id}</span></div>
</div>""", unsafe_allow_html=True
        )

        st.markdown('<div class="panel-heading" style="border:none; margin-top:50px;">💾 Export Recommendation Artifacts</div>', unsafe_allow_html=True)
        
        col_exp1, col_exp2 = st.columns(2)
        
        export_df = recs.copy()
        for col in export_df.columns:
            if export_df[col].dtype == 'object':
                export_df[col] = export_df[col].astype(str)
                
        dict_records = export_df.to_dict(orient='records')
        
        json_payload = {
            "metadata": {
                "transaction_id": sess_id,
                "timestamp": ts,
                "source_query": st.session_state["user_input"],
                "resolved_match": matched,
                "levenshtein_confidence": st.session_state["match_confidence"]
            },
            "recommendations": dict_records
        }
        json_str = json.dumps(json_payload, indent=4)
        b64_json = base64.b64encode(json_str.encode()).decode()
        
        csv_data = export_df.to_csv(index=False)
        b64_csv = base64.b64encode(csv_data.encode()).decode()
        
        with col_exp1:
            href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="RetailNexus_Recommendations_{sess_id}.csv" style="display:block; text-align:center; padding:25px; background:rgba(6, 182, 212, 0.1); border:1px solid var(--cyan-core); color:var(--cyan-core); text-decoration:none; font-family:\'Space Mono\'; font-weight:700; font-size:16px; border-radius:8px; letter-spacing:2px; transition:all 0.3s ease;">⬇️ DOWNLOAD CSV LEDGER</a>'
            st.markdown(href_csv, unsafe_allow_html=True)
            
        with col_exp2:
            href_json = f'<a href="data:application/json;base64,{b64_json}" download="RetailNexus_Payload_{sess_id}.json" style="display:block; text-align:center; padding:25px; background:rgba(245, 158, 11, 0.1); border:1px solid var(--gold-core); color:var(--gold-core); text-decoration:none; font-family:\'Space Mono\'; font-weight:700; font-size:16px; border-radius:8px; letter-spacing:2px; transition:all 0.3s ease;">⬇️ DOWNLOAD JSON PAYLOAD</a>'
            st.markdown(href_json, unsafe_allow_html=True)

        st.markdown('<div class="panel-heading" style="border:none; margin-top:70px;">💻 Raw Transmission Payload</div>', unsafe_allow_html=True)
        st.json(json_payload)

# =========================================================================================
# 8. GLOBAL FOOTER
# =========================================================================================
st.markdown(
"""<div style="text-align:center; padding:70px; margin-top:100px; border-top:1px solid rgba(6,182,212,0.15); font-family:'Space Mono'; font-size:11px; color:rgba(148,163,184,0.3); letter-spacing:4px; text-transform:uppercase;">
&copy; 2026 | RetailNexus Omni-Channel Terminal v11.2<br>
<span style="color:rgba(6,182,212,0.5); font-size:10px; display:block; margin-top:10px;">Strictly Confidential E-Commerce Data | Powered by TF-IDF & Cosine Similarity Architecture</span>
</div>""",
    unsafe_allow_html=True,
)