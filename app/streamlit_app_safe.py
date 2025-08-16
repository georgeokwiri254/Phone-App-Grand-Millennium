"""
Grand Millennium Revenue Analytics - Safe Version
Handles missing dependencies gracefully
"""

# Core imports that should always work
import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Setup basic page config first
st.set_page_config(
    page_title="Grand Millennium Revenue Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Show basic title
st.title("🏨 Grand Millennium Revenue Analytics")

try:
    # Try plotly imports
    import plotly.express as px
    import plotly.graph_objects as go
    plotly_available = True
except ImportError:
    st.warning("Plotly not available - charts will be limited")
    plotly_available = False

try:
    # Try statistical imports
    import numpy as np
    import sqlite3
    stats_available = True
except ImportError:
    st.warning("Statistical packages not available - some features will be limited")
    stats_available = False

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Try local imports
database_available = False
converters_available = False

try:
    from app.database import get_database
    database_available = True
except ImportError:
    st.warning("Database module not available")

try:
    from converters.segment_converter import run_segment_conversion
    from converters.occupancy_converter import run_occupancy_conversion
    converters_available = True
except ImportError:
    st.warning("Converter modules not available")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Dashboard"

# Simple sidebar
with st.sidebar:
    st.markdown("### 📊 Navigation")
    
    tabs = ["Dashboard", "Daily Occupancy", "Segment Analysis", "ADR Analysis"]
    selected_tab = st.radio("Choose a section:", tabs)
    
    if selected_tab != st.session_state.current_tab:
        st.session_state.current_tab = selected_tab
        st.rerun()

# Main content
st.markdown(f"**Current Section:** {st.session_state.current_tab}")

if st.session_state.current_tab == "Dashboard":
    st.header("📊 Dashboard")
    
    st.subheader("System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Plotly Charts", "✅ Available" if plotly_available else "❌ Missing")
    with col2:
        st.metric("Statistics", "✅ Available" if stats_available else "❌ Missing")
    with col3:
        st.metric("Database", "✅ Available" if database_available else "❌ Missing")
    
    if converters_available:
        st.success("✅ All converters are available")
        
        # File upload
        uploaded_file = st.file_uploader("Choose Excel file", type=['xlsm', 'xlsx'])
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
    else:
        st.error("❌ Converters not available")

elif st.session_state.current_tab == "Daily Occupancy":
    st.header("📈 Daily Occupancy Analysis")
    if not st.session_state.data_loaded:
        st.warning("Please load data in Dashboard first")
    else:
        st.info("Daily occupancy analysis would appear here")

elif st.session_state.current_tab == "Segment Analysis":
    st.header("🎯 Segment Analysis")
    if not st.session_state.data_loaded:
        st.warning("Please load data in Dashboard first")
    else:
        st.info("Segment analysis would appear here")

elif st.session_state.current_tab == "ADR Analysis":
    st.header("💰 ADR Analysis")
    if not st.session_state.data_loaded:
        st.warning("Please load data in Dashboard first")
    else:
        st.info("ADR analysis would appear here")

st.markdown("---")
st.info("This is a safe version to test basic functionality. If this works, we can identify what's causing the crash in the full version.")