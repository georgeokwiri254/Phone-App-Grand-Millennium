"""
Grand Millennium Revenue Analytics - Streamlit Dashboard
Clean implementation with default collapsible sidebar
"""

# Core imports
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import base64
from pathlib import Path
from datetime import datetime, timedelta
import io
import traceback

# Statistical and ML imports
try:
    import sqlite3
    import numpy as np
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    import warnings
    warnings.filterwarnings('ignore')
except ImportError as e:
    st.error(f"Missing required packages: {e}")

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'converters'))

# Local imports
try:
    from app.database import get_database
    from app.logging_config import get_conversion_logger, get_app_logger, get_log_content
    from app.advanced_forecasting import get_advanced_forecaster
    from converters.segment_converter import run_segment_conversion
    from converters.occupancy_converter import run_occupancy_conversion
except ImportError as e:
    st.error(f"Error importing local modules: {e}")

# Initialize loggers
try:
    conversion_logger = get_conversion_logger()
    app_logger = get_app_logger()
except:
    conversion_logger = None
    app_logger = None

# Configure Streamlit page
st.set_page_config(
    page_title="Grand Millennium Revenue Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'segment_data' not in st.session_state:
    st.session_state.segment_data = None
if 'occupancy_data' not in st.session_state:
    st.session_state.occupancy_data = None
if 'last_run_timestamp' not in st.session_state:
    st.session_state.last_run_timestamp = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Dashboard"

# Show loading requirements if data not loaded
def show_loading_requirements():
    """Show requirements to load data first"""
    st.warning("⚠️ No data loaded yet!")
    st.info("👆 Please go to the **Dashboard** tab first to upload your Excel file and run the converters.")
    if st.button("🔄 Go to Dashboard Tab", type="primary"):
        st.session_state.current_tab = "Dashboard"
        st.rerun()

# PLACEHOLDER FUNCTIONS - These will be populated with the actual implementations
def show_dashboard_kpis():
    """Show daily pickup KPI cards with conditional formatting"""
    st.subheader("📊 Daily Pickup KPI Cards")
    
    try:
        # Get segment data
        if hasattr(st.session_state, 'segment_data') and st.session_state.segment_data is not None:
            df = st.session_state.segment_data.copy()
        else:
            db = get_database()
            df = db.get_segment_data()
        
        if df.empty:
            st.error("No segment data available for KPIs")
            return
        
        # Get current month and next 2 months
        current_date = datetime.now()
        months_to_show = []
        
        for i in range(3):  # Current + next 2 months
            if current_date.month + i > 12:
                month_date = datetime(current_date.year + 1, (current_date.month + i) % 12, 1)
            else:
                month_date = datetime(current_date.year, current_date.month + i, 1)
            months_to_show.append(month_date)
        
        # Create 3 columns for the KPI cards
        col1, col2, col3 = st.columns(3)
        
        for i, month_date in enumerate(months_to_show):
            month_str = month_date.strftime('%Y-%m-01')
            month_name = month_date.strftime('%B %Y')
            
            # Get month data
            month_data = df[df['Month'] == month_str]
            
            if not month_data.empty:
                daily_pickup = month_data['Daily_Pick_up_Revenue'].sum()
                
                # Determine color based on positive/negative
                if daily_pickup >= 0:
                    delta_color = "normal"  # Green
                    emoji = "✅"
                else:
                    delta_color = "inverse"  # Red
                    emoji = "❌"
                
                # Show in appropriate column
                with [col1, col2, col3][i]:
                    st.metric(
                        label=f"{emoji} {month_name}",
                        value=f"AED {daily_pickup:,.0f}",
                        delta=f"Daily Pickup",
                        delta_color=delta_color
                    )
            else:
                with [col1, col2, col3][i]:
                    st.metric(
                        label=f"❓ {month_name}",
                        value="No Data",
                        delta="Daily Pickup"
                    )
                    
    except Exception as e:
        st.error(f"Error displaying KPIs: {str(e)}")

def show_bob_vs_budget_table():
    """Show Business on Books vs Budget table with variance conditional formatting"""
    st.subheader("💰 Business on Books vs Budget Analysis")
    
    try:
        # Get segment data
        if hasattr(st.session_state, 'segment_data') and st.session_state.segment_data is not None:
            df = st.session_state.segment_data.copy()
        else:
            db = get_database()
            df = db.get_segment_data()
        
        if df.empty:
            st.error("No segment data available for BOB vs Budget analysis")
            return
        
        # Create table data
        table_data = []
        
        for month_str in sorted(df['Month'].unique()):
            month_data = df[df['Month'] == month_str]
            month_date = pd.to_datetime(month_str)
            month_name = month_date.strftime('%B %Y')
            
            bob_revenue = month_data['Business_on_the_Books_Revenue'].sum()
            budget_revenue = month_data['Budget_This_Year_Revenue'].sum()
            
            if budget_revenue > 0:
                variance = bob_revenue - budget_revenue
                variance_pct = (variance / budget_revenue) * 100
            else:
                variance = 0
                variance_pct = 0
            
            table_data.append({
                'Month': month_name,
                'BOB Revenue': bob_revenue,
                'Budget': budget_revenue,
                'Variance': variance,
                'Variance %': variance_pct
            })
        
        # Create DataFrame
        table_df = pd.DataFrame(table_data)
        
        if not table_df.empty:
            # Format the table for display (keep variance % as numeric for conditional formatting)
            display_df = table_df.copy()
            
            # Apply conditional formatting to variance percentage BEFORE formatting as text
            def highlight_variance(val):
                if isinstance(val, (int, float)):
                    if val >= 0:
                        return 'color: green; font-weight: bold'
                    else:
                        return 'color: red; font-weight: bold'
                return ''
            
            # Format the dataframe with conditional formatting
            formatted_df = display_df.style.format({
                'BOB Revenue': 'AED {:,.0f}',
                'Budget': 'AED {:,.0f}',
                'Variance': 'AED {:+,.0f}',
                'Variance %': '{:+.1f}%'
            }).applymap(highlight_variance, subset=['Variance %'])
            
            st.dataframe(formatted_df, use_container_width=True)
        else:
            st.error("No data available for BOB vs Budget table")
            
    except Exception as e:
        st.error(f"Error displaying BOB vs Budget table: {str(e)}")

def dashboard_tab():
    """Dashboard tab - data processing and key metrics"""
    st.header("📊 Dashboard")
    
    # File upload section
    st.subheader("1. Upload Excel File")
    uploaded_file = st.file_uploader(
        "Choose MHR Excel file",
        type=['xlsm', 'xlsx'],
        help="Upload an MHR Pick Up Report with DPR sheet"
    )
    
    # Alternative: Use existing files
    st.subheader("2. Or Select Existing File")
    
    # Look for existing Excel files in project root
    excel_files = []
    for ext in ['*.xlsm', '*.xlsx']:
        excel_files.extend(project_root.glob(ext))
    
    if excel_files:
        file_options = [""] + [f.name for f in excel_files]
        selected_file = st.selectbox(
            "Select from existing files:",
            file_options,
            help="Choose from Excel files found in the project directory"
        )
    else:
        selected_file = ""
        st.info("No Excel files found in project directory")
    
    # Determine which file to process
    file_to_process = None
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = project_root / f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_to_process = temp_path
        st.success(f"✅ Uploaded: {uploaded_file.name}")
    elif selected_file:
        file_to_process = project_root / selected_file
        st.success(f"✅ Selected: {selected_file}")
    
    # Auto-processing section
    st.subheader("3. Data Processing")
    
    if file_to_process:
        st.info(f"Processing: {file_to_process.name}")
        
        # Auto-trigger processing
        success = run_conversion_process(file_to_process)
        
        # Clean up temp file if it was uploaded
        if uploaded_file is not None and temp_path.exists():
            temp_path.unlink()
            
        if success:
            st.balloons()
    else:
        st.warning("Please upload or select an Excel file first")
    
    # Show current status
    st.subheader("4. Current Status")
    show_data_status()
    
    # Try to load existing data if not already loaded
    if not st.session_state.data_loaded:
        try:
            segment_file = project_root / "data" / "processed" / "segment.csv"
            occupancy_file = project_root / "data" / "processed" / "occupancy.csv"
            
            if segment_file.exists() and occupancy_file.exists():
                st.session_state.segment_data = pd.read_csv(segment_file)
                st.session_state.occupancy_data = pd.read_csv(occupancy_file)
                st.session_state.data_loaded = True
                st.session_state.last_run_timestamp = datetime.now()
                st.success("✅ Existing data loaded automatically!")
                st.rerun()  # Force a rerun to update all components
        except Exception as e:
            st.info("📁 No existing data found. Upload an Excel file to get started.")
    
    # Always show KPI cards and tables (they handle no-data cases internally)
    show_dashboard_kpis()
    show_bob_vs_budget_table()

def run_conversion_process(file_path):
    """
    Run the complete conversion process
    
    Args:
        file_path: Path to Excel file to process
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Use a simpler approach - just show status without complex UI updates
    status_placeholder = st.empty()
    
    try:
        if app_logger:
            app_logger.info(f"Starting conversion process for: {file_path}")
        status_placeholder.info("🔄 Starting conversion process...")
        
        # Step 1: Run segment conversion
        if conversion_logger:
            conversion_logger.info("=== STARTING SEGMENT CONVERSION ===")
        segment_df, segment_path = run_segment_conversion(str(file_path))
        if conversion_logger:
            conversion_logger.info(f"Segment conversion completed: {segment_path}")
        
        status_placeholder.info("🔄 Segment conversion completed, running occupancy conversion...")
        
        # Step 2: Run occupancy conversion
        status_placeholder.info("🔄 Running occupancy conversion...")
        
        if conversion_logger:
            conversion_logger.info("=== STARTING OCCUPANCY CONVERSION ===")
        occupancy_df, occupancy_path = run_occupancy_conversion(str(file_path))
        if conversion_logger:
            conversion_logger.info(f"Occupancy conversion completed: {occupancy_path}")
        
        # Step 3: Ingest to database
        status_placeholder.info("🔄 Ingesting data to database...")
        
        db = get_database()
        
        # Ingest segment data
        if db.ingest_segment_data(segment_df):
            if conversion_logger:
                conversion_logger.info("Segment data ingested to database")
        else:
            raise Exception("Failed to ingest segment data")
        
        # Ingest occupancy data
        if db.ingest_occupancy_data(occupancy_df):
            if conversion_logger:
                conversion_logger.info("Occupancy data ingested to database")
        else:
            raise Exception("Failed to ingest occupancy data")
        
        # Step 4: Cache data in session state
        status_placeholder.info("🔄 Caching data...")
        
        st.session_state.segment_data = segment_df
        st.session_state.occupancy_data = occupancy_df
        st.session_state.data_loaded = True
        st.session_state.last_run_timestamp = datetime.now()
        
        status_placeholder.success("✅ Conversion completed successfully!")
        
        if conversion_logger:
            conversion_logger.info("=== CONVERSION PROCESS COMPLETED SUCCESSFULLY ===")
        if app_logger:
            app_logger.info("Data loaded and cached in session state")
        
        # Show summary
        show_conversion_summary(segment_df, occupancy_df, segment_path, occupancy_path)
        
        return True
        
    except Exception as e:
        error_msg = f"Conversion failed: {str(e)}"
        status_placeholder.error(f"❌ {error_msg}")
        if conversion_logger:
            conversion_logger.error(error_msg)
            conversion_logger.error(traceback.format_exc())
        
        # Load existing data from database instead of failing completely
        st.warning("⚠️ Processing failed. Loading existing data from database...")
        try:
            # Try to load from database
            db = get_database()
            segment_data = db.get_segment_data()
            occupancy_data = db.get_occupancy_data()
            
            if not segment_data.empty and not occupancy_data.empty:
                st.session_state.segment_data = segment_data
                st.session_state.occupancy_data = occupancy_data
                st.session_state.data_loaded = True
                st.session_state.last_run_timestamp = datetime.now()
                st.info("ℹ️ Existing data loaded successfully from database.")
                return True
            else:
                st.error("No existing data found in database.")
                return False
        except Exception as load_error:
            st.error(f"Failed to load existing data: {str(load_error)}")
            return False

def show_conversion_summary(segment_df, occupancy_df, segment_path, occupancy_path):
    """Show summary of successful conversion"""
    st.success("🎉 Conversion completed successfully!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Segment Data", f"{len(segment_df)} rows")
        st.metric("Segments", f"{segment_df['MergedSegment'].nunique()}" if 'MergedSegment' in segment_df.columns else "N/A")
        
    with col2:
        st.metric("Occupancy Data", f"{len(occupancy_df)} rows")
        st.metric("Date Range", f"{occupancy_df['Date'].min()} to {occupancy_df['Date'].max()}" if 'Date' in occupancy_df.columns else "N/A")
    
    # Show file paths
    st.info(f"📄 Segment CSV: {segment_path}")
    st.info(f"📄 Occupancy CSV: {occupancy_path}")
    
    # Show data previews
    st.subheader("📊 Data Preview")
    
    tab1, tab2 = st.tabs(["Segment Data (Top 5)", "Occupancy Data (Top 5)"])
    
    with tab1:
        st.dataframe(segment_df.head())
    
    with tab2:
        st.dataframe(occupancy_df.head())

def show_data_status():
    """Show current data loading status"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.data_loaded:
            st.success("✅ Data Loaded")
        else:
            st.error("❌ No Data")
    
    with col2:
        if st.session_state.last_run_timestamp:
            st.info(f"🕒 Last Run: {st.session_state.last_run_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("🕒 Never Run")
    
    with col3:
        try:
            db_stats = get_database().get_database_stats()
            if db_stats:
                st.info(f"📊 DB: {db_stats.get('segment_analysis_rows', 0)} seg, {db_stats.get('occupancy_analysis_rows', 0)} occ")
        except:
            st.info("📊 DB: Not available")

def loading_tab():
    """Loading tab for file upload and data processing"""
    st.header("📁 Data Loading & Processing")
    st.info("Please use the Dashboard tab to upload and process data files.")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Go to Dashboard Tab", use_container_width=True):
            st.session_state.current_tab = "Dashboard"
            st.rerun()
    
    # Show current status
    st.subheader("Current Status")
    show_data_status()

def daily_occupancy_tab():
    """Daily Occupancy Analysis Tab"""
    st.header("📈 Daily Occupancy Analysis")
    
    if not st.session_state.data_loaded:
        show_loading_requirements()
        return
    
    # Get data from session state or database
    if st.session_state.occupancy_data is not None:
        df = st.session_state.occupancy_data.copy()
    else:
        db = get_database()
        df = db.get_occupancy_data()
    
    if df.empty:
        st.error("No occupancy data available")
        return
    
    # Ensure Date column is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Date range filter
    st.subheader("📅 Date Range Filter")
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", df['Date'].min().date() if 'Date' in df.columns else datetime.now().date())
    with col2:
        end_date = st.date_input("End Date", df['Date'].max().date() if 'Date' in df.columns else datetime.now().date())
    
    # Filter data
    if 'Date' in df.columns:
        mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
        filtered_df = df[mask]
    else:
        filtered_df = df
    
    # Main metrics
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_occ = filtered_df['Occ%'].mean() if 'Occ%' in filtered_df.columns else 0
        st.metric("Average Occupancy", f"{avg_occ:.1f}%")
    
    with col2:
        total_revenue = filtered_df['Revenue'].sum() if 'Revenue' in filtered_df.columns else 0
        st.metric("Total Revenue (AED)", f"{total_revenue:,.0f}")
    
    with col3:
        avg_adr = filtered_df['ADR'].mean() if 'ADR' in filtered_df.columns else 0
        st.metric("Average ADR (AED)", f"{avg_adr:.0f}")
    
    with col4:
        avg_revpar = filtered_df['RevPar'].mean() if 'RevPar' in filtered_df.columns else 0
        st.metric("Average RevPAR (AED)", f"{avg_revpar:.0f}")
    
    # Occupancy trend chart
    st.subheader("📈 Daily Occupancy & ADR Analysis")
    
    if 'Date' in filtered_df.columns and 'Occ%' in filtered_df.columns and 'ADR' in filtered_df.columns:
        # Create dual-axis chart
        fig = go.Figure()
        
        # Add occupancy as bar chart
        fig.add_trace(go.Bar(
            x=filtered_df['Date'],
            y=filtered_df['Occ%'],
            name='Daily Occupancy %',
            marker_color='lightblue',
            opacity=0.7,
            yaxis='y1'
        ))
        
        # Add ADR as line chart on secondary y-axis
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['ADR'],
            mode='lines+markers',
            name='ADR Trend',
            line=dict(color='red', width=3),
            marker=dict(size=4),
            yaxis='y2'
        ))
        
        # Update layout for dual axis
        fig.update_layout(
            title="Daily Occupancy (Bars) & ADR Trend (Line) - Full Year View",
            xaxis_title="Date",
            yaxis=dict(
                title="Occupancy %",
                side="left",
                range=[0, 100]  # Fixed range 0-100% for occupancy
            ),
            yaxis2=dict(
                title="ADR (AED)",
                side="right",
                overlaying="y",
                range=[0, max(filtered_df['ADR'].max() * 1.1, 1000)]  # Dynamic range for ADR
            ),
            hovermode='x unified',
            height=600,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.8)"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("📋 Occupancy Data Table")
    
    # Format display columns
    if not filtered_df.empty:
        display_columns = ['Date', 'DOW', 'Rms', 'Rm Sold', 'Revenue', 'ADR', 'Occ%']
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        
        if available_columns:
            formatted_df = filtered_df[available_columns].copy()
            
            # Format monetary columns
            for col in ['Revenue', 'ADR']:
                if col in formatted_df.columns:
                    formatted_df[col] = formatted_df[col].apply(lambda x: f"AED {x:,.0f}" if pd.notna(x) else "AED 0")
            
            # Format percentage
            if 'Occ%' in formatted_df.columns:
                formatted_df['Occ%'] = formatted_df['Occ%'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "0.0%")
            
            st.dataframe(formatted_df, use_container_width=True, height=400)
        else:
            st.dataframe(filtered_df, use_container_width=True, height=400)
    
    st.info(f"Showing {len(filtered_df)} rows")

def segment_analysis_tab():
    """Segment Analysis Tab"""
    st.header("🎯 Segment Analysis")
    
    if not st.session_state.data_loaded:
        show_loading_requirements()
        return
    
    # Get data from session state or database
    if st.session_state.segment_data is not None:
        df = st.session_state.segment_data.copy()
    else:
        db = get_database()
        df = db.get_segment_data()
    
    if df.empty:
        st.error("No segment data available")
        return
    
    # Ensure Month column is datetime
    if 'Month' in df.columns:
        df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
        df = df.dropna(subset=['Month'])  # Remove rows with invalid dates
    
    # Controls
    st.subheader("📊 Analysis Controls")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Aggregation selector
        aggregation = st.selectbox(
            "Aggregation Level:",
            ["Monthly", "Weekly", "Daily"],
            index=0
        )
    
    with col2:
        # Segment type selector
        use_merged = st.checkbox(
            "Use Merged Segments",
            value=True,
            help="Use grouped segments (Retail, Corporate, etc.) vs original segments"
        )
    
    with col3:
        # Date range
        if 'Month' in df.columns and len(df) > 0:
            min_date = df['Month'].min().date()
            max_date = df['Month'].max().date()
            date_range = st.date_input(
                "Date Range:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            if len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date, end_date = min_date, max_date
        else:
            start_date = end_date = datetime.now().date()
    
    # Filter data by date range
    if 'Month' in df.columns:
        mask = (df['Month'].dt.date >= start_date) & (df['Month'].dt.date <= end_date)
        filtered_df = df[mask]
    else:
        filtered_df = df
    
    # Choose segment column
    segment_col = 'MergedSegment' if use_merged and 'MergedSegment' in filtered_df.columns else 'Segment'
    
    if segment_col not in filtered_df.columns:
        st.error(f"Segment column '{segment_col}' not found in data")
        return
    
    # Key metrics
    st.subheader("📈 Key Performance Indicators")
    
    if 'Business_on_the_Books_Revenue' in filtered_df.columns:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_revenue = filtered_df['Business_on_the_Books_Revenue'].sum()
            st.metric("Total BOB Revenue (AED)", f"{total_revenue:,.0f}")
        
        with col2:
            avg_revenue = filtered_df['Business_on_the_Books_Revenue'].mean()
            st.metric("Average BOB Revenue (AED)", f"{avg_revenue:,.0f}")
        
        with col3:
            segments_count = filtered_df[segment_col].nunique()
            st.metric("Active Segments", segments_count)
        
        with col4:
            if 'Business_on_the_Books_Rooms' in filtered_df.columns:
                total_rooms = filtered_df['Business_on_the_Books_Rooms'].sum()
                st.metric("Total BOB Rooms", f"{total_rooms:,.0f}")
    
    # Top 5 segments by revenue
    st.subheader("🏆 Top 5 Segments by Revenue")
    
    if 'Business_on_the_Books_Revenue' in filtered_df.columns:
        top_segments = filtered_df.groupby(segment_col)['Business_on_the_Books_Revenue'].sum().sort_values(ascending=False).head(5)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar chart
            fig_bar = px.bar(
                x=top_segments.values,
                y=top_segments.index,
                orientation='h',
                title="Top 5 Segments by Business on the Books Revenue",
                labels={'x': 'Revenue (AED)', 'y': 'Segment'}
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Table
            top_df = pd.DataFrame({
                'Segment': top_segments.index,
                'Revenue (AED)': [f"{x:,.0f}" for x in top_segments.values],
                'Share': [f"{(x/top_segments.sum())*100:.1f}%" for x in top_segments.values]
            })
            st.dataframe(top_df, hide_index=True)
    
    # Time series analysis
    st.subheader("📈 Revenue Time Series")
    
    if 'Month' in filtered_df.columns and 'Business_on_the_Books_Revenue' in filtered_df.columns:
        # Prepare time series data
        ts_data = filtered_df.groupby(['Month', segment_col])['Business_on_the_Books_Revenue'].sum().reset_index()
        
        # Create time series chart
        fig_ts = px.line(
            ts_data,
            x='Month',
            y='Business_on_the_Books_Revenue',
            color=segment_col,
            title=f"Business on the Books Revenue by Segment ({aggregation})",
            labels={'Business_on_the_Books_Revenue': 'Revenue (AED)', 'Month': 'Date'}
        )
        fig_ts.update_layout(height=500)
        st.plotly_chart(fig_ts, use_container_width=True)
    
    # Data summary table
    st.subheader("📋 Segment Data Summary")
    
    if 'Business_on_the_Books_Revenue' in filtered_df.columns:
        summary_stats = filtered_df.groupby(segment_col).agg({
            'Business_on_the_Books_Revenue': ['sum', 'mean', 'count']
        }).round(2)
        
        # Flatten column names
        summary_stats.columns = ['Total_Revenue', 'Avg_Revenue', 'Record_Count']
        summary_stats = summary_stats.reset_index()
        
        # Format monetary columns
        summary_stats['Total_Revenue'] = summary_stats['Total_Revenue'].apply(lambda x: f"AED {x:,.0f}")
        summary_stats['Avg_Revenue'] = summary_stats['Avg_Revenue'].apply(lambda x: f"AED {x:,.0f}")
        
        summary_stats = summary_stats.rename(columns={
            segment_col: 'Segment',
            'Total_Revenue': 'Total Revenue',
            'Avg_Revenue': 'Average Revenue',
            'Record_Count': 'Records'
        })
        
        st.dataframe(summary_stats, use_container_width=True)

def adr_analysis_tab():
    """ADR Analysis Tab"""
    st.header("💰 ADR Analysis")
    
    if not st.session_state.data_loaded:
        show_loading_requirements()
        return
    
    # Get occupancy data for ADR analysis
    if st.session_state.occupancy_data is not None:
        df = st.session_state.occupancy_data.copy()
    else:
        db = get_database()
        df = db.get_occupancy_data()
    
    if df.empty:
        st.error("No occupancy data available for ADR analysis")
        return
    
    # Key ADR metrics
    st.subheader("📊 ADR Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    if 'ADR' in df.columns:
        with col1:
            avg_adr = df['ADR'].mean()
            st.metric("Average ADR", f"AED {avg_adr:.0f}")
        
        with col2:
            median_adr = df['ADR'].median()
            st.metric("Median ADR", f"AED {median_adr:.0f}")
        
        with col3:
            max_adr = df['ADR'].max()
            st.metric("Peak ADR", f"AED {max_adr:.0f}")
        
        with col4:
            min_adr = df['ADR'].min()
            st.metric("Lowest ADR", f"AED {min_adr:.0f}")
        
        # ADR distribution histogram
        st.subheader("📊 ADR Distribution")
        fig_hist = px.histogram(df, x='ADR', nbins=30, title="ADR Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # ADR boxplot
        st.subheader("📦 ADR Statistical Analysis")
        fig_box = px.box(df, y='ADR', title="ADR Boxplot - Quartiles and Outliers")
        st.plotly_chart(fig_box, use_container_width=True)

def block_analysis_tab():
    """Block Analysis Tab"""
    st.header("📊 Block Analysis")
    st.info("Upload block data files to analyze booking patterns and company performance.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a Block Data file", type=['txt'])
    
    if uploaded_file:
        st.success(f"✅ Block data file uploaded: {uploaded_file.name}")
        st.info("Block data processing functionality can be implemented here.")

def block_dashboard_tab():
    """Block Dashboard Tab"""
    st.header("📊 Block Dashboard")
    st.info("KPI dashboard for block bookings with metrics on confirmed/prospect blocks and conversion rates.")

def events_analysis_tab():
    """Events Analysis Tab"""
    st.header("🎉 Events Analysis")
    st.info("Dubai Events Calendar analysis with correlation to occupancy and block bookings.")

def enhanced_forecasting_tab():
    """Enhanced Forecasting Tab"""
    st.header("🔮 Enhanced Forecasting")
    
    if not st.session_state.data_loaded:
        show_loading_requirements()
        return
    
    st.info("Advanced forecasting models with ExponentialSmoothing and machine learning.")
    
    # Simple forecast button
    if st.button("Generate Basic Forecast"):
        st.success("✅ Forecast functionality can be implemented here using statsmodels.")

def machine_learning_tab():
    """Machine Learning Tab"""
    st.header("🤖 Machine Learning")
    
    if not st.session_state.data_loaded:
        show_loading_requirements()
        return
    
    st.info("ML models for revenue prediction, demand forecasting, and pattern recognition.")

def insights_analysis_tab():
    """Insights Analysis Tab"""
    st.header("💡 Insights Analysis")
    
    if not st.session_state.data_loaded:
        show_loading_requirements()
        return
    
    st.info("Advanced analytics and business insights from revenue and occupancy data.")

def controls_logs_tab():
    """Controls and Logs Tab"""
    st.header("⚙️ Controls & Logs")
    
    st.subheader("📊 System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.data_loaded:
            st.success("✅ Data Loaded")
        else:
            st.error("❌ No Data")
    
    with col2:
        if st.session_state.last_run_timestamp:
            st.info(f"🕒 Last Run: {st.session_state.last_run_timestamp.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.info("🕒 Never Run")
    
    with col3:
        try:
            db_stats = get_database().get_database_stats()
            if db_stats:
                st.info(f"📊 DB Records Available")
        except:
            st.warning("📊 DB Connection Issue")
    
    st.subheader("🔄 Re-run Data Processing")
    if st.button("Go to Dashboard to Load Data", type="primary"):
        st.session_state.current_tab = "Dashboard"
        st.rerun()
    
    st.subheader("📝 Application Logs")
    st.info("Application logs and conversion logs can be displayed here.")

def main():
    """Main application function with clean sidebar implementation"""
    
    # Get logo for sidebar (80px as requested)
    logo_img = '<div style="width:80px;height:80px;background:#fff;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:32px;">🏨</div>'
    try:
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Pictures', 'xkrIXuKn_400x400.jpg')
        if not os.path.exists(logo_path):
            logo_path = 'Pictures/xkrIXuKn_400x400.jpg'
        with open(logo_path, 'rb') as f:
            logo_data = base64.b64encode(f.read()).decode()
            logo_img = f'<img src="data:image/jpeg;base64,{logo_data}" alt="Logo" style="width:80px;height:80px;border-radius:3px;object-fit:cover;">'
    except:
        pass
    
    # SIDEBAR with default Streamlit collapsible behavior
    with st.sidebar:
        # Logo and title in sidebar
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px; margin: 20px 0;">
            {logo_img}
            <div style="font-size: 16px; font-weight: 600; font-style: italic; color: #333; text-align: center; line-height: 1.2;">
                Grand Millennium<br>Revenue Analytics
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation menu
        st.markdown("### 📊 Navigation")
        
        # Navigation options
        navigation_options = [
            "Dashboard",
            "Daily Occupancy", 
            "Segment Analysis",
            "ADR Analysis",
            "Block Analysis",
            "Block Dashboard", 
            "Events Analysis",
            "Enhanced Forecasting",
            "Machine Learning",
            "Insights Analysis",
            "Controls & Logs"
        ]
        
        current_tab = st.session_state.current_tab
        
        # Create radio buttons for navigation
        selected_tab = st.radio(
            "Choose a section:",
            navigation_options,
            index=navigation_options.index(current_tab) if current_tab in navigation_options else 0,
            key="sidebar_navigation"
        )
        
        # Update session state if selection changed
        if selected_tab != current_tab:
            st.session_state.current_tab = selected_tab
            st.rerun()
        
        st.markdown("---")
        
        # Data status in sidebar
        if st.session_state.data_loaded:
            st.success("✅ Data Loaded")
            if st.session_state.last_run_timestamp:
                st.caption(f"Last updated: {st.session_state.last_run_timestamp.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.warning("⚠️ No Data")
            st.caption("Upload data in Loading tab")
    
    # Main content area with header
    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 10px;">
            {logo_img}
            <h1 style="color: #333; font-style: italic; margin: 0;">Grand Millennium Revenue Analytics</h1>
        </div>
        <p style="color: #666; font-size: 16px;">Current Section: <strong>{st.session_state.current_tab}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add hotel exterior image after title
    try:
        hotel_image_path = project_root / "Pictures" / "Grand Millennium Dubai Hotel Exterior 2.jpg"
        if hotel_image_path.exists():
            st.image(str(hotel_image_path), caption="Grand Millennium Dubai", use_container_width=True)
    except:
        pass
    
    # Route to appropriate tab
    current_tab = st.session_state.current_tab
    if current_tab == "Dashboard":
        dashboard_tab()
    elif current_tab == "Daily Occupancy":
        daily_occupancy_tab()
    elif current_tab == "Segment Analysis":
        segment_analysis_tab()
    elif current_tab == "ADR Analysis":
        adr_analysis_tab()
    elif current_tab == "Block Analysis":
        block_analysis_tab()
    elif current_tab == "Block Dashboard":
        block_dashboard_tab()
    elif current_tab == "Events Analysis":
        events_analysis_tab()
    elif current_tab == "Enhanced Forecasting":
        enhanced_forecasting_tab()
    elif current_tab == "Machine Learning":
        machine_learning_tab()
    elif current_tab == "Insights Analysis":
        insights_analysis_tab()
    elif current_tab == "Controls & Logs":
        controls_logs_tab()

if __name__ == "__main__":
    main()