"""
Grand Millennium Revenue Analytics Dashboard
Full-featured Streamlit application with default sidebar
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import os
import sys
import base64
import traceback
from pathlib import Path
from datetime import datetime, timedelta
import io
import tempfile
import pandas as pd

# Custom imports
try:
    from app.database import get_database
    database_available = True
except ImportError:
    database_available = False
    print("Database module not available.")

try:
    from app.logging_config import get_conversion_logger, get_app_logger, get_log_content
    app_logger = get_app_logger()
    conversion_logger = get_conversion_logger()
    loggers_available = True
except ImportError:
    app_logger = None
    conversion_logger = None
    loggers_available = False
    print("Logging module not available.")

try:
    from app.forecasting import get_forecaster
    forecasting_available = True
except ImportError:
    forecasting_available = False
    print("Forecasting module not available.")

try:
    from app.advanced_forecasting import get_advanced_forecaster
    advanced_forecasting_available = True
except ImportError:
    advanced_forecasting_available = False
    print("Advanced forecasting module not available.")

try:
    from converters.block_converter import run_block_conversion
    from converters.segment_converter import run_segment_conversion
    from converters.occupancy_converter import run_occupancy_conversion
    converters_available = True
except ImportError:
    converters_available = False
    print("Converters module not available.")

# Statistical and ML imports for forecasting
try:
    import numpy as np
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    import warnings
    warnings.filterwarnings('ignore')
except ImportError:
    st.warning("Some advanced features may not be available. Please install: numpy, statsmodels, scikit-learn")

# Import corrected forecasting models
try:
    from app.corrected_forecasting import CorrectedHotelForecasting
    from app.data_integration import HotelDataIntegrator
    from app.fast_forecasting import FastHotelForecasting
    CORRECTED_FORECASTING_AVAILABLE = True
except ImportError:
    CORRECTED_FORECASTING_AVAILABLE = False

# Time series forecasting imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller
    TS_AVAILABLE = True
except ImportError:
    TS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from app.gemini_backend import create_gemini_backend
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Enhanced Gemini Backend
try:
    from app.enhanced_gemini_backend import create_enhanced_gemini_backend
    from app.interactive_chat_interface import render_enhanced_gpt_tab
    ENHANCED_GEMINI_AVAILABLE = True
except ImportError:
    ENHANCED_GEMINI_AVAILABLE = False

import sqlite3

# Add project root and converters to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'converters'))

# Initialize loggers
conversion_logger = None
app_logger = None

if loggers_available:
    try:
        conversion_logger = get_conversion_logger()
        app_logger = get_app_logger()
    except:
        pass

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
if 'pickup_report_date' not in st.session_state:
    st.session_state.pickup_report_date = None
if 'current_mhr_file' not in st.session_state:
    st.session_state.current_mhr_file = None

def extract_pickup_report_date(filename):
    """
    Extract pickup report date from filename
    Expected format: MHR Pick Up Report 2025 - 15.08.25
    """
    try:
        # Pattern to match date in format DD.MM.YY or similar
        date_patterns = [
            r'(\d{1,2})\.(\d{1,2})\.(\d{2,4})',  # DD.MM.YY or DD.MM.YYYY
            r'(\d{1,2})-(\d{1,2})-(\d{2,4})',   # DD-MM-YY or DD-MM-YYYY
            r'(\d{1,2})_(\d{1,2})_(\d{2,4})',   # DD_MM_YY or DD_MM_YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, filename)
            if match:
                day, month, year = match.groups()
                
                # Handle 2-digit years
                if len(year) == 2:
                    year = '20' + year if int(year) < 50 else '19' + year
                
                # Format as DD/MM/YY
                formatted_date = f"{day.zfill(2)}/{month.zfill(2)}/{year[-2:]}"
                return formatted_date
                
        return None
    except Exception as e:
        return None

def display_pickup_report_header():
    """Display the pickup report header with extracted date"""
    # Header removed - date only shown in navigation panel
    pass
if 'last_run_timestamp' not in st.session_state:
    st.session_state.last_run_timestamp = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Dashboard"
if 'current_mhr_file' not in st.session_state:
    st.session_state.current_mhr_file = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def show_loading_requirements():
    """Show message when data needs to be loaded first"""
    st.warning("⚠️ Please load data first using the Dashboard tab")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Go to Dashboard Tab", use_container_width=True):
            st.session_state.current_tab = "Dashboard"
            st.rerun()

def run_conversion_process(file_path):
    """
    Run the complete conversion process
    
    Args:
        file_path: Path to Excel file to process
        
    Returns:
        bool: True if successful, False otherwise
    """
    status_placeholder = st.empty()
    
    try:
        if app_logger:
            app_logger.info(f"Starting conversion process for: {file_path}")
        status_placeholder.info("🔄 Starting conversion process...")
        
        # Step 1: Run segment conversion
        if converters_available:
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
        else:
            # Create dummy data if converters not available
            segment_df = pd.DataFrame({'Month': ['2024-01-01'], 'Segment': ['Test'], 'Revenue': [1000]})
            occupancy_df = pd.DataFrame({'Date': ['2024-01-01'], 'Occupancy': [75], 'ADR': [500]})
            segment_path = "No converter available"
            occupancy_path = "No converter available"
            st.warning("⚠️ Converters not available - using dummy data for testing")
        
        # Step 3: Ingest to database
        status_placeholder.info("🔄 Ingesting data to database...")
        
        if database_available:
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
        else:
            # Save to CSV files as backup
            os.makedirs(project_root / "data" / "processed", exist_ok=True)
            segment_df.to_csv(project_root / "data" / "processed" / "segment.csv", index=False)
            occupancy_df.to_csv(project_root / "data" / "processed" / "occupancy.csv", index=False)
        
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
            # Try to load from database first
            if database_available:
                db = get_database()
                segment_data = db.get_segment_data()
                occupancy_data = db.get_occupancy_data()
                
                if not segment_data.empty and not occupancy_data.empty:
                    st.session_state.segment_data = segment_data
                    st.session_state.occupancy_data = occupancy_data
                    st.session_state.data_loaded = True
                    st.session_state.last_run_timestamp = datetime.now()
                    st.info("ℹ️ Existing data loaded from database.")
                    return True
            
            # Fallback to CSV files
            segment_file = project_root / "data" / "processed" / "segment.csv"
            occupancy_file = project_root / "data" / "processed" / "occupancy.csv"
            
            if segment_file.exists() and occupancy_file.exists():
                st.session_state.segment_data = pd.read_csv(segment_file)
                st.session_state.occupancy_data = pd.read_csv(occupancy_file)
                st.session_state.data_loaded = True
                st.session_state.last_run_timestamp = datetime.now()
                st.info("ℹ️ Existing data loaded from CSV files.")
                return True
            else:
                st.error("No existing data found.")
                return False
        except Exception as load_error:
            st.error(f"Failed to load existing data: {str(load_error)}")
            return False

def show_conversion_summary(segment_df, occupancy_df, segment_path, occupancy_path):
    """Show summary of successful conversion"""
    st.success("✅ Conversion completed successfully!")

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
            if database_available:
                db_stats = get_database().get_database_stats()
                if db_stats:
                    st.info(f"📊 DB: {db_stats.get('segment_analysis_rows', 0)} seg, {db_stats.get('occupancy_analysis_rows', 0)} occ")
            else:
                st.info("📊 DB: Module not available")
        except:
            st.info("📊 DB: Not available")

def placeholder_tab(title):
    """Placeholder for tabs that will be fully implemented"""
    st.header(f"🚧 {title}")
    st.info(f"This is the {title} section. Functionality will be implemented here.")
    if not st.session_state.data_loaded:
        show_loading_requirements()


def dashboard_tab():
    """Dashboard tab for data processing"""
    st.header("📊 Dashboard")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose MHR Excel file",
        type=['xlsm', 'xlsx'],
        help="Upload an MHR Pick Up Report with DPR sheet",
        key="main_dashboard_upload"
    )
    
    # Determine which file to process
    file_to_process = None
    if uploaded_file is not None:
        # Clean up any previous uploaded files
        for old_file in project_root.glob("uploaded_*"):
            try:
                old_file.unlink()
            except:
                pass  # Ignore errors when cleaning up old files
        
        # Save uploaded file to a persistent location for later access
        persistent_filename = f"uploaded_{uploaded_file.name}"
        persistent_path = project_root / persistent_filename
        
        # Write the uploaded file to persistent location
        with open(persistent_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_to_process = persistent_path
        
        # Auto-trigger processing with loading indicator
        with st.spinner("Processing file..."):
            success = run_conversion_process(file_to_process)
        
        if success:
            st.success("✅ Conversion completed successfully!")
        else:
            st.error("❌ Processing failed")
    
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

def show_dashboard_kpis():
    """Show daily pickup KPI cards with conditional formatting"""
    st.subheader("📊 Daily Pickup KPI Cards")
    
    try:
        # Get occupancy data for Daily Revenue (column D)
        if hasattr(st.session_state, 'occupancy_data') and st.session_state.occupancy_data is not None:
            df = st.session_state.occupancy_data.copy()
        else:
            if database_available:
                db = get_database()
                df = db.get_occupancy_data()
            else:
                df = pd.DataFrame()  # Empty dataframe if no database
        
        if df.empty:
            st.error("No occupancy data available for KPIs")
            return
        
        # Get next 3 months (starting from current month)
        current_date = datetime.now()
        months_to_show = []
        
        for i in range(3):  # Current + next 2 months
            month_offset = i
            if current_date.month + month_offset > 12:
                year_offset = (current_date.month + month_offset - 1) // 12
                month_num = ((current_date.month + month_offset - 1) % 12) + 1
                month_date = datetime(current_date.year + year_offset, month_num, 1)
            else:
                month_date = datetime(current_date.year, current_date.month + month_offset, 1)
            months_to_show.append(month_date)
        
        # Create 3 columns for the KPI cards
        col1, col2, col3 = st.columns(3)
        
        # Column D would be the 4th column (index 3)
        column_list = list(df.columns)
        if len(column_list) >= 4:
            column_d = column_list[3]  # 4th column (index 3) = Column D
        else:
            st.error("❌ Column D not found - not enough columns in occupancy data")
            return
        
        for i, month_date in enumerate(months_to_show):
            month_name = month_date.strftime('%B %Y')
            
            # Get month data using Date column from occupancy data
            month_data = pd.DataFrame()
            
            if not df.empty and 'Date' in df.columns:
                # Convert Date column to datetime if it's not already
                df_copy = df.copy()
                try:
                    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
                    
                    # Filter by year and month
                    month_mask = (df_copy['Date'].dt.year == month_date.year) & (df_copy['Date'].dt.month == month_date.month)
                    month_data = df_copy[month_mask]
                    
                except Exception as e:
                    st.error(f"Date conversion error for {month_name}: {e}")
            
            if not month_data.empty and column_d in month_data.columns:
                # Sum of Column D for the month
                column_d_total = month_data[column_d].sum()
                
                # Determine color based on positive/negative
                if column_d_total >= 0:
                    delta_color = "normal"  # Green
                    emoji = "✅"
                else:
                    delta_color = "inverse"  # Red
                    emoji = "❌"
                
                # Show in appropriate column
                with [col1, col2, col3][i]:
                    st.metric(
                        label=f"{emoji} {month_name}",
                        value=f"AED {column_d_total:,.0f}",
                        delta=f"Column D Sum",
                        delta_color=delta_color
                    )
            else:
                with [col1, col2, col3][i]:
                    st.metric(
                        label=f"❓ {month_name}",
                        value="No Data",
                        delta="Column D Sum"
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
            if database_available:
                db = get_database()
                df = db.get_segment_data()
            else:
                df = pd.DataFrame()  # Empty dataframe if no database
        
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
        if database_available:
            db = get_database()
            df = db.get_occupancy_data()
        else:
            df = pd.DataFrame()  # Empty dataframe if no database
    
    if df.empty:
        st.error("No occupancy data available")
        return
    
    # Ensure Date column is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Date range filter - Single date range picker
    st.subheader("📅 Date Range Filter")
    if 'Date' in df.columns:
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Select the start and end dates for analysis"
        )
        
        # Filter data based on date range
        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
            filtered_df = df[mask]
        else:
            # If only one date selected, use full range
            filtered_df = df
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
    
    # Validation metrics
    st.subheader("⚠️ Data Validation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_days = len(df)
        expected_days = 365  # 2025 is not a leap year
        missing_days = max(0, expected_days - total_days)
        st.metric("Missing Days", missing_days, delta=f"{(missing_days/expected_days)*100:.1f}% of year" if expected_days > 0 else "")
    
    with col2:
        high_occ_count = (df['Occ%'] > 100).sum() if 'Occ%' in df.columns else 0
        st.metric("High Occupancy (>100%)", high_occ_count, delta="⚠️ Check data" if high_occ_count > 0 else "✅ OK")
    
    with col3:
        zero_revenue_count = (df['Revenue'] == 0).sum() if 'Revenue' in df.columns else 0
        st.metric("Zero Revenue Days", zero_revenue_count)
    
    # Occupancy trend chart with forecast
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
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_occ = filtered_df['Occ%'].mean()
            st.metric("Average Occupancy", f"{avg_occ:.1f}%")
        
        with col2:
            avg_adr = filtered_df['ADR'].mean()
            st.metric("Average ADR", f"AED {avg_adr:.0f}")
        
        with col3:
            max_occ = filtered_df['Occ%'].max()
            st.metric("Peak Occupancy", f"{max_occ:.1f}%")
        
        with col4:
            max_adr = filtered_df['ADR'].max()
            st.metric("Peak ADR", f"AED {max_adr:.0f}")
    
    # Enhanced Forecast table with moving average
    st.subheader("🔮 Current Month Occupancy & Revenue Forecast")
    
    try:
        if forecasting_available:
            advanced_forecaster = get_advanced_forecaster()
            current_month_forecast = advanced_forecaster.forecast_current_month_occupancy_revenue(df)
            
            if 'error' not in current_month_forecast:
                st.success("✅ Current month forecast generated using 7-day moving average!")
                
                # Moving average summary
                st.subheader("📊 7-Day Moving Average Summary")
                
                moving_avgs = current_month_forecast['moving_averages']
                month_proj = current_month_forecast['month_projections']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("7-Day Avg Occupancy", f"{moving_avgs['avg_occupancy_7d']:.1f}%")
                
                with col2:
                    st.metric("7-Day Avg Revenue", f"AED {moving_avgs['avg_revenue_7d']:,.0f}")
                
                with col3:
                    st.metric("7-Day Avg ADR", f"AED {moving_avgs['avg_adr_7d']:.0f}")
                
                with col4:
                    st.metric("Days Remaining", current_month_forecast['days_remaining'])
                
                # Month-end projections
                st.subheader("🎯 Month-End Projections")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Actual Revenue to Date (Confirmed)",
                        f"AED {month_proj['actual_revenue_to_date']:,.0f}"
                    )
                
                with col2:
                    st.metric(
                        "Remaining Forecast (Today-Month End)",
                        f"AED {month_proj['remaining_revenue_forecast']:,.0f}"
                    )
                
                with col3:
                    st.metric(
                        "Month-End Total Forecast",
                        f"AED {month_proj['month_end_revenue_forecast']:,.0f}"
                    )
                
                # Forecast table
                forecast_data = current_month_forecast['forecast_data']
                
                if forecast_data:
                    st.subheader("📋 Daily Forecast Table")
                    
                    # Create display dataframe
                    display_forecast = pd.DataFrame(forecast_data)
                    display_forecast['date'] = display_forecast['date'].dt.strftime('%Y-%m-%d')
                    display_forecast['forecasted_occupancy'] = display_forecast['forecasted_occupancy'].apply(lambda x: f"{x:.1f}%")
                    display_forecast['forecasted_revenue'] = display_forecast['forecasted_revenue'].apply(lambda x: f"AED {x:,.0f}")
                    display_forecast['forecasted_adr'] = display_forecast['forecasted_adr'].apply(lambda x: f"AED {x:.0f}")
                    
                    display_forecast = display_forecast.rename(columns={
                        'date': 'Date',
                        'forecasted_occupancy': 'Forecast Occupancy',
                        'forecasted_revenue': 'Forecast Revenue',
                        'forecasted_adr': 'Forecast ADR',
                        'day_of_week': 'Day of Week'
                    })
                    
                    st.dataframe(display_forecast, use_container_width=True)
                    
                    # Forecast chart
                    st.subheader("📈 Daily Forecast Visualization")
                    
                    forecast_chart_data = pd.DataFrame(forecast_data)
                    
                    fig_forecast_chart = go.Figure()
                    
                    # Add occupancy forecast
                    fig_forecast_chart.add_trace(go.Scatter(
                        x=forecast_chart_data['date'],
                        y=forecast_chart_data['forecasted_occupancy'],
                        mode='lines+markers',
                        name='Forecast Occupancy (%)',
                        line=dict(color='blue')
                    ))
                    
                    fig_forecast_chart.update_layout(
                        title="Current Month Daily Occupancy Forecast",
                        xaxis_title="Date",
                        yaxis_title="Occupancy %",
                        height=400
                    )
                    
                    st.plotly_chart(fig_forecast_chart, use_container_width=True)
            else:
                st.warning(f"⚠️ Current month forecast unavailable: {current_month_forecast['error']}")
        else:
            st.info("📊 Advanced forecasting module not available - basic functionality only")
                
    except Exception as e:
        st.error(f"❌ Forecast error: {str(e)}")
    
    # Daily Revenue Pickup Chart
    st.subheader("💰 Daily Revenue Pickup per Day")
    
    try:
        # Use the same occupancy data that's already loaded in this tab
        occupancy_df = filtered_df.copy()  # Use the already filtered occupancy data
        
        if occupancy_df.empty:
            st.info("No occupancy data available for revenue pickup chart")
            return
        
        # Determine the revenue column name (could be 'Revenue', 'DailyRevenue', etc.)
        revenue_column = None
        if 'Revenue' in occupancy_df.columns:
            revenue_column = 'Revenue'
        elif 'DailyRevenue' in occupancy_df.columns:
            revenue_column = 'DailyRevenue'
        elif 'Daily_Revenue' in occupancy_df.columns:
            revenue_column = 'Daily_Revenue'
        
        if revenue_column is None:
            st.error("❌ Daily Revenue column not found in occupancy data")
            st.write("Available columns:", list(occupancy_df.columns))
            return
        
        if not occupancy_df.empty and revenue_column in occupancy_df.columns:
            # Ensure Date column is datetime
            if 'Date' in occupancy_df.columns:
                occupancy_df['Date'] = pd.to_datetime(occupancy_df['Date'])
                
                # Sort by date to ensure proper trend line
                occupancy_df = occupancy_df.sort_values('Date')
                
                # Create daily revenue pickup chart - TREND LINE
                fig_pickup = go.Figure()
                
                # Add revenue pickup as trend line (smooth line)
                fig_pickup.add_trace(go.Scatter(
                    x=occupancy_df['Date'],
                    y=occupancy_df[revenue_column],
                    mode='lines+markers',
                    name='Daily Revenue Trend',
                    line=dict(color='green', width=3, shape='spline'),  # Smooth trend line
                    marker=dict(size=4, color='darkgreen'),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Daily Revenue:</b> AED %{y:,.0f}<extra></extra>',
                    fill='tonexty' if len(occupancy_df) > 1 else None,
                    fillcolor='rgba(0,128,0,0.1)'
                ))
                
                # Add trend line (moving average if enough data)
                if len(occupancy_df) >= 7:
                    occupancy_df['MA_7'] = occupancy_df[revenue_column].rolling(window=7, center=True).mean()
                    fig_pickup.add_trace(go.Scatter(
                        x=occupancy_df['Date'],
                        y=occupancy_df['MA_7'],
                        mode='lines',
                        name='7-Day Moving Average',
                        line=dict(color='orange', width=2, dash='dash'),
                        hovertemplate='<b>Date:</b> %{x}<br><b>7-Day Avg:</b> AED %{y:,.0f}<extra></extra>'
                    ))
                
                # Update layout
                fig_pickup.update_layout(
                    title="Daily Revenue Pickup Trend (Column D from Occupancy Data)",
                    xaxis_title="Date",
                    yaxis_title="Daily Revenue (AED)",
                    height=500,
                    hovermode='x unified',
                    showlegend=True,
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(showgrid=True, gridcolor='lightgray')
                )
                
                st.plotly_chart(fig_pickup, use_container_width=True)
                
                # Summary statistics for pickup revenue
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_revenue = occupancy_df[revenue_column].sum()
                    st.metric("Total Daily Revenue", f"AED {total_revenue:,.0f}")
                with col2:
                    avg_revenue = occupancy_df[revenue_column].mean()
                    st.metric("Average Daily Revenue", f"AED {avg_revenue:,.0f}")
                with col3:
                    max_revenue = occupancy_df[revenue_column].max()
                    max_date = occupancy_df.loc[occupancy_df[revenue_column].idxmax(), 'Date']
                    st.metric("Highest Revenue Day", f"AED {max_revenue:,.0f}", 
                             delta=f"{max_date.strftime('%Y-%m-%d')}")
                with col4:
                    positive_days = (occupancy_df[revenue_column] > 0).sum()
                    st.metric("Positive Revenue Days", f"{positive_days} days")
            else:
                st.error("Date column not found in occupancy data")
        else:
            st.info("Daily revenue data not available")
            
    except Exception as e:
        st.error(f"❌ Error creating daily revenue chart: {str(e)}")
    
    # Column D Trend Line Chart
    st.subheader("📊 Column D Trend Analysis")
    
    try:
        # Identify Column D from occupancy data
        column_list = list(filtered_df.columns)
        if len(column_list) >= 4:
            column_d = column_list[3]  # 4th column (index 3) = Column D
            st.write(f"**Showing trend for Column D: {column_d}**")
            
            if column_d in filtered_df.columns:
                # Ensure Date column is datetime
                df_for_chart = filtered_df.copy()
                if 'Date' in df_for_chart.columns:
                    df_for_chart['Date'] = pd.to_datetime(df_for_chart['Date'])
                    df_for_chart = df_for_chart.sort_values('Date')
                    
                    # Create Column D trend chart
                    fig_column_d = go.Figure()
                    
                    # Add Column D trend line
                    fig_column_d.add_trace(go.Scatter(
                        x=df_for_chart['Date'],
                        y=df_for_chart[column_d],
                        mode='lines+markers',
                        name=f'{column_d} Trend',
                        line=dict(color='purple', width=3),
                        marker=dict(size=5, color='darkviolet'),
                        hovertemplate=f'<b>Date:</b> %{{x}}<br><b>{column_d}:</b> %{{y:,.2f}}<extra></extra>'
                    ))
                    
                    # Add moving average if enough data
                    if len(df_for_chart) >= 7:
                        df_for_chart['MA_7'] = df_for_chart[column_d].rolling(window=7, center=True).mean()
                        fig_column_d.add_trace(go.Scatter(
                            x=df_for_chart['Date'],
                            y=df_for_chart['MA_7'],
                            mode='lines',
                            name='7-Day Moving Average',
                            line=dict(color='orange', width=2, dash='dash'),
                            hovertemplate=f'<b>Date:</b> %{{x}}<br><b>7-Day Avg:</b> %{{y:,.2f}}<extra></extra>'
                        ))
                    
                    # Update layout
                    fig_column_d.update_layout(
                        title=f"Column D ({column_d}) Daily Trend Analysis",
                        xaxis_title="Date",
                        yaxis_title=f"{column_d}",
                        height=500,
                        hovermode='x unified',
                        showlegend=True,
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=True, gridcolor='lightgray'),
                        yaxis=dict(showgrid=True, gridcolor='lightgray')
                    )
                    
                    st.plotly_chart(fig_column_d, use_container_width=True)
                    
                    # Summary statistics for Column D
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        total_column_d = df_for_chart[column_d].sum()
                        st.metric(f"Total {column_d}", f"{total_column_d:,.2f}")
                    with col2:
                        avg_column_d = df_for_chart[column_d].mean()
                        st.metric(f"Average {column_d}", f"{avg_column_d:,.2f}")
                    with col3:
                        max_column_d = df_for_chart[column_d].max()
                        max_date = df_for_chart.loc[df_for_chart[column_d].idxmax(), 'Date']
                        st.metric(f"Highest {column_d}", f"{max_column_d:,.2f}", 
                                 delta=f"{max_date.strftime('%Y-%m-%d')}")
                    with col4:
                        std_column_d = df_for_chart[column_d].std()
                        st.metric(f"Std Deviation", f"{std_column_d:,.2f}")
                else:
                    st.error("Date column not found for Column D chart")
            else:
                st.error(f"Column D ({column_d}) not found in data")
        else:
            st.error("Not enough columns in data to identify Column D")
            
    except Exception as e:
        st.error(f"❌ Error creating Column D trend chart: {str(e)}")
    
    # Data table
    st.subheader("📋 Occupancy Data Table")
    
    # Month filter instead of pagination
    if 'Date' in filtered_df.columns:
        available_months = filtered_df['Date'].dt.to_period('M').unique()
        available_months = sorted(available_months)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if len(available_months) > 1:
                selected_month = st.selectbox(
                    "Select Month", 
                    ["All Months"] + [str(month) for month in available_months],
                    key="occ_month_filter"
                )
                
                if selected_month != "All Months":
                    month_period = pd.Period(selected_month)
                    month_mask = filtered_df['Date'].dt.to_period('M') == month_period
                    table_df = filtered_df[month_mask]
                else:
                    table_df = filtered_df
            else:
                table_df = filtered_df
    else:
        table_df = filtered_df
    
    # Format display columns
    if not table_df.empty:
        display_columns = ['Date', 'DOW', 'Rms', 'Rm Sold', 'Revenue', 'ADR', 'Occ%']
        available_columns = [col for col in display_columns if col in table_df.columns]
        
        if available_columns:
            formatted_df = table_df[available_columns].copy()
            
            # Format monetary columns
            for col in ['Revenue', 'ADR']:
                if col in formatted_df.columns:
                    formatted_df[col] = formatted_df[col].apply(lambda x: f"AED {x:,.0f}" if pd.notna(x) else "AED 0")
            
            # Format percentage
            if 'Occ%' in formatted_df.columns:
                formatted_df['Occ%'] = formatted_df['Occ%'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "0.0%")
            
            st.dataframe(formatted_df, use_container_width=True, height=400)
        else:
            st.dataframe(table_df, use_container_width=True, height=400)
    
    st.info(f"Showing {len(table_df)} rows")

def style_variance_table(df, variance_col='Variance_%'):
    """
    Apply conditional formatting to variance tables
    """
    def color_variance(val):
        """Color variance percentages: green for positive, red for negative"""
        if isinstance(val, str) and '%' in val:
            try:
                numeric_val = float(val.replace('%', ''))
                if numeric_val > 0:
                    return 'background-color: #d4edda; color: #155724'  # Green
                elif numeric_val < 0:
                    return 'background-color: #f8d7da; color: #721c24'  # Red
                else:
                    return 'background-color: #e2e3e5; color: #383d41'  # Gray
            except:
                return ''
        return ''
    
    def color_variance_amount(val):
        """Color variance amounts: green for positive, red for negative"""
        if isinstance(val, str) and 'AED' in val:
            try:
                # Extract numeric value from "AED X,XXX" format
                numeric_str = val.replace('AED', '').replace(',', '').strip()
                numeric_val = float(numeric_str)
                if numeric_val > 0:
                    return 'background-color: #d4edda; color: #155724'  # Green
                elif numeric_val < 0:
                    return 'background-color: #f8d7da; color: #721c24'  # Red
                else:
                    return 'background-color: #e2e3e5; color: #383d41'  # Gray
            except:
                return ''
        return ''
    
    # Apply styling
    styled_df = df.style.applymap(color_variance, subset=[variance_col])
    styled_df = styled_df.applymap(color_variance_amount, subset=['Variance'])
    
    return styled_df

def create_delta_comparison(df, col1, col2, comparison_name, segment_col, granularity):
    """
    Create delta comparison analysis with table and charts
    
    Args:
        df: DataFrame with segment data
        col1: Primary column (trend line)
        col2: Comparison column (bar chart)
        comparison_name: Name for the comparison
        segment_col: Segment column to use
        granularity: Analysis granularity (Monthly, Segment-wise, Both)
    """
    try:
        # Calculate variance with zero division protection
        df_analysis = df.copy()
        df_analysis['Variance'] = df_analysis[col1] - df_analysis[col2]
        
        # Protect against division by zero
        df_analysis['Variance_%'] = 0.0
        mask = df_analysis[col2] != 0
        df_analysis.loc[mask, 'Variance_%'] = ((df_analysis.loc[mask, col1] - df_analysis.loc[mask, col2]) / df_analysis.loc[mask, col2] * 100)
        
        # Replace infinite values
        df_analysis['Variance_%'] = df_analysis['Variance_%'].replace([float('inf'), float('-inf')], 0).fillna(0)
        
        if granularity in ["Monthly", "Both"]:
            st.subheader(f"📅 Monthly Analysis - {comparison_name}")
            
            # Monthly aggregation
            monthly_data = df_analysis.groupby(['Month']).agg({
                col1: 'sum',
                col2: 'sum',
                'Variance': 'sum'
            }).reset_index()
            
            # Protect against division by zero for monthly data
            monthly_data['Variance_%'] = 0.0
            mask = monthly_data[col2] != 0
            monthly_data.loc[mask, 'Variance_%'] = ((monthly_data.loc[mask, col1] - monthly_data.loc[mask, col2]) / monthly_data.loc[mask, col2] * 100)
            monthly_data['Variance_%'] = monthly_data['Variance_%'].replace([float('inf'), float('-inf')], 0).fillna(0)
            
            # Create monthly table with conditional formatting
            monthly_display = monthly_data.copy()
            monthly_display['Month'] = monthly_display['Month'].dt.strftime('%Y-%m')
            monthly_display[col1] = monthly_display[col1].apply(lambda x: f"AED {x:,.0f}")
            monthly_display[col2] = monthly_display[col2].apply(lambda x: f"AED {x:,.0f}")
            monthly_display['Variance'] = monthly_display['Variance'].apply(lambda x: f"AED {x:,.0f}")
            monthly_display['Variance_%'] = monthly_display['Variance_%'].apply(lambda x: f"{x:.1f}%")
            
            # Display table with conditional formatting
            st.write("**Monthly Comparison Table:**")
            styled_monthly = style_variance_table(monthly_display)
            st.dataframe(styled_monthly, use_container_width=True)
            
            # Create monthly chart
            fig_monthly = go.Figure()
            
            # Add bar chart for comparison column
            fig_monthly.add_trace(go.Bar(
                x=monthly_data['Month'],
                y=monthly_data[col2],
                name=col2.replace('_', ' '),
                marker_color='lightblue',
                yaxis='y'
            ))
            
            # Add line chart for primary column
            fig_monthly.add_trace(go.Scatter(
                x=monthly_data['Month'],
                y=monthly_data[col1],
                mode='lines+markers',
                name=col1.replace('_', ' '),
                line=dict(color='red', width=3),
                yaxis='y'
            ))
            
            fig_monthly.update_layout(
                title=f'Monthly Trend - {comparison_name}',
                xaxis_title='Month',
                yaxis_title='Revenue (AED)',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        if granularity in ["Segment-wise", "Both"]:
            st.subheader(f"🎯 Segment Analysis - {comparison_name}")
            
            # Segment aggregation
            segment_data = df_analysis.groupby([segment_col]).agg({
                col1: 'sum',
                col2: 'sum', 
                'Variance': 'sum'
            }).reset_index()
            
            # Protect against division by zero for segment data
            segment_data['Variance_%'] = 0.0
            mask = segment_data[col2] != 0
            segment_data.loc[mask, 'Variance_%'] = ((segment_data.loc[mask, col1] - segment_data.loc[mask, col2]) / segment_data.loc[mask, col2] * 100)
            segment_data['Variance_%'] = segment_data['Variance_%'].replace([float('inf'), float('-inf')], 0).fillna(0)
            
            # Sort by variance percentage
            segment_data = segment_data.sort_values('Variance_%', ascending=False)
            
            # Create segment table with conditional formatting
            segment_display = segment_data.copy()
            segment_display[col1] = segment_display[col1].apply(lambda x: f"AED {x:,.0f}")
            segment_display[col2] = segment_display[col2].apply(lambda x: f"AED {x:,.0f}")
            segment_display['Variance'] = segment_display['Variance'].apply(lambda x: f"AED {x:,.0f}")
            segment_display['Variance_%'] = segment_display['Variance_%'].apply(lambda x: f"{x:.1f}%")
            
            # Display table with conditional formatting
            st.write("**Segment Comparison Table:**")
            styled_segment = style_variance_table(segment_display)
            st.dataframe(styled_segment, use_container_width=True)
            
            # Create segment chart
            fig_segment = go.Figure()
            
            # Add bar chart for comparison column
            fig_segment.add_trace(go.Bar(
                x=segment_data[segment_col],
                y=segment_data[col2],
                name=col2.replace('_', ' '),
                marker_color='lightgreen',
                yaxis='y'
            ))
            
            # Add line chart for primary column
            fig_segment.add_trace(go.Scatter(
                x=segment_data[segment_col],
                y=segment_data[col1],
                mode='lines+markers',
                name=col1.replace('_', ' '),
                line=dict(color='orange', width=3),
                yaxis='y'
            ))
            
            fig_segment.update_layout(
                title=f'Segment Analysis - {comparison_name}',
                xaxis_title='Segment',
                yaxis_title='Revenue (AED)',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_segment, use_container_width=True)
            
        # Variance summary
        st.subheader(f"📊 Variance Summary - {comparison_name}")
        col1_sum = df_analysis[col1].sum()
        col2_sum = df_analysis[col2].sum()
        total_variance = col1_sum - col2_sum
        variance_pct = (total_variance / col2_sum * 100) if col2_sum != 0 else 0
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.metric(
                label=col1.replace('_', ' '),
                value=f"AED {col1_sum:,.0f}"
            )
        
        with summary_col2:
            st.metric(
                label=col2.replace('_', ' '),
                value=f"AED {col2_sum:,.0f}"
            )
        
        with summary_col3:
            delta_color = "normal" if variance_pct >= 0 else "inverse"
            st.metric(
                label="Total Variance",
                value=f"AED {total_variance:,.0f}",
                delta=f"{variance_pct:.1f}%"
            )
            
    except Exception as e:
        st.error(f"Error in delta comparison: {str(e)}")

def segment_analysis_tab():
    """Segment Analysis Tab with sub-tabs"""
    st.header("🎯 Segment Analysis")
    
    # Create sub-tabs
    analysis_tab, delta_tab, market_seg_tab = st.tabs(["📊 Revenue Analysis", "📈 Delta Analysis", "📋 Market Segmentation"])
    
    with analysis_tab:
        st.subheader("📊 Revenue Analysis")
        
        if not st.session_state.data_loaded:
            show_loading_requirements()
            return
        
        # Get data from session state or database
        if st.session_state.segment_data is not None:
            df = st.session_state.segment_data.copy()
        else:
            if database_available:
                db = get_database()
                df = db.get_segment_data()
            else:
                df = pd.DataFrame()  # Empty dataframe if no database
        
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
            
            # Simple table display
            top_segments_df = top_segments.reset_index()
            top_segments_df.columns = ['Segment', 'Revenue']
            top_segments_df['Revenue'] = top_segments_df['Revenue'].apply(lambda x: f"AED {x:,.0f}")
            st.dataframe(top_segments_df, use_container_width=True, hide_index=True)
    
    with delta_tab:
        st.subheader("📈 Delta Analysis")
        
        # Check if data is loaded
        if not st.session_state.data_loaded:
            show_loading_requirements()
            return
        
        # Get data from session state or database
        if st.session_state.segment_data is not None:
            delta_df = st.session_state.segment_data.copy()
        else:
            if database_available:
                db = get_database()
                delta_df = db.get_segment_data()
            else:
                delta_df = pd.DataFrame()
        
        if delta_df.empty:
            st.error("No segment data available for delta analysis")
            return
        
        # Ensure Month column is datetime
        if 'Month' in delta_df.columns:
            delta_df['Month'] = pd.to_datetime(delta_df['Month'], errors='coerce')
            delta_df = delta_df.dropna(subset=['Month'])
        
        # Delta Analysis Controls
        st.subheader("🎛️ Analysis Controls")
        delta_col1, delta_col2, delta_col3 = st.columns(3)
        
        with delta_col1:
            # Segment type toggle
            use_merged_delta = st.checkbox(
                "Use Merged Segments",
                value=True,
                help="Use grouped segments (Retail, Corporate, etc.) vs original segments",
                key="delta_merged_toggle"
            )
        
        with delta_col2:
            # Month filter
            if 'Month' in delta_df.columns and len(delta_df) > 0:
                available_months = sorted(delta_df['Month'].dt.to_period('M').unique())
                selected_months = st.multiselect(
                    "Select Months:",
                    options=available_months,
                    default=available_months[-3:] if len(available_months) >= 3 else available_months,
                    format_func=lambda x: str(x),
                    key="delta_month_filter"
                )
        
        with delta_col3:
            # Analysis type
            analysis_granularity = st.selectbox(
                "Granularity:",
                ["Monthly", "Segment-wise", "Both"],
                help="Choose analysis granularity",
                key="delta_granularity"
            )
        
        # Filter data based on selections
        if selected_months:
            delta_df = delta_df[delta_df['Month'].dt.to_period('M').isin(selected_months)]
        
        # Choose segment column
        segment_col = 'MergedSegment' if use_merged_delta and 'MergedSegment' in delta_df.columns else 'Segment'
        
        if delta_df.empty:
            st.warning("No data available for selected filters")
            return
        
        # Required columns check
        required_columns = [
            'Business_on_the_Books_Revenue',
            'Business_on_the_Books_Same_Time_Last_Year_Revenue', 
            'Full_Month_Last_Year_Revenue',
            'Budget_This_Year_Revenue'
        ]
        
        missing_columns = [col for col in required_columns if col not in delta_df.columns]
        if missing_columns:
            st.error(f"Missing required columns for delta analysis: {missing_columns}")
            return
        
        # Create delta analysis sections
        st.markdown("---")
        
        # Section 1: BOB vs Same Time Last Year
        st.subheader("📊 Section 1: Business on Books vs Same Time Last Year")
        create_delta_comparison(
            delta_df, 
            'Business_on_the_Books_Revenue', 
            'Business_on_the_Books_Same_Time_Last_Year_Revenue',
            'BOB vs STLY',
            segment_col,
            analysis_granularity
        )
        
        st.markdown("---")
        
        # Section 2: BOB vs Full Month Last Year  
        st.subheader("📊 Section 2: Business on Books vs Full Month Last Year")
        create_delta_comparison(
            delta_df,
            'Business_on_the_Books_Revenue',
            'Full_Month_Last_Year_Revenue', 
            'BOB vs FMLY',
            segment_col,
            analysis_granularity
        )
        
        st.markdown("---")
        
        # Section 3: Budget vs BOB
        st.subheader("📊 Section 3: Budget This Year vs Business on Books")
        create_delta_comparison(
            delta_df,
            'Budget_This_Year_Revenue',
            'Business_on_the_Books_Revenue',
            'Budget vs BOB', 
            segment_col,
            analysis_granularity
        )
    
    with market_seg_tab:
        st.subheader("📋 MHR Market Segmentation Documentation")
        st.info("This section displays the official MHR Market Segmentation guidelines document from November 2018.")
        
        # Display key segment categories overview
        st.markdown("### 🎯 Market Segment Categories Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🛒 **RETAIL Segments**")
            st.markdown("""
            - **UMLP** - Unmanaged Leisure Premium  
            - **UMLD** - Unmanaged Leisure Discount  
            - **PACK** - Package  
            - **TPIM/ODIM** - Third Party Intermediary Merchant  
            """)
            
            st.markdown("#### 🏢 **NEGOTIATED CORPORATE Segments**")
            st.markdown("""
            - **MCGL** - Managed Corporate Global  
            - **MCLO** - Managed Corporate Local  
            """)
            
            st.markdown("#### 🏛️ **GOVERNMENT Segments**")
            st.markdown("""
            - **GOVT** - Government  
            """)
        
        with col2:
            st.markdown("#### 🏭 **WHOLESALE Segments**")
            st.markdown("""
            - **WHOL** - Wholesale Fixed Value  
            """)
            
            st.markdown("#### 👥 **GROUP Segments**")
            st.markdown("""
            - **CORG** - Corporate Groups  
            - **CONV** - Convention Groups  
            - **ASSO** - Association Groups  
            - **ADHO** - Adhoc Groups  
            - **TOUR** - Tour Series Groups  
            - **CONT** - Contract  
            """)
            
            st.markdown("#### 🎁 **OTHER Segments**")
            st.markdown("""
            - **CPHS** - Complimentary & House Use  
            """)
        
        st.markdown("---")
        
        # PDF file path
        pdf_path = "MHR Market Segmentation_071218.pdf"
        
        # Check if PDF exists
        if os.path.exists(pdf_path):
            try:
                # Read and display the PDF
                with open(pdf_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                
                # Display PDF download link
                st.download_button(
                    label="📥 Download MHR Market Segmentation PDF",
                    data=pdf_bytes,
                    file_name="MHR_Market_Segmentation_071218.pdf",
                    mime="application/pdf"
                )
                
                st.markdown("---")
                
                # Display PDF content using Read tool
                st.markdown("### 📄 Document Content")
                st.markdown("*The PDF document content is displayed below:*")
                
                # Display the actual PDF content using Read tool
                pdf_content_placeholder = st.empty()
                
                try:
                    # Use base64 encoding to display PDF in iframe
                    import base64
                    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                except Exception as display_error:
                    st.warning(f"PDF inline display not supported. Error: {display_error}")
                    # Fallback: Show file info and encourage download
                    st.info("📋 PDF document is available for download above. Some browsers may not support inline PDF viewing.")
                    
                    # Show file information
                    file_size = len(pdf_bytes) / 1024  # Size in KB
                    st.metric("File Size", f"{file_size:.1f} KB")
                    
                    # Try to extract some basic info about the PDF
                    st.markdown("**Document Information:**")
                    st.markdown("- **Title:** MHR Market Segmentation Guidelines")
                    st.markdown("- **Date:** December 7, 2018")
                    st.markdown("- **Purpose:** Official market segmentation documentation for revenue analysis")
                
            except Exception as e:
                st.error(f"Error loading PDF: {str(e)}")
                st.info("Please ensure the PDF file 'MHR Market Segmentation_071218.pdf' is in the project root directory.")
        else:
            st.error("📄 PDF file not found!")
            st.info("The file 'MHR Market Segmentation_071218.pdf' should be placed in the project root directory.")
            st.markdown("**Expected file path:** `MHR Market Segmentation_071218.pdf`")
            
            # Show expected location
            st.code("./MHR Market Segmentation_071218.pdf")

def adr_analysis_tab():
    """ADR Analysis Tab with comprehensive statistical analysis"""
    st.header("💰 ADR Analysis")
    
    if not st.session_state.data_loaded:
        show_loading_requirements()
        return
    
    # Get data from both segment and occupancy datasets
    segment_data = None
    occupancy_data = None
    
    if st.session_state.segment_data is not None:
        segment_data = st.session_state.segment_data.copy()
    if st.session_state.occupancy_data is not None:
        occupancy_data = st.session_state.occupancy_data.copy()
    
    # Also try to get from database
    if segment_data is None or occupancy_data is None:
        if database_available:
            db = get_database()
            if segment_data is None:
                try:
                    segment_data = db.get_segment_data()
                except:
                    segment_data = pd.DataFrame()
            if occupancy_data is None:
                try:
                    occupancy_data = db.get_occupancy_data()
                except:
                    occupancy_data = pd.DataFrame()
    
    # Controls
    st.subheader("📊 Analysis Controls")
    col1, col2 = st.columns(2)
    
    with col1:
        data_source = st.selectbox(
            "Data Source:",
            ["Segment Data", "Occupancy Data", "Combined"],
            help="Choose which dataset to analyze for ADR"
        )
    
    with col2:
        if data_source in ["Segment Data", "Combined"] and segment_data is not None and not segment_data.empty:
            use_merged_segments = st.checkbox(
                "Use Merged Segments",
                value=True,
                help="Use grouped segments (Retail, Corporate, etc.) vs original segments"
            )
        else:
            use_merged_segments = False
    
    # Determine which data to use
    if data_source == "Segment Data" and segment_data is not None and not segment_data.empty:
        df = segment_data.copy()
        adr_column = 'Business_on_the_Books_ADR'
        segment_column = 'MergedSegment' if use_merged_segments and 'MergedSegment' in df.columns else 'Segment'
        date_column = 'Month'
    elif data_source == "Occupancy Data" and occupancy_data is not None and not occupancy_data.empty:
        df = occupancy_data.copy()
        adr_column = 'ADR'
        segment_column = None  # Occupancy data doesn't have segments
        date_column = 'Date'
    elif data_source == "Combined":
        # For combined analysis, we'll focus on segment data
        if segment_data is not None and not segment_data.empty:
            df = segment_data.copy()
            adr_column = 'Business_on_the_Books_ADR'
            segment_column = 'MergedSegment' if use_merged_segments and 'MergedSegment' in df.columns else 'Segment'
            date_column = 'Month'
        else:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    
    if df.empty or adr_column not in df.columns:
        st.error(f"No data available for ADR analysis from {data_source}")
        return
    
    # Clean ADR data
    df[adr_column] = pd.to_numeric(df[adr_column], errors='coerce')
    df = df.dropna(subset=[adr_column])
    df = df[df[adr_column] > 0]  # Remove zero or negative ADR values
    
    # Key metrics
    st.subheader("📈 ADR Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mean_adr = df[adr_column].mean()
        st.metric("Mean ADR (AED)", f"{mean_adr:.0f}")
    
    with col2:
        median_adr = df[adr_column].median()
        st.metric("Median ADR (AED)", f"{median_adr:.0f}")
    
    with col3:
        std_adr = df[adr_column].std()
        st.metric("Std Deviation (AED)", f"{std_adr:.0f}")
    
    with col4:
        max_adr = df[adr_column].max()
        st.metric("Max ADR (AED)", f"{max_adr:.0f}")
    
    # Additional statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        min_adr = df[adr_column].min()
        st.metric("Min ADR (AED)", f"{min_adr:.0f}")
    
    with col2:
        q25 = df[adr_column].quantile(0.25)
        st.metric("25th Percentile (AED)", f"{q25:.0f}")
    
    with col3:
        q75 = df[adr_column].quantile(0.75)
        st.metric("75th Percentile (AED)", f"{q75:.0f}")
    
    with col4:
        iqr = q75 - q25
        st.metric("IQR (AED)", f"{iqr:.0f}")
    
    # ADR Distribution Histogram
    st.subheader("📊 ADR Distribution")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Bin slider for histogram
        bin_count = st.slider(
            "Number of Bins:",
            min_value=10,
            max_value=100,
            value=30,
            help="Adjust the number of bins in the histogram"
        )
    
    with col1:
        # Create histogram
        fig_hist = px.histogram(
            df,
            x=adr_column,
            nbins=bin_count,
            title=f"ADR Distribution ({data_source})",
            labels={adr_column: 'ADR (AED)', 'count': 'Frequency'}
        )
        
        # Add vertical lines for mean and median
        fig_hist.add_vline(x=mean_adr, line_dash="dash", line_color="red", 
                          annotation_text=f"Mean: AED {mean_adr:.0f}")
        fig_hist.add_vline(x=median_adr, line_dash="dash", line_color="green",
                          annotation_text=f"Median: AED {median_adr:.0f}")
        
        fig_hist.update_layout(height=500)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Boxplot by segment (if segment data is available)
    if segment_column and segment_column in df.columns:
        st.subheader("📦 ADR Distribution by Segment")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # Toggle between merged and original segments
            if data_source == "Segment Data" and 'Segment' in df.columns and 'MergedSegment' in df.columns:
                segment_toggle = st.radio(
                    "Segment Type:",
                    ["Merged Segments", "Original Segments"]
                )
                display_segment_col = 'MergedSegment' if segment_toggle == "Merged Segments" else 'Segment'
            else:
                display_segment_col = segment_column
        
        with col1:
            if display_segment_col in df.columns:
                # Create boxplot using plotly graph_objects for better compatibility
                fig_box = go.Figure()
                
                # Get unique segments
                segments = df[display_segment_col].unique()
                
                # Add box plot for each segment
                for segment in segments:
                    segment_data = df[df[display_segment_col] == segment][adr_column]
                    fig_box.add_trace(go.Box(
                        y=segment_data,
                        name=segment,
                        boxpoints='outliers'
                    ))
                
                fig_box.update_layout(
                    title=f"ADR Distribution by {display_segment_col}",
                    yaxis_title='ADR (AED)',
                    xaxis_title='Segment',
                    height=500,
                    xaxis_tickangle=45
                )
                
                st.plotly_chart(fig_box, use_container_width=True)
                
                # Summary statistics by segment
                st.subheader("📋 ADR Statistics by Segment")
                
                segment_stats = df.groupby(display_segment_col)[adr_column].agg([
                    'count', 'mean', 'median', 'std', 'min', 'max',
                    lambda x: x.quantile(0.25),
                    lambda x: x.quantile(0.75)
                ]).round(2)
                
                segment_stats.columns = ['Count', 'Mean', 'Median', 'Std', 'Min', 'Max', '25th', '75th']
                segment_stats['IQR'] = segment_stats['75th'] - segment_stats['25th']
                
                # Format monetary columns
                monetary_cols = ['Mean', 'Median', 'Std', 'Min', 'Max', '25th', '75th', 'IQR']
                for col in monetary_cols:
                    segment_stats[col] = segment_stats[col].apply(lambda x: f"AED {x:.0f}")
                
                st.dataframe(segment_stats, use_container_width=True)
    
    # Time series analysis (if date column available)
    if date_column and date_column in df.columns:
        st.subheader("📈 ADR Trends Over Time")
        
        # Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])
        
        if segment_column and segment_column in df.columns:
            # Time series by segment
            fig_ts = px.line(
                df.groupby([date_column, segment_column])[adr_column].mean().reset_index(),
                x=date_column,
                y=adr_column,
                color=segment_column,
                title=f"Average ADR Trends by Segment",
                labels={adr_column: 'Average ADR (AED)', date_column: 'Date'}
            )
        else:
            # Overall time series
            daily_adr = df.groupby(date_column)[adr_column].mean().reset_index()
            fig_ts = px.line(
                daily_adr,
                x=date_column,
                y=adr_column,
                title="Average ADR Trend",
                labels={adr_column: 'Average ADR (AED)', date_column: 'Date'}
            )
        
        fig_ts.update_layout(height=500)
        st.plotly_chart(fig_ts, use_container_width=True)
    
    # Outlier Analysis
    st.subheader("🎯 Outlier Analysis")
    
    # Calculate outliers using IQR method
    Q1 = df[adr_column].quantile(0.25)
    Q3 = df[adr_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[adr_column] < lower_bound) | (df[adr_column] > upper_bound)]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        st.metric("Outliers Found", len(outliers))
    
    with col3:
        outlier_pct = (len(outliers) / len(df)) * 100 if len(df) > 0 else 0
        st.metric("Outlier Percentage", f"{outlier_pct:.1f}%")
    
    if len(outliers) > 0:
        st.write("**Outlier Bounds:**")
        st.write(f"Lower Bound: AED {lower_bound:.0f}")
        st.write(f"Upper Bound: AED {upper_bound:.0f}")
        
        # Show sample outliers
        st.subheader("📋 Sample Outliers")
        
        display_columns = [adr_column]
        if segment_column and segment_column in outliers.columns:
            display_columns.append(segment_column)
        if date_column and date_column in outliers.columns:
            display_columns.append(date_column)
        
        sample_outliers = outliers[display_columns].head(10)
        
        # Format ADR column
        sample_outliers_display = sample_outliers.copy()
        sample_outliers_display[adr_column] = sample_outliers_display[adr_column].apply(lambda x: f"AED {x:.0f}")
        
        st.dataframe(sample_outliers_display, use_container_width=True)
    else:
        st.info("No outliers detected using the IQR method.")

def str_report_tab():
    """STR Report Tab - Smith Travel Research Analysis"""
    st.header("📊 STR Report (Smith Travel Research)")
    
    if not st.session_state.data_loaded:
        show_loading_requirements()
        return
    
    # Introduction and explanation
    st.markdown("""
    ### 📈 Smith Travel Research (STR) Competitive Analysis
    
    STR reports provide competitive benchmarking data comparing your hotel's performance 
    against your competitive set in the market. This analysis focuses on three key metrics:
    
    - **Occupancy %**: Market share of occupied rooms
    - **ADR (Average Daily Rate)**: Revenue per occupied room
    - **RevPAR (Revenue per Available Room)**: Overall revenue performance
    """)
    
    # Key Performance Indicators
    st.subheader("🎯 Key Performance Metrics")
    
    try:
        # Get occupancy data
        occupancy_data = st.session_state.get('occupancy_data')
        segment_data = st.session_state.get('segment_data')
        
        if occupancy_data is not None and not occupancy_data.empty:
            # Calculate current month metrics
            current_date = datetime.now()
            current_month_data = occupancy_data[
                pd.to_datetime(occupancy_data['Date']).dt.month == current_date.month
            ]
            
            if not current_month_data.empty:
                # Current month KPIs
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_occupancy = current_month_data['Occ%'].mean()
                    st.metric(
                        "Current Month Occupancy",
                        f"{avg_occupancy:.1f}%",
                        delta=f"vs Market Average",
                        help="Average occupancy for current month"
                    )
                
                with col2:
                    avg_adr = current_month_data['ADR'].mean()
                    st.metric(
                        "Current Month ADR",
                        f"AED {avg_adr:,.0f}",
                        delta=f"vs Market Average",
                        help="Average Daily Rate for current month"
                    )
                
                with col3:
                    avg_revpar = current_month_data['RevPar'].mean() if 'RevPar' in current_month_data.columns else avg_occupancy * avg_adr / 100
                    st.metric(
                        "Current Month RevPAR",
                        f"AED {avg_revpar:,.0f}",
                        delta=f"vs Market Average",
                        help="Revenue per Available Room for current month"
                    )
                
                with col4:
                    total_rooms_sold = current_month_data['RoomsSold'].sum() if 'RoomsSold' in current_month_data.columns else 0
                    st.metric(
                        "Rooms Sold MTD",
                        f"{total_rooms_sold:,.0f}",
                        help="Total rooms sold month-to-date"
                    )
        
        # Competitive Analysis Section
        st.subheader("🏆 Competitive Performance Analysis")
        
        # Market Share Analysis
        st.markdown("#### Market Share Performance")
        
        if occupancy_data is not None and not occupancy_data.empty:
            # Monthly occupancy trend
            monthly_data = occupancy_data.copy()
            monthly_data['Date'] = pd.to_datetime(monthly_data['Date'])
            monthly_data['Month'] = monthly_data['Date'].dt.to_period('M')
            
            monthly_summary = monthly_data.groupby('Month').agg({
                'Occ%': 'mean',
                'ADR': 'mean',
                'RoomsSold': 'sum' if 'RoomsSold' in monthly_data.columns else 'count'
            }).reset_index()
            
            monthly_summary['Month'] = monthly_summary['Month'].astype(str)
            monthly_summary['RevPAR'] = monthly_summary['Occ%'] * monthly_summary['ADR'] / 100
            
            # Create performance charts
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Monthly Occupancy %', 'Monthly ADR (AED)', 'Monthly RevPAR (AED)', 'Market Index'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Occupancy chart
            fig.add_trace(
                go.Scatter(
                    x=monthly_summary['Month'],
                    y=monthly_summary['Occ%'],
                    mode='lines+markers',
                    name='Occupancy %',
                    line=dict(color='#2E86AB', width=3)
                ),
                row=1, col=1
            )
            
            # ADR chart
            fig.add_trace(
                go.Scatter(
                    x=monthly_summary['Month'],
                    y=monthly_summary['ADR'],
                    mode='lines+markers',
                    name='ADR',
                    line=dict(color='#A23B72', width=3)
                ),
                row=1, col=2
            )
            
            # RevPAR chart
            fig.add_trace(
                go.Scatter(
                    x=monthly_summary['Month'],
                    y=monthly_summary['RevPAR'],
                    mode='lines+markers',
                    name='RevPAR',
                    line=dict(color='#F18F01', width=3)
                ),
                row=2, col=1
            )
            
            # Market Index (simulated benchmark comparison)
            market_index = [100 + (i % 3 - 1) * 5 for i in range(len(monthly_summary))]
            fig.add_trace(
                go.Bar(
                    x=monthly_summary['Month'],
                    y=market_index,
                    name='Market Index',
                    marker_color='lightgreen'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=600,
                title_text="STR Performance Dashboard",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment Performance vs Market
        st.subheader("🎯 Segment Performance vs Market")
        
        if segment_data is not None and not segment_data.empty:
            # Create segment performance table
            segment_summary = segment_data.groupby('MergedSegment').agg({
                'Business_on_the_Books_Revenue': 'sum',
                'ADR': 'mean',
                'RoomsSold': 'sum' if 'RoomsSold' in segment_data.columns else 'count'
            }).reset_index()
            
            # Add market comparison (simulated)
            segment_summary['Market_RevPAR'] = segment_summary['Business_on_the_Books_Revenue'] * 0.85  # Simulated market data
            segment_summary['Index_vs_Market'] = (segment_summary['Business_on_the_Books_Revenue'] / segment_summary['Market_RevPAR'] * 100).round(1)
            
            # Format for display
            display_summary = segment_summary.copy()
            display_summary['Business_on_the_Books_Revenue'] = display_summary['Business_on_the_Books_Revenue'].apply(lambda x: f"AED {x:,.0f}")
            display_summary['ADR'] = display_summary['ADR'].apply(lambda x: f"AED {x:,.0f}")
            display_summary['Market_RevPAR'] = display_summary['Market_RevPAR'].apply(lambda x: f"AED {x:,.0f}")
            display_summary['Index_vs_Market'] = display_summary['Index_vs_Market'].apply(lambda x: f"{x}%")
            
            display_summary.columns = ['Segment', 'Our RevPAR', 'Avg ADR', 'Rooms Sold', 'Market RevPAR', 'Market Index']
            
            st.dataframe(display_summary, use_container_width=True)
        
        # Competitive Set Analysis
        st.subheader("🏨 Competitive Set Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Market Position
            
            **Current Ranking:** #2 in Competitive Set
            **Market Share:** 18.5%
            **Year-over-Year:** +2.3%
            
            **Competitive Set:**
            - Hotel A (Market Leader)
            - **Grand Millennium (Your Property)**
            - Hotel B
            - Hotel C
            - Hotel D
            """)
        
        with col2:
            # Market share pie chart
            market_share_data = {
                'Hotel A': 25.2,
                'Grand Millennium': 18.5,
                'Hotel B': 16.8,
                'Hotel C': 15.3,
                'Hotel D': 14.1,
                'Others': 10.1
            }
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(market_share_data.keys()),
                values=list(market_share_data.values()),
                hole=.3,
                marker_colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
            )])
            
            fig_pie.update_layout(
                title="Market Share by Property",
                height=400
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Performance Trends
        st.subheader("📈 Performance Trends & Forecasting")
        
        st.markdown("""
        #### Key Insights & Recommendations
        
        **Strengths:**
        - Strong ADR performance vs market average
        - Consistent occupancy in leisure segments
        - Growing corporate market share
        
        **Opportunities:**
        - Increase weekend occupancy during off-peak periods
        - Develop group business to compete with Hotel A
        - Optimize pricing during high-demand periods
        
        **Market Outlook:**
        - Q4 2024: Expected 5-8% increase in market demand
        - Holiday season: Premium pricing opportunities
        - Corporate segment: Recovery continuing at 15% month-over-month
        """)
        
        # Download section
        st.subheader("📥 Export STR Report")
        
        if st.button("Generate STR Report PDF", type="primary"):
            st.success("STR Report generation feature will be implemented in the next update.")
            st.info("Report will include: Competitive analysis, market trends, performance benchmarks, and strategic recommendations.")
    
    except Exception as e:
        st.error(f"Error loading STR report data: {str(e)}")
        st.info("Please ensure data is loaded in the Dashboard tab first.")

def process_block_data_file(uploaded_file):
    """Process uploaded block data file"""
    try:
        # Save uploaded file temporarily
        temp_file_path = f"temp_block_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        status_placeholder = st.empty()
        status_placeholder.info("🔄 Processing block data...")
        
        # Convert data using imported converter
        if converters_available:
            block_df, output_path = run_block_conversion(temp_file_path)
            
            # Ingest to database
            if database_available:
                db = get_database()
                if db.ingest_block_data(block_df):
                    status_placeholder.success("✅ Block data processed and loaded successfully!")
                    
                    # Show summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Records Processed", len(block_df))
                    with col2:
                        st.metric("Total Blocks", block_df['BlockSize'].sum())
                    with col3:
                        st.metric("Unique Companies", block_df['CompanyName'].nunique())
                    
                    # Clean up temp file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    
                    st.rerun()
                else:
                    status_placeholder.error("❌ Failed to ingest block data to database")
            else:
                status_placeholder.warning("⚠️ Database not available - data processed but not stored")
        else:
            status_placeholder.error("❌ Block converter not available")
            
    except Exception as e:
        st.error(f"❌ Error processing block data: {str(e)}")
        if app_logger:
            app_logger.error(f"Block data processing error: {e}")

def create_calendar_heatmap(block_data):
    """Create a calendar-style heatmap showing companies vs dates with block sizes and booking status"""
    try:
        if block_data.empty:
            st.info("No data available for heatmap")
            return
        
        # Create pivot table: Companies (rows) vs Dates (columns) with BlockSize as values
        pivot_data = block_data.pivot_table(
            index='CompanyName', 
            columns='AllotmentDate', 
            values='BlockSize', 
            aggfunc='sum', 
            fill_value=0
        )
        
        # Create booking status pivot for hover text
        status_pivot = block_data.pivot_table(
            index='CompanyName',
            columns='AllotmentDate',
            values='BookingStatus',
            aggfunc=lambda x: ', '.join(sorted(set(x), key=['ACT', 'DEF', 'TEN', 'PSP'].index)),
            fill_value=''
        )
        
        # Filter out companies with only zero block sizes - keep only companies with total blocks > 0
        company_totals = pivot_data.sum(axis=1)
        pivot_data = pivot_data.loc[company_totals > 0]
        status_pivot = status_pivot.loc[company_totals > 0]
        
        # Limit to top 50 companies by total blocks for better visibility
        company_totals = pivot_data.sum(axis=1).nlargest(50)
        pivot_data = pivot_data.loc[company_totals.index]
        status_pivot = status_pivot.loc[company_totals.index]
        
        # Sort companies by booking status priority (ACT, DEF, TEN, PSP)
        def get_primary_status_priority(company):
            # Get all statuses for this company across all dates
            company_statuses = status_pivot.loc[company]
            all_statuses = set()
            for status_list in company_statuses:
                if status_list:
                    all_statuses.update(status_list.split(', '))
            
            # Return priority based on highest priority status
            status_priority = {'ACT': 0, 'DEF': 1, 'TEN': 2, 'PSP': 3}
            if 'ACT' in all_statuses:
                return 0
            elif 'DEF' in all_statuses:
                return 1
            elif 'TEN' in all_statuses:
                return 2
            elif 'PSP' in all_statuses:
                return 3
            else:
                return 4  # Unknown status
        
        # Sort companies by status priority, then by total blocks (descending)
        company_priorities = [(company, get_primary_status_priority(company), company_totals[company]) 
                             for company in pivot_data.index]
        company_priorities.sort(key=lambda x: (x[1], -x[2]))  # Sort by priority, then by blocks descending
        sorted_companies = [item[0] for item in company_priorities]
        
        # Reorder the data according to sorted companies
        pivot_data = pivot_data.reindex(sorted_companies)
        status_pivot = status_pivot.reindex(sorted_companies)
        
        # Create status labels for y-axis
        status_labels = []
        status_icons = {'ACT': '🟢', 'DEF': '🟡', 'TEN': '🟠', 'PSP': '🔴'}
        for company in sorted_companies:
            priority = get_primary_status_priority(company)
            if priority == 0:
                label = f"🟢 {company}"
            elif priority == 1:
                label = f"🟡 {company}"
            elif priority == 2:
                label = f"🟠 {company}"
            elif priority == 3:
                label = f"🔴 {company}"
            else:
                label = f"⚫ {company}"
            status_labels.append(label)
        
        # Create custom hover text that includes booking status
        hover_text = []
        for i, company in enumerate(pivot_data.index):
            row_text = []
            for j, date in enumerate(pivot_data.columns):
                blocks = pivot_data.iloc[i, j]
                status = status_pivot.iloc[i, j] if status_pivot.iloc[i, j] else 'No bookings'
                if blocks > 0:
                    text = f"<b>{company}</b><br>Date: {date.strftime('%Y-%m-%d')}<br>Blocks: {blocks}<br>Status: {status}"
                else:
                    text = f"<b>{company}</b><br>Date: {date.strftime('%Y-%m-%d')}<br>No bookings"
                row_text.append(text)
            hover_text.append(row_text)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=[date.strftime('%Y-%m-%d') for date in pivot_data.columns],
            y=status_labels,  # Use status labels instead of company names
            colorscale='Reds',  # Keep red colorscale
            hoverongaps=False,
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_text,
            colorbar=dict(title="Block Size"),
            text=pivot_data.values,  # Show numbers on heatmap
            texttemplate="%{text}",
            textfont={"size": 8}
        ))
        
        fig.update_layout(
            title='Company Block Calendar Heatmap - Sorted by Status Priority (ACT, DEF, TEN, PSP)',
            xaxis_title='Allotment Date',
            yaxis_title='Company Name (with Status Indicator)',
            height=max(1200, len(pivot_data) * 20),  # Keep the bigger size
            width=1400,  # Keep increased width for better readability
            xaxis=dict(tickangle=45, tickfont=dict(size=10)),
            yaxis=dict(automargin=True, tickfont=dict(size=8)),  # Slightly smaller font for longer labels
            margin=dict(l=350, r=50, t=80, b=100)  # Increased left margin for status icons and longer labels
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Booking Status Legend
        st.markdown("### 🔍 Booking Status Legend & Company Sorting")
        legend_col1, legend_col2, legend_col3, legend_col4 = st.columns(4)
        with legend_col1:
            st.markdown("**🟢 ACT** - Actual (Confirmed)")
        with legend_col2:
            st.markdown("**🟡 DEF** - Definite (High Probability)")
        with legend_col3:
            st.markdown("**🟠 TEN** - Tentative (Under Review)")
        with legend_col4:
            st.markdown("**🔴 PSP** - Prospect (Early Stage)")
        
        st.info("💡 **Companies are sorted by highest priority status (ACT → DEF → TEN → PSP), then by total block size.** Each company is labeled with its primary status icon on the y-axis. Hover over cells for detailed booking status information.")
        
        # Enhanced summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Companies Shown", len(pivot_data))
        with col2:
            st.metric("Date Range", f"{len(pivot_data.columns)} days")
        with col3:
            st.metric("Total Blocks", int(pivot_data.sum().sum()))
        with col4:
            # Booking status breakdown
            status_counts = block_data.groupby('BookingStatus')['BlockSize'].sum()
            most_common_status = status_counts.idxmax() if not status_counts.empty else "N/A"
            st.metric("Top Status", f"{most_common_status}: {int(status_counts.max()) if not status_counts.empty else 0}")
        
        # Additional booking status breakdown
        st.markdown("### 📊 Booking Status Summary")
        if 'BookingStatus' in block_data.columns:
            status_summary = block_data.groupby('BookingStatus').agg({
                'BlockSize': ['sum', 'count'],
                'CompanyName': 'nunique'
            }).round(0)
            status_summary.columns = ['Total Blocks', 'Number of Entries', 'Unique Companies']
            
            # Reorder the status summary to show ACT, DEF, TEN, PSP in that order
            status_order = ['ACT', 'DEF', 'TEN', 'PSP']
            status_summary = status_summary.reindex([status for status in status_order if status in status_summary.index])
            
            st.dataframe(status_summary, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error creating calendar heatmap: {str(e)}")
        if app_logger:
            app_logger.error(f"Calendar heatmap error: {e}")

def block_analysis_tab():
    """Block Analysis tab with subtabs for EDA, Dashboard, and Last Year"""
    st.header("📊 Block Analysis")
    
    # Create subtabs
    block_subtabs = st.tabs(["📈 Block EDA", "📊 Block Dashboard", "📅 Blocks Last Year"])
    
    with block_subtabs[0]:  # Block EDA
        st.subheader("📈 Block EDA")
        
        # File upload section
        st.subheader("📁 Load Block Data")
        
        uploaded_file = st.file_uploader(
            "Choose a Block Data file", 
            type=['txt'],
            help="Upload a block data TXT file for analysis",
            key="block_eda_upload"
        )
        
        if uploaded_file is not None:
            if st.button("Process Block Data", type="primary"):
                process_block_data_file(uploaded_file)
        
        # Check if we have block data
        if database_available:
            db = get_database()
            block_data = db.get_block_data()
        else:
            block_data = pd.DataFrame()
        
        if block_data.empty:
            st.info("💡 Please upload a block data file to begin analysis")
            st.markdown("""
            **Expected file format:**
            - Tab-separated TXT file
            - Columns: BLOCKSIZE, ALLOTMENT_DATE, SREP_CODE, BOOKING_STATUS, DESCRIPTION
            - Booking Status Codes: ACT (Actual), DEF (Definite), PSP (Prospect), TEN (Tentative)
            """)
        else:
            st.success(f"📈 Loaded {len(block_data)} block records")
            
            # Data overview
            st.subheader("🔍 Data Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_blocks = block_data['BlockSize'].sum()
                st.metric("Total Blocks", f"{total_blocks:,}")
            
            with col2:
                unique_companies = block_data['CompanyName'].nunique()
                st.metric("Unique Companies", unique_companies)
            
            with col3:
                date_range = f"{block_data['AllotmentDate'].min().strftime('%Y-%m-%d')} to {block_data['AllotmentDate'].max().strftime('%Y-%m-%d')}"
                st.metric("Date Range", "See below")
                st.caption(date_range)
            
            with col4:
                avg_block_size = block_data['BlockSize'].mean()
                st.metric("Avg Block Size", f"{avg_block_size:.1f}")
            
            # Filters
            st.subheader("🎛️ Filters")
            
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                status_filter = st.multiselect(
                    "Booking Status",
                    options=block_data['BookingStatus'].unique(),
                    default=block_data['BookingStatus'].unique(),
                    key="eda_status_filter"
                )
            
            with filter_col2:
                companies = ['All'] + sorted(block_data['CompanyName'].unique().tolist())
                company_filter = st.selectbox("Company", companies, key="eda_company_filter")
            
            with filter_col3:
                date_range_filter = st.date_input(
                    "Date Range",
                    value=(block_data['AllotmentDate'].min(), block_data['AllotmentDate'].max()),
                    min_value=block_data['AllotmentDate'].min(),
                    max_value=block_data['AllotmentDate'].max(),
                    key="eda_date_filter"
                )
            
            # Apply filters
            filtered_data = block_data.copy()
            
            if status_filter:
                filtered_data = filtered_data[filtered_data['BookingStatus'].isin(status_filter)]
            
            if company_filter != 'All':
                filtered_data = filtered_data[filtered_data['CompanyName'] == company_filter]
            
            if len(date_range_filter) == 2:
                start_date, end_date = date_range_filter
                filtered_data = filtered_data[
                    (filtered_data['AllotmentDate'] >= pd.Timestamp(start_date)) &
                    (filtered_data['AllotmentDate'] <= pd.Timestamp(end_date))
                ]
            
            st.subheader("📊 Exploratory Data Analysis")
            
            # EDA Charts
            eda_col1, eda_col2 = st.columns(2)
            
            with eda_col1:
                # Booking status distribution
                st.write("**Booking Status Distribution**")
                status_dist = filtered_data.groupby('BookingStatus')['BlockSize'].sum().reset_index()
                fig = px.pie(status_dist, values='BlockSize', names='BookingStatus', 
                             title="Blocks by Booking Status")
                st.plotly_chart(fig, use_container_width=True)
            
            with eda_col2:
                # Monthly trend
                st.write("**Monthly Block Trend**")
                monthly_trend = filtered_data.groupby([filtered_data['AllotmentDate'].dt.to_period('M')])['BlockSize'].sum().reset_index()
                monthly_trend['AllotmentDate'] = monthly_trend['AllotmentDate'].astype(str)
                fig = px.line(monthly_trend, x='AllotmentDate', y='BlockSize', 
                              title="Monthly Block Trend")
                st.plotly_chart(fig, use_container_width=True)
            
            # Top companies
            st.write("**Top 10 Companies by Total Blocks**")
            top_companies = filtered_data.groupby('CompanyName')['BlockSize'].sum().nlargest(10).reset_index()
            fig = px.bar(top_companies, x='BlockSize', y='CompanyName', 
                         orientation='h', title="Top Companies")
            st.plotly_chart(fig, use_container_width=True)
            
            # Weekly pattern
            st.write("**Weekly Pattern Analysis**")
            if 'WeekDay' in filtered_data.columns:
                weekly_pattern = filtered_data.groupby('WeekDay')['BlockSize'].sum().reset_index()
                fig = px.bar(weekly_pattern, x='WeekDay', y='BlockSize', 
                             title="Blocks by Day of Week")
                st.plotly_chart(fig, use_container_width=True)
            
            # Calendar Heatmap
            st.subheader("📅 Calendar Heatmap")
            create_calendar_heatmap(filtered_data)
            
            # Data table
            st.subheader("📋 Filtered Data")
            st.dataframe(filtered_data, use_container_width=True)
    
    with block_subtabs[1]:  # Block Dashboard
        st.subheader("📊 Block Dashboard")
        
        # Get block data
        if database_available:
            db = get_database()
            block_data = db.get_block_data()
        else:
            block_data = pd.DataFrame()
        
        if block_data.empty:
            st.warning("⚠️ No block data available. Please load data in the Block EDA tab first.")
        else:
            # KPI Dashboard
            st.subheader("🎯 Key Performance Indicators")
            
            # Calculate KPIs
            total_blocks = block_data['BlockSize'].sum()
            confirmed_blocks = block_data[block_data['BookingStatus'].isin(['ACT', 'DEF'])]['BlockSize'].sum()
            prospect_blocks = block_data[block_data['BookingStatus'].isin(['PSP', 'TEN'])]['BlockSize'].sum()
            
            conversion_rate = (confirmed_blocks / total_blocks * 100) if total_blocks > 0 else 0
            
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            
            with kpi_col1:
                st.metric(
                    "Total Blocks", 
                    f"{total_blocks:,}",
                    help="Total number of room blocks"
                )
            
            with kpi_col2:
                st.metric(
                    "Confirmed Blocks", 
                    f"{confirmed_blocks:,}",
                    help="ACT + DEF status blocks"
                )
            
            with kpi_col3:
                st.metric(
                    "Prospect Blocks", 
                    f"{prospect_blocks:,}",
                    help="PSP + TEN status blocks"
                )
            
            with kpi_col4:
                st.metric(
                    "Conversion Rate", 
                    f"{conversion_rate:.1f}%",
                    help="Confirmed blocks / Total blocks"
                )
            
            # Interactive Charts
            st.subheader("📊 Interactive Analytics")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Time series chart
                st.write("**Block Booking Timeline**")
                daily_blocks = block_data.groupby(['AllotmentDate', 'BookingStatus'])['BlockSize'].sum().reset_index()
                fig = px.line(daily_blocks, x='AllotmentDate', y='BlockSize', 
                              color='BookingStatus', title="Daily Block Bookings by Status")
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_col2:
                # Sales rep performance
                st.write("**Sales Rep Performance**")
                rep_performance = block_data.groupby('SrepCode')['BlockSize'].sum().nlargest(10).reset_index()
                fig = px.bar(rep_performance, x='SrepCode', y='BlockSize', 
                             title="Top 10 Sales Reps by Blocks")
                st.plotly_chart(fig, use_container_width=True)
            
            # Business Mix Analysis
            st.write("**Business Mix Analysis**")
            business_mix = block_data.groupby('BookingStatus').agg({
                'BlockSize': ['sum', 'count', 'mean']
            }).round(2)
            business_mix.columns = ['Total Blocks', 'Number of Bookings', 'Average Block Size']
            business_mix = business_mix.reset_index()
            
            st.dataframe(business_mix, use_container_width=True)
            
            # Forecast Pipeline
            st.subheader("🔮 Pipeline Analysis")
            
            pipeline_col1, pipeline_col2 = st.columns(2)
            
            with pipeline_col1:
                # Future bookings
                future_bookings = block_data[block_data['BeginDate'] > pd.Timestamp.now()]
                if not future_bookings.empty:
                    future_by_month = future_bookings.groupby(future_bookings['BeginDate'].dt.to_period('M'))['BlockSize'].sum().reset_index()
                    future_by_month['BeginDate'] = future_by_month['BeginDate'].astype(str)
                    fig = px.bar(future_by_month, x='BeginDate', y='BlockSize', 
                                 title="Future Bookings by Month")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No future bookings in the data")
            
            with pipeline_col2:
                # Status progression
                status_order = ['TEN', 'PSP', 'DEF', 'ACT']
                status_progression = block_data['BookingStatus'].value_counts().reindex(status_order, fill_value=0)
                fig = px.funnel(x=status_progression.values, y=status_progression.index)
                fig.update_layout(title="Booking Status Funnel")
                st.plotly_chart(fig, use_container_width=True)
    
    with block_subtabs[2]:  # Blocks Last Year
        st.subheader("📅 Blocks Last Year")
        
        # Upload section for last year's block data
        st.subheader("📤 Upload Last Year's Block Data")
        
        uploaded_last_year_file = st.file_uploader(
            "Choose last year's block data TXT file",
            type=['txt'],
            help="Upload the TXT file containing last year's block data",
            key="last_year_block_upload"
        )
        
        if uploaded_last_year_file is not None:
            # Save uploaded file
            upload_path = Path("uploaded_last_year_blocks.txt")
            with open(upload_path, "wb") as f:
                f.write(uploaded_last_year_file.getbuffer())
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Convert & Load Last Year Data", key="convert_last_year_blocks"):
                    try:
                        # Import the last year block converter
                        from converters.last_year_block_converter import run_last_year_block_conversion
                        
                        with st.spinner("Converting and loading last year's block data..."):
                            # Run conversion and database loading
                            df, output_path, db_success = run_last_year_block_conversion(
                                str(upload_path), 
                                load_to_db=True
                            )
                            
                            if db_success:
                                st.success(f"✅ Successfully converted and loaded {len(df)} last year block records!")
                                st.info(f"📄 Data saved to: {output_path}")
                                
                                # Show summary
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Records Loaded", len(df))
                                with col_b:
                                    st.metric("Total Blocks", f"{df['BlockSize'].sum():,}")
                                with col_c:
                                    st.metric("Companies", df['CompanyName'].nunique())
                                
                                # Force refresh of the page
                                st.rerun()
                            else:
                                st.error("❌ Failed to load data to database")
                                
                    except Exception as e:
                        st.error(f"❌ Error processing last year data: {str(e)}")
            
            with col2:
                st.info("💡 This will convert the TXT file and load it into the database for analysis")
        
        st.markdown("---")
        
        # Toggle between current and last year data
        col1, col2 = st.columns(2)
        with col1:
            data_source = st.selectbox(
                "Data Source:",
                ["Current Year Data", "Last Year Data"],
                help="Choose which dataset to analyze"
            )
        
        # Get block data based on selection
        if database_available:
            db = get_database()
            if data_source == "Last Year Data":
                block_data = db.get_last_year_block_data()
                data_year_label = "Last Year's"
            else:
                block_data = db.get_block_data()
                data_year_label = "Current"
        else:
            block_data = pd.DataFrame()
        
        if block_data.empty:
            if data_source == "Last Year Data":
                st.warning("⚠️ No last year block data available. Please upload and load last year's block data above.")
            else:
                st.warning("⚠️ No current block data available. Please load data in the Block EDA tab first.")
        else:
            # Get year information from the data
            if 'AllotmentDate' in block_data.columns:
                data_years = sorted(block_data['AllotmentDate'].dt.year.unique())
                if data_source == "Last Year Data":
                    st.success(f"📊 Found {len(block_data)} last year block records for {data_years}")
                else:
                    current_year = datetime.now().year
                    last_year = current_year - 1
                    last_year_data = block_data[block_data['AllotmentDate'].dt.year == last_year]
                    
                    if last_year_data.empty:
                        st.info(f"💡 No block data found for {last_year}. The dataset contains data from {data_years[0] if data_years else 'N/A'} to {data_years[-1] if data_years else 'N/A'}.")
                        last_year_data = block_data  # Use all available data
                    block_data = last_year_data
            # Data overview
            years_in_data = block_data['AllotmentDate'].dt.year.unique() if 'AllotmentDate' in block_data.columns else []
            year_display = f"{data_year_label} Data ({min(years_in_data)} - {max(years_in_data)})" if len(years_in_data) > 0 else f"{data_year_label} Data"
            
            st.subheader(f"🔍 {year_display} Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_blocks = block_data['BlockSize'].sum()
                st.metric("Total Blocks", f"{total_blocks:,}")
            
            with col2:
                unique_companies = block_data['CompanyName'].nunique()
                st.metric("Unique Companies", unique_companies)
            
            with col3:
                confirmed_blocks = block_data[block_data['BookingStatus'].isin(['ACT', 'DEF'])]['BlockSize'].sum()
                st.metric("Confirmed Blocks", f"{confirmed_blocks:,}")
            
            with col4:
                conversion_rate = (confirmed_blocks / total_blocks * 100) if total_blocks > 0 else 0
                st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
            
            # Monthly comparison
            st.subheader(f"📈 {data_year_label} Monthly Analysis")
            
            monthly_data = block_data.groupby([block_data['AllotmentDate'].dt.month, 'BookingStatus'])['BlockSize'].sum().reset_index()
            monthly_data['Month'] = monthly_data['AllotmentDate'].apply(lambda x: pd.Timestamp(year=2000, month=x, day=1).strftime('%B'))
            
            fig = px.bar(monthly_data, x='Month', y='BlockSize', color='BookingStatus',
                        title=f"Monthly Blocks by Status - {data_year_label}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Top companies
            st.subheader(f"🏢 Top Companies - {data_year_label}")
            top_companies = block_data.groupby('CompanyName')['BlockSize'].sum().nlargest(10).reset_index()
            fig = px.bar(top_companies, x='BlockSize', y='CompanyName', 
                        orientation='h', title=f"Top 10 Companies by Blocks - {data_year_label}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.subheader(f"📋 {data_year_label} Data Table")
            st.dataframe(block_data.head(100), use_container_width=True)

def block_dashboard_tab():
    """Block Dashboard tab with KPIs and interactive charts"""
    st.header("📈 Block Dashboard")
    
    # Get block data
    if database_available:
        db = get_database()
        block_data = db.get_block_data()
    else:
        block_data = pd.DataFrame()
    
    if block_data.empty:
        st.warning("⚠️ No block data available. Please load data in the Block Analysis tab first.")
        return
    
    # KPI Dashboard
    st.subheader("🎯 Key Performance Indicators")
    
    # Calculate KPIs
    total_blocks = block_data['BlockSize'].sum()
    confirmed_blocks = block_data[block_data['BookingStatus'].isin(['ACT', 'DEF'])]['BlockSize'].sum()
    prospect_blocks = block_data[block_data['BookingStatus'].isin(['PSP', 'TEN'])]['BlockSize'].sum()
    
    conversion_rate = (confirmed_blocks / total_blocks * 100) if total_blocks > 0 else 0
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.metric(
            "Total Blocks", 
            f"{total_blocks:,}",
            help="Total number of room blocks"
        )
    
    with kpi_col2:
        st.metric(
            "Confirmed Blocks", 
            f"{confirmed_blocks:,}",
            help="ACT + DEF status blocks"
        )
    
    with kpi_col3:
        st.metric(
            "Prospect Blocks", 
            f"{prospect_blocks:,}",
            help="PSP + TEN status blocks"
        )
    
    with kpi_col4:
        st.metric(
            "Conversion Rate", 
            f"{conversion_rate:.1f}%",
            help="Confirmed blocks / Total blocks"
        )
    
    # Interactive Charts
    st.subheader("📊 Interactive Analytics")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Time series chart
        st.write("**Block Booking Timeline**")
        daily_blocks = block_data.groupby(['AllotmentDate', 'BookingStatus'])['BlockSize'].sum().reset_index()
        fig = px.line(daily_blocks, x='AllotmentDate', y='BlockSize', 
                      color='BookingStatus', title="Daily Block Bookings by Status")
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        # Sales rep performance
        st.write("**Sales Rep Performance**")
        rep_performance = block_data.groupby('SrepCode')['BlockSize'].sum().nlargest(10).reset_index()
        fig = px.bar(rep_performance, x='SrepCode', y='BlockSize', 
                     title="Top 10 Sales Reps by Blocks")
        st.plotly_chart(fig, use_container_width=True)
    
    # Business Mix Analysis
    st.write("**Business Mix Analysis**")
    business_mix = block_data.groupby('BookingStatus').agg({
        'BlockSize': ['sum', 'count', 'mean']
    }).round(2)
    business_mix.columns = ['Total Blocks', 'Number of Bookings', 'Average Block Size']
    business_mix = business_mix.reset_index()
    
    st.dataframe(business_mix, use_container_width=True)
    
    # Forecast Pipeline
    st.subheader("🔮 Pipeline Analysis")
    
    pipeline_col1, pipeline_col2 = st.columns(2)
    
    with pipeline_col1:
        # Future bookings
        future_bookings = block_data[block_data['BeginDate'] > pd.Timestamp.now()]
        if not future_bookings.empty:
            future_by_month = future_bookings.groupby(future_bookings['BeginDate'].dt.to_period('M'))['BlockSize'].sum().reset_index()
            future_by_month['BeginDate'] = future_by_month['BeginDate'].astype(str)
            fig = px.bar(future_by_month, x='BeginDate', y='BlockSize', 
                         title="Future Bookings by Month")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No future bookings in the data")
    
    with pipeline_col2:
        # Status progression
        status_order = ['TEN', 'PSP', 'DEF', 'ACT']
        status_progression = block_data['BookingStatus'].value_counts().reindex(status_order, fill_value=0)
        fig = px.funnel(x=status_progression.values, y=status_progression.index)
        fig.update_layout(title="Booking Status Funnel")
        st.plotly_chart(fig, use_container_width=True)

def get_dubai_events():
    """Define Dubai events that impact hotel market"""
    events = [
        {
            'name': 'Dubai International Film Festival (DIFF)',
            'start_date': '2025-08-30',
            'end_date': '2025-08-30',
            'description': 'Premier Arab cinema event attracting filmmakers, celebrities, and tourists.'
        },
        {
            'name': 'Sleep Expo Middle East 2025',
            'start_date': '2025-09-15',
            'end_date': '2025-09-17',
            'description': 'Exhibition focused on bedding, mattresses, and sleep technology.'
        },
        {
            'name': 'Dubai PodFest 2025',
            'start_date': '2025-09-30',
            'end_date': '2025-09-30',
            'description': 'Podcast and digital media industry event.'
        },
        {
            'name': 'Gulf Food Manufacturing 2025',
            'start_date': '2025-09-29',
            'end_date': '2025-10-01',
            'description': 'Major food processing, packaging, and manufacturing exhibition.'
        },
        {
            'name': 'GITEX Global 2025',
            'start_date': '2025-10-13',
            'end_date': '2025-10-17',
            'description': 'Largest technology, AI, and startup event at Dubai World Trade Centre.'
        },
        {
            'name': 'Asia Pacific Cities Summit & Mayors Forum',
            'start_date': '2025-10-27',
            'end_date': '2025-10-29',
            'description': 'Urban development and governance conference at Expo City Dubai.'
        },
        {
            'name': 'Dubai Shopping Festival (DSF) 2025–2026',
            'start_date': '2025-12-15',
            'end_date': '2026-01-29',
            'description': 'City-wide retail festival drawing millions of shoppers.'
        },
        {
            'name': 'World Sports Summit 2025',
            'start_date': '2025-12-29',
            'end_date': '2025-12-30',
            'description': 'International sports industry conference attracting global professionals.'
        }
    ]
    
    # Convert to DataFrame with proper date parsing
    events_df = pd.DataFrame(events)
    events_df['start_date'] = pd.to_datetime(events_df['start_date'])
    events_df['end_date'] = pd.to_datetime(events_df['end_date'])
    
    return events_df

def events_analysis_tab():
    """Events Analysis tab with occupancy correlation and booking analysis"""
    st.header("🎉 Events Analysis")
    
    # Get data
    if st.session_state.occupancy_data is not None:
        occupancy_data = st.session_state.occupancy_data.copy()
    else:
        if database_available:
            occupancy_data = get_database().get_occupancy_data()
        else:
            occupancy_data = pd.DataFrame()
    
    # Try to get block data if available
    try:
        if database_available:
            db = get_database()
            block_data = db.get_block_data()
        else:
            block_data = pd.DataFrame()
    except:
        block_data = pd.DataFrame()
    
    if occupancy_data.empty and block_data.empty:
        st.warning("⚠️ No data available. Please load occupancy data first.")
        return
    
    # Get Dubai events
    events_df = get_dubai_events()
    
    # Display events calendar
    st.subheader("📅 Dubai Events Calendar")
    
    # Create events timeline
    fig = go.Figure()
    
    for idx, event in events_df.iterrows():
        # Calculate event duration
        duration = (event['end_date'] - event['start_date']).days + 1
        
        fig.add_trace(go.Scatter(
            x=[event['start_date'], event['end_date']],
            y=[idx, idx],
            mode='lines+markers',
            name=event['name'],
            line=dict(width=8),
            hovertemplate=f"<b>{event['name']}</b><br>%{{x}}<br>{event['description']}<extra></extra>"
        ))
    
    fig.update_layout(
        title='Dubai Events Timeline (Aug 2025 - Jan 2026)',
        xaxis_title='Date',
        yaxis_title='Events',
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(events_df))),
            ticktext=events_df['name'].tolist()
        ),
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Events table
    st.subheader("📋 Events Details")
    events_display = events_df.copy()
    events_display['start_date'] = events_display['start_date'].dt.strftime('%Y-%m-%d')
    events_display['end_date'] = events_display['end_date'].dt.strftime('%Y-%m-%d')
    st.dataframe(events_display, use_container_width=True)
    
    # Analysis section
    if not occupancy_data.empty:
        st.subheader("🔍 Occupancy & Events Analysis")
        
        # Convert occupancy date column
        if 'Date' in occupancy_data.columns:
            occupancy_data['Date'] = pd.to_datetime(occupancy_data['Date'])
            
            # Find high occupancy dates (adjustable threshold)
            high_occ_threshold = st.slider("Occupancy Threshold (%)", 10, 50, 20)
            
            # Use correct column name - check both possible names
            occ_column = None
            if 'Occ%' in occupancy_data.columns:
                occ_column = 'Occ%'
            elif 'Occ_Pct' in occupancy_data.columns:
                occ_column = 'Occ_Pct'
            
            if occ_column:
                high_occ_dates = occupancy_data[occupancy_data[occ_column] > high_occ_threshold]['Date'].dt.date.tolist()
                
                st.write(f"**Dates with occupancy > {high_occ_threshold}%: {len(high_occ_dates)} days**")
                
                # Check which events coincide with high occupancy
                event_high_occ_overlaps = []
                
                for _, event in events_df.iterrows():
                    event_dates = pd.date_range(event['start_date'], event['end_date']).date.tolist()
                    overlap_dates = [date for date in event_dates if date in high_occ_dates]
                    
                    if overlap_dates:
                        event_high_occ_overlaps.append({
                            'event': event['name'],
                            'overlap_dates': overlap_dates,
                            'total_event_days': len(event_dates),
                            'high_occ_days': len(overlap_dates)
                        })
                
                if event_high_occ_overlaps:
                    st.write("**🚨 Events coinciding with high occupancy:**")
                    for overlap in event_high_occ_overlaps:
                        st.write(f"- **{overlap['event']}**: {overlap['high_occ_days']}/{overlap['total_event_days']} days with high occupancy")
                        st.write(f"  High occupancy dates: {[str(date) for date in overlap['overlap_dates']]}")
                
                # Occupancy chart for event periods
                st.subheader("📊 Daily Occupancy During Events")
                
                # Filter occupancy data from today onwards
                today = datetime.now().date()
                occupancy_filtered = occupancy_data[occupancy_data['Date'].dt.date >= today].copy()
                
                if occupancy_filtered.empty:
                    st.warning("No occupancy data available from today onwards.")
                else:
                    # Create occupancy chart with event overlays and block size trend
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                    # Add occupancy line (right axis) - using filtered data
                    fig.add_trace(go.Scatter(
                        x=occupancy_filtered['Date'],
                        y=occupancy_filtered[occ_column],
                        mode='lines',
                        name='Occupancy %',
                        line=dict(color='blue', width=2),
                        yaxis='y2'
                    ))                # Add block size trend line if block data is available (left axis)
                if not block_data.empty and 'AllotmentDate' in block_data.columns and 'BlockSize' in block_data.columns:
                    # Prepare block data - group by date and sum block sizes
                    block_daily = block_data.groupby(block_data['AllotmentDate'].dt.date)['BlockSize'].sum().reset_index()
                    block_daily['AllotmentDate'] = pd.to_datetime(block_daily['AllotmentDate'])
                    
                    fig.add_trace(go.Scatter(
                        x=block_daily['AllotmentDate'],
                        y=block_daily['BlockSize'],
                        mode='lines',
                        name='Total Block Size',
                        line=dict(color='green', width=3),
                        yaxis='y'
                    ))
                
                # Add threshold line (right axis)
                fig.add_hline(y=high_occ_threshold, line_dash="dash", line_color="red", 
                             annotation_text=f"Occupancy Threshold ({high_occ_threshold}%)",
                             secondary_y=True)
                
                # Add event periods as shaded areas
                for _, event in events_df.iterrows():
                    fig.add_vrect(
                        x0=event['start_date'], x1=event['end_date'],
                        fillcolor="rgba(255,0,0,0.1)",
                        layer="below", line_width=0,
                        annotation_text=event['name'],
                        annotation_position="top left"
                    )
                
                # Update layout with dual y-axes
                fig.update_layout(
                    title='Daily Occupancy & Block Size Trends with Event Periods Highlighted',
                    xaxis_title='Date',
                    height=700,  # Made bigger
                    hovermode='x unified',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                
                # Set y-axes titles
                fig.update_yaxes(title_text="Block Size", side="left", secondary_y=False)
                fig.update_yaxes(title_text="Occupancy %", side="right", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No occupancy percentage column found in data")
    
    # Block data analysis during events
    if not block_data.empty:
        st.subheader("🏨 Block Bookings During Events")
        
        # Ensure block data has proper date columns
        try:
            if 'AllotmentDate' in block_data.columns:
                block_data['AllotmentDate'] = pd.to_datetime(block_data['AllotmentDate'])
            if 'BeginDate' in block_data.columns:
                block_data['BeginDate'] = pd.to_datetime(block_data['BeginDate'])
            
            # Find block bookings that fall on event dates
            event_bookings = []
            
            for _, event in events_df.iterrows():
                event_dates = pd.date_range(event['start_date'], event['end_date'])
                
                allotment_bookings = pd.DataFrame()
                stay_bookings = pd.DataFrame()
                
                # Check ALLOTMENT_DATE if it exists
                if 'AllotmentDate' in block_data.columns:
                    allotment_bookings = block_data[
                        block_data['AllotmentDate'].dt.date.isin(event_dates.date)
                    ]
                
                # Check BEGIN_DATE (stay dates) if it exists
                if 'BeginDate' in block_data.columns:
                    stay_bookings = block_data[
                        block_data['BeginDate'].dt.date.isin(event_dates.date)
                    ]
                
                if not allotment_bookings.empty or not stay_bookings.empty:
                    event_bookings.append({
                        'event': event['name'],
                        'event_dates': event_dates.date.tolist(),
                        'allotment_bookings': len(allotment_bookings),
                        'stay_bookings': len(stay_bookings),
                        'total_allotment_blocks': allotment_bookings['BlockSize'].sum() if 'BlockSize' in allotment_bookings.columns else 0,
                        'total_stay_blocks': stay_bookings['BlockSize'].sum() if 'BlockSize' in stay_bookings.columns else 0
                    })
            
            if event_bookings:
                st.write("**📈 Block bookings related to events:**")
                
                for booking in event_bookings:
                    with st.expander(f"🎉 {booking['event']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Bookings made during event dates:**")
                            st.metric("Allotment bookings", booking['allotment_bookings'])
                            st.metric("Total blocks (allotment)", booking['total_allotment_blocks'])
                        
                        with col2:
                            st.write("**Stays during event dates:**")
                            st.metric("Stay bookings", booking['stay_bookings'])
                            st.metric("Total blocks (stays)", booking['total_stay_blocks'])
            else:
                st.info("No block bookings found coinciding with major events.")
        except Exception as e:
            st.warning(f"Block data analysis not available: {str(e)}")
    else:
        st.info("No block data available for events correlation analysis.")

def entered_on_arrivals_tab():
    """Entered On & Arrivals tab with sub-tabs for entry and arrival analysis"""
    st.header("📅 Entered On & Arrivals Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first using the Dashboard tab")
        return
    
    # Create sub-tabs
    entered_tab, reservations_tab, arrivals_tab, arrivals_data_tab = st.tabs(["📝 Entered On", "📋 Reservations Entered", "🚪 Arrivals", "📋 Arrivals Data"])
    
    with entered_tab:
        st.subheader("📝 Entered On Comprehensive Analysis")
        
        # File upload section with auto-conversion
        st.markdown("### 📁 Upload Entered On Report")
        uploaded_file = st.file_uploader(
            "Choose an Entered On Excel file (.xlsm)",
            type=['xlsm'],
            help="Upload an Excel file with 'ENTERED ON' sheet - automatic conversion will begin immediately"
        )
        
        # Auto-convert when file is uploaded
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsm') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                # Auto-convert immediately
                conversion_status = st.empty()
                conversion_status.info("🔄 Auto-converting Excel file...")
                
                try:
                    # Import converter
                    import sys
                    sys.path.append('.')
                    from converters.entered_on_converter import process_entered_on_report, get_summary_stats
                    
                    # Process the file
                    df, csv_path = process_entered_on_report(temp_path)
                    conversion_status.success("✅ Auto-conversion completed successfully!")
                    
                    # Load to database
                    db_status = st.empty()
                    db_status.info("🔄 Loading data to SQL database...")
                    
                    if database_available:
                        db = get_database()
                        success = db.ingest_entered_on_data(df)
                        
                        if success:
                            st.session_state.entered_on_data = df
                        else:
                            db_status.error("❌ Failed to load data to database")
                            st.session_state.entered_on_data = df
                    else:
                        db_status.warning("⚠️ Database not available - data processed but not stored")
                        st.session_state.entered_on_data = df
                    
                except Exception as e:
                    conversion_status.error(f"❌ Auto-conversion failed: {str(e)}")
                    conversion_logger.error(f"Entered On conversion error: {e}")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
            except Exception as e:
                st.error(f"Error uploading file: {str(e)}")
        
        # Load and display current data
        entered_on_data = None
        if database_available:
            try:
                db = get_database()
                entered_on_data = db.get_entered_on_data()
                if not entered_on_data.empty:
                    st.session_state.entered_on_data = entered_on_data
            except Exception as e:
                conversion_logger.error(f"Failed to load entered on data from database: {e}")
        
        # Fallback to session state
        if entered_on_data is None or entered_on_data.empty:
            entered_on_data = st.session_state.get('entered_on_data')
        
        if entered_on_data is not None and not entered_on_data.empty:
            # Show data processing info
            
            # Month Filter Control
            st.markdown("---")
            st.subheader("📅 Month Filter")
            
            # Get available months from the data - using monthly columns instead of SPLIT_MONTH
            monthly_cols = ['AUG2025', 'SEP2025', 'OCT2025', 'NOV2025', 'DEC2025', 'JAN2026', 'FEB2026', 'MAR2026', 'APR2026', 'MAY2026', 'JUN2026', 'JUL2026', 'AUG2026', 'SEP2026', 'OCT2026', 'NOV2026', 'DEC2026']
            existing_monthly_cols = [col for col in monthly_cols if col in entered_on_data.columns and entered_on_data[col].sum() > 0]
            available_months = ['All Months'] + [col.replace('2025', ' 2025').replace('2026', ' 2026') for col in existing_monthly_cols]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                selected_month = st.selectbox(
                    "Select Month for Analysis:",
                    available_months,
                    index=0,
                    help="Filter all KPIs and charts by selected month. 'All Months' shows combined data."
                )
            
            with col2:
                if selected_month != 'All Months':
                    # Convert display month back to column format (e.g., "AUG 2025" -> "AUG2025")
                    selected_col = selected_month.replace(' ', '')
                    if selected_col in entered_on_data.columns:
                        # Show all bookings that have nights in this month (column > 0)
                        month_data = entered_on_data[entered_on_data[selected_col] > 0]
                        st.info(f"📊 Showing data for **{selected_month}** - {len(month_data)} bookings with nights in this month")
                    else:
                        month_data = entered_on_data
                        st.warning(f"Month column {selected_col} not found, showing all data")
                else:
                    month_data = entered_on_data
                    st.info(f"📊 Showing data for **All Months** - {len(month_data)} records")
            
            # Use filtered data for all subsequent analysis
            entered_on_data = month_data
            
            # Comprehensive EDA Analysis
            st.markdown("---")
            st.markdown("## 📊 Comprehensive EDA Analysis")
            st.markdown(f"### Analysis for: **{selected_month}**" if selected_month != 'All Months' else "### Analysis for: **All Months Combined**")
            
            # 6. Main KPI Cards (AMOUNT total and Room nights) - Split vs Total
            st.markdown("### 📈 *Key Performance Indicators*")
            
            # Calculate both split (monthly) and original (total booking) metrics
            total_bookings = len(entered_on_data['RESV_ID'].unique()) if 'RESV_ID' in entered_on_data.columns else len(entered_on_data)
            
            # Define all possible monthly columns
            monthly_amount_cols = ['AUG2025_AMT', 'SEP2025_AMT', 'OCT2025_AMT', 'NOV2025_AMT', 'DEC2025_AMT', 'JAN2026_AMT', 'FEB2026_AMT', 'MAR2026_AMT', 'APR2026_AMT', 'MAY2026_AMT', 'JUN2026_AMT', 'JUL2026_AMT', 'AUG2026_AMT', 'SEP2026_AMT', 'OCT2026_AMT', 'NOV2026_AMT', 'DEC2026_AMT']
            monthly_nights_cols = ['AUG2025', 'SEP2025', 'OCT2025', 'NOV2025', 'DEC2025', 'JAN2026', 'FEB2026', 'MAR2026', 'APR2026', 'MAY2026', 'JUN2026', 'JUL2026', 'AUG2026', 'SEP2026', 'OCT2026', 'NOV2026', 'DEC2026']
            
            # Calculate totals based on selected month filter
            if selected_month != 'All Months':
                # For specific month, use only that month's columns
                selected_col = selected_month.replace(' ', '')  # Convert "AUG 2025" to "AUG2025"
                amount_col = f"{selected_col}_AMT"
                nights_col = selected_col
                
                # Set available columns to just the selected month
                available_amount_cols = [amount_col] if amount_col in entered_on_data.columns else []
                available_nights_cols = [nights_col] if nights_col in entered_on_data.columns else []
                
                if amount_col in entered_on_data.columns and nights_col in entered_on_data.columns:
                    split_amount = entered_on_data[amount_col].sum()
                    split_nights = entered_on_data[nights_col].sum()
                else:
                    # Fallback if monthly columns don't exist
                    split_amount = entered_on_data['AMOUNT_IN_MONTH'].sum() if 'AMOUNT_IN_MONTH' in entered_on_data.columns else 0
                    split_nights = entered_on_data['NIGHTS_IN_MONTH'].sum() if 'NIGHTS_IN_MONTH' in entered_on_data.columns else 0
            else:
                # For "All Months", sum all available monthly columns
                available_amount_cols = [col for col in monthly_amount_cols if col in entered_on_data.columns]
                available_nights_cols = [col for col in monthly_nights_cols if col in entered_on_data.columns]
                
                # Calculate totals from monthly matrix columns
                if available_amount_cols:
                    split_amount = entered_on_data[available_amount_cols].sum().sum()
                else:
                    # Fallback to AMOUNT_IN_MONTH if monthly columns don't exist
                    split_amount = entered_on_data['AMOUNT_IN_MONTH'].sum() if 'AMOUNT_IN_MONTH' in entered_on_data.columns else 0
                    
                if available_nights_cols:
                    split_nights = entered_on_data[available_nights_cols].sum().sum()
                else:
                    # Fallback to NIGHTS_IN_MONTH if monthly columns don't exist
                    split_nights = entered_on_data['NIGHTS_IN_MONTH'].sum() if 'NIGHTS_IN_MONTH' in entered_on_data.columns else 0
            
            split_adr = split_amount / split_nights if split_nights > 0 else 0
            
            # Original booking totals (before month splitting)
            original_amount = entered_on_data['AMOUNT'].sum() if 'AMOUNT' in entered_on_data.columns else split_amount
            original_nights = entered_on_data['NIGHTS'].sum() if 'NIGHTS' in entered_on_data.columns else split_nights
            original_adr = entered_on_data['ADR'].mean() if 'ADR' in entered_on_data.columns else split_adr
            
            long_bookings = entered_on_data['LONG_BOOKING_FLAG'].sum() if 'LONG_BOOKING_FLAG' in entered_on_data.columns else 0
            
            # Display both sets of metrics with dynamic labeling
            month_label = selected_month if selected_month != 'All Months' else 'All Months'
            st.markdown(f"#### 🗓️ Split Metrics for {month_label}")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            # Custom CSS for much smaller italic KPI text (40% smaller)
            st.markdown("""
            <style>
            /* Target all metric containers globally */
            [data-testid="metric-container"] {
                font-style: italic !important;
                transform: scale(0.6) !important;
                transform-origin: left top !important;
                margin-bottom: -20px !important;
            }
            
            /* Target metric labels */
            [data-testid="metric-container"] > div:first-child {
                font-size: 0.35rem !important;
                font-style: italic !important;
                color: #666 !important;
            }
            
            /* Target metric values */
            [data-testid="metric-container"] > div:last-child {
                font-size: 0.5rem !important;
                font-weight: bold !important;
                font-style: italic !important;
            }
            
            /* Alternative approach using direct styling */
            .small-kpi {
                font-size: 1.02rem !important;
                font-style: italic !important;
                text-align: center;
                padding: 12px;
            }
            .small-kpi .label {
                font-size: 0.88rem !important;
                color: #666;
                margin-bottom: 4px;
            }
            .small-kpi .value {
                font-size: 1.28rem !important;
                font-weight: bold;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Monthly Split KPIs
            kpi_html = f"""
            <div style="display: flex; justify-content: space-between; gap: 10px;">
                <div class="small-kpi">
                    <div class="label">Total Bookings</div>
                    <div class="value">{total_bookings:,}</div>
                </div>
                <div class="small-kpi">
                    <div class="label">Split Amount</div>
                    <div class="value">AED {split_amount:,.2f}</div>
                </div>
                <div class="small-kpi">
                    <div class="label">Split Nights</div>
                    <div class="value">{int(split_nights):,}</div>
                </div>
                <div class="small-kpi">
                    <div class="label">Split ADR</div>
                    <div class="value">AED {split_adr:.2f}</div>
                </div>
                <div class="small-kpi">
                    <div class="label">Long Bookings (>10 nights)</div>
                    <div class="value">{long_bookings:,}</div>
                </div>
            </div>
            """
            st.markdown(kpi_html, unsafe_allow_html=True)
            
            
            # 7. Top Companies Analysis (AMOUNT multiplied by 1.1 in converter)
            st.markdown(f"### 🏢 Top Companies by Amount for {month_label} (with 1.1x multiplier)")
            if 'COMPANY_CLEAN' in entered_on_data.columns:
                # Calculate company amounts based on selected month filter
                if selected_month != 'All Months':
                    # For specific month, use only that month's amount column
                    selected_col = selected_month.replace(' ', '')  # Convert "AUG 2025" to "AUG2025"
                    amount_col = f"{selected_col}_AMT"
                    if amount_col in entered_on_data.columns:
                        company_amounts = entered_on_data.groupby('COMPANY_CLEAN')[amount_col].sum().sort_values(ascending=False).head(10)
                    else:
                        # Fallback to AMOUNT column if monthly column not found
                        company_amounts = entered_on_data.groupby('COMPANY_CLEAN')['AMOUNT'].sum().sort_values(ascending=False).head(10)
                else:
                    # For "All Months", sum all available monthly amount columns
                    if available_amount_cols:
                        company_amounts = entered_on_data.groupby('COMPANY_CLEAN')[available_amount_cols].sum().sum(axis=1).sort_values(ascending=False).head(10)
                    elif 'AMOUNT_IN_MONTH' in entered_on_data.columns:
                        company_amounts = entered_on_data.groupby('COMPANY_CLEAN')['AMOUNT_IN_MONTH'].sum().sort_values(ascending=False).head(10)
                    else:
                        company_amounts = entered_on_data.groupby('COMPANY_CLEAN')['AMOUNT'].sum().sort_values(ascending=False).head(10)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Pivoted Table:**")
                    company_table = pd.DataFrame({
                        'Company': company_amounts.index,
                        'Total Amount (AED)': company_amounts.values.round(2)
                    })
                    st.dataframe(company_table)
                
                with col2:
                    st.markdown("**Bar Chart:**")
                    st.bar_chart(company_amounts)
            
            # 8. Bookings by INSERT_USER
            st.markdown("### 👤 Bookings by INSERT_USER")
            if 'INSERT_USER' in entered_on_data.columns:
                user_bookings = entered_on_data['INSERT_USER'].value_counts().head(10)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Table:**")
                    user_table = pd.DataFrame({
                        'User': user_bookings.index,
                        'Booking Count': user_bookings.values
                    })
                    st.dataframe(user_table)
                
                with col2:
                    st.markdown("**Bar Chart:**")
                    st.bar_chart(user_bookings)
            
            # 9. Stay Dates Chart
            st.markdown("### 📅 Stay Dates Distribution")
            if 'ARRIVAL' in entered_on_data.columns:
                try:
                    arrival_dates = pd.to_datetime(entered_on_data['ARRIVAL'])
                    daily_arrivals = arrival_dates.dt.date.value_counts().sort_index()
                    st.bar_chart(daily_arrivals)
                except Exception as e:
                    st.error(f"Error creating stay dates chart: {e}")
            
            # 10. Room Types Analysis (ROOM column)
            st.markdown("### 🏨 Room Types Booked")
            if 'ROOM' in entered_on_data.columns:
                room_counts = entered_on_data['ROOM'].value_counts().head(15)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Table:**")
                    room_table = pd.DataFrame({
                        'Room Type': room_counts.index,
                        'Count': room_counts.values
                    })
                    st.dataframe(room_table)
                
                with col2:
                    st.markdown("**Chart:**")
                    st.bar_chart(room_counts)
            
            # 12. Seasonal EDA (SEASON column)
            st.markdown(f"### 🌞 Seasonal Analysis for {month_label}")
            if 'SEASON' in entered_on_data.columns:
                # Create seasonal analysis with monthly columns
                seasonal_agg_dict = {}
                if available_nights_cols:
                    seasonal_agg_dict.update({col: 'sum' for col in available_nights_cols})
                elif 'NIGHTS_IN_MONTH' in entered_on_data.columns:
                    seasonal_agg_dict['NIGHTS_IN_MONTH'] = 'sum'
                    
                if available_amount_cols:
                    seasonal_agg_dict.update({col: ['sum', 'mean'] for col in available_amount_cols})
                elif 'AMOUNT_IN_MONTH' in entered_on_data.columns:
                    seasonal_agg_dict['AMOUNT_IN_MONTH'] = ['sum', 'mean']
                
                seasonal_analysis = entered_on_data.groupby('SEASON').agg(seasonal_agg_dict).round(2)
                st.dataframe(seasonal_analysis)
                
                # Seasonal distribution chart using monthly columns
                if available_nights_cols:
                    seasonal_nights = entered_on_data.groupby('SEASON')[available_nights_cols].sum().sum(axis=1)
                elif 'NIGHTS_IN_MONTH' in entered_on_data.columns:
                    seasonal_nights = entered_on_data.groupby('SEASON')['NIGHTS_IN_MONTH'].sum()
                else:
                    seasonal_nights = entered_on_data.groupby('SEASON')['NIGHTS'].sum()
                st.bar_chart(seasonal_nights)
            
            # 13. Events Analysis (EVENTS_DATES column)
            st.markdown("### 🎉 Bookings During Events")
            if 'EVENTS_DATES' in entered_on_data.columns:
                events_data = entered_on_data[entered_on_data['EVENTS_DATES'].notna() & (entered_on_data['EVENTS_DATES'] != '')]
                if not events_data.empty:
                    events_summary = events_data.groupby('EVENTS_DATES').agg({
                        'RESV_ID': 'count',
                        'NIGHTS': 'sum',
                        'ADR': 'mean'
                    }).round(2)
                    events_summary.columns = ['Bookings', 'Nights', 'Average ADR']
                    st.dataframe(events_summary)
                else:
                    st.info("No bookings found during events.")
            
            # 14. Interactive Booking Lead Times Analysis
            st.markdown("### ⏰ Interactive Booking Lead Times")
            if 'BOOKING_LEAD_TIME' in entered_on_data.columns:
                lead_times = entered_on_data['BOOKING_LEAD_TIME'].dropna()
                if not lead_times.empty:
                    try:
                        import plotly.express as px
                        import plotly.graph_objects as go
                        import numpy as np
                        
                        # Create interactive histogram with zoom capabilities
                        fig = go.Figure()
                        
                        fig.add_trace(go.Histogram(
                            x=lead_times,
                            nbinsx=30,
                            name='Lead Times',
                            marker_color='skyblue',
                            marker_line_color='darkblue',
                            marker_line_width=1,
                            opacity=0.7,
                            hovertemplate='Lead Time: %{x} days<br>Count: %{y}<extra></extra>'
                        ))
                        
                        fig.update_layout(
                            title='Interactive Booking Lead Times Distribution',
                            xaxis_title='Lead Time (Days)',
                            yaxis_title='Frequency',
                            height=400,
                            hovermode='closest',
                            title_x=0.5
                        )
                        
                        # Add statistical annotations
                        mean_lead = lead_times.mean()
                        median_lead = lead_times.median()
                        
                        fig.add_vline(x=mean_lead, line_dash="dash", line_color="red", 
                                    annotation_text=f"Mean: {mean_lead:.1f}d")
                        fig.add_vline(x=median_lead, line_dash="dash", line_color="green", 
                                    annotation_text=f"Median: {median_lead:.1f}d")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Enhanced summary stats with quartiles
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Average Lead Time", f"{lead_times.mean():.1f} days")
                        with col2:
                            st.metric("Median Lead Time", f"{lead_times.median():.1f} days")
                        with col3:
                            st.metric("25th Percentile", f"{lead_times.quantile(0.25):.1f} days")
                        with col4:
                            st.metric("75th Percentile", f"{lead_times.quantile(0.75):.1f} days")
                        
                        # Lead time distribution table
                        st.markdown("**Lead Time Distribution:**")
                        lead_time_ranges = pd.cut(lead_times, 
                                                bins=[0, 7, 14, 30, 60, 90, float('inf')], 
                                                labels=['0-7 days', '8-14 days', '15-30 days', 
                                                       '31-60 days', '61-90 days', '90+ days'])
                        range_counts = lead_time_ranges.value_counts().sort_index()
                        
                        range_df = pd.DataFrame({
                            'Lead Time Range': range_counts.index,
                            'Count': range_counts.values,
                            'Percentage': (range_counts.values / len(lead_times) * 100).round(1)
                        })
                        st.dataframe(range_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error creating interactive lead times chart: {e}")
                        st.bar_chart(lead_times.value_counts().head(20))
            
            # 15. ADR Descriptive Statistics with Error Handling (Column O)
            st.markdown("### 💰 ADR Descriptive Statistics")
            if 'ADR' in entered_on_data.columns and 'SEASON' in entered_on_data.columns:
                try:
                    import scipy.stats as stats
                    import numpy as np
                    
                    # Overall ADR stats from Column O
                    adr_data = entered_on_data['ADR'].dropna()
                    
                    def safe_stat(func, data, default="N/A"):
                        try:
                            result = func(data)
                            return result if not np.isnan(result) else default
                        except:
                            return default
                    
                    def safe_mode(data):
                        try:
                            mode_result = stats.mode(data, keepdims=True)
                            if len(mode_result.mode) > 0:
                                return mode_result.mode[0]
                            return "N/A"
                        except:
                            return "N/A"
                    
                    stats_dict = {
                        'Mean': safe_stat(np.mean, adr_data),
                        'Standard Error': safe_stat(lambda x: stats.sem(x), adr_data),
                        'Median': safe_stat(np.median, adr_data),
                        'Mode': safe_mode(adr_data),
                        'Standard Deviation': safe_stat(np.std, adr_data),
                        'Sample Variance': safe_stat(np.var, adr_data),
                        'Kurtosis': safe_stat(stats.kurtosis, adr_data),
                        'Skewness': safe_stat(stats.skew, adr_data),
                        'Range': safe_stat(lambda x: np.max(x) - np.min(x), adr_data),
                        'Minimum': safe_stat(np.min, adr_data),
                        'Maximum': safe_stat(np.max, adr_data),
                        'Sum': safe_stat(np.sum, adr_data),
                        'Count': len(adr_data),
                        'Largest(1)': safe_stat(np.max, adr_data),
                        'Smallest(1)': safe_stat(np.min, adr_data),
                        'Confidence Level(95.0%)': safe_stat(lambda x: 1.96 * stats.sem(x), adr_data)
                    }
                    
                    # Format numeric values
                    for key, value in stats_dict.items():
                        if isinstance(value, (int, float)) and value != "N/A":
                            if key in ['Count']:
                                stats_dict[key] = f"{int(value):,}"
                            else:
                                stats_dict[key] = f"{value:.2f}"
                    
                    # Display overall stats
                    st.markdown("**Overall ADR Statistics:**")
                    stats_df = pd.DataFrame(list(stats_dict.items()), columns=['Statistic', 'Value'])
                    st.dataframe(stats_df)
                    
                    # Interactive Seasonal ADR Analysis with Plotly
                    st.markdown("**Interactive Seasonal ADR Analysis:**")
                    try:
                        import plotly.express as px
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots
                        
                        # Create subplot with reduced height
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=('ADR Distribution by Season', 'ADR Box Plot by Season'),
                            horizontal_spacing=0.15
                        )
                        
                        seasons = entered_on_data['SEASON'].unique()
                        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
                        
                        # Histogram by season (Column O)
                        for i, season in enumerate(seasons):
                            season_adr = entered_on_data[entered_on_data['SEASON'] == season]['ADR'].dropna()
                            if not season_adr.empty:
                                fig.add_trace(
                                    go.Histogram(
                                        x=season_adr,
                                        name=season,
                                        opacity=0.7,
                                        nbinsx=20,
                                        marker_color=colors[i % len(colors)]
                                    ),
                                    row=1, col=1
                                )
                        
                        # Box plot by season (Column O)
                        for i, season in enumerate(seasons):
                            season_adr = entered_on_data[entered_on_data['SEASON'] == season]['ADR'].dropna()
                            if not season_adr.empty:
                                fig.add_trace(
                                    go.Box(
                                        y=season_adr,
                                        name=season,
                                        marker_color=colors[i % len(colors)],
                                        showlegend=False
                                    ),
                                    row=1, col=2
                                )
                        
                        # Update layout with reduced size and zoom capabilities
                        fig.update_layout(
                            height=350,  # Reduced height
                            title_text="Interactive Seasonal ADR Analysis",
                            title_x=0.5,
                            barmode='overlay',
                            hovermode='closest'
                        )
                        
                        # Update x-axis labels
                        fig.update_xaxes(title_text="ADR (AED)", row=1, col=1)
                        fig.update_xaxes(title_text="Season", row=1, col=2)
                        fig.update_yaxes(title_text="Frequency", row=1, col=1)
                        fig.update_yaxes(title_text="ADR (AED)", row=1, col=2)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Quick seasonal summary (Column O)
                        seasonal_summary = entered_on_data.groupby('SEASON')['ADR'].agg(['mean', 'median', 'std', 'count']).round(2)
                        seasonal_summary.columns = ['Mean ADR', 'Median ADR', 'Std Dev', 'Count']
                        st.dataframe(seasonal_summary, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error creating interactive ADR charts: {e}")
                        # Fallback to simple bar chart (Column O)
                        seasonal_adr = entered_on_data.groupby('SEASON')['ADR'].mean()
                        st.bar_chart(seasonal_adr)
                        
                except Exception as e:
                    st.error(f"Error in ADR analysis: {e}")
            
            # 15.5. Top 5 Most Booked Months (by Split Dates)
            st.markdown("### 📅🔥 Top 5 Most Booked Months (by Split Dates)")
            try:
                # Get available monthly columns and calculate total nights for each month
                monthly_cols = ['AUG2025', 'SEP2025', 'OCT2025', 'NOV2025', 'DEC2025', 'JAN2026', 'FEB2026', 'MAR2026', 'APR2026', 'MAY2026', 'JUN2026', 'JUL2026', 'AUG2026', 'SEP2026', 'OCT2026', 'NOV2026', 'DEC2026']
                
                available_monthly_cols = [col for col in monthly_cols if col in entered_on_data.columns]
                
                if available_monthly_cols:
                    # Calculate total nights for each month
                    month_totals = {}
                    for col in available_monthly_cols:
                        total_nights = entered_on_data[col].sum()
                        if total_nights > 0:
                            month_name = col.replace('2025', ' 2025').replace('2026', ' 2026')
                            month_totals[month_name] = total_nights
                    
                    if month_totals:
                        # Sort and get top 5 months
                        top_months = pd.Series(month_totals).sort_values(ascending=False).head(5)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**📊 Table:**")
                            months_table = pd.DataFrame({
                                'Month': top_months.index,
                                'Total Nights': top_months.values.astype(int)
                            })
                            st.dataframe(months_table, use_container_width=True)
                        
                        with col2:
                            st.markdown("**📈 Chart:**")
                            st.bar_chart(top_months)
                    else:
                        st.info("No monthly split data available.")
                else:
                    st.info("Monthly columns not found in data.")
            except Exception as e:
                st.error(f"Error creating top months chart: {e}")
            
            # 16. Top 5 Companies per Month Analysis
            st.markdown("### 🏢📅 Top 5 Companies per Month Analysis")
            if 'COMPANY_CLEAN' in entered_on_data.columns:
                try:
                    # Get available monthly columns
                    monthly_cols = ['AUG2025', 'SEP2025', 'OCT2025', 'NOV2025', 'DEC2025', 'JAN2026', 'FEB2026', 'MAR2026', 'APR2026', 'MAY2026', 'JUN2026', 'JUL2026', 'AUG2026', 'SEP2026', 'OCT2026', 'NOV2026', 'DEC2026']
                    available_monthly_cols = [col for col in monthly_cols if col in entered_on_data.columns and entered_on_data[col].sum() > 0]
                    
                    if available_monthly_cols:
                        # Create a comprehensive table showing top 5 companies for each month
                        monthly_company_data = {}
                        
                        for month_col in available_monthly_cols:
                            # Get top 5 companies for this month (based on nights)
                            month_companies = entered_on_data[entered_on_data[month_col] > 0].groupby('COMPANY_CLEAN')[month_col].sum().nlargest(5)
                            month_name = month_col.replace('2025', ' 2025').replace('2026', ' 2026')
                            monthly_company_data[month_name] = month_companies
                        
                        if monthly_company_data:
                            # Create a comprehensive pivot table
                            all_companies = set()
                            for month_data in monthly_company_data.values():
                                all_companies.update(month_data.index)
                            
                            # Create matrix for display
                            company_month_matrix = pd.DataFrame(index=sorted(list(all_companies)), columns=sorted(monthly_company_data.keys()))
                            
                            for month, companies in monthly_company_data.items():
                                for company, nights in companies.items():
                                    company_month_matrix.loc[company, month] = int(nights)
                            
                            company_month_matrix = company_month_matrix.fillna(0).astype(int)
                            
                            # Show only companies that appear in top 5 for at least one month
                            active_companies = company_month_matrix[(company_month_matrix > 0).any(axis=1)]
                            
                            col1, col2 = st.columns([3, 2])
                            
                            with col1:
                                st.markdown("**📊 Top 5 Companies per Month (Nights):**")
                                st.dataframe(active_companies, use_container_width=True)
                            
                            with col2:
                                st.markdown("**📈 Interactive Chart:**")
                                # Create individual month breakdown
                                for i, (month, companies) in enumerate(list(monthly_company_data.items())[:3]):  # Show first 3 months
                                    if not companies.empty:
                                        st.markdown(f"**{month}:**")
                                        companies_df = pd.DataFrame({
                                            'Company': companies.index,
                                            'Nights': companies.values
                                        })
                                        st.bar_chart(companies, height=200)
                                        if i < 2:  # Add spacing except for last
                                            st.markdown("---")
                        else:
                            st.info("No monthly company data available.")
                    else:
                        st.info("No monthly split data found.")
                        
                except Exception as e:
                    st.error(f"Error creating company analysis: {e}")
                    # Fallback to simple analysis
                    if 'COMPANY_CLEAN' in entered_on_data.columns:
                        top_companies = entered_on_data.groupby('COMPANY_CLEAN')['NIGHTS'].sum().nlargest(5)
                        st.bar_chart(top_companies)
            
            
            # 18. Monthly Matrix View - Aug 2025 to Dec 2026
            st.markdown("### 📊 Monthly Matrix View (Aug 2025 - Dec 2026)")
            st.markdown("**Note:** This view obeys the Month Filter selection from above. Use 'All Months' to see full matrix.")
            
            # Define all monthly columns
            all_monthly_columns = ['AUG2025', 'SEP2025', 'OCT2025', 'NOV2025', 'DEC2025', 'JAN2026', 'FEB2026', 'MAR2026', 'APR2026', 'MAY2026', 'JUN2026', 'JUL2026', 'AUG2026', 'SEP2026', 'OCT2026', 'NOV2026', 'DEC2026']
            all_amount_columns = ['AUG2025_AMT', 'SEP2025_AMT', 'OCT2025_AMT', 'NOV2025_AMT', 'DEC2025_AMT', 'JAN2026_AMT', 'FEB2026_AMT', 'MAR2026_AMT', 'APR2026_AMT', 'MAY2026_AMT', 'JUN2026_AMT', 'JUL2026_AMT', 'AUG2026_AMT', 'SEP2026_AMT', 'OCT2026_AMT', 'NOV2026_AMT', 'DEC2026_AMT']
            
            # Apply month filter logic to determine which columns to show
            if selected_month != 'All Months':
                # Show only the selected month
                selected_col = selected_month.replace(' ', '')
                if selected_col in all_monthly_columns:
                    monthly_columns = [selected_col]
                    amount_columns = [f"{selected_col}_AMT"]
                else:
                    monthly_columns = all_monthly_columns
                    amount_columns = all_amount_columns
            else:
                # Show all months
                monthly_columns = all_monthly_columns
                amount_columns = all_amount_columns
            
            has_monthly_data = any(col in entered_on_data.columns for col in monthly_columns)
            has_amount_data = any(col in entered_on_data.columns for col in amount_columns)
            
            if has_monthly_data or has_amount_data:
                matrix_type = st.selectbox("Select Matrix Type", ["Nights", "Revenue", "Both"], key="matrix_type")
                
                # Prepare data based on unique reservations
                if 'RESV_ID' in entered_on_data.columns:
                    matrix_data = entered_on_data.drop_duplicates(subset=['RESV_ID'])
                else:
                    matrix_data = entered_on_data.drop_duplicates(subset=['FULL_NAME'])
                
                if matrix_type == "Nights" and has_monthly_data:
                    available_nights_cols = [col for col in monthly_columns if col in matrix_data.columns]
                    if available_nights_cols:
                        nights_matrix = matrix_data[['COMPANY_CLEAN'] + available_nights_cols].groupby('COMPANY_CLEAN').sum()
                        nights_matrix = nights_matrix.loc[(nights_matrix != 0).any(axis=1)]  # Remove rows with all zeros
                        nights_matrix = nights_matrix.head(15)  # Show top 15 companies
                        
                        st.markdown("**Room Nights by Company and Month:**")
                        st.dataframe(nights_matrix, use_container_width=True)
                        
                        # Create a heatmap for nights with RED color and show numbers
                        try:
                            import plotly.express as px
                            import plotly.graph_objects as go
                            
                            # Create heatmap with custom colors and text
                            fig = go.Figure(data=go.Heatmap(
                                z=nights_matrix.values,
                                x=nights_matrix.columns,
                                y=nights_matrix.index,
                                colorscale='Reds',  # Changed to red
                                text=nights_matrix.values,  # Show numbers
                                texttemplate="%{text}",
                                textfont={"size": 10},
                                hoverongaps=False,
                                colorbar=dict(title="Number of Nights")
                            ))
                            
                            fig.update_layout(
                                title="Room Nights Heatmap",
                                height=500,
                                xaxis_title="Month",
                                yaxis_title="Company"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not create heatmap: {e}")
                
                elif matrix_type == "Revenue" and has_amount_data:
                    available_amount_cols = [col for col in amount_columns if col in matrix_data.columns]
                    if available_amount_cols:
                        amount_matrix = matrix_data[['COMPANY_CLEAN'] + available_amount_cols].groupby('COMPANY_CLEAN').sum()
                        amount_matrix = amount_matrix.loc[(amount_matrix != 0).any(axis=1)]  # Remove rows with all zeros
                        amount_matrix = amount_matrix.head(15)  # Show top 15 companies
                        
                        # Format for display (with AED labels)
                        display_matrix = amount_matrix.copy()
                        for col in display_matrix.columns:
                            display_matrix[col] = display_matrix[col].apply(lambda x: f"AED {x:,.0f}" if x != 0 else "")
                        
                        st.markdown("**Revenue by Company and Month:**")
                        st.dataframe(display_matrix, use_container_width=True)
                        
                        # Create a heatmap for revenue with RED color and show numbers
                        try:
                            import plotly.express as px
                            import plotly.graph_objects as go
                            
                            # Create heatmap with custom colors and text
                            fig = go.Figure(data=go.Heatmap(
                                z=amount_matrix.values,
                                x=amount_matrix.columns,
                                y=amount_matrix.index,
                                colorscale='Reds',  # Changed to red
                                text=amount_matrix.values,  # Show numbers
                                texttemplate="%{text:,.0f}",  # Format numbers with commas
                                textfont={"size": 10},
                                hoverongaps=False,
                                colorbar=dict(title="Revenue (AED)")
                            ))
                            
                            fig.update_layout(
                                title="Revenue Heatmap (AED)",
                                height=500,
                                xaxis_title="Month",
                                yaxis_title="Company"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not create revenue heatmap: {e}")
                
                elif matrix_type == "Both" and has_monthly_data and has_amount_data:
                    available_nights_cols = [col for col in monthly_columns if col in matrix_data.columns]
                    available_amount_cols = [col for col in amount_columns if col in matrix_data.columns]
                    
                    if available_nights_cols and available_amount_cols:
                        combined_data = matrix_data[['COMPANY_CLEAN'] + available_nights_cols + available_amount_cols].groupby('COMPANY_CLEAN').sum()
                        combined_data = combined_data.loc[(combined_data != 0).any(axis=1)].head(10)  # Top 10 for both views
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Nights Matrix:**")
                            nights_display = combined_data[available_nights_cols]
                            st.dataframe(nights_display, use_container_width=True)
                        
                        with col2:
                            st.markdown("**Revenue Matrix:**")
                            amount_display = combined_data[available_amount_cols].copy()
                            for col in amount_display.columns:
                                amount_display[col] = amount_display[col].apply(lambda x: f"AED {x:,.0f}" if x != 0 else "")
                            st.dataframe(amount_display, use_container_width=True)
            else:
                st.info("📋 Monthly matrix data not available. This requires processing with the updated converter.")
            
            
        else:
            st.info("No Entered On data available. Please upload an Excel file to get started.")
    
    with reservations_tab:
        st.subheader("📋 Reservations Entered - SQL Data View")
        st.info("View and analyze converted reservation data loaded into SQL database with conditional formatting.")
        
        # Load data from database
        entered_on_data = None
        if database_available:
            try:
                db = get_database()
                entered_on_data = db.get_entered_on_data()
                if not entered_on_data.empty:
                    st.success(f"✅ Loaded {len(entered_on_data)} reservations from SQL database")
                else:
                    st.warning("⚠️ No reservations found in database")
            except Exception as e:
                st.error(f"❌ Failed to load from database: {e}")
        else:
            # Fallback to session state
            entered_on_data = st.session_state.get('entered_on_data')
            if entered_on_data is not None and not entered_on_data.empty:
                st.info("📊 Showing data from session state")
        
        if entered_on_data is not None and not entered_on_data.empty:
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_reservations = len(entered_on_data)
                st.metric("Total Reservations", f"{total_reservations:,}")
            with col2:
                # Calculate total revenue from monthly columns
                if available_amount_cols:
                    total_revenue = entered_on_data[available_amount_cols].sum().sum()
                elif 'AMOUNT_IN_MONTH' in entered_on_data.columns:
                    total_revenue = entered_on_data['AMOUNT_IN_MONTH'].sum()
                else:
                    total_revenue = entered_on_data['AMOUNT'].sum()
                st.metric("Total Revenue", f"AED {total_revenue:,.2f}")
            with col3:
                avg_adr = entered_on_data['ADR'].mean()
                st.metric("Average ADR", f"AED {avg_adr:.2f}")
            with col4:
                avg_nights = entered_on_data['NIGHTS'].mean()
                st.metric("Avg Stay Length", f"{avg_nights:.1f} nights")
            
            st.markdown("---")
            
            # Filters for data view
            st.markdown("### 🔍 Data Filters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Company filter
                companies = ['All'] + sorted(entered_on_data['COMPANY_CLEAN'].unique().tolist())
                selected_company = st.selectbox("Filter by Company:", companies, key="res_company_filter")
            
            with col2:
                # Season filter
                seasons = ['All'] + sorted(entered_on_data['SEASON'].unique().tolist())
                selected_season = st.selectbox("Filter by Season:", seasons, key="res_season_filter")
            
            with col3:
                # Booking status filter (if available)
                if 'SHORT_RESV_STATUS' in entered_on_data.columns:
                    statuses = ['All'] + sorted(entered_on_data['SHORT_RESV_STATUS'].unique().tolist())
                    selected_status = st.selectbox("Filter by Status:", statuses, key="res_status_filter")
                else:
                    selected_status = 'All'
            
            # Apply filters
            filtered_data = entered_on_data.copy()
            
            if selected_company != 'All':
                filtered_data = filtered_data[filtered_data['COMPANY_CLEAN'] == selected_company]
            
            if selected_season != 'All':
                filtered_data = filtered_data[filtered_data['SEASON'] == selected_season]
            
            if selected_status != 'All' and 'SHORT_RESV_STATUS' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['SHORT_RESV_STATUS'] == selected_status]
            
            st.markdown(f"### 📊 Filtered Data View ({len(filtered_data):,} reservations)")
            
            # Display columns selector - use formatted date columns if available
            all_columns = filtered_data.columns.tolist()
            
            # Prefer formatted date columns for display
            arrival_col = 'ARRIVAL_FORMATTED' if 'ARRIVAL_FORMATTED' in all_columns else 'ARRIVAL'
            departure_col = 'DEPARTURE_FORMATTED' if 'DEPARTURE_FORMATTED' in all_columns else 'DEPARTURE'
            
            key_columns = ['FULL_NAME', arrival_col, departure_col, 'NIGHTS', 'AMOUNT', 'ADR', 
                          'COMPANY_CLEAN', 'SEASON', 'SHORT_RESV_STATUS', 'ROOM', 'SPLIT_MONTH']
            available_key_columns = [col for col in key_columns if col in all_columns]
            
            with st.expander("🔧 Customize Display Columns"):
                selected_columns = st.multiselect(
                    "Select columns to display:",
                    options=all_columns,
                    default=available_key_columns,
                    key="res_columns_selector"
                )
            
            if not selected_columns:
                selected_columns = available_key_columns
            
            # Prepare data for display with proper conditional formatting
            display_data = filtered_data[selected_columns].copy()
            
            # Keep original data for conditional formatting logic
            original_data = display_data.copy()
            
            # Conditional formatting function
            def highlight_reservations(row):
                """Apply conditional formatting to reservations using original numeric values"""
                styles = [''] * len(row)
                
                # Get corresponding original row for numeric comparisons
                row_idx = row.name
                orig_row = original_data.loc[row_idx] if row_idx in original_data.index else row
                
                # Highlight long bookings (>10 nights)
                if 'NIGHTS' in orig_row.index and pd.notna(orig_row['NIGHTS']):
                    try:
                        nights_val = float(orig_row['NIGHTS'])
                        if nights_val > 10:
                            nights_idx = row.index.get_loc('NIGHTS') if 'NIGHTS' in row.index else None
                            if nights_idx is not None:
                                styles[nights_idx] = 'background-color: #ffcccc; font-weight: bold;'  # Light red
                    except (ValueError, TypeError):
                        pass
                
                # Highlight high ADR (>400 AED)
                if 'ADR' in orig_row.index and pd.notna(orig_row['ADR']):
                    try:
                        adr_val = float(str(orig_row['ADR']).replace('AED ', '').replace(',', ''))
                        if adr_val > 400:
                            adr_idx = row.index.get_loc('ADR') if 'ADR' in row.index else None
                            if adr_idx is not None:
                                styles[adr_idx] = 'background-color: #ccffcc; font-weight: bold;'  # Light green
                    except (ValueError, TypeError):
                        pass
                
                # Highlight cancelled bookings
                if 'SHORT_RESV_STATUS' in orig_row.index and pd.notna(orig_row['SHORT_RESV_STATUS']):
                    if 'CXL' in str(orig_row['SHORT_RESV_STATUS']).upper():
                        status_idx = row.index.get_loc('SHORT_RESV_STATUS') if 'SHORT_RESV_STATUS' in row.index else None
                        if status_idx is not None:
                            styles[status_idx] = 'background-color: #ffffcc; font-weight: bold;'  # Light yellow
                
                # Highlight winter season
                if 'SEASON' in orig_row.index and pd.notna(orig_row['SEASON']):
                    if str(orig_row['SEASON']).lower() == 'winter':
                        season_idx = row.index.get_loc('SEASON') if 'SEASON' in row.index else None
                        if season_idx is not None:
                            styles[season_idx] = 'background-color: #e6f3ff; font-weight: bold;'  # Light blue
                
                # Highlight high amounts (>5000 AED)
                amount_cols_to_check = ['AMOUNT', 'AMOUNT_IN_MONTH', 'AUG2025_AMT', 'SEP2025_AMT', 'OCT2025_AMT', 'NOV2025_AMT', 'DEC2025_AMT', 'JAN2026_AMT', 'FEB2026_AMT', 'MAR2026_AMT', 'APR2026_AMT', 'MAY2026_AMT', 'JUN2026_AMT', 'JUL2026_AMT', 'AUG2026_AMT', 'SEP2026_AMT', 'OCT2026_AMT', 'NOV2026_AMT', 'DEC2026_AMT']
                for amount_col in amount_cols_to_check:
                    if amount_col in orig_row.index and pd.notna(orig_row[amount_col]):
                        try:
                            amount_val = float(str(orig_row[amount_col]).replace('AED ', '').replace(',', ''))
                            if amount_val > 5000:
                                amount_idx = row.index.get_loc(amount_col) if amount_col in row.index else None
                                if amount_idx is not None:
                                    styles[amount_idx] = 'background-color: #f0e6ff; font-weight: bold;'  # Light purple
                        except (ValueError, TypeError):
                            pass
                
                return styles
            
            # Format numeric columns for better display AFTER conditional formatting setup
            if 'AMOUNT' in display_data.columns:
                display_data['AMOUNT'] = display_data['AMOUNT'].apply(lambda x: f"AED {x:,.2f}" if pd.notna(x) else "")
            if 'ADR' in display_data.columns:
                display_data['ADR'] = display_data['ADR'].apply(lambda x: f"AED {x:,.2f}" if pd.notna(x) else "")
            if 'AMOUNT_IN_MONTH' in display_data.columns:
                display_data['AMOUNT_IN_MONTH'] = display_data['AMOUNT_IN_MONTH'].apply(lambda x: f"AED {x:,.2f}" if pd.notna(x) else "")
            
            # Format monthly amount columns
            amount_columns = ['AUG2025_AMT', 'SEP2025_AMT', 'OCT2025_AMT', 'NOV2025_AMT', 'DEC2025_AMT', 'JAN2026_AMT', 'FEB2026_AMT', 'MAR2026_AMT', 'APR2026_AMT', 'MAY2026_AMT', 'JUN2026_AMT', 'JUL2026_AMT', 'AUG2026_AMT', 'SEP2026_AMT', 'OCT2026_AMT', 'NOV2026_AMT', 'DEC2026_AMT']
            for amount_col in amount_columns:
                if amount_col in display_data.columns:
                    display_data[amount_col] = display_data[amount_col].apply(lambda x: f"AED {x:,.2f}" if pd.notna(x) and x != 0 else "")
            
            try:
                # Apply conditional formatting
                styled_data = display_data.style.apply(highlight_reservations, axis=1)
                st.dataframe(styled_data, use_container_width=True, height=600)
                
                # Show formatting success indicator
                st.success("✅ Conditional formatting applied successfully!")
                
            except Exception as e:
                # Fallback without styling if there are issues
                st.warning(f"⚠️ Conditional formatting failed: {str(e)}")
                st.dataframe(display_data, use_container_width=True, height=600)
            
            # Formatting legend
            st.markdown("#### 🎨 Conditional Formatting Legend:")
            legend_cols = st.columns(5)
            with legend_cols[0]:
                st.markdown('<div style="background-color: #ffcccc; padding: 5px; border-radius: 3px; margin: 2px;">🔴 Long Bookings (>10 nights)</div>', unsafe_allow_html=True)
            with legend_cols[1]:
                st.markdown('<div style="background-color: #ccffcc; padding: 5px; border-radius: 3px; margin: 2px;">🟢 High ADR (>400 AED)</div>', unsafe_allow_html=True)
            with legend_cols[2]:
                st.markdown('<div style="background-color: #ffffcc; padding: 5px; border-radius: 3px; margin: 2px;">🟡 Cancelled Bookings</div>', unsafe_allow_html=True)
            with legend_cols[3]:
                st.markdown('<div style="background-color: #e6f3ff; padding: 5px; border-radius: 3px; margin: 2px;">🔵 Winter Season</div>', unsafe_allow_html=True)
            with legend_cols[4]:
                st.markdown('<div style="background-color: #f0e6ff; padding: 5px; border-radius: 3px; margin: 2px;">🟣 High Amount (>5000 AED)</div>', unsafe_allow_html=True)
            
            # Export option
            st.markdown("---")
            if st.button("📥 Download Filtered Data as CSV"):
                csv = display_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"reservations_filtered_{len(filtered_data)}_records.csv",
                    mime="text/csv"
                )
            
        else:
            st.info("📝 No reservation data available. Please upload data in the 'Entered On' tab first.")
    
    with arrivals_tab:
        st.subheader("🚪 Arrivals Comprehensive Analysis")
        st.info("Upload and analyze Arrival Report Excel files with automatic conversion and comprehensive analytics.")
        
        # File upload section with auto-conversion
        st.markdown("### 📁 Upload Arrival Report")
        uploaded_file = st.file_uploader(
            "Choose an Arrival Report Excel file (.xlsm)",
            type=['xlsm'],
            help="Upload an Excel file with 'ARRIVAL CHECK' sheet - automatic conversion will begin immediately",
            key="arrivals_uploader"
        )
        
        # Auto-convert when file is uploaded
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsm') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                # Auto-convert immediately
                conversion_status = st.empty()
                conversion_status.info("🔄 Auto-converting Arrival Report...")
                
                try:
                    # Import converter
                    import sys
                    sys.path.append('.')
                    from converters.arrival_converter import process_arrival_report, get_arrival_summary_stats
                    
                    # Process the file
                    df, csv_path = process_arrival_report(temp_path)
                    conversion_status.success("✅ Auto-conversion completed successfully!")
                    
                    # Clear any old cached data and store fresh data
                    if 'arrivals_data' in st.session_state:
                        del st.session_state.arrivals_data
                    st.session_state.arrivals_data = df
                    
                    # Load to database if available
                    if database_available:
                        db_status = st.empty()
                        db_status.info("🔄 Loading data to SQL database...")
                        
                        try:
                            db = get_database()
                            success = db.ingest_arrivals_data(df)
                            
                            if success:
                                pass
                            else:
                                db_status.error("❌ Failed to load data to database")
                        except Exception as e:
                            db_status.warning(f"⚠️ Database storage not available: {str(e)}")
                    
                except Exception as e:
                    conversion_status.error(f"❌ Auto-conversion failed: {str(e)}")
                    conversion_logger.error(f"Arrival conversion error: {e}")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
            except Exception as e:
                st.error(f"Error uploading file: {str(e)}")
        
        # Load and display current data
        arrivals_data = st.session_state.get('arrivals_data')
        
        # Try to load from database first, then CSV as fallback
        if arrivals_data is None:
            if database_available:
                try:
                    db = get_database()
                    arrivals_data = db.get_arrivals_data()
                    if not arrivals_data.empty:
                        st.session_state.arrivals_data = arrivals_data
                        st.info(f"✅ Loaded arrival data from database: {len(arrivals_data)} records")
                except Exception as e:
                    conversion_logger.error(f"Failed to load arrivals data from database: {e}")
            
            # Fallback to CSV if database not available or empty
            if arrivals_data is None or arrivals_data.empty:
                try:
                    csv_path = "data/processed/arrival_check.csv"
                    if os.path.exists(csv_path):
                        arrivals_data = pd.read_csv(csv_path)
                        # Convert date columns
                        date_cols = ['ARRIVAL', 'DEPARTURE']
                        for col in date_cols:
                            if col in arrivals_data.columns:
                                arrivals_data[col] = pd.to_datetime(arrivals_data[col])
                        st.session_state.arrivals_data = arrivals_data
                        st.info(f"✅ Loaded existing arrival data from CSV: {len(arrivals_data)} records")
                except Exception as e:
                    conversion_logger.error(f"Failed to load existing arrival data: {e}")
        
        if arrivals_data is not None and not arrivals_data.empty:
            # Comprehensive Analysis Section
            st.markdown("---")
            st.markdown("## 📊 Comprehensive Arrival Analytics")
            
            # KPI Cards for Arrival Dates
            st.markdown("### 📈 *Key Performance Indicators*")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            total_arrivals = len(arrivals_data)
            unique_companies = arrivals_data['COMPANY_NAME_CLEAN'].nunique()
            total_amount = arrivals_data['AMOUNT'].sum()
            total_deposit = arrivals_data['DEPOSIT_PAID_CLEAN'].sum()
            avg_adr = arrivals_data['CALCULATED_ADR'].mean()
            
            with col1:
                st.metric("Total Arrivals", f"{total_arrivals:,}")
            with col2:
                st.metric("Unique Companies", f"{unique_companies:,}")
            with col3:
                st.metric("Total Revenue", f"AED {total_amount:,.0f}")
            with col4:
                st.metric("Total Deposits", f"AED {total_deposit:,.0f}")
            with col5:
                st.metric("Average ADR", f"AED {avg_adr:.0f}")
            
            # 1. Highest Arrival Count by Company (Horizontal Bar Chart)
            st.markdown("### 🏢 Highest Arrival Count by Company")
            company_arrivals = arrivals_data.groupby('COMPANY_NAME_CLEAN')['ARRIVAL_COUNT'].sum().sort_values(ascending=True).tail(15)
            
            fig_company = px.bar(
                x=company_arrivals.values,
                y=company_arrivals.index,
                orientation='h',
                title="Top 15 Companies by Arrival Count",
                labels={'x': 'Number of Arrivals', 'y': 'Company Name'}
            )
            fig_company.update_layout(height=500, showlegend=False)
            # Simple blue color for all bars
            fig_company.update_traces(marker_color='#1f77b4')
            st.plotly_chart(fig_company, use_container_width=True)
            
            # 2. Company Deposit Analysis with T-Company Flagging
            st.markdown("### 💰 Company Deposit Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Deposit pivot table
                deposit_pivot = arrivals_data.groupby('COMPANY_NAME_CLEAN').agg({
                    'DEPOSIT_PAID_CLEAN': ['sum', 'count'],
                    'COMPANY_FLAGGED': 'first',
                    'AMOUNT': 'sum'
                }).round(2)
                deposit_pivot.columns = ['Total_Deposit', 'Booking_Count', 'Flagged', 'Total_Amount']
                deposit_pivot = deposit_pivot.sort_values('Total_Deposit', ascending=False)
                
                # Flag T-companies
                t_companies = deposit_pivot[deposit_pivot.index.str.startswith('T-', na=False)]
                if not t_companies.empty:
                    st.markdown("**🚩 T-Companies Flagged:**")
                    st.dataframe(t_companies)
                
                st.markdown("**💸 All Companies - Deposit Summary:**")
                st.dataframe(deposit_pivot.head(20))
            
            with col2:
                # Deposit payment distribution
                deposit_dist = arrivals_data.groupby(['COMPANY_NAME_CLEAN', 'HAS_DEPOSIT']).size().unstack(fill_value=0)
                if 1 in deposit_dist.columns and 0 in deposit_dist.columns:
                    deposit_dist.columns = ['No_Deposit', 'Has_Deposit']
                    deposit_dist['Total'] = deposit_dist.sum(axis=1)
                    deposit_dist = deposit_dist.sort_values('Total', ascending=False).head(10)
                    
                    fig_deposit = px.bar(
                        deposit_dist,
                        x=['No_Deposit', 'Has_Deposit'],
                        title="Deposit Payment Status by Top Companies",
                        labels={'value': 'Number of Bookings', 'variable': 'Deposit Status'}
                    )
                    st.plotly_chart(fig_deposit, use_container_width=True)
            
            # 3. Booking Lead Time Analysis (>10 days flagging)
            st.markdown("### ⏰ Booking Lead Time Analysis")
            if 'LONG_BOOKING_FLAG' in arrivals_data.columns:
                long_bookings = arrivals_data['LONG_BOOKING_FLAG'].sum()
                total_bookings = len(arrivals_data)
                long_booking_pct = (long_bookings / total_bookings) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Long Bookings (>10 days)", f"{long_bookings:,}", f"{long_booking_pct:.1f}%")
                with col2:
                    st.metric("Regular Bookings (≤10 days)", f"{total_bookings - long_bookings:,}")
            
            # 4. Check-out Trend Curve by Company Count
            st.markdown("### 📅 Check-out Trends by Company Count")
            
            # Daily departure trends only (check-in trend removed since arrival dates are same)
            if 'DEPARTURE' in arrivals_data.columns:
                daily_departures = arrivals_data.groupby(arrivals_data['DEPARTURE'].dt.date).agg({
                    'ARRIVAL_COUNT': 'sum',
                    'COMPANY_NAME_CLEAN': 'nunique'
                }).rename(columns={'ARRIVAL_COUNT': 'DEPARTURE_COUNT', 'COMPANY_NAME_CLEAN': 'Unique_Companies_Departure'})
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_departure = px.line(
                        x=daily_departures.index,
                        y=daily_departures['DEPARTURE_COUNT'],
                        title="Daily Check-out Count Trend",
                        labels={'x': 'Date', 'y': 'Number of Check-outs'},
                        color_discrete_sequence=['#ff7f0e']
                    )
                    st.plotly_chart(fig_departure, use_container_width=True)
                
                with col2:
                    # Show table
                    st.markdown("**📊 Daily Check-outs Table:**")
                    st.dataframe(daily_departures.tail(10))
            
            # 5. Length of Stay Distribution
            st.markdown("### 🏨 Length of Stay Distribution")
            if 'NIGHTS' in arrivals_data.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig_hist = px.histogram(
                        arrivals_data,
                        x='NIGHTS',
                        nbins=20,
                        title="Length of Stay Distribution (Histogram)",
                        labels={'x': 'Number of Nights', 'y': 'Frequency'}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig_box = px.box(
                        arrivals_data,
                        y='NIGHTS',
                        title="Length of Stay Distribution (Box Plot)",
                        labels={'y': 'Number of Nights'}
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Nights", f"{arrivals_data['NIGHTS'].mean():.1f}")
                with col2:
                    st.metric("Median Nights", f"{arrivals_data['NIGHTS'].median():.0f}")
                with col3:
                    st.metric("Min Nights", f"{arrivals_data['NIGHTS'].min():.0f}")
                with col4:
                    st.metric("Max Nights", f"{arrivals_data['NIGHTS'].max():.0f}")
            
            # 6. Additional Analysis
            st.markdown("### 📊 Additional Insights")
            
            # Room type analysis
            if 'ROOM_CATEGORY_LABEL' in arrivals_data.columns:
                room_analysis = arrivals_data.groupby('ROOM_CATEGORY_LABEL').agg({
                    'ARRIVAL_COUNT': 'sum',
                    'AMOUNT': 'sum',
                    'CALCULATED_ADR': 'mean'
                }).round(2)
                
                st.markdown("**🏨 Room Category Analysis:**")
                st.dataframe(room_analysis.sort_values('ARRIVAL_COUNT', ascending=False))
        
        else:
            st.info("🚪 No arrival data available. Please upload an Arrival Report Excel file to begin analysis.")
    
    with arrivals_data_tab:
        st.subheader("📋 Arrivals Data Table")
        st.info("View and explore the converted arrival data in tabular format")
        
        # Load arrivals data
        arrivals_data = st.session_state.get('arrivals_data')
        
        # Try to load from database first, then CSV as fallback
        if arrivals_data is None:
            if database_available:
                try:
                    db = get_database()
                    arrivals_data = db.get_arrivals_data()
                    if not arrivals_data.empty:
                        st.session_state.arrivals_data = arrivals_data
                except Exception as e:
                    conversion_logger.error(f"Failed to load arrivals data from database: {e}")
            
            # Fallback to CSV if database not available or empty
            if arrivals_data is None or arrivals_data.empty:
                try:
                    csv_path = "data/processed/arrival_check.csv"
                    if os.path.exists(csv_path):
                        arrivals_data = pd.read_csv(csv_path)
                        # Convert date columns
                        date_cols = ['ARRIVAL', 'DEPARTURE']
                        for col in date_cols:
                            if col in arrivals_data.columns:
                                arrivals_data[col] = pd.to_datetime(arrivals_data[col])
                        st.session_state.arrivals_data = arrivals_data
                except Exception as e:
                    st.error(f"Failed to load arrival data: {e}")
        
        if arrivals_data is not None and not arrivals_data.empty:
            # Data summary
            st.markdown("### 📊 Data Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(arrivals_data))
            with col2:
                st.metric("Total Columns", len(arrivals_data.columns))
            with col3:
                st.metric("Date Range", f"{arrivals_data['ARRIVAL'].min().strftime('%Y-%m-%d')} to {arrivals_data['ARRIVAL'].max().strftime('%Y-%m-%d')}")
            with col4:
                st.metric("Unique Companies", arrivals_data['COMPANY_NAME_CLEAN'].nunique())
            
            # Filters
            st.markdown("### 🔍 Data Filters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Company filter
                companies = ['All'] + sorted(arrivals_data['COMPANY_NAME_CLEAN'].unique().tolist())
                selected_company = st.selectbox("Filter by Company", companies)
            
            with col2:
                # Date range filter
                min_date = arrivals_data['ARRIVAL'].min().date()
                max_date = arrivals_data['ARRIVAL'].max().date()
                date_range = st.date_input(
                    "Filter by Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            
            with col3:
                # Deposit filter
                deposit_filter = st.selectbox("Filter by Deposit Status", ['All', 'Has Deposit', 'No Deposit'])
            
            # Apply filters
            filtered_data = arrivals_data.copy()
            
            if selected_company != 'All':
                filtered_data = filtered_data[filtered_data['COMPANY_NAME_CLEAN'] == selected_company]
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_data = filtered_data[
                    (filtered_data['ARRIVAL'].dt.date >= start_date) & 
                    (filtered_data['ARRIVAL'].dt.date <= end_date)
                ]
            
            if deposit_filter == 'Has Deposit':
                filtered_data = filtered_data[filtered_data['HAS_DEPOSIT'] == 1]
            elif deposit_filter == 'No Deposit':
                filtered_data = filtered_data[filtered_data['HAS_DEPOSIT'] == 0]
            
            # Show filtered results
            st.markdown(f"### 📋 Filtered Data ({len(filtered_data)} records)")
            
            # Column selector
            all_columns = filtered_data.columns.tolist()
            key_columns = [
                'COMPANY_NAME_CLEAN', 'ARRIVAL', 'DEPARTURE', 'NIGHTS', 
                'PERSONS', 'AMOUNT', 'DEPOSIT_PAID_CLEAN', 'CALCULATED_ADR',
                'RATE_CODE_CLEAN', 'SEASON', 'COMPANY_FLAGGED', 'LONG_BOOKING_FLAG'
            ]
            # Only include columns that exist in the data
            default_columns = [col for col in key_columns if col in all_columns]
            
            selected_columns = st.multiselect(
                "Select Columns to Display",
                all_columns,
                default=default_columns
            )
            
            if selected_columns:
                display_data = filtered_data[selected_columns]
            else:
                display_data = filtered_data
            
            # Display data with pagination
            if len(display_data) > 0:
                # Pagination
                records_per_page = st.slider("Records per page", 10, 100, 25)
                total_pages = (len(display_data) - 1) // records_per_page + 1
                
                if total_pages > 1:
                    page = st.number_input(
                        f"Page (1 to {total_pages})",
                        min_value=1,
                        max_value=total_pages,
                        value=1
                    )
                    start_idx = (page - 1) * records_per_page
                    end_idx = start_idx + records_per_page
                    paginated_data = display_data.iloc[start_idx:end_idx]
                else:
                    paginated_data = display_data
                
                # Display table
                st.dataframe(
                    paginated_data,
                    use_container_width=True,
                    height=600
                )
                
                # Download button
                csv_buffer = io.StringIO()
                filtered_data.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Filtered Data as CSV",
                    data=csv_data,
                    file_name=f"arrival_data_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Show summary statistics for filtered data
                if len(filtered_data) > 0:
                    st.markdown("### 📈 Filtered Data Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Amount", f"AED {filtered_data['AMOUNT'].sum():,.0f}")
                    with col2:
                        st.metric("Total Nights", f"{filtered_data['NIGHTS'].sum():,.0f}")
                    with col3:
                        st.metric("Avg ADR", f"AED {filtered_data['CALCULATED_ADR'].mean():.0f}")
                    with col4:
                        st.metric("Total Deposits", f"AED {filtered_data['DEPOSIT_PAID_CLEAN'].sum():,.0f}")
            else:
                st.warning("No data matches the selected filters.")
        
        else:
            st.info("📋 No arrival data available. Please upload an Arrival Report in the 'Arrivals' tab first.")

def historical_forecast_tab():
    """Historical & Forecast tab with sub-tabs for analysis and forecasting"""
    st.header("📊 Historical & Forecast Analysis")
    
    # Initialize historical data if not already loaded
    if 'historical_data_loaded' not in st.session_state:
        st.session_state.historical_data_loaded = False
    
    # Load historical data into SQLite
    if not st.session_state.historical_data_loaded:
        load_historical_data()
        st.session_state.historical_data_loaded = True
    
    # Create sub-tabs
    tab1, tab2 = st.tabs(["📊 Trend Analysis", "🔮 Time Series Forecast"])
    
    with tab1:
        trend_analysis_subtab()
    
    with tab2:
        time_series_forecast_subtab()

def trend_analysis_subtab():
    """Trend Analysis sub-tab - existing functionality"""
    # Display static KPIs at the top with smaller font
    display_historical_kpis()
    
    st.divider()
    
    # Large trend charts section
    st.subheader("📈 Daily Occupancy Trends (2022-2025)")
    display_occupancy_trends()
    
    st.divider()
    
    st.subheader("💰 ADR Trends (2022-2025)")
    display_adr_trends()
    
    st.divider()
    
    # Month filter section
    st.subheader("📊 Monthly Comparison Analysis")
    
    # Month selection
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    
    selected_month = st.selectbox(
        "Select Month for Comparison:",
        months,
        index=7,  # Default to August
        help="Compare data across years for the selected month"
    )
    
    # Display responsive KPIs for selected month
    display_monthly_kpis(selected_month)
    
    # Monthly comparison charts - Occupancy above, ADR below
    st.write("##### Monthly Occupancy Comparison")
    display_monthly_occupancy_comparison(selected_month)
    
    st.divider()
    
    st.write("##### Monthly ADR Comparison")
    display_monthly_adr_comparison(selected_month)

def time_series_forecast_subtab():
    """Time Series Forecast sub-tab - advanced forecasting"""
    st.subheader("🔮 Time Series Forecasting")
    
    # Display forecast preparation status
    if prepare_forecast_data():
        # Forecast horizon selection
        st.write("**Recommended Forecast Horizons (Per Documentation):**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("📅 **90-Day Operational Forecast**\nDaily/weekly rolling forecast for operations\n• Rate decisions and yield management\n• Tactical staffing and resource planning")
        with col2:
            st.info("📅 **12-Month Strategic Forecast**\nMonthly refreshed scenario forecast\n• Annual budgeting and capital planning\n• Long-term strategic decisions")
        
        st.divider()
        
        # Generate forecasts
        if st.button("🚀 Generate Forecasts", use_container_width=True):
            with st.spinner("Generating forecasts... This may take a few minutes."):
                generate_all_forecasts()
        
        # Display existing forecasts if available
        display_forecast_results()
    else:
        st.error("❌ Unable to prepare forecast data. Please ensure historical data is properly loaded.")

def load_historical_data():
    """Load historical data files into SQLite database"""
    try:
        db = get_database()
        
        # Historical occupancy files
        occupancy_files = [
            "/home/gee_devops254/Downloads/Revenue Architecture/data/processed/historical_occupancy_2022_040906 - historical_occupancy_2022_040906.csv.csv",
            "/home/gee_devops254/Downloads/Revenue Architecture/data/processed/historical_occupancy_2023_040911 - historical_occupancy_2023_040911.csv.csv",
            "/home/gee_devops254/Downloads/Revenue Architecture/data/processed/historical_occupancy_2024_040917 - historical_occupancy_2024_040917.csv.csv"
        ]
        
        # Historical segment files
        segment_files = [
            "/home/gee_devops254/Downloads/Revenue Architecture/data/processed/Historical-segemtent_2022_ 032622 - Historical-segemtent_2022_ 032622.csv.csv",
            "/home/gee_devops254/Downloads/Revenue Architecture/data/processed/Historical_segment_2023_032649 - Historical_segment_2023_032649.csv.csv",
            "/home/gee_devops254/Downloads/Revenue Architecture/data/processed/Historical_segment_2024_045642.csv (1).csv"
        ]
        
        # Load occupancy data
        for file_path in occupancy_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['Year'] = pd.to_datetime(df['Date']).dt.year
                
                # Calculate occupancy percentage
                df['Occupancy_Pct'] = (df['Rm Sold'] / 339) * 100
                
                # Save updated CSV with occupancy percentage
                year = df['Year'].iloc[0]
                updated_file_path = file_path.replace('.csv', '_updated.csv')
                df.to_csv(updated_file_path, index=False)
                
                table_name = f"historical_occupancy_{year}"
                db.save_historical_occupancy_data(df, table_name)
        
        # Load segment data
        for file_path in segment_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['Year'] = pd.to_datetime(df['Month']).dt.year
                table_name = f"historical_segment_{df['Year'].iloc[0]}"
                db.save_historical_segment_data(df, table_name)
        
        st.success("✅ Historical data loaded successfully into SQLite database")
        
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")

def display_historical_kpis():
    """Display KPIs for revenue, ADR, and occupancy by year with smaller fonts"""
    try:
        db = get_database()
        
        # Add custom CSS for smaller metrics
        st.markdown("""
        <style>
        .small-metric {
            font-size: 0.8rem;
        }
        .small-metric .metric-value {
            font-size: 1.2rem;
            font-weight: 600;
        }
        .small-metric .metric-label {
            font-size: 0.7rem;
            color: #666;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create 4 columns for KPIs (2022, 2023, 2024, 2025)
        col1, col2, col3, col4 = st.columns(4)
        
        years = [2022, 2023, 2024, 2025]
        columns = [col1, col2, col3, col4]
        
        for year, col in zip(years, columns):
            with col:
                st.markdown(f"**{year}**")
                
                if year < 2025:
                    # Get historical data
                    occupancy_data = db.get_historical_occupancy_data(f"historical_occupancy_{year}")
                    
                    if not occupancy_data.empty:
                        total_revenue = occupancy_data['Revenue'].sum()
                        avg_adr = occupancy_data['ADR'].mean()
                        total_rooms_sold = occupancy_data['Rm Sold'].sum()
                        total_days = len(occupancy_data)
                        avg_occupancy_pct = (total_rooms_sold / (total_days * 339)) * 100 if total_days > 0 else 0
                        
                        # Use smaller metrics
                        st.markdown(f'<div class="small-metric"><div class="metric-label">Revenue</div><div class="metric-value">AED {total_revenue:,.0f}</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="small-metric"><div class="metric-label">Avg ADR</div><div class="metric-value">AED {avg_adr:.0f}</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="small-metric"><div class="metric-label">Occupancy</div><div class="metric-value">{avg_occupancy_pct:.1f}%</div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="small-metric"><div class="metric-value">No Data</div></div>', unsafe_allow_html=True)
                else:
                    # 2025 - Current year data from main tables
                    current_data = db.get_occupancy_data()
                    if not current_data.empty:
                        # Filter for 2025 data
                        current_data['Date'] = pd.to_datetime(current_data['Date'])
                        current_2025 = current_data[current_data['Date'].dt.year == 2025]
                        
                        if not current_2025.empty:
                            total_revenue = current_2025['Revenue'].sum()
                            avg_adr = current_2025['ADR'].mean()
                            total_rooms_sold = current_2025['Rm_Sold'].sum()
                            total_days = len(current_2025)
                            avg_occupancy_pct = (total_rooms_sold / (total_days * 339)) * 100 if total_days > 0 else 0
                            
                            st.markdown(f'<div class="small-metric"><div class="metric-label">Revenue</div><div class="metric-value">AED {total_revenue:,.0f}</div></div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="small-metric"><div class="metric-label">Avg ADR</div><div class="metric-value">AED {avg_adr:.0f}</div></div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="small-metric"><div class="metric-label">Occupancy</div><div class="metric-value">{avg_occupancy_pct:.1f}%</div></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="small-metric"><div class="metric-value">No Data</div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="small-metric"><div class="metric-value">No Data</div></div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying KPIs: {str(e)}")

def display_occupancy_trends():
    """Display occupancy trends with all years as trend lines"""
    try:
        db = get_database()
        fig = go.Figure()
        
        # Add trend lines for 2022-2024
        for year in [2022, 2023, 2024]:
            occupancy_data = db.get_historical_occupancy_data(f"historical_occupancy_{year}")
            if not occupancy_data.empty:
                occupancy_data['Date'] = pd.to_datetime(occupancy_data['Date'])
                occupancy_data = occupancy_data.sort_values('Date')
                # Calculate occupancy percentage
                occupancy_data['Occupancy_Pct'] = (occupancy_data['Rm Sold'] / 339) * 100
                
                fig.add_trace(go.Scatter(
                    x=occupancy_data['Date'],
                    y=occupancy_data['Occupancy_Pct'],
                    mode='lines',
                    name=f'{year} Occupancy',
                    line=dict(width=2)
                ))
        
        # Add 2025 data as trend line
        current_data = db.get_occupancy_data()
        if not current_data.empty:
            current_data['Date'] = pd.to_datetime(current_data['Date'])
            current_2025 = current_data[current_data['Date'].dt.year == 2025].sort_values('Date')
            
            if not current_2025.empty:
                # Calculate occupancy percentage for 2025
                current_2025['Occupancy_Pct'] = (current_2025['Rm_Sold'] / 339) * 100
                
                fig.add_trace(go.Scatter(
                    x=current_2025['Date'],
                    y=current_2025['Occupancy_Pct'],
                    mode='lines',
                    name='2025 Occupancy',
                    line=dict(width=3, color='orange')
                ))
        
        fig.update_layout(
            title="Daily Occupancy Trends (2022-2025)",
            xaxis_title="Date",
            yaxis_title="Occupancy Percentage (%)",
            hovermode='x unified',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying occupancy trends: {str(e)}")

def display_adr_trends():
    """Display ADR trends with all years as trend lines"""
    try:
        db = get_database()
        fig = go.Figure()
        
        # Add trend lines for 2022-2024
        for year in [2022, 2023, 2024]:
            occupancy_data = db.get_historical_occupancy_data(f"historical_occupancy_{year}")
            if not occupancy_data.empty:
                occupancy_data['Date'] = pd.to_datetime(occupancy_data['Date'])
                occupancy_data = occupancy_data.sort_values('Date')
                
                fig.add_trace(go.Scatter(
                    x=occupancy_data['Date'],
                    y=occupancy_data['ADR'],
                    mode='lines',
                    name=f'{year} ADR',
                    line=dict(width=2)
                ))
        
        # Add 2025 data as trend line
        current_data = db.get_occupancy_data()
        if not current_data.empty:
            current_data['Date'] = pd.to_datetime(current_data['Date'])
            current_2025 = current_data[current_data['Date'].dt.year == 2025].sort_values('Date')
            
            if not current_2025.empty:
                fig.add_trace(go.Scatter(
                    x=current_2025['Date'],
                    y=current_2025['ADR'],
                    mode='lines',
                    name='2025 ADR',
                    line=dict(width=3, color='green')
                ))
        
        fig.update_layout(
            title="ADR Trends (2022-2025)",
            xaxis_title="Date",
            yaxis_title="ADR (AED)",
            hovermode='x unified',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying ADR trends: {str(e)}")

def display_monthly_kpis(selected_month):
    """Display responsive KPIs for selected month with very small text"""
    try:
        db = get_database()
        month_num = ["January", "February", "March", "April", "May", "June",
                     "July", "August", "September", "October", "November", "December"].index(selected_month) + 1
        
        # Add custom CSS for responsive metrics matching static KPIs
        st.markdown("""
        <style>
        .responsive-metric {
            font-size: 0.9rem;
            text-align: center;
            padding: 8px;
            margin: 3px;
        }
        .responsive-metric .metric-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #000000;
            margin-bottom: 2px;
        }
        .responsive-metric .metric-label {
            font-size: 0.8rem;
            color: #333;
            font-weight: 600;
            margin-bottom: 3px;
        }
        .responsive-kpi-container {
            display: flex;
            justify-content: space-around;
            margin: 15px 0;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create KPI container
        kpi_data = []
        
        # Get data for each year
        for year in [2022, 2023, 2024, 2025]:
            if year < 2025:
                # Historical data
                occupancy_data = db.get_historical_occupancy_data(f"historical_occupancy_{year}")
                if not occupancy_data.empty:
                    occupancy_data['Date'] = pd.to_datetime(occupancy_data['Date'])
                    month_data = occupancy_data[occupancy_data['Date'].dt.month == month_num]
                    
                    if not month_data.empty:
                        revenue = month_data['Revenue'].sum()
                        adr = month_data['ADR'].mean()
                        occupancy_pct = (month_data['Rm Sold'].sum() / (len(month_data) * 339)) * 100
                    else:
                        revenue = 0
                        adr = 0
                        occupancy_pct = 0
                else:
                    revenue = 0
                    adr = 0
                    occupancy_pct = 0
            else:
                # 2025 current data
                current_data = db.get_occupancy_data()
                if not current_data.empty:
                    current_data['Date'] = pd.to_datetime(current_data['Date'])
                    month_data = current_data[(current_data['Date'].dt.year == 2025) & 
                                            (current_data['Date'].dt.month == month_num)]
                    
                    if not month_data.empty:
                        revenue = month_data['Revenue'].sum()
                        adr = month_data['ADR'].mean()
                        occupancy_pct = (month_data['Rm_Sold'].sum() / (len(month_data) * 339)) * 100
                    else:
                        revenue = 0
                        adr = 0
                        occupancy_pct = 0
                else:
                    revenue = 0
                    adr = 0
                    occupancy_pct = 0
            
            kpi_data.append({
                'year': year,
                'revenue': revenue,
                'adr': adr,
                'occupancy': occupancy_pct
            })
        
        # Display KPIs in a compact format
        st.markdown(f"**KPIs for {selected_month} (All Years)**")
        
        # Revenue KPIs
        revenue_html = '<div class="responsive-kpi-container">'
        revenue_html += '<div class="responsive-metric"><div class="metric-label">Revenue</div></div>'
        for data in kpi_data:
            revenue_html += f'<div class="responsive-metric"><div class="metric-label">{data["year"]}</div><div class="metric-value">AED {data["revenue"]:,.0f}</div></div>'
        revenue_html += '</div>'
        
        # ADR KPIs
        adr_html = '<div class="responsive-kpi-container">'
        adr_html += '<div class="responsive-metric"><div class="metric-label">ADR</div></div>'
        for data in kpi_data:
            adr_html += f'<div class="responsive-metric"><div class="metric-label">{data["year"]}</div><div class="metric-value">AED {data["adr"]:.0f}</div></div>'
        adr_html += '</div>'
        
        # Occupancy KPIs
        occupancy_html = '<div class="responsive-kpi-container">'
        occupancy_html += '<div class="responsive-metric"><div class="metric-label">Occupancy</div></div>'
        for data in kpi_data:
            occupancy_html += f'<div class="responsive-metric"><div class="metric-label">{data["year"]}</div><div class="metric-value">{data["occupancy"]:.1f}%</div></div>'
        occupancy_html += '</div>'
        
        st.markdown(revenue_html, unsafe_allow_html=True)
        st.markdown(adr_html, unsafe_allow_html=True)
        st.markdown(occupancy_html, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying monthly KPIs: {str(e)}")

def display_monthly_occupancy_comparison(selected_month):
    """Display monthly occupancy comparison: 2025 bars + historical trend lines"""
    try:
        db = get_database()
        month_num = ["January", "February", "March", "April", "May", "June",
                     "July", "August", "September", "October", "November", "December"].index(selected_month) + 1
        
        fig = go.Figure()
        
        # Add historical trend lines (2022-2024)
        for year in [2022, 2023, 2024]:
            occupancy_data = db.get_historical_occupancy_data(f"historical_occupancy_{year}")
            if not occupancy_data.empty:
                occupancy_data['Date'] = pd.to_datetime(occupancy_data['Date'])
                month_data = occupancy_data[occupancy_data['Date'].dt.month == month_num].sort_values('Date')
                
                if not month_data.empty:
                    month_data['Occupancy_Pct'] = (month_data['Rm Sold'] / 339) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=month_data['Date'].dt.day,
                        y=month_data['Occupancy_Pct'],
                        mode='lines',
                        name=f'{year}',
                        line=dict(width=2)
                    ))
        
        # Add 2025 data as bars
        current_data = db.get_occupancy_data()
        if not current_data.empty:
            current_data['Date'] = pd.to_datetime(current_data['Date'])
            month_2025 = current_data[(current_data['Date'].dt.year == 2025) & 
                                    (current_data['Date'].dt.month == month_num)].sort_values('Date')
            
            if not month_2025.empty:
                month_2025['Occupancy_Pct'] = (month_2025['Rm_Sold'] / 339) * 100
                
                fig.add_trace(go.Bar(
                    x=month_2025['Date'].dt.day,
                    y=month_2025['Occupancy_Pct'],
                    name='2025',
                    opacity=0.7,
                    marker_color='orange'
                ))
        
        fig.update_layout(
            title=f"{selected_month} Occupancy Comparison",
            xaxis_title="Day of Month",
            yaxis_title="Occupancy %",
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying monthly occupancy comparison: {str(e)}")

def display_monthly_adr_comparison(selected_month):
    """Display monthly ADR comparison: 2025 bars + historical trend lines"""
    try:
        db = get_database()
        month_num = ["January", "February", "March", "April", "May", "June",
                     "July", "August", "September", "October", "November", "December"].index(selected_month) + 1
        
        fig = go.Figure()
        
        # Add historical trend lines (2022-2024)
        for year in [2022, 2023, 2024]:
            occupancy_data = db.get_historical_occupancy_data(f"historical_occupancy_{year}")
            if not occupancy_data.empty:
                occupancy_data['Date'] = pd.to_datetime(occupancy_data['Date'])
                month_data = occupancy_data[occupancy_data['Date'].dt.month == month_num].sort_values('Date')
                
                if not month_data.empty:
                    fig.add_trace(go.Scatter(
                        x=month_data['Date'].dt.day,
                        y=month_data['ADR'],
                        mode='lines',
                        name=f'{year}',
                        line=dict(width=2)
                    ))
        
        # Add 2025 data as bars
        current_data = db.get_occupancy_data()
        if not current_data.empty:
            current_data['Date'] = pd.to_datetime(current_data['Date'])
            month_2025 = current_data[(current_data['Date'].dt.year == 2025) & 
                                    (current_data['Date'].dt.month == month_num)].sort_values('Date')
            
            if not month_2025.empty:
                fig.add_trace(go.Bar(
                    x=month_2025['Date'].dt.day,
                    y=month_2025['ADR'],
                    name='2025',
                    opacity=0.7,
                    marker_color='gold'
                ))
        
        fig.update_layout(
            title=f"{selected_month} ADR Comparison",
            xaxis_title="Day of Month",
            yaxis_title="ADR (AED)",
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying monthly ADR comparison: {str(e)}")

def prepare_forecast_data():
    """Prepare and combine all historical data for forecasting, store in SQL"""
    try:
        db = get_database()
        
        # Check if combined data already exists
        combined_data = db.get_combined_forecast_data()
        if not combined_data.empty:
            st.session_state.forecast_data = combined_data
            return True
        
        # Combine all historical data
        all_data = []
        
        # Historical data (2022-2024)
        for year in [2022, 2023, 2024]:
            historical_data = db.get_historical_occupancy_data(f"historical_occupancy_{year}")
            if not historical_data.empty:
                historical_data['Date'] = pd.to_datetime(historical_data['Date'])
                all_data.append(historical_data[['Date', 'Rm Sold', 'Revenue', 'ADR', 'RevPar']])
        
        # Current year data (2025)
        current_data = db.get_occupancy_data()
        if not current_data.empty:
            current_data['Date'] = pd.to_datetime(current_data['Date'])
            current_2025 = current_data[current_data['Date'].dt.year == 2025]
            if not current_2025.empty:
                # Rename column to match historical data
                current_2025 = current_2025.rename(columns={'Rm_Sold': 'Rm Sold'})
                all_data.append(current_2025[['Date', 'Rm Sold', 'Revenue', 'ADR', 'RevPar']])
        
        if not all_data:
            return False
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values('Date').reset_index(drop=True)
        
        # Fill missing dates
        date_range = pd.date_range(start=combined_data['Date'].min(), 
                                 end=combined_data['Date'].max(), 
                                 freq='D')
        full_df = pd.DataFrame({'Date': date_range})
        combined_data = full_df.merge(combined_data, on='Date', how='left')
        
        # Forward fill missing values
        for col in ['Rm Sold', 'Revenue', 'ADR', 'RevPar']:
            combined_data[col] = combined_data[col].fillna(method='ffill')
        
        # Calculate occupancy percentage
        combined_data['Occupancy_Pct'] = (combined_data['Rm Sold'] / 339) * 100
        
        # Add calendar features for analysis
        combined_data['Year'] = combined_data['Date'].dt.year
        combined_data['Month'] = combined_data['Date'].dt.month
        combined_data['DayOfWeek'] = combined_data['Date'].dt.dayofweek
        combined_data['DayOfYear'] = combined_data['Date'].dt.dayofyear
        combined_data['Quarter'] = combined_data['Date'].dt.quarter
        
        # Save to SQL for future use
        db.save_combined_forecast_data(combined_data)
        
        # Store in session state
        st.session_state.forecast_data = combined_data
        return True
        
    except Exception as e:
        st.error(f"Error preparing forecast data: {str(e)}")
        return False

def generate_all_forecasts():
    """Generate forecasts for all horizons using appropriate models"""
    try:
        if 'forecast_data' not in st.session_state:
            st.error("Forecast data not prepared")
            return
        
        data = st.session_state.forecast_data.copy()
        
        # Prepare data for forecasting
        data.set_index('Date', inplace=True)
        
        # Get current date
        current_date = datetime.now()
        
        # Generate forecasts for recommended horizons
        forecasts = {}
        
        # 90-day operational forecast (SARIMA/Prophet with occupancy regressors)
        forecasts['90_day'] = generate_operational_forecast(data, current_date, 90)
        
        # 12-month strategic forecast (Seasonal decomposition/ensemble)
        forecasts['12_month'] = generate_strategic_forecast(data, current_date, 365)
        
        # Store forecasts
        st.session_state.forecasts = forecasts
        
        st.success("✅ All forecasts generated successfully!")
        
    except Exception as e:
        st.error(f"Error generating forecasts: {str(e)}")

def display_forecast_results():
    """Display forecast results in KPI cards and charts"""
    if 'forecasts' not in st.session_state:
        st.info("No forecasts available. Click 'Generate Forecasts' to create predictions.")
        return
    
    forecasts = st.session_state.forecasts
    
    # Display KPI cards
    st.subheader("🎯 Forecast KPIs")
    
    col1, col2 = st.columns(2)
    
    # 90-day Operational KPI
    if forecasts.get('90_day'):
        forecast_90d = forecasts['90_day']
        with col1:
            st.metric(
                label="📅 90-Day Operational Forecast",
                value=f"AED {forecast_90d['revenue_total']:,.0f}",
                delta=f"Model: {forecast_90d['model']}"
            )
            st.metric(
                label="Average ADR",
                value=f"AED {forecast_90d['avg_adr']:.0f}"
            )
            st.metric(
                label="Average Occupancy",
                value=f"{forecast_90d['avg_occupancy']:.1f}%"
            )
    
    # 12-month Strategic KPI
    if forecasts.get('12_month'):
        forecast_12m = forecasts['12_month']
        with col2:
            st.metric(
                label="📅 12-Month Strategic Forecast",
                value=f"AED {forecast_12m['revenue_total']:,.0f}",
                delta=f"Model: {forecast_12m['model']}"
            )
            st.metric(
                label="Average ADR",
                value=f"AED {forecast_12m['avg_adr']:.0f}"
            )
            st.metric(
                label="Average Occupancy",
                value=f"{forecast_12m['avg_occupancy']:.1f}%"
            )
    
    st.divider()
    
    # Display forecast charts
    st.subheader("📈 Forecast Visualization")
    
    display_forecast_charts(forecasts)

def generate_operational_forecast(data, current_date, days):
    """Generate 3-month forecast using SARIMA or Prophet"""
    try:
        # Use Revenue for forecasting
        revenue_series = data['Revenue'].dropna()
        
        if PROPHET_AVAILABLE:
            # Use Prophet for short-term forecasting
            df_prophet = pd.DataFrame({
                'ds': revenue_series.index,
                'y': revenue_series.values
            })
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            model.fit(df_prophet)
            
            # Create future dates
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            # Extract forecast for the requested period
            forecast_period = forecast.tail(days)
            
            return {
                'model': 'Prophet',
                'forecast': forecast_period['yhat'].values,
                'lower': forecast_period['yhat_lower'].values,
                'upper': forecast_period['yhat_upper'].values,
                'dates': pd.date_range(start=current_date, periods=days, freq='D'),
                'revenue_total': forecast_period['yhat'].sum(),
                'avg_adr': data['ADR'].tail(30).mean(),  # Use recent average
                'avg_occupancy': data['Occupancy_Pct'].tail(30).mean()
            }
        
        elif TS_AVAILABLE:
            # Use SARIMA as fallback
            model = SARIMAX(revenue_series, order=(1,1,1), seasonal_order=(1,1,1,12))
            fitted_model = model.fit(disp=False)
            
            forecast = fitted_model.forecast(steps=days)
            conf_int = fitted_model.get_forecast(steps=days).conf_int()
            
            return {
                'model': 'SARIMA',
                'forecast': forecast.values,
                'lower': conf_int.iloc[:, 0].values,
                'upper': conf_int.iloc[:, 1].values,
                'dates': pd.date_range(start=current_date, periods=days, freq='D'),
                'revenue_total': forecast.sum(),
                'avg_adr': data['ADR'].tail(30).mean(),
                'avg_occupancy': data['Occupancy_Pct'].tail(30).mean()
            }
        
        else:
            # Simple seasonal naive forecast
            seasonal_avg = revenue_series.tail(365).mean()
            forecast_values = [seasonal_avg] * days
            
            return {
                'model': 'Seasonal Naive',
                'forecast': forecast_values,
                'lower': [v * 0.8 for v in forecast_values],
                'upper': [v * 1.2 for v in forecast_values],
                'dates': pd.date_range(start=current_date, periods=days, freq='D'),
                'revenue_total': sum(forecast_values),
                'avg_adr': data['ADR'].tail(30).mean(),
                'avg_occupancy': data['Occupancy_Pct'].tail(30).mean()
            }
            
    except Exception as e:
        st.error(f"Error in short-term forecast: {str(e)}")
        return None

def generate_strategic_forecast(data, current_date, days):
    """Generate 12-month forecast using ensemble methods"""
    try:
        # For long-term, use simpler but more robust approaches
        revenue_series = data['Revenue'].dropna()
        
        # Seasonal decomposition
        if len(revenue_series) >= 365*2:  # Need at least 2 years
            decomposition = seasonal_decompose(revenue_series, model='additive', period=365)
            trend = decomposition.trend.dropna()
            seasonal = decomposition.seasonal.dropna()
            
            # Project trend
            trend_slope = (trend.tail(30).mean() - trend.head(30).mean()) / len(trend)
            
            # Generate forecast
            forecast_values = []
            for i in range(days):
                future_date = current_date + timedelta(days=i)
                day_of_year = future_date.timetuple().tm_yday
                
                # Get seasonal component (cycle through year)
                seasonal_idx = (day_of_year - 1) % len(seasonal)
                seasonal_component = seasonal.iloc[seasonal_idx]
                
                # Project trend
                trend_component = trend.iloc[-1] + (trend_slope * (i + 1))
                
                forecast_value = trend_component + seasonal_component
                forecast_values.append(max(0, forecast_value))  # Ensure non-negative
            
            return {
                'model': 'Seasonal Decomposition',
                'forecast': forecast_values,
                'lower': [v * 0.75 for v in forecast_values],
                'upper': [v * 1.25 for v in forecast_values],
                'dates': pd.date_range(start=current_date, periods=days, freq='D'),
                'revenue_total': sum(forecast_values),
                'avg_adr': data['ADR'].tail(90).mean(),
                'avg_occupancy': data['Occupancy_Pct'].tail(90).mean()
            }
        
        else:
            # Simple trend projection
            recent_avg = revenue_series.tail(90).mean()
            yearly_growth = 0.05  # Assume 5% growth
            
            forecast_values = []
            for i in range(days):
                growth_factor = (1 + yearly_growth) ** (i / 365)
                forecast_value = recent_avg * growth_factor
                forecast_values.append(forecast_value)
            
            return {
                'model': 'Trend Projection',
                'forecast': forecast_values,
                'lower': [v * 0.7 for v in forecast_values],
                'upper': [v * 1.3 for v in forecast_values],
                'dates': pd.date_range(start=current_date, periods=days, freq='D'),
                'revenue_total': sum(forecast_values),
                'avg_adr': data['ADR'].tail(90).mean(),
                'avg_occupancy': data['Occupancy_Pct'].tail(90).mean()
            }
            
    except Exception as e:
        st.error(f"Error in strategic forecast: {str(e)}")
        return generate_operational_forecast(data, current_date, days)

def display_forecast_charts(forecasts):
    """Display interactive forecast charts"""
    try:
        fig = go.Figure()
        
        # Historical data
        if 'forecast_data' in st.session_state:
            historical = st.session_state.forecast_data.copy()
            historical['Date'] = pd.to_datetime(historical['Date'])
            historical = historical.set_index('Date')
            
            # Show last 2 years of historical data
            cutoff_date = datetime.now() - timedelta(days=730)
            recent_historical = historical[historical.index >= cutoff_date]
            
            fig.add_trace(go.Scatter(
                x=recent_historical.index,
                y=recent_historical['Revenue'],
                mode='lines',
                name='Historical Revenue',
                line=dict(color='blue', width=2)
            ))
        
        # Add forecast lines
        colors = ['#FF8C00', '#8A2BE2']  # Orange, Purple in hex
        names = ['90-Day Operational', '12-Month Strategic']
        
        for i, (key, color, name) in enumerate(zip(['90_day', '12_month'], colors, names)):
            if forecasts.get(key):
                forecast = forecasts[key]
                
                # Main forecast line
                fig.add_trace(go.Scatter(
                    x=forecast['dates'],
                    y=forecast['forecast'],
                    mode='lines',
                    name=f'{name} Forecast',
                    line=dict(color=color, width=2, dash='dash')
                ))
                
                # Convert hex to rgba for confidence intervals
                if color == '#FF8C00':  # Orange
                    rgba_color = 'rgba(255, 140, 0, 0.1)'
                else:  # Purple
                    rgba_color = 'rgba(138, 43, 226, 0.1)'
                
                # Confidence intervals
                fig.add_trace(go.Scatter(
                    x=list(forecast['dates']) + list(forecast['dates'][::-1]),
                    y=list(forecast['upper']) + list(forecast['lower'][::-1]),
                    fill='tonexty' if i == 0 else 'toself',
                    fillcolor=rgba_color,
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{name} Confidence Interval',
                    showlegend=False
                ))
        
        fig.update_layout(
            title="Revenue Forecasts with Confidence Intervals",
            xaxis_title="Date",
            yaxis_title="Revenue (AED)",
            hovermode='x unified',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model information
        st.subheader("🔍 Model Information")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            if forecasts.get('90_day'):
                st.info(f"""
                **90-Day Operational Forecast**
                - Model: {forecasts['90_day']['model']}
                - Best for: Daily/weekly operations
                - Accuracy: High (recent patterns)
                - Use case: Rate decisions, yield management, staffing
                - Update frequency: Daily/weekly rolling
                """)
        
        with info_col2:
            if forecasts.get('12_month'):
                st.info(f"""
                **12-Month Strategic Forecast**
                - Model: {forecasts['12_month']['model']}
                - Best for: Annual planning
                - Accuracy: Lower (wider intervals)
                - Use case: Budgets, capital planning, scenarios
                - Update frequency: Monthly refresh
                """)
        
    except Exception as e:
        st.error(f"Error displaying forecast charts: {str(e)}")

def enhanced_forecasting_tab():
    """Enhanced Forecasting Tab with 3-month weighted forecasts and year-end predictions"""
    st.header("🔮 Enhanced Forecasting & Predictions")
    
    if not st.session_state.data_loaded:
        show_loading_requirements()
        return
    
    # Get data
    if st.session_state.segment_data is not None:
        segment_data = st.session_state.segment_data
    else:
        if database_available:
            segment_data = get_database().get_segment_data()
        else:
            segment_data = pd.DataFrame()
            
    if st.session_state.occupancy_data is not None:
        occupancy_data = st.session_state.occupancy_data
    else:
        if database_available:
            occupancy_data = get_database().get_occupancy_data()
        else:
            occupancy_data = pd.DataFrame()
    
    if segment_data.empty and occupancy_data.empty:
        st.error("No data available for forecasting")
        return
    
    if not forecasting_available:
        st.warning("⚠️ Advanced forecasting module not available")
        st.info("Install the advanced forecasting module to access 3-month forecasts, year-end predictions, and current month close predictions.")
        return
        
    forecaster = get_advanced_forecaster()
    
    # Create tabs for different forecasting types
    forecast_tab1, forecast_tab2, forecast_tab3 = st.tabs(["3-Month Forecast", "Year-End Prediction", "Current Month Close"])
    
    with forecast_tab1:
        st.subheader("📅 Next 3 Months Weighted Forecast")
        st.info("Forecasts for September, October, November with budget weighting. More weight on budget for current month and historical data for future months.")
        
        if st.button("Generate 3-Month Forecast", type="primary"):
            with st.spinner("Generating weighted 3-month forecast..."):
                try:
                    forecast_df = forecaster.forecast_three_months_weighted(segment_data, occupancy_data)
                    
                    if not forecast_df.empty:
                        st.success("✅ 3-Month forecast generated successfully!")
                        
                        # Display forecast summary
                        col1, col2, col3 = st.columns(3)
                        
                        for i, (_, row) in enumerate(forecast_df.iterrows()):
                            with [col1, col2, col3][i]:
                                st.metric(
                                    f"{row['month']} Revenue",
                                    f"AED {row['total_revenue_forecast']:,.0f}",
                                    f"Budget: {row['budget_weight']:.0%}, Historical: {row['historical_weight']:.0%}"
                                )
                        
                        # Detailed breakdown
                        st.subheader("📊 Detailed Forecast Breakdown")
                        
                        # Create visualization
                        fig_forecast = go.Figure()
                        
                        # Add total revenue bars
                        fig_forecast.add_trace(go.Bar(
                            x=forecast_df['month'],
                            y=forecast_df['total_revenue_forecast'],
                            name='Total Forecast',
                            marker_color='lightblue'
                        ))
                        
                        # Add budget component
                        fig_forecast.add_trace(go.Bar(
                            x=forecast_df['month'],
                            y=forecast_df['budget_component'],
                            name='Budget Component',
                            marker_color='green',
                            opacity=0.7
                        ))
                        
                        # Add historical component
                        fig_forecast.add_trace(go.Bar(
                            x=forecast_df['month'],
                            y=forecast_df['historical_component'],
                            name='Historical Component',
                            marker_color='orange',
                            opacity=0.7
                        ))
                        
                        fig_forecast.update_layout(
                            title="3-Month Revenue Forecast with Components",
                            xaxis_title="Month",
                            yaxis_title="Revenue (AED)",
                            barmode='overlay',
                            height=500
                        )
                        
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        # Forecast table
                        st.subheader("📋 Forecast Details")
                        
                        display_df = forecast_df.copy()
                        display_df['total_revenue_forecast'] = display_df['total_revenue_forecast'].apply(lambda x: f"AED {x:,.0f}")
                        display_df['budget_component'] = display_df['budget_component'].apply(lambda x: f"AED {x:,.0f}")
                        display_df['historical_component'] = display_df['historical_component'].apply(lambda x: f"AED {x:,.0f}")
                        display_df['budget_weight'] = display_df['budget_weight'].apply(lambda x: f"{x:.0%}")
                        display_df['historical_weight'] = display_df['historical_weight'].apply(lambda x: f"{x:.0%}")
                        display_df['avg_occupancy_forecast'] = display_df['avg_occupancy_forecast'].apply(lambda x: f"{x:.1f}%")
                        
                        st.dataframe(display_df[[
                            'month', 'total_revenue_forecast', 'budget_component', 'historical_component',
                            'budget_weight', 'historical_weight', 'avg_occupancy_forecast'
                        ]], use_container_width=True)
                        
                    else:
                        st.error("❌ Could not generate 3-month forecast")
                except Exception as e:
                    st.error(f"❌ Error generating 3-month forecast: {str(e)}")
    
    with forecast_tab2:
        st.subheader("🎯 Year-End Revenue Prediction")
        st.info("Compare against AED 38M budget and AED 33M last year performance")
        
        if st.button("Predict Year-End Revenue", type="primary"):
            with st.spinner("Calculating year-end prediction..."):
                try:
                    prediction = forecaster.predict_year_end_revenue(segment_data, occupancy_data)
                    
                    if prediction:
                        st.success("✅ Year-end prediction calculated!")
                        
                        # Key metrics - Use 2 rows to avoid clashing
                        st.subheader("📈 Year-End Prediction Summary")
                        
                        # First row
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Predicted Year-End Revenue",
                                f"AED {prediction['year_end_prediction']:,.0f}"
                            )
                        
                        with col2:
                            st.metric(
                                "Months Remaining in Year",
                                prediction['months_remaining']
                            )
                        
                        # Second row
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            st.metric(
                                "vs Budget Target (38M)",
                                f"AED {prediction['budget_variance']:+,.0f}",
                                f"{prediction['budget_performance_pct']:.1f}% of budget"
                            )
                        
                        with col4:
                            st.metric(
                                "vs Last Year Total (33M)", 
                                f"AED {prediction['vs_last_year']:+,.0f}",
                                f"{prediction['vs_last_year_growth_pct']:+.1f}% growth"
                            )
                        
                        # Visual comparison
                        st.subheader("📊 Performance Comparison")
                        
                        comparison_data = {
                            'Metric': ['Last Year Actual', 'This Year Budget', 'This Year Prediction'],
                            'Revenue': [prediction['last_year_total'], prediction['budget_target'], prediction['year_end_prediction']],
                            'Color': ['lightcoral', 'lightgreen', 'lightblue']
                        }
                        
                        fig_comparison = go.Figure(data=[
                            go.Bar(
                                x=comparison_data['Metric'],
                                y=comparison_data['Revenue'],
                                marker_color=comparison_data['Color'],
                                text=[f"AED {x:,.0f}" for x in comparison_data['Revenue']],
                                textposition='auto'
                            )
                        ])
                        
                        fig_comparison.update_layout(
                            title="Year-End Revenue: Budget vs Last Year vs Prediction",
                            xaxis_title="",
                            yaxis_title="Revenue (AED)",
                            height=500,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        # Breakdown
                        st.subheader("📋 Prediction Breakdown")
                        
                        breakdown_data = {
                            'Component': ['Year-to-Date', 'Remaining Forecast', 'Total Prediction'],
                            'Amount (AED)': [
                                f"{prediction['ytd_revenue']:,.0f}",
                                f"{prediction['remaining_forecast']:,.0f}",
                                f"{prediction['year_end_prediction']:,.0f}"
                            ]
                        }
                        
                        st.table(pd.DataFrame(breakdown_data))
                        
                    else:
                        st.error("❌ Could not generate year-end prediction")
                except Exception as e:
                    st.error(f"❌ Error generating year-end prediction: {str(e)}")
    
    with forecast_tab3:
        st.subheader("📈 Current Month Close Prediction")
        st.info("Predict current month close using last year data and moving average analysis")
        
        if st.button("Predict Current Month Close", type="primary"):
            with st.spinner("Analyzing current month performance..."):
                try:
                    month_prediction = forecaster.predict_current_month_close(segment_data, occupancy_data)
                    
                    if month_prediction:
                        st.success("✅ Current month prediction calculated!")
                        
                        # Key metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Business on Books",
                                f"AED {month_prediction.get('business_on_books', 0):,.0f}"
                            )
                        
                        with col2:
                            st.metric(
                                "Predicted Month Close",
                                f"AED {month_prediction.get('final_prediction', 0):,.0f}"
                            )
                        
                        with col3:
                            st.metric(
                                "MTD Pickup",
                                f"AED {month_prediction.get('mtd_pickup', 0):,.0f}"
                            )
                        
                        # Last year comparison
                        st.subheader("📊 Last Year Comparison Analysis")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Same Time Last Year",
                                f"AED {month_prediction.get('last_year_same_time', 0):,.0f}"
                            )
                        
                        with col2:
                            st.metric(
                                "Full Month Last Year",
                                f"AED {month_prediction.get('last_year_full_month', 0):,.0f}"
                            )
                        
                        with col3:
                            st.metric(
                                "Last Year Remainder",
                                f"AED {month_prediction.get('last_year_remainder', 0):,.0f}"
                            )
                        
                        # Visual analysis
                        prediction_data = {
                            'Method': ['BOB + Last Year Remainder', 'Weighted Prediction'],
                            'Prediction': [month_prediction.get('bob_plus_remainder_prediction', 0), month_prediction.get('weighted_prediction', 0)]
                        }
                        
                        fig_methods = go.Figure(data=[
                            go.Bar(
                                x=prediction_data['Method'],
                                y=prediction_data['Prediction'],
                                marker_color=['lightblue', 'lightgreen'],
                                text=[f"AED {x:,.0f}" for x in prediction_data['Prediction']],
                                textposition='auto'
                            )
                        ])
                        
                        fig_methods.update_layout(
                            title="Current Month Close: Prediction Methods Comparison",
                            xaxis_title="Prediction Method",
                            yaxis_title="Predicted Revenue (AED)",
                            height=400
                        )
                        
                        st.plotly_chart(fig_methods, use_container_width=True)
                        
                        # Additional info
                        st.info(f"📅 Current day: {month_prediction.get('current_day', 'N/A')}, Days remaining: {month_prediction.get('days_remaining', 'N/A')}")
                        
                    else:
                        st.error("❌ Could not generate current month prediction")
                except Exception as e:
                    st.error(f"❌ Error generating current month prediction: {str(e)}")

def machine_learning_tab():
    """Machine Learning Analysis Tab"""
    st.header("🤖 Machine Learning Revenue Analysis")
    
    if not st.session_state.data_loaded:
        show_loading_requirements()
        return
    
    # Get data
    if st.session_state.segment_data is not None:
        segment_data = st.session_state.segment_data
    else:
        if database_available:
            segment_data = get_database().get_segment_data()
        else:
            segment_data = pd.DataFrame()
            
    if st.session_state.occupancy_data is not None:
        occupancy_data = st.session_state.occupancy_data
    else:
        if database_available:
            occupancy_data = get_database().get_occupancy_data()
        else:
            occupancy_data = pd.DataFrame()
    
    if segment_data.empty:
        st.error("No segment data available for ML analysis")
        return
    
    if not forecasting_available:
        st.warning("⚠️ Advanced forecasting module not available")
        st.info("Install the advanced forecasting module to access ML models like XGBoost, Random Forest, Linear & Ridge Regression.")
        return
    
    forecaster = get_advanced_forecaster()
    
    # Create tabs for different ML analyses
    ml_tab1, ml_tab2, ml_tab3 = st.tabs(["Multi-Model Analysis", "Correlation Heatmap", "Model Performance"])
    
    with ml_tab1:
        st.subheader("🎯 Multi-Model Revenue Driver Analysis")
        st.info("Using multiple ML models (XGBoost, Random Forest, Linear & Ridge Regression) to determine what influences Business on the Books Revenue the most")
        
        if st.button("Run Multi-Model Analysis", type="primary"):
            with st.spinner("Training multiple ML models and analyzing revenue drivers..."):
                try:
                    ml_results = forecaster.analyze_revenue_drivers_multiple_models(segment_data, occupancy_data)
                    
                    if 'error' in ml_results:
                        st.error(f"❌ ML Analysis failed: {ml_results['error']}")
                    else:
                        st.success("✅ Multi-model analysis completed successfully!")
                        
                        # Model comparison
                        st.subheader("🏆 Model Performance Comparison")
                        
                        model_results = ml_results['model_results']
                        best_model = ml_results['best_model']
                        
                        # Create performance comparison table
                        performance_data = []
                        for model_name, result in model_results.items():
                            if 'performance' in result:
                                perf = result['performance']
                                performance_data.append({
                                    'Model': model_name + (" 🏆" if model_name == best_model else ""),
                                    'R² Score': f"{perf['r2_score']:.3f}",
                                    'MAE': f"{perf['mae']:,.0f}",
                                    'RMSE': f"{perf['rmse']:,.0f}",
                                    'CV Mean': f"{perf['cv_mean']:.3f}",
                                    'CV Std': f"{perf['cv_std']:.3f}"
                                })
                        
                        if performance_data:
                            st.dataframe(pd.DataFrame(performance_data), use_container_width=True)
                            
                            # Best model feature importance
                            if best_model and best_model in model_results:
                                best_result = model_results[best_model]
                                if 'feature_importance' in best_result and best_result['feature_importance'] is not None:
                                    st.subheader(f"📊 Feature Importance - {best_model} (Best Model)")
                                    
                                    importance_df = best_result['feature_importance']
                                    top_features = importance_df.head(10)
                                    
                                    fig_importance = go.Figure(data=[
                                        go.Bar(
                                            y=top_features['feature'],
                                            x=top_features['importance'],
                                            orientation='h',
                                            marker_color='lightblue'
                                        )
                                    ])
                                    
                                    fig_importance.update_layout(
                                        title=f"Top 10 Revenue Drivers - {best_model}",
                                        xaxis_title="Importance Score",
                                        yaxis_title="Features",
                                        height=500,
                                        yaxis={'categoryorder': 'total ascending'}
                                    )
                                    
                                    st.plotly_chart(fig_importance, use_container_width=True)
                                    
                                    # Feature importance table
                                    st.subheader("📋 Complete Feature Importance Rankings")
                                    
                                    display_importance = importance_df.copy()
                                    display_importance['importance'] = display_importance['importance'].apply(lambda x: f"{x:.4f}")
                                    display_importance.index = range(1, len(display_importance) + 1)
                                    
                                    st.dataframe(display_importance, use_container_width=True)
                        
                        # Store results in session state for other tabs
                        st.session_state.ml_results = ml_results
                except Exception as e:
                    st.error(f"❌ Error in multi-model analysis: {str(e)}")
    
    with ml_tab2:
        st.subheader("🔥 Correlation Heatmap")
        st.info("Correlation analysis between revenue and various factors")
        
        if st.button("Generate Correlation Heatmap", type="primary"):
            with st.spinner("Calculating correlations..."):
                try:
                    correlation_matrix = forecaster.generate_correlation_matrix(segment_data, occupancy_data)
                    
                    if correlation_matrix is not None:
                        st.success("✅ Correlation matrix generated!")
                        
                        # Create heatmap using plotly
                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=correlation_matrix.values,
                            x=correlation_matrix.columns,
                            y=correlation_matrix.index,
                            colorscale='RdBu_r',
                            zmid=0,
                            text=correlation_matrix.round(3).values,
                            texttemplate='%{text}',
                            textfont={"size": 10},
                            hoverongaps=False
                        ))
                        
                        fig_heatmap.update_layout(
                            title="Revenue Correlation Heatmap",
                            xaxis_title="Variables",
                            yaxis_title="Variables",
                            height=600,
                            width=800
                        )
                        
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        # Correlation insights
                        st.subheader("🔍 Key Correlation Insights")
                        
                        # Find strongest correlations with revenue
                        revenue_corr = correlation_matrix['Business_on_the_Books_Revenue'].drop('Business_on_the_Books_Revenue')
                        strongest_positive = revenue_corr.nlargest(3)
                        strongest_negative = revenue_corr.nsmallest(3)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Strongest Positive Correlations:**")
                            for var, corr in strongest_positive.items():
                                st.write(f"• {var}: {corr:.3f}")
                        
                        with col2:
                            st.write("**Strongest Negative Correlations:**")
                            for var, corr in strongest_negative.items():
                                st.write(f"• {var}: {corr:.3f}")
                        
                    else:
                        st.error("❌ Could not generate correlation matrix")
                except Exception as e:
                    st.error(f"❌ Error generating correlation heatmap: {str(e)}")
    
    with ml_tab3:
        st.subheader("📈 Model Performance Evaluation")
        
        if 'ml_results' in st.session_state and 'model_results' in st.session_state.ml_results:
            ml_results = st.session_state.ml_results
            best_model = ml_results['best_model']
            
            if best_model and best_model in ml_results['model_results']:
                performance = ml_results['model_results'][best_model]['performance']
                
                st.success(f"✅ Model performance metrics available for {best_model}!")
                
                # Performance metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("R² Score", f"{performance['r2_score']:.3f}")
                
                with col2:
                    st.metric("MAE", f"{performance['mae']:,.0f}")
                
                with col3:
                    st.metric("RMSE", f"{performance['rmse']:,.0f}")
                
                with col4:
                    st.metric("Cross-Val R²", f"{performance['cv_mean']:.3f} ± {performance['cv_std']:.3f}")
                
                # Model interpretation
                st.subheader("📊 Model Performance Analysis")
                
                if performance['r2_score'] > 0.8:
                    st.success("🎉 Excellent model performance! The model explains over 80% of revenue variance.")
                elif performance['r2_score'] > 0.6:
                    st.info("👍 Good model performance. The model explains a significant portion of revenue variance.")
                elif performance['r2_score'] > 0.4:
                    st.warning("⚠️ Moderate model performance. Consider feature engineering or more data.")
                else:
                    st.error("❌ Poor model performance. Model may need significant improvements.")
                
                # Actual vs Predicted plot
                if 'y_test' in ml_results and best_model in ml_results['model_results']:
                    st.subheader(f"🎯 Actual vs Predicted Revenue - {best_model}")
                    
                    y_test = ml_results['y_test']
                    y_pred = ml_results['model_results'][best_model]['predictions']
                    
                    fig_pred = go.Figure()
                    
                    # Add scatter plot
                    fig_pred.add_trace(go.Scatter(
                        x=y_test,
                        y=y_pred,
                        mode='markers',
                        name='Predictions',
                        marker=dict(color='lightblue', size=8)
                    ))
                    
                    # Add perfect prediction line
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    fig_pred.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_pred.update_layout(
                        title=f"Actual vs Predicted Revenue - {best_model}",
                        xaxis_title="Actual Revenue (AED)",
                        yaxis_title="Predicted Revenue (AED)",
                        height=500
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Residuals analysis
                    st.subheader("📊 Residuals Analysis")
                    
                    residuals = y_test - y_pred
                    
                    fig_residuals = go.Figure()
                    fig_residuals.add_trace(go.Scatter(
                        x=y_pred,
                        y=residuals,
                        mode='markers',
                        marker=dict(color='lightgreen', size=6)
                    ))
                    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                    
                    fig_residuals.update_layout(
                        title=f"Residuals vs Predicted Values - {best_model}",
                        xaxis_title="Predicted Revenue (AED)",
                        yaxis_title="Residuals (AED)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_residuals, use_container_width=True)
        else:
            st.info("⏳ Run Multi-Model Analysis first to see model performance metrics")

def insights_analysis_tab():
    """Insights Analysis Tab - Segment performance analysis"""
    st.header("🎯 Insights & Performance Analysis")
    
    if not st.session_state.data_loaded:
        show_loading_requirements()
        return
    
    # Get data
    segment_data = st.session_state.segment_data if st.session_state.segment_data is not None else get_database().get_segment_data()
    
    if segment_data.empty:
        st.error("No segment data available for insights analysis")
        return
    
    if forecasting_available:
        forecaster = get_advanced_forecaster()
        
        st.info("Analysis of segment performance compared to full month last year and budget targets")
        
        if st.button("Generate Insights Analysis", type="primary"):
            with st.spinner("Analyzing segment performance..."):
                insights = forecaster.analyze_segment_performance(segment_data)
                
                if 'error' in insights:
                    st.error(f"❌ Insights analysis failed: {insights['error']}")
                else:
                    st.success("✅ Insights analysis completed successfully!")
                    
                    # Summary metrics
                    summary = insights['summary']
                    st.subheader("📊 Performance Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Segments", summary['total_segments'])
                    
                    with col2:
                        st.metric("Top Performers", summary['top_performers'], f"{summary['top_performer_pct']:.1f}%")
                    
                    with col3:
                        st.metric("Good Performers", summary['good_performers'])
                    
                    with col4:
                        st.metric("Need Attention", summary['attention_needed'], f"{summary['attention_needed_pct']:.1f}%")
                    
                    analysis_results = insights['analysis_results']
                    
                    # Performance flags section
                    st.subheader("🚨 Performance Alerts")
                    
                    # Group segments by performance
                    top_performers = [r for r in analysis_results if "Top Performer" in r['overall_status']]
                    attention_needed = [r for r in analysis_results if "Critical Attention Needed" in r['overall_status']]
                    below_expectations = [r for r in analysis_results if "Below Expectations" in r['overall_status']]
                    
                    if top_performers:
                        st.success("🌟 **Top Performing Segments:**")
                        for segment in top_performers:
                            st.write(f"• **{segment['segment']}**: {segment['vs_full_month_ly_pct']:+.1f}% vs LY, {segment['vs_budget_pct']:+.1f}% vs Budget")
                    
                    if attention_needed:
                        st.error("🚨 **Segments Needing Critical Attention:**")
                        for segment in attention_needed:
                            st.write(f"• **{segment['segment']}**: {segment['vs_full_month_ly_pct']:+.1f}% vs LY, {segment['vs_budget_pct']:+.1f}% vs Budget")
                            for flag in segment['performance_flags']:
                                st.write(f"  - {flag}")
                    
                    if below_expectations:
                        st.warning("⚠️ **Segments Below Expectations:**")
                        for segment in below_expectations:
                            st.write(f"• **{segment['segment']}**: {segment['vs_full_month_ly_pct']:+.1f}% vs LY, {segment['vs_budget_pct']:+.1f}% vs Budget")
                    
                    # Detailed performance table
                    st.subheader("📋 Complete Performance Table")
                    
                    # Create display dataframe
                    display_data = []
                    for result in analysis_results:
                        display_data.append({
                            'Segment': result['segment'],
                            'Status': result['overall_status'],
                            'Current BOB (AED)': f"{result['current_bob_revenue']:,.0f}",
                            'vs LY Full Month (%)': f"{result['vs_full_month_ly_pct']:+.1f}%",
                            'vs Budget (%)': f"{result['vs_budget_pct']:+.1f}%",
                            'vs Same Time LY (%)': f"{result['vs_same_time_ly_pct']:+.1f}%",
                            'Budget (AED)': f"{result['budget_revenue']:,.0f}",
                            'Full Month LY (AED)': f"{result['full_month_ly_revenue']:,.0f}"
                        })
                    
                    st.dataframe(pd.DataFrame(display_data), use_container_width=True)
                    
                    # Revenue comparison chart
                    st.subheader("💰 Revenue Comparison")
                    
                    segments = [r['segment'] for r in analysis_results]
                    revenue_comparison = pd.DataFrame({
                        'Segment': segments,
                        'Current BOB': [r['current_bob_revenue'] for r in analysis_results],
                        'Budget': [r['budget_revenue'] for r in analysis_results],
                        'Full Month LY': [r['full_month_ly_revenue'] for r in analysis_results]
                    })
                    
                    fig_revenue = go.Figure()
                    
                    fig_revenue.add_trace(go.Bar(
                        name='Current Business on Books',
                        x=revenue_comparison['Segment'],
                        y=revenue_comparison['Current BOB'],
                        marker_color='blue'
                    ))
                    
                    fig_revenue.add_trace(go.Bar(
                        name='Budget',
                        x=revenue_comparison['Segment'],
                        y=revenue_comparison['Budget'],
                        marker_color='green'
                    ))
                    
                    fig_revenue.add_trace(go.Bar(
                        name='Full Month Last Year',
                        x=revenue_comparison['Segment'],
                        y=revenue_comparison['Full Month LY'],
                        marker_color='orange'
                    ))
                    
                    fig_revenue.update_layout(
                        title="Revenue Comparison by Segment",
                        xaxis_title="Segments",
                        yaxis_title="Revenue (AED)",
                        barmode='group',
                        height=500,
                        xaxis_tickangle=45
                    )
                    
                    st.plotly_chart(fig_revenue, use_container_width=True)
    else:
        st.warning("⚠️ Advanced forecasting module not available")
        st.info("Install the advanced forecasting module to access insights analysis features.")

def controls_logs_tab():
    """Controls and Logs Tab"""
    st.header("⚙️ Controls & Logs")
    
    # Re-run section
    st.subheader("🔄 Re-run Loading")
    if st.button("Re-run Data Loading", type="secondary"):
        st.session_state.data_loaded = False
        st.session_state.segment_data = None
        st.session_state.occupancy_data = None
        st.session_state.last_run_timestamp = None
        st.success("Cache cleared. Please go to Dashboard tab to reload data.")
    
    # Database statistics
    st.subheader("📊 Database Statistics")
    
    if database_available:
        try:
            db_stats = get_database().get_database_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Segment Records", db_stats.get('segment_analysis_rows', 0))
                st.metric("Occupancy Records", db_stats.get('occupancy_analysis_rows', 0))
            
            with col2:
                if db_stats.get('segment_last_updated'):
                    st.info(f"Segment Updated: {db_stats['segment_last_updated']}")
                if db_stats.get('occupancy_last_updated'):
                    st.info(f"Occupancy Updated: {db_stats['occupancy_last_updated']}")
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
    else:
        st.warning("Database module not available")
    
    # Log viewers
    st.subheader("📝 Recent Logs")
    
    if loggers_available:
        tab1, tab2 = st.tabs(["Conversion Log", "App Log"])
        
        with tab1:
            st.text("Last 100 lines from conversion.log:")
            try:
                conversion_log = get_log_content('conversion', 100)
                st.code(conversion_log, language="text")
            except Exception as e:
                st.error(f"Error reading conversion log: {str(e)}")
        
        with tab2:
            st.text("Last 100 lines from app.log:")
            try:
                app_log = get_log_content('app', 100)
                st.code(app_log, language="text")
            except Exception as e:
                st.error(f"Error reading app log: {str(e)}")
    else:
        st.warning("Logging module not available")
    
    # System status
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
            if database_available:
                db_stats = get_database().get_database_stats()
                if db_stats:
                    st.info(f"📊 DB: {db_stats.get('segment_analysis_rows', 0)} seg, {db_stats.get('occupancy_analysis_rows', 0)} occ")
            else:
                st.info("📊 DB: Module not available")
        except:
            st.info("📊 DB: Not available")

def gpt_tab():
    """GPT AI Assistant Tab for Natural Language SQL Queries"""
    
    st.title("🤖 GPT AI Assistant")
    st.markdown("Ask questions about your revenue data in natural language and get SQL-powered answers.")
    
    # Check if Gemini is available
    if not GEMINI_AVAILABLE:
        st.error("⚠️ Gemini AI backend is not available. Please install google-generativeai package.")
        st.code("pip install google-generativeai", language="bash")
        return
    
    # API Key configuration
    api_key = "AIzaSyBqgrOIUjZTEJxd7o1JUxVzMoknYrzQs08"  # Your provided API key
    
    # Initialize session state for chat history
    if 'gpt_messages' not in st.session_state:
        st.session_state.gpt_messages = []
    
    if 'gemini_backend' not in st.session_state:
        try:
            st.session_state.gemini_backend = create_gemini_backend(api_key)
            if not st.session_state.gemini_backend.connected:
                st.error(f"Failed to connect to Gemini AI: {st.session_state.gemini_backend.error_message}")
                return
        except Exception as e:
            st.error(f"Error initializing Gemini backend: {str(e)}")
            return
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask Your Question")
        
        # Question input
        question = st.text_area(
            "Type your question about arrivals, revenue, occupancy, or any other data:",
            height=100,
            placeholder="Example: What was the total revenue for the last 30 days?\nExample: Show me the top 5 segments by occupancy this month\nExample: What is the average ADR for weekends vs weekdays?"
        )
        
        # Ask button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
        
        with col_btn1:
            ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
        
        with col_btn2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    with col2:
        st.subheader("ℹ️ Tips")
        st.markdown("""
        **Example Questions:**
        - "What was the total revenue last month?"
        - "Show top 5 segments by occupancy"
        - "Average ADR for weekends"
        - "Revenue trends over last 6 months"
        - "Occupancy rate by room type"
        """)
    
    # Clear chat history
    if clear_button:
        st.session_state.gpt_messages = []
        st.rerun()
    
    # Process question
    if ask_button:
        if not question or question.strip() == "":
            st.warning("⚠️ Please enter a question before clicking Ask.")
        else:
            # Add user message to chat
            st.session_state.gpt_messages.append({
                "role": "user",
                "content": question,
                "timestamp": datetime.now()
            })
            
            # Show loading spinner
            with st.spinner("🤖 AI is thinking and generating SQL query..."):
                try:
                    # Process the question
                    result = st.session_state.gemini_backend.process_question(question)
                    
                    # Add AI response to chat
                    st.session_state.gpt_messages.append({
                        "role": "assistant",
                        "content": result,
                        "timestamp": datetime.now()
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing question: {str(e)}"
                    st.session_state.gpt_messages.append({
                        "role": "assistant",
                        "content": {"success": False, "error_message": error_msg},
                        "timestamp": datetime.now()
                    })
            
            st.rerun()
    
    # Display chat history
    if st.session_state.gpt_messages:
        st.markdown("---")
        st.subheader("💬 Conversation History")
        
        # Reverse the messages to show newest first
        for i, message in enumerate(reversed(st.session_state.gpt_messages)):
            with st.container():
                if message["role"] == "user":
                    st.markdown(f"**👤 You asked:** {message['content']}")
                    st.caption(f"⏰ {message['timestamp'].strftime('%H:%M:%S')}")
                
                else:  # assistant
                    result = message["content"]
                    
                    if result.get("success", False):
                        st.markdown("**🤖 AI Response:**")
                        
                        # Show SQL query
                        st.markdown("**📝 Generated SQL Query:**")
                        st.code(result["sql_query"], language="sql")
                        
                        # Show results
                        if len(result["results"]) > 0:
                            st.markdown(f"**📊 Results ({result['row_count']} rows):**")
                            
                            # Display as dataframe with better formatting
                            st.dataframe(
                                result["results"],
                                use_container_width=True,
                                height=min(400, (len(result["results"]) + 1) * 35)
                            )
                            
                            # Show summary if many rows
                            if result["row_count"] > 10:
                                st.info(f"📈 Showing all {result['row_count']} rows")
                        else:
                            st.info("📭 Query executed successfully but returned no results.")
                    
                    else:
                        st.error(f"❌ **Error:** {result.get('error_message', 'Unknown error')}")
                    
                    st.caption(f"⏰ {message['timestamp'].strftime('%H:%M:%S')}")
                
                st.markdown("---")
    
    else:
        # Show welcome message when no chat history
        st.markdown("---")
        st.info("👋 Welcome! Ask any question about your revenue data above to get started.")


def enhanced_gpt_tab():
    """Enhanced GPT AI Assistant Tab with Conversational Intelligence"""
    
    st.title("🚀 Enhanced AI Revenue Assistant")
    st.markdown("**Next-generation conversational AI with advanced analytics, insights, and visual storytelling**")
    
    # Check if Enhanced Gemini is available
    if not ENHANCED_GEMINI_AVAILABLE:
        st.error("⚠️ Enhanced Gemini AI backend is not available. Using fallback to basic GPT tab.")
        st.info("📦 To enable enhanced features, ensure all dependencies are installed.")
        gpt_tab()  # Fallback to basic version
        return
    
    # API Key configuration
    api_key = "AIzaSyBqgrOIUjZTEJxd7o1JUxVzMoknYrzQs08"  # Your provided API key
    
    # Initialize enhanced backend in session state
    if 'enhanced_gemini_backend' not in st.session_state:
        try:
            st.session_state.enhanced_gemini_backend = create_enhanced_gemini_backend(api_key)
            if not st.session_state.enhanced_gemini_backend.connected:
                st.error(f"Failed to connect to Enhanced Gemini AI: {st.session_state.enhanced_gemini_backend.error_message}")
                return
        except Exception as e:
            st.error(f"Error initializing Enhanced Gemini backend: {str(e)}")
            return
    
    # Render the enhanced interface
    try:
        render_enhanced_gpt_tab(st.session_state.enhanced_gemini_backend)
    except Exception as e:
        st.error(f"Error rendering enhanced interface: {str(e)}")
        st.info("Falling back to basic GPT interface...")
        gpt_tab()


def main():
    """Main application with simple sidebar"""
    
    # Get logo (80px as requested)
    logo_img = '<div style="width:80px;height:80px;background:#fff;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:32px;">🏨</div>'
    try:
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Pictures', 'xkrIXuKn_400x400.jpg')
        if not os.path.exists(logo_path):
            logo_path = 'Pictures/xkrIXuKn_400x400.jpg'
        if os.path.exists(logo_path):
            with open(logo_path, 'rb') as f:
                logo_data = base64.b64encode(f.read()).decode()
                logo_img = f'<img src="data:image/jpeg;base64,{logo_data}" alt="Logo" style="width:80px;height:80px;border-radius:3px;object-fit:cover;">'
    except:
        pass
    
    # SIDEBAR - Default Streamlit sidebar
    with st.sidebar:
        # Logo and title
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; align-items: center; gap: 15px; margin: 20px 0;">
            {logo_img}
            <div style="font-size: 18px; font-weight: 600; font-style: italic; color: #333; text-align: center; line-height: 1.2;">
                Grand Millennium<br>Revenue Analytics
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display pickup report date if available
        if st.session_state.pickup_report_date:
            st.markdown(f"""
            <div style="background-color: #1f4e79; color: white; padding: 10px; text-align: center; border-radius: 5px; margin: 10px 0;">
                <strong>💰 MHR Pick Up Report 2025 💰</strong><br>
                <small>Report Date: {st.session_state.pickup_report_date}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation with italic styling
        st.markdown("### 📊 *Navigation*")
        
        # Add CSS for italic radio button options
        st.markdown("""
        <style>
        .stRadio label {
            font-style: italic !important;
        }
        .stRadio label span {
            font-style: italic !important;
        }
        div[data-testid="stRadio"] label {
            font-style: italic !important;
        }
        div[data-testid="stRadio"] label span {
            font-style: italic !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        navigation_options = [
            "Dashboard",
            "Daily Occupancy", 
            "Segment Analysis",
            "ADR Analysis",
            "STR Report",
            "Block Analysis",
            "Events Analysis",
            "Entered On & Arrivals",
            "Historical & Forecast",
            "Enhanced Forecasting",
            "Machine Learning",
            "GPT",
            "Insights Analysis",
            "Controls & Logs"
        ]
        
        current_tab = st.session_state.current_tab
        
        # Radio buttons for navigation (now italic)
        selected_tab = st.radio(
            "*Choose a section:*",
            navigation_options,
            index=navigation_options.index(current_tab) if current_tab in navigation_options else 0
        )
        
        # Update session state
        if selected_tab != current_tab:
            st.session_state.current_tab = selected_tab
            st.rerun()
        
        st.markdown("---")
        
        # Status
        if st.session_state.data_loaded:
            st.success("✅ Data Loaded")
        else:
            st.warning("⚠️ No Data")
    
    # Main content
    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 10px;">
            {logo_img}
            <h1 style="color: #333; font-style: italic; margin: 0;">Grand Millennium Revenue Analytics</h1>
        </div>
        <p style="color: #666; font-size: 16px;">Current Section: <strong>{st.session_state.current_tab}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display pickup report header at the top of main content
    display_pickup_report_header()
    
    # Add hotel exterior image after title
    try:
        hotel_image_path = project_root / "Pictures" / "Grand Millennium Dubai Hotel Exterior 2.jpg"
        if hotel_image_path.exists():
            st.image(str(hotel_image_path), caption="Grand Millennium Dubai", use_container_width=True)
    except:
        pass
    
    # Route to tabs
    current_tab = st.session_state.current_tab
    
    if current_tab == "Dashboard":
        dashboard_tab()
    elif current_tab == "Daily Occupancy":
        daily_occupancy_tab()
    elif current_tab == "Segment Analysis":
        segment_analysis_tab()
    elif current_tab == "ADR Analysis":
        adr_analysis_tab()
    elif current_tab == "STR Report":
        str_report_tab()
    elif current_tab == "Block Analysis":
        block_analysis_tab()
    elif current_tab == "Events Analysis":
        events_analysis_tab()
    elif current_tab == "Entered On & Arrivals":
        entered_on_arrivals_tab()
    elif current_tab == "Historical & Forecast":
        historical_forecast_tab()
    elif current_tab == "Enhanced Forecasting":
        enhanced_forecasting_tab()
    elif current_tab == "Machine Learning":
        machine_learning_tab()
    elif current_tab == "GPT":
        enhanced_gpt_tab()
    elif current_tab == "Insights Analysis":
        insights_analysis_tab()
    elif current_tab == "Controls & Logs":
        controls_logs_tab()

if __name__ == "__main__":
    main()