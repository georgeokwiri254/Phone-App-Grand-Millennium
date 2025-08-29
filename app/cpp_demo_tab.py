"""
C++ Integration Demo Tab for Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta

try:
    from app.cpp_wrapper import get_revenue_analytics, is_cpp_available
    cpp_wrapper_available = True
except ImportError:
    cpp_wrapper_available = False


def render_cpp_demo_tab():
    """Render the C++ integration demo tab"""
    
    st.header("🚀 C++ Performance Analytics")
    
    if not cpp_wrapper_available:
        st.error("C++ wrapper module not available. Please check the installation.")
        st.info("Run: `python build_cpp.py` to build the C++ module")
        return
    
    # Check for database availability
    try:
        from app.database import get_database
        database_available = True
    except ImportError:
        database_available = False
    
    # Show C++ availability status
    col1, col2 = st.columns(2)
    with col1:
        if is_cpp_available():
            st.success("✅ C++ Module: Active")
        else:
            st.warning("⚠️ C++ Module: Using Python Fallback")
    
    with col2:
        if st.button("🔧 Build C++ Module"):
            with st.spinner("Building C++ module..."):
                import subprocess
                import sys
                result = subprocess.run([sys.executable, "build_cpp.py"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("C++ module built successfully!")
                    st.rerun()
                else:
                    st.error(f"Build failed: {result.stderr}")
    
    st.markdown("---")
    
    # Real data from database
    st.subheader("📊 Real Data Analytics with C++ Performance")
    
    if database_available:
        try:
            db = get_database()
            
            # Data source selection
            data_source = st.selectbox("Select Data Source", 
                                     ["Daily Revenue Data", "Monthly Aggregated", "Segment Data"])
            
            if data_source == "Daily Revenue Data":
                # Query daily revenue data
                query = """
                SELECT date, revenue, rooms_sold, available_rooms, occupancy_rate, adr, revpar
                FROM daily_revenue 
                ORDER BY date DESC
                LIMIT 1000
                """
                df = db.fetch_data(query)
                
                if not df.empty:
                    revenues = df['revenue'].fillna(0).tolist()
                    rooms_sold = df['rooms_sold'].fillna(0).astype(int).tolist()
                    available_rooms = df['available_rooms'].fillna(150).astype(int).tolist()
                    dates = pd.to_datetime(df['date']).tolist()
                else:
                    st.warning("No data found in daily_revenue table")
                    revenues, rooms_sold, available_rooms, dates = [], [], [], []
                    
            elif data_source == "Monthly Aggregated":
                # Query monthly data
                query = """
                SELECT * FROM monthly_forecast_data_actuals_only 
                ORDER BY date DESC
                """
                df = db.fetch_data(query)
                
                if not df.empty:
                    revenues = (df['adr'] * df['occupancy_rooms']).fillna(0).tolist()
                    rooms_sold = df['occupancy_rooms'].fillna(0).astype(int).tolist() 
                    available_rooms = df['available_rooms'].fillna(150).astype(int).tolist()
                    dates = pd.to_datetime(df['date']).tolist()
                else:
                    st.warning("No data found in monthly tables")
                    revenues, rooms_sold, available_rooms, dates = [], [], [], []
                    
            else:  # Segment Data
                query = """
                SELECT date, revenue_amount as revenue, rooms_count as rooms_sold
                FROM segments
                ORDER BY date DESC
                LIMIT 500
                """
                df = db.fetch_data(query)
                
                if not df.empty:
                    revenues = df['revenue'].fillna(0).tolist()
                    rooms_sold = df['rooms_sold'].fillna(0).astype(int).tolist()
                    available_rooms = [200] * len(revenues)  # Default assumption
                    dates = pd.to_datetime(df['date']).tolist()
                else:
                    st.warning("No data found in segments table")
                    revenues, rooms_sold, available_rooms, dates = [], [], [], []
                    
        except Exception as e:
            st.error(f"Database error: {e}")
            st.info("Using sample data for demonstration")
            # Fallback to sample data
            revenues, rooms_sold, available_rooms, dates = generate_sample_data()
    else:
        st.warning("Database not available. Using sample data.")
        revenues, rooms_sold, available_rooms, dates = generate_sample_data()


def generate_sample_data():
    """Generate sample data when database is not available"""
    data_size = 365
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=data_size, freq='D')
    
    # Use AED base amounts (Dubai hotel)
    base_revenue = 45000  # AED instead of USD
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * np.arange(data_size) / 365.25)
    noise = np.random.normal(0, 0.1, data_size)
    
    revenues = base_revenue * seasonal_factor * (1 + noise)
    rooms_sold = np.random.randint(80, 150, data_size)
    available_rooms = np.random.randint(150, 200, data_size)
    
    return revenues.tolist(), rooms_sold.tolist(), available_rooms.tolist(), dates.tolist()
    
    # Get analytics instance
    analytics = get_revenue_analytics()
    
    # Performance comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏨 Revenue Metrics")
        
        with st.spinner("Calculating metrics..."):
            start_time = time.time()
            
            # Calculate key metrics
            adr = analytics.calculate_adr(revenues.tolist(), rooms_sold.tolist())
            revpar = analytics.calculate_revpar(revenues.tolist(), available_rooms.tolist())
            occupancy = analytics.calculate_occupancy_rate(rooms_sold.tolist(), available_rooms.tolist())
            
            calculation_time = time.time() - start_time
        
        # Display metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Average Daily Rate", f"AED {adr:,.2f}")
        
        with metric_col2:
            st.metric("RevPAR", f"AED {revpar:,.2f}")
        
        with metric_col3:
            st.metric("Occupancy Rate", f"{occupancy:.1f}%")
        
        st.info(f"⚡ Calculation time: {calculation_time*1000:.2f}ms")
    
    with col2:
        st.subheader("📈 Time Series Analysis")
        
        window_size = st.selectbox("Moving Average Window", [7, 14, 30], index=1)
        alpha = st.slider("Exponential Smoothing Alpha", 0.1, 0.9, 0.3, 0.1)
        
        with st.spinner("Computing time series..."):
            start_time = time.time()
            
            # Calculate moving averages and smoothing
            ma_revenues = analytics.moving_average(revenues.tolist(), window_size)
            smooth_revenues = analytics.exponential_smoothing(revenues.tolist(), alpha)
            
            # Statistical measures
            variance = analytics.calculate_variance(revenues.tolist())
            std_dev = analytics.calculate_standard_deviation(revenues.tolist())
            
            ts_calculation_time = time.time() - start_time
        
        # Display statistics
        stat_col1, stat_col2 = st.columns(2)
        
        with stat_col1:
            st.metric("Variance", f"{variance:.0f}")
        
        with stat_col2:
            st.metric("Std Dev", f"{std_dev:.2f}")
        
        st.info(f"⚡ Time series calculation: {ts_calculation_time*1000:.2f}ms")
    
    # Visualization
    st.subheader("📊 Revenue Trend Visualization")
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Date': dates,
        'Revenue': revenues,
        'Rooms_Sold': rooms_sold,
        'Available_Rooms': available_rooms
    })
    
    # Add moving average data (align with dates)
    ma_dates = dates[window_size-1:]
    ma_df = pd.DataFrame({
        'Date': ma_dates,
        'Moving_Average': ma_revenues
    })
    
    # Add smoothed data
    smooth_df = pd.DataFrame({
        'Date': dates,
        'Exponential_Smoothing': smooth_revenues
    })
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Revenue Trends', 'Occupancy Analysis'],
        vertical_spacing=0.1
    )
    
    # Revenue plot
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Revenue'], name='Daily Revenue', 
                  line=dict(color='blue', width=1), opacity=0.7),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=ma_df['Date'], y=ma_df['Moving_Average'], 
                  name=f'{window_size}d Moving Average',
                  line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=smooth_df['Date'], y=smooth_df['Exponential_Smoothing'],
                  name='Exponential Smoothing',
                  line=dict(color='green', width=2)),
        row=1, col=1
    )
    
    # Occupancy plot
    occupancy_rate = (df['Rooms_Sold'] / df['Available_Rooms']) * 100
    
    fig.add_trace(
        go.Scatter(x=df['Date'], y=occupancy_rate, name='Daily Occupancy %',
                  line=dict(color='orange', width=1), fill='tonexty'),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        title_text="Revenue Analytics Dashboard (C++ Accelerated)",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Revenue (AED)", row=1, col=1)
    fig.update_yaxes(title_text="Occupancy (%)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Correlation Analysis")
    
    correlation = analytics.calculate_correlation(revenues.tolist(), rooms_sold.tolist())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Revenue vs Rooms Sold Correlation", f"{correlation:.3f}")
        
        if correlation > 0.7:
            st.success("Strong positive correlation")
        elif correlation > 0.3:
            st.info("Moderate positive correlation")
        else:
            st.warning("Weak correlation")
    
    with col2:
        # Correlation scatter plot
        fig_scatter = px.scatter(
            x=rooms_sold, y=revenues,
            title="Revenue vs Rooms Sold",
            labels={'x': 'Rooms Sold', 'y': 'Revenue (AED)'},
            trendline="ols"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        ### C++ Integration Features:
        
        - **Fast Numerical Computations**: Core revenue calculations optimized in C++
        - **Memory Efficient**: Direct memory access without Python overhead  
        - **Automatic Fallback**: Pure Python implementations when C++ unavailable
        - **Pybind11 Integration**: Modern C++/Python binding
        - **Statistical Functions**: Variance, correlation, smoothing algorithms
        
        ### Performance Benefits:
        - 10-100x faster for large datasets
        - Lower memory usage
        - Parallelization ready
        - Type safety
        
        ### Usage in Streamlit:
        ```python
        from app.cpp_wrapper import get_revenue_analytics
        
        analytics = get_revenue_analytics()
        adr = analytics.calculate_adr(revenues, rooms_sold)
        ```
        """)


if __name__ == "__main__":
    render_cpp_demo_tab()