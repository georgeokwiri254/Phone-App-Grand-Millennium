"""
Example of integrating C++ accelerated analytics into existing tabs
This demonstrates how to enhance your current revenue analysis with C++ performance
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

try:
    from app.revenue_calculator import get_revenue_calculator, enhanced_metric_card, format_aed
    from app.database import get_database
    ENHANCED_ANALYTICS_AVAILABLE = True
except ImportError:
    ENHANCED_ANALYTICS_AVAILABLE = False


def enhanced_revenue_analysis_example():
    """
    Example function showing how to integrate C++ accelerated analytics
    This can be applied to your existing dashboard, segment analysis, etc.
    """
    
    if not ENHANCED_ANALYTICS_AVAILABLE:
        st.error("Enhanced analytics not available")
        return
    
    st.subheader("🚀 Enhanced Revenue Analysis (C++ Accelerated)")
    
    # Get calculator instance
    calc = get_revenue_calculator()
    
    # Show performance status
    perf_status = "C++ Acceleration Active 🚀" if calc.is_accelerated() else "Python Fallback ⚡"
    st.info(f"Performance Status: {perf_status}")
    
    # Data source selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Daily Performance", "Segment Analysis", "Monthly Trends", "ADR Analysis"]
        )
    
    with col2:
        time_period = st.selectbox(
            "Time Period", 
            ["Last 30 Days", "Last 90 Days", "Year to Date", "All Time"]
        )
    
    # Load and process data with timing
    with st.spinner("Loading data and calculating metrics..."):
        start_time = time.time()
        
        try:
            db = get_database()
            
            # Example queries based on analysis type
            if analysis_type == "Daily Performance":
                query = """
                SELECT date, revenue, rooms_sold, available_rooms, adr, revpar, occupancy_rate
                FROM daily_revenue 
                ORDER BY date DESC 
                LIMIT 500
                """
                
            elif analysis_type == "Segment Analysis":
                query = """
                SELECT date, segment_name, revenue_amount as revenue, rooms_count as rooms_sold
                FROM segments 
                ORDER BY date DESC 
                LIMIT 1000
                """
                
            elif analysis_type == "Monthly Trends":
                query = """
                SELECT date, adr, occupancy_rooms as rooms_sold, available_rooms,
                       (adr * occupancy_rooms) as revenue
                FROM monthly_forecast_data_actuals_only
                ORDER BY date DESC
                """
                
            else:  # ADR Analysis
                query = """
                SELECT date, revenue, rooms_sold, available_rooms, adr
                FROM daily_revenue 
                WHERE adr > 0
                ORDER BY date DESC 
                LIMIT 300
                """
            
            df = db.fetch_data(query)
            
            if df.empty:
                st.warning("No data found for selected criteria")
                return
            
            # Calculate metrics with C++ acceleration
            metrics = calc.calculate_all_metrics(df)
            
            calculation_time = time.time() - start_time
            
        except Exception as e:
            st.error(f"Data loading error: {e}")
            return
    
    # Display performance metrics
    st.markdown("---")
    
    # Key metrics with acceleration indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        enhanced_metric_card("Average Daily Rate", metrics['adr'], "aed")
    
    with col2:
        enhanced_metric_card("RevPAR", metrics['revpar'], "aed")
    
    with col3:
        enhanced_metric_card("Occupancy Rate", metrics['occupancy_rate'], "percentage")
    
    with col4:
        data_points = len(df)
        st.metric("Data Points", f"{data_points:,}")
        if calc.is_accelerated():
            st.markdown("🚀")
        else:
            st.markdown("⚡")
    
    # Performance timing
    st.success(f"⚡ Calculated in {calculation_time*1000:.1f}ms with {perf_status}")
    
    # Advanced analytics with C++ acceleration
    st.markdown("---")
    st.subheader("📈 Advanced Analytics")
    
    if len(df) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Time Series Analysis**")
            
            # Moving average calculation
            window_size = st.slider("Moving Average Window", 3, 30, 7)
            
            if 'revenue' in df.columns:
                ma_values = calc.moving_average(df['revenue'], window_size)
                
                if ma_values:
                    # Create visualization
                    fig = go.Figure()
                    
                    # Original data
                    fig.add_trace(go.Scatter(
                        x=df['date'][:len(ma_values)],
                        y=df['revenue'][:len(ma_values)],
                        name='Daily Revenue',
                        line=dict(color='lightblue', width=1),
                        opacity=0.6
                    ))
                    
                    # Moving average
                    fig.add_trace(go.Scatter(
                        x=df['date'][:len(ma_values)],
                        y=ma_values,
                        name=f'{window_size}-Day Moving Average',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"Revenue Trend Analysis ({perf_status})",
                        yaxis_title="Revenue (AED)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Statistical Analysis**")
            
            if 'revenue' in df.columns:
                # Statistical measures with C++ acceleration
                variance = calc.calculate_variance(df['revenue'])
                
                stat_col1, stat_col2 = st.columns(2)
                
                with stat_col1:
                    st.metric("Revenue Variance", f"{variance:,.0f}")
                    if calc.is_accelerated():
                        st.markdown("🚀")
                
                with stat_col2:
                    std_dev = variance ** 0.5
                    st.metric("Standard Deviation", f"{std_dev:,.0f}")
                    if calc.is_accelerated():
                        st.markdown("🚀")
                
                # Correlation analysis if multiple numeric columns
                if 'rooms_sold' in df.columns and 'revenue' in df.columns:
                    correlation = calc.calculate_correlation(df['revenue'], df['rooms_sold'])
                    
                    st.markdown("**Correlation Analysis**")
                    st.metric("Revenue vs Rooms Correlation", f"{correlation:.3f}")
                    
                    if correlation > 0.7:
                        st.success("Strong positive correlation")
                    elif correlation > 0.3:
                        st.info("Moderate correlation")
                    else:
                        st.warning("Weak correlation")
    
    # Data preview
    with st.expander("📋 Data Preview"):
        st.dataframe(df.head(20), use_container_width=True)
    
    # Integration guide
    with st.expander("🔧 How This Works"):
        st.markdown("""
        ### C++ Integration in Your Existing Tabs
        
        This example shows how to enhance your current revenue analysis tabs with C++ acceleration:
        
        1. **Import the calculator**: `from app.revenue_calculator import get_revenue_calculator`
        2. **Get instance**: `calc = get_revenue_calculator()`
        3. **Use accelerated functions**: `calc.calculate_adr(data)`
        4. **Show performance status**: Display 🚀 for C++ or ⚡ for Python
        
        ### Benefits in Your Current Tabs:
        - **Dashboard Tab**: Faster metric calculations for uploaded files
        - **Segment Analysis**: Accelerated segment performance comparisons  
        - **Daily Occupancy**: Real-time occupancy rate calculations
        - **ADR Analysis**: High-speed ADR trend analysis
        - **Weekly Analysis**: Fast weekly aggregations
        - **Machine Learning**: Faster feature calculations
        
        ### Easy Integration:
        ```python
        # Replace existing calculations like this:
        # OLD: adr = df['revenue'].sum() / df['rooms_sold'].sum()
        # NEW: adr = calc.calculate_adr(df)
        
        # Use enhanced metric cards:
        enhanced_metric_card("ADR", adr, "aed")
        ```
        
        The calculator automatically falls back to Python if C++ isn't available.
        """)


def add_cpp_acceleration_to_existing_function(df: pd.DataFrame) -> dict:
    """
    Example function showing how to add C++ acceleration to existing revenue calculations
    You can use this pattern in your existing dashboard_tab(), segment_analysis_tab(), etc.
    """
    
    if not ENHANCED_ANALYTICS_AVAILABLE:
        # Fallback to original calculations
        return {
            'adr': df['revenue'].sum() / df['rooms_sold'].sum() if 'rooms_sold' in df.columns else 0,
            'revpar': df['revenue'].sum() / df['available_rooms'].sum() if 'available_rooms' in df.columns else 0,
            'occupancy': (df['rooms_sold'].sum() / df['available_rooms'].sum()) * 100 if 'available_rooms' in df.columns else 0,
            'accelerated': False
        }
    
    # Use C++ accelerated calculations
    calc = get_revenue_calculator()
    return calc.calculate_all_metrics(df)


if __name__ == "__main__":
    # For testing
    enhanced_revenue_analysis_example()