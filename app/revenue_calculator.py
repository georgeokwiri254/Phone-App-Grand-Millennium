"""
High-performance revenue calculation utilities
Integrates C++ acceleration across all Streamlit tabs
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Union, Optional, Tuple
import warnings

# Try to import C++ accelerated functions
try:
    from app.cpp_wrapper import get_revenue_analytics, is_cpp_available
    cpp_available = True
except ImportError:
    cpp_available = False

class RevenueCalculator:
    """High-performance revenue calculations with C++ acceleration"""
    
    def __init__(self):
        """Initialize calculator with C++ backend if available"""
        if cpp_available:
            self.backend = get_revenue_analytics()
            self.accelerated = is_cpp_available()
        else:
            self.backend = None
            self.accelerated = False
    
    def is_accelerated(self) -> bool:
        """Check if C++ acceleration is active"""
        return self.accelerated
    
    def calculate_adr(self, data: Union[pd.DataFrame, dict], 
                     revenue_col: str = 'revenue', 
                     rooms_col: str = 'rooms_sold') -> float:
        """
        Calculate Average Daily Rate with C++ acceleration
        
        Args:
            data: DataFrame or dict containing revenue and rooms data
            revenue_col: Name of revenue column
            rooms_col: Name of rooms sold column
            
        Returns:
            ADR value in AED
        """
        try:
            if isinstance(data, pd.DataFrame):
                revenues = data[revenue_col].fillna(0).tolist()
                rooms = data[rooms_col].fillna(0).astype(int).tolist()
            elif isinstance(data, dict):
                revenues = data.get(revenue_col, [])
                rooms = data.get(rooms_col, [])
            else:
                return 0.0
            
            if self.backend and revenues and rooms:
                return self.backend.calculate_adr(revenues, rooms)
            else:
                # Fallback calculation
                total_revenue = sum(revenues)
                total_rooms = sum(rooms)
                return total_revenue / total_rooms if total_rooms > 0 else 0.0
                
        except Exception as e:
            st.warning(f"ADR calculation error: {e}")
            return 0.0
    
    def calculate_revpar(self, data: Union[pd.DataFrame, dict],
                        revenue_col: str = 'revenue',
                        available_rooms_col: str = 'available_rooms') -> float:
        """
        Calculate Revenue Per Available Room with C++ acceleration
        
        Args:
            data: DataFrame or dict containing revenue and available rooms data
            revenue_col: Name of revenue column
            available_rooms_col: Name of available rooms column
            
        Returns:
            RevPAR value in AED
        """
        try:
            if isinstance(data, pd.DataFrame):
                revenues = data[revenue_col].fillna(0).tolist()
                available = data[available_rooms_col].fillna(0).astype(int).tolist()
            elif isinstance(data, dict):
                revenues = data.get(revenue_col, [])
                available = data.get(available_rooms_col, [])
            else:
                return 0.0
            
            if self.backend and revenues and available:
                return self.backend.calculate_revpar(revenues, available)
            else:
                # Fallback calculation
                total_revenue = sum(revenues)
                total_available = sum(available)
                return total_revenue / total_available if total_available > 0 else 0.0
                
        except Exception as e:
            st.warning(f"RevPAR calculation error: {e}")
            return 0.0
    
    def calculate_occupancy_rate(self, data: Union[pd.DataFrame, dict],
                                rooms_sold_col: str = 'rooms_sold',
                                available_rooms_col: str = 'available_rooms') -> float:
        """
        Calculate occupancy rate with C++ acceleration
        
        Args:
            data: DataFrame or dict containing rooms data
            rooms_sold_col: Name of rooms sold column
            available_rooms_col: Name of available rooms column
            
        Returns:
            Occupancy rate as percentage
        """
        try:
            if isinstance(data, pd.DataFrame):
                sold = data[rooms_sold_col].fillna(0).astype(int).tolist()
                available = data[available_rooms_col].fillna(0).astype(int).tolist()
            elif isinstance(data, dict):
                sold = data.get(rooms_sold_col, [])
                available = data.get(available_rooms_col, [])
            else:
                return 0.0
            
            if self.backend and sold and available:
                return self.backend.calculate_occupancy_rate(sold, available)
            else:
                # Fallback calculation
                total_sold = sum(sold)
                total_available = sum(available)
                return (total_sold / total_available) * 100.0 if total_available > 0 else 0.0
                
        except Exception as e:
            st.warning(f"Occupancy calculation error: {e}")
            return 0.0
    
    def calculate_all_metrics(self, data: Union[pd.DataFrame, dict]) -> dict:
        """
        Calculate all key revenue metrics at once
        
        Args:
            data: DataFrame or dict with revenue data
            
        Returns:
            Dictionary with ADR, RevPAR, and occupancy rate
        """
        return {
            'adr': self.calculate_adr(data),
            'revpar': self.calculate_revpar(data),
            'occupancy_rate': self.calculate_occupancy_rate(data),
            'accelerated': self.accelerated
        }
    
    def moving_average(self, data: Union[pd.Series, list], window: int = 7) -> List[float]:
        """
        Calculate moving average with C++ acceleration
        
        Args:
            data: Time series data
            window: Moving average window size
            
        Returns:
            List of moving average values
        """
        try:
            if isinstance(data, pd.Series):
                values = data.fillna(0).tolist()
            else:
                values = list(data)
            
            if self.backend and values:
                return self.backend.moving_average(values, window)
            else:
                # Fallback calculation
                if len(values) < window:
                    return []
                
                result = []
                for i in range(window - 1, len(values)):
                    window_sum = sum(values[i - window + 1:i + 1])
                    result.append(window_sum / window)
                
                return result
                
        except Exception as e:
            st.warning(f"Moving average calculation error: {e}")
            return []
    
    def exponential_smoothing(self, data: Union[pd.Series, list], alpha: float = 0.3) -> List[float]:
        """
        Apply exponential smoothing with C++ acceleration
        
        Args:
            data: Time series data
            alpha: Smoothing parameter (0-1)
            
        Returns:
            List of smoothed values
        """
        try:
            if isinstance(data, pd.Series):
                values = data.fillna(0).tolist()
            else:
                values = list(data)
            
            if self.backend and values:
                return self.backend.exponential_smoothing(values, alpha)
            else:
                # Fallback calculation
                if not values:
                    return []
                
                result = [values[0]]
                for i in range(1, len(values)):
                    smoothed = alpha * values[i] + (1.0 - alpha) * result[i-1]
                    result.append(smoothed)
                
                return result
                
        except Exception as e:
            st.warning(f"Exponential smoothing error: {e}")
            return []
    
    def calculate_variance(self, data: Union[pd.Series, list]) -> float:
        """Calculate variance with C++ acceleration"""
        try:
            if isinstance(data, pd.Series):
                values = data.fillna(0).tolist()
            else:
                values = list(data)
            
            if self.backend and values:
                return self.backend.calculate_variance(values)
            else:
                # Fallback calculation
                if len(values) < 2:
                    return 0.0
                
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
                return variance
                
        except Exception as e:
            st.warning(f"Variance calculation error: {e}")
            return 0.0
    
    def calculate_correlation(self, x_data: Union[pd.Series, list], 
                            y_data: Union[pd.Series, list]) -> float:
        """Calculate correlation with C++ acceleration"""
        try:
            if isinstance(x_data, pd.Series):
                x_values = x_data.fillna(0).tolist()
            else:
                x_values = list(x_data)
                
            if isinstance(y_data, pd.Series):
                y_values = y_data.fillna(0).tolist()
            else:
                y_values = list(y_data)
            
            if self.backend and x_values and y_values and len(x_values) == len(y_values):
                return self.backend.calculate_correlation(x_values, y_values)
            else:
                # Fallback calculation using numpy
                if len(x_values) != len(y_values) or len(x_values) < 2:
                    return 0.0
                
                return np.corrcoef(x_values, y_values)[0, 1] if len(x_values) > 1 else 0.0
                
        except Exception as e:
            st.warning(f"Correlation calculation error: {e}")
            return 0.0


# Global calculator instance
_calculator = None

def get_revenue_calculator() -> RevenueCalculator:
    """Get global revenue calculator instance"""
    global _calculator
    if _calculator is None:
        _calculator = RevenueCalculator()
    return _calculator


def format_aed(amount: float, decimals: int = 2) -> str:
    """Format amount in AED currency"""
    return f"AED {amount:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage"""
    return f"{value:.{decimals}f}%"


def show_performance_badge():
    """Show C++ performance status badge in sidebar"""
    calc = get_revenue_calculator()
    
    if calc.is_accelerated():
        st.sidebar.success("🚀 C++ Acceleration: Active")
    else:
        st.sidebar.info("⚡ Performance: Python Fallback")


def enhanced_metric_card(title: str, value: float, format_type: str = "aed", 
                        delta: Optional[float] = None):
    """
    Display enhanced metric card with performance indicator
    
    Args:
        title: Metric title
        value: Metric value
        format_type: 'aed', 'percentage', or 'number'
        delta: Optional delta value for comparison
    """
    calc = get_revenue_calculator()
    
    # Format value based on type
    if format_type == "aed":
        formatted_value = format_aed(value)
    elif format_type == "percentage":
        formatted_value = format_percentage(value)
    else:
        formatted_value = f"{value:,.2f}"
    
    # Format delta if provided
    delta_str = None
    if delta is not None:
        if format_type == "aed":
            delta_str = format_aed(delta)
        elif format_type == "percentage":
            delta_str = format_percentage(delta)
        else:
            delta_str = f"{delta:,.2f}"
    
    # Display metric with acceleration indicator
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.metric(title, formatted_value, delta_str)
    
    with col2:
        if calc.is_accelerated():
            st.markdown("🚀")
        else:
            st.markdown("⚡")