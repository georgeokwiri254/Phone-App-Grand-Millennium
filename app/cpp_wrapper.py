"""
C++ wrapper module for fast revenue analytics computations
"""

import warnings
import sys
import os

# Fallback pure Python implementations
class PythonRevenueAnalytics:
    """Pure Python fallback for revenue analytics calculations"""
    
    def calculate_adr(self, revenues, rooms_sold):
        """Calculate Average Daily Rate (ADR)"""
        if not revenues or not rooms_sold or len(revenues) != len(rooms_sold):
            return 0.0
        
        total_revenue = sum(revenues)
        total_rooms = sum(rooms_sold)
        
        return total_revenue / total_rooms if total_rooms > 0 else 0.0
    
    def calculate_revpar(self, revenues, available_rooms):
        """Calculate Revenue Per Available Room (RevPAR)"""
        if not revenues or not available_rooms or len(revenues) != len(available_rooms):
            return 0.0
        
        total_revenue = sum(revenues)
        total_rooms = sum(available_rooms)
        
        return total_revenue / total_rooms if total_rooms > 0 else 0.0
    
    def calculate_occupancy_rate(self, rooms_sold, available_rooms):
        """Calculate occupancy rate percentage"""
        if not rooms_sold or not available_rooms or len(rooms_sold) != len(available_rooms):
            return 0.0
        
        total_sold = sum(rooms_sold)
        total_available = sum(available_rooms)
        
        return (total_sold / total_available) * 100.0 if total_available > 0 else 0.0
    
    def moving_average(self, data, window_size):
        """Calculate moving average"""
        if len(data) < window_size:
            return []
        
        result = []
        for i in range(window_size - 1, len(data)):
            window_sum = sum(data[i - window_size + 1:i + 1])
            result.append(window_sum / window_size)
        
        return result
    
    def exponential_smoothing(self, data, alpha):
        """Apply exponential smoothing"""
        if not data:
            return []
        
        result = [data[0]]
        
        for i in range(1, len(data)):
            smoothed = alpha * data[i] + (1.0 - alpha) * result[i-1]
            result.append(smoothed)
        
        return result
    
    def calculate_variance(self, data):
        """Calculate variance"""
        if len(data) < 2:
            return 0.0
        
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        
        return variance
    
    def calculate_standard_deviation(self, data):
        """Calculate standard deviation"""
        return self.calculate_variance(data) ** 0.5
    
    def calculate_correlation(self, x, y):
        """Calculate correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
        sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
        
        denominator = (sum_sq_x * sum_sq_y) ** 0.5
        
        return numerator / denominator if denominator > 0 else 0.0


# Try to import the C++ module, fall back to Python if not available
try:
    import revenue_analytics_cpp
    RevenueAnalytics = revenue_analytics_cpp.RevenueAnalytics
    CPP_AVAILABLE = True
    print("Using fast C++ revenue analytics module")
except ImportError:
    RevenueAnalytics = PythonRevenueAnalytics
    CPP_AVAILABLE = False
    warnings.warn("C++ module not available, using Python fallback. "
                 "Run 'python setup.py build_ext --inplace' to compile C++ module.")


def get_revenue_analytics():
    """Factory function to get revenue analytics instance"""
    return RevenueAnalytics()


def is_cpp_available():
    """Check if C++ module is available"""
    return CPP_AVAILABLE