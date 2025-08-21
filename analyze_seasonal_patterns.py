#!/usr/bin/env python3
"""
Analyze seasonal patterns in historical data to ensure proper forecasting
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_seasonality():
    """Analyze seasonal patterns in the actuals-only monthly data"""
    
    # Read the actuals-only monthly data
    data_file = Path(__file__).parent / 'data' / 'processed' / 'monthly_forecast_data_actuals_only.csv'
    
    if not data_file.exists():
        print(f"Actuals-only data file not found at: {data_file}")
        return False
    
    try:
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} months of actual data")
        print(f"Data range: {df['Year_Month'].min()} to {df['Year_Month'].max()}")
        
        # Analyze monthly patterns
        monthly_patterns = df.groupby('Month').agg({
            'Total_Revenue': ['mean', 'std', 'min', 'max'],
            'Avg_Occupancy_Pct': ['mean', 'std'],
            'Avg_ADR': ['mean', 'std']
        }).round(2)
        
        print("\n=== MONTHLY SEASONAL PATTERNS ===")
        print("Month | Avg Revenue | Std Dev | Min Revenue | Max Revenue | Avg Occ%")
        print("-" * 70)
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month in range(1, 13):
            if month <= 12:  # All months
                month_data = df[df['Month'] == month]
                if len(month_data) > 0:
                    avg_rev = month_data['Total_Revenue'].mean()
                    std_rev = month_data['Total_Revenue'].std()
                    min_rev = month_data['Total_Revenue'].min()
                    max_rev = month_data['Total_Revenue'].max()
                    avg_occ = month_data['Avg_Occupancy_Pct'].mean()
                    
                    print(f"{month_names[month-1]:>3} | {avg_rev:>11,.0f} | {std_rev:>7,.0f} | {min_rev:>11,.0f} | {max_rev:>11,.0f} | {avg_occ:>6.1f}%")
        
        # Identify peak and low seasons
        monthly_avg = df.groupby('Month')['Total_Revenue'].mean()
        overall_avg = df['Total_Revenue'].mean()
        
        print(f"\n=== SEASONALITY ANALYSIS ===")
        print(f"Overall Average Monthly Revenue: AED {overall_avg:,.0f}")
        
        peak_months = monthly_avg[monthly_avg > overall_avg * 1.2].index.tolist()
        low_months = monthly_avg[monthly_avg < overall_avg * 0.8].index.tolist()
        
        print(f"PEAK SEASON months (>20% above avg): {[month_names[m-1] for m in peak_months]}")
        print(f"LOW SEASON months (<20% below avg): {[month_names[m-1] for m in low_months]}")
        
        # Year-over-year trends
        print(f"\n=== YEAR-OVER-YEAR TRENDS ===")
        for year in sorted(df['Year'].unique()):
            year_data = df[df['Year'] == year]
            total_rev = year_data['Total_Revenue'].sum()
            avg_occ = year_data['Avg_Occupancy_Pct'].mean()
            avg_adr = year_data['Avg_ADR'].mean()
            months_count = len(year_data)
            
            print(f"{year}: AED {total_rev:>10,.0f} | {avg_occ:>5.1f}% occ | AED {avg_adr:>5.0f} ADR | ({months_count} months)")
        
        return True
        
    except Exception as e:
        print(f"Error analyzing seasonality: {e}")
        return False

if __name__ == "__main__":
    analyze_seasonality()