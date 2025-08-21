#!/usr/bin/env python3
"""
Update monthly aggregation to use ONLY actual data up to August 2025
Exclude projected data from Sep-Dec 2025 for proper forecasting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os

# Add app directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

try:
    from database import get_database
    database_available = True
except ImportError:
    database_available = False
    print("Database module not available")

def create_actuals_only_monthly_data():
    """Create monthly aggregated data using ONLY actual historical data"""
    
    # Read the combined data
    data_file = Path(__file__).parent / 'data' / 'processed' / 'combined_all_years_data.csv'
    
    if not data_file.exists():
        print(f"Combined data file not found at: {data_file}")
        print("Please run extract_all_years_data.py first")
        return False
    
    try:
        # Read the data
        df = pd.read_csv(data_file)
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # CRITICAL: Filter to actual data only (up to August 2025)
        # This removes projected data from Sep-Dec 2025
        cutoff_date = datetime(2025, 8, 31)
        df_actuals = df[df['Date'] <= cutoff_date].copy()
        
        print(f"FILTERED DAILY DATA:")
        print(f"   Original records: {len(df)}")
        print(f"   Actual records (up to Aug 2025): {len(df_actuals)}")
        print(f"   Removed projected records: {len(df) - len(df_actuals)}")
        print(f"   Date range: {df_actuals['Date'].min().strftime('%Y-%m-%d')} to {df_actuals['Date'].max().strftime('%Y-%m-%d')}")
        
        # Create Year-Month column for grouping
        df_actuals['Year'] = df_actuals['Date'].dt.year
        df_actuals['Month'] = df_actuals['Date'].dt.month
        
        # Clean and standardize columns
        if 'Rm_Sold' in df_actuals.columns:
            df_actuals['Rooms_Sold'] = df_actuals['Rm_Sold']
        elif 'Rm Sold' in df_actuals.columns:
            df_actuals['Rooms_Sold'] = df_actuals['Rm Sold']
        
        # Ensure we have the key columns
        required_cols = ['Revenue', 'Rooms_Sold']
        missing_cols = [col for col in required_cols if col not in df_actuals.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            print(f"Available columns: {list(df_actuals.columns)}")
            return False
        
        # Group by Year and Month and aggregate
        monthly_agg = df_actuals.groupby(['Year', 'Month']).agg({
            'Revenue': ['sum', 'mean', 'count'],
            'Rooms_Sold': ['sum', 'mean'],
            'ADR': ['mean', 'median'],
            'RevPar': ['mean', 'median'],
            'Occupancy_Pct': ['mean', 'median'],
            'Date': ['min', 'max']  # First and last date of month
        }).round(2)
        
        # Flatten column names
        monthly_agg.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in monthly_agg.columns]
        
        # Reset index to make Year and Month regular columns
        monthly_agg = monthly_agg.reset_index()
        
        # Rename columns for clarity
        column_mapping = {
            'Revenue_sum': 'Total_Revenue',
            'Revenue_mean': 'Avg_Daily_Revenue',
            'Revenue_count': 'Days_in_Month',
            'Rooms_Sold_sum': 'Total_Rooms_Sold',
            'Rooms_Sold_mean': 'Avg_Daily_Rooms_Sold',
            'ADR_mean': 'Avg_ADR',
            'ADR_median': 'Median_ADR',
            'RevPar_mean': 'Avg_RevPar',
            'RevPar_median': 'Median_RevPar',
            'Occupancy_Pct_mean': 'Avg_Occupancy_Pct',
            'Occupancy_Pct_median': 'Median_Occupancy_Pct',
            'Date_min': 'Month_Start_Date',
            'Date_max': 'Month_End_Date'
        }
        
        monthly_agg = monthly_agg.rename(columns=column_mapping)
        
        # Add calculated fields
        monthly_agg['Month_Name'] = pd.to_datetime(monthly_agg[['Year', 'Month']].assign(day=1)).dt.strftime('%B')
        monthly_agg['Year_Month'] = pd.to_datetime(monthly_agg[['Year', 'Month']].assign(day=1)).dt.strftime('%Y-%m')
        
        # Calculate average occupancy based on rooms sold (assuming 339 total rooms)
        monthly_agg['Calculated_Avg_Occupancy_Pct'] = (monthly_agg['Avg_Daily_Rooms_Sold'] / 339) * 100
        
        # Add actuals flag
        monthly_agg['Data_Type'] = 'Actual'
        
        # Reorder columns
        column_order = [
            'Year', 'Month', 'Month_Name', 'Year_Month', 'Data_Type',
            'Days_in_Month', 'Month_Start_Date', 'Month_End_Date',
            'Total_Revenue', 'Avg_Daily_Revenue',
            'Total_Rooms_Sold', 'Avg_Daily_Rooms_Sold',
            'Avg_ADR', 'Median_ADR',
            'Avg_RevPar', 'Median_RevPar',
            'Avg_Occupancy_Pct', 'Median_Occupancy_Pct', 'Calculated_Avg_Occupancy_Pct'
        ]
        
        # Select columns that exist
        available_columns = [col for col in column_order if col in monthly_agg.columns]
        monthly_agg = monthly_agg[available_columns]
        
        # Sort by Year and Month
        monthly_agg = monthly_agg.sort_values(['Year', 'Month']).reset_index(drop=True)
        
        # Save to CSV
        output_file = Path(__file__).parent / 'data' / 'processed' / 'monthly_forecast_data_actuals_only.csv'
        monthly_agg.to_csv(output_file, index=False)
        
        print(f"\n=== ACTUALS-ONLY MONTHLY DATA ===")
        print(f"Total months: {len(monthly_agg)}")
        print(f"Years covered: {sorted(monthly_agg['Year'].unique())}")
        print(f"Months per year:")
        for year in sorted(monthly_agg['Year'].unique()):
            year_data = monthly_agg[monthly_agg['Year'] == year]
            print(f"  {year}: {len(year_data)} months")
        
        # Show last few months to verify cutoff
        print(f"\nLast 6 months (should end at Aug 2025):")
        print(monthly_agg[['Year_Month', 'Data_Type', 'Total_Revenue']].tail(6).to_string(index=False))
        
        print(f"Saved to: {output_file}")
        
        # Update database if available
        if database_available:
            try:
                db = get_database()
                success = db.ingest_monthly_forecast_data(monthly_agg)
                if success:
                    print("Updated database with actuals-only monthly data")
                else:
                    print("Failed to update database")
            except Exception as e:
                print(f"Database update error: {e}")
        
        return monthly_agg
        
    except Exception as e:
        print(f"Error aggregating actuals-only data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    create_actuals_only_monthly_data()