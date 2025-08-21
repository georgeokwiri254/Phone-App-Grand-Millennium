#!/usr/bin/env python3
"""
Aggregate daily data to monthly summaries from combined all years dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

def aggregate_to_monthly():
    """Aggregate daily data to monthly summaries"""
    
    # Read the combined data
    data_file = Path(__file__).parent / 'data' / 'processed' / 'combined_all_years_data.csv'
    
    if not data_file.exists():
        print(f"Combined data file not found at: {data_file}")
        print("Please run extract_all_years_data.py first")
        return
    
    try:
        # Read the data
        df = pd.read_csv(data_file)
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create Year-Month column for grouping
        df['Year_Month'] = df['Date'].dt.to_period('M')
        
        print(f"Loaded {len(df)} daily records")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Clean and standardize columns
        # Handle different column names between years
        if 'Rm_Sold' in df.columns:
            df['Rooms_Sold'] = df['Rm_Sold']
        elif 'Rm Sold' in df.columns:
            df['Rooms_Sold'] = df['Rm Sold']
        
        # Ensure we have the key columns
        required_cols = ['Revenue', 'Rooms_Sold']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Group by Year and Month and aggregate
        monthly_agg = df.groupby(['Year', 'Month']).agg({
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
        
        # Reorder columns
        column_order = [
            'Year', 'Month', 'Month_Name', 'Year_Month',
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
        output_file = Path(__file__).parent / 'data' / 'processed' / 'monthly_aggregated_data.csv'
        monthly_agg.to_csv(output_file, index=False)
        
        print(f"\n=== MONTHLY AGGREGATED DATA ===")
        print(f"Total months: {len(monthly_agg)}")
        print(f"Years covered: {sorted(monthly_agg['Year'].unique())}")
        print(f"Months per year:")
        for year in sorted(monthly_agg['Year'].unique()):
            year_data = monthly_agg[monthly_agg['Year'] == year]
            print(f"  {year}: {len(year_data)} months")
        print(f"Saved to: {output_file}")
        
        # Show sample data
        print(f"\nSample monthly data (first 5 months):")
        display_cols = ['Year_Month', 'Total_Revenue', 'Total_Rooms_Sold', 'Avg_ADR', 'Avg_Occupancy_Pct']
        available_display_cols = [col for col in display_cols if col in monthly_agg.columns]
        print(monthly_agg[available_display_cols].head().to_string(index=False))
        
        # Show summary statistics
        print(f"\nSummary by Year:")
        yearly_summary = monthly_agg.groupby('Year').agg({
            'Total_Revenue': 'sum',
            'Total_Rooms_Sold': 'sum',
            'Avg_ADR': 'mean',
            'Avg_Occupancy_Pct': 'mean'
        }).round(2)
        print(yearly_summary.to_string())
        
        return monthly_agg
        
    except Exception as e:
        print(f"Error aggregating data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    aggregate_to_monthly()