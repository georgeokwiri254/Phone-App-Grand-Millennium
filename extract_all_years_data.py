#!/usr/bin/env python3
"""
Extract ALL years data (2022-2025) from SQLite database and create truly combined CSV
"""

import sqlite3
import pandas as pd
from pathlib import Path

def extract_all_years_data():
    """Extract data from all years and create combined dataset"""
    
    # Database path
    db_path = Path(__file__).parent / 'db' / 'revenue.db'
    
    if not db_path.exists():
        print(f"Database not found at: {db_path}")
        return
    
    try:
        # Connect to database
        conn = sqlite3.connect(str(db_path))
        
        all_data = []
        
        # Extract historical occupancy data (2022-2024)
        for year in [2022, 2023, 2024]:
            table_name = f"historical_occupancy_{year}"
            try:
                query = f"SELECT * FROM {table_name} ORDER BY Date"
                df = pd.read_sql_query(query, conn)
                if not df.empty:
                    df['Source'] = f"historical_{year}"
                    df['Year'] = year
                    all_data.append(df)
                    print(f"Extracted {len(df)} records from {table_name}")
                    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
            except Exception as e:
                print(f"Error extracting {table_name}: {e}")
        
        # Extract current occupancy data (2025)
        try:
            query = "SELECT * FROM occupancy_analysis ORDER BY Date"
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                df['Source'] = "current_2025"
                # Parse dates to extract year
                df['Date'] = pd.to_datetime(df['Date'])
                df['Year'] = df['Date'].dt.year
                all_data.append(df)
                print(f"Extracted {len(df)} records from occupancy_analysis")
                print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        except Exception as e:
            print(f"Error extracting occupancy_analysis: {e}")
        
        if all_data:
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True, sort=False)
            
            # Ensure Date column is datetime
            combined_df['Date'] = pd.to_datetime(combined_df['Date'])
            
            # Sort by date
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)
            
            # Add additional time features for analysis
            combined_df['Month'] = combined_df['Date'].dt.month
            combined_df['DayOfWeek'] = combined_df['Date'].dt.dayofweek
            combined_df['DayOfYear'] = combined_df['Date'].dt.dayofyear
            combined_df['Quarter'] = combined_df['Date'].dt.quarter
            
            # Standardize column names (handle different naming between historical and current)
            if 'Rm Sold' in combined_df.columns and 'Rm_Sold' in combined_df.columns:
                # Fill missing values between the two columns
                combined_df['Rm_Sold'] = combined_df['Rm_Sold'].fillna(combined_df['Rm Sold'])
                combined_df = combined_df.drop('Rm Sold', axis=1)
            
            # Calculate occupancy percentage if not present (assuming 339 total rooms)
            if 'Occupancy_Pct' not in combined_df.columns and 'Occ%' not in combined_df.columns:
                if 'Rm_Sold' in combined_df.columns:
                    combined_df['Occupancy_Pct'] = (combined_df['Rm_Sold'] / 339) * 100
                elif 'Rm Sold' in combined_df.columns:
                    combined_df['Occupancy_Pct'] = (combined_df['Rm Sold'] / 339) * 100
            
            # Save to CSV
            output_file = Path(__file__).parent / 'data' / 'processed' / 'combined_all_years_data.csv'
            combined_df.to_csv(output_file, index=False)
            
            print(f"\n=== COMBINED ALL YEARS DATA ===")
            print(f"Total records: {len(combined_df)}")
            print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
            print(f"Years covered: {sorted(combined_df['Year'].unique())}")
            print(f"Records per year:")
            for year in sorted(combined_df['Year'].unique()):
                year_data = combined_df[combined_df['Year'] == year]
                print(f"  {year}: {len(year_data)} records")
            print(f"Saved to: {output_file}")
            
            # Show column names
            print(f"\nColumns: {list(combined_df.columns)}")
            
            # Show sample data from each year
            print(f"\nSample data from each year:")
            for year in sorted(combined_df['Year'].unique()):
                year_data = combined_df[combined_df['Year'] == year]
                if len(year_data) > 0:
                    print(f"\n{year} (showing first 3 rows):")
                    print(year_data.head(3)[['Date', 'Year', 'Source']].to_string(index=False))
            
        else:
            print("No data found to combine")
        
        conn.close()
        
    except Exception as e:
        print(f"Error extracting data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_all_years_data()