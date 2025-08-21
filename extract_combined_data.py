#!/usr/bin/env python3
"""
Extract combined forecast data from SQLite database and save as CSV
"""

import sqlite3
import pandas as pd
from pathlib import Path

def extract_combined_data():
    """Extract combined forecast data from database"""
    
    # Database path
    db_path = Path(__file__).parent / 'db' / 'revenue.db'
    
    if not db_path.exists():
        print(f"Database not found at: {db_path}")
        return
    
    try:
        # Connect to database
        conn = sqlite3.connect(str(db_path))
        
        # Check if combined_forecast_data table exists
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables_df = pd.read_sql_query(tables_query, conn)
        print("Available tables:")
        for table in tables_df['name']:
            print(f"  - {table}")
        
        if 'combined_forecast_data' in tables_df['name'].values:
            # Extract combined forecast data
            query = "SELECT * FROM combined_forecast_data ORDER BY Date"
            combined_df = pd.read_sql_query(query, conn)
            
            if not combined_df.empty:
                # Save to CSV
                output_file = Path(__file__).parent / 'data' / 'processed' / 'combined_forecast_data.csv'
                combined_df.to_csv(output_file, index=False)
                
                print(f"\nCombined forecast data extracted:")
                print(f"  Records: {len(combined_df)}")
                print(f"  Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
                print(f"  Saved to: {output_file}")
                
                # Show first few rows
                print(f"\nFirst 10 rows:")
                print(combined_df.head(10).to_string())
                
            else:
                print("\nCombined forecast data table is empty")
                
        else:
            print("\nCombined forecast data table not found")
            
            # Try to extract historical occupancy data instead
            print("\nLooking for historical occupancy tables...")
            historical_tables = [table for table in tables_df['name'].values if 'historical_occupancy' in table]
            
            if historical_tables:
                print(f"Found historical tables: {historical_tables}")
                
                all_data = []
                for table in historical_tables:
                    query = f"SELECT * FROM {table} ORDER BY Date"
                    df = pd.read_sql_query(query, conn)
                    if not df.empty:
                        df['Source'] = table
                        all_data.append(df)
                        print(f"  {table}: {len(df)} records")
                
                if all_data:
                    combined_historical = pd.concat(all_data, ignore_index=True)
                    output_file = Path(__file__).parent / 'data' / 'processed' / 'combined_historical_data.csv'
                    combined_historical.to_csv(output_file, index=False)
                    
                    print(f"\nCombined historical data:")
                    print(f"  Total records: {len(combined_historical)}")
                    print(f"  Date range: {combined_historical['Date'].min()} to {combined_historical['Date'].max()}")
                    print(f"  Saved to: {output_file}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error extracting data: {e}")

if __name__ == "__main__":
    extract_combined_data()