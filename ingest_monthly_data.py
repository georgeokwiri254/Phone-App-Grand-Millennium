#!/usr/bin/env python3
"""
Script to ingest monthly aggregated data into SQL database
"""

import pandas as pd
from pathlib import Path
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

def ingest_monthly_data():
    """Load and ingest monthly aggregated data into database"""
    
    if not database_available:
        print("Cannot proceed without database module")
        return False
    
    # Path to monthly aggregated data
    data_file = Path(__file__).parent / 'data' / 'processed' / 'monthly_aggregated_data.csv'
    
    if not data_file.exists():
        print(f"Monthly data file not found at: {data_file}")
        print("Please run aggregate_monthly_data.py first")
        return False
    
    try:
        # Read the CSV
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} monthly records")
        
        # Get database instance
        db = get_database()
        
        # Ingest data
        success = db.ingest_monthly_forecast_data(df)
        
        if success:
            print("Monthly forecast data successfully ingested into database")
            
            # Verify ingestion
            test_data = db.get_monthly_forecast_data()
            print(f"Verification: {len(test_data)} records found in database")
            
            return True
        else:
            print("Failed to ingest monthly forecast data")
            return False
            
    except Exception as e:
        print(f"Error ingesting monthly data: {e}")
        return False

if __name__ == "__main__":
    ingest_monthly_data()