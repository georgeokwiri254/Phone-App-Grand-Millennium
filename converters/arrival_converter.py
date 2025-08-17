"""
Arrival Report Converter
Processes the ARRIVAL CHECK sheet from Excel files.

Key Features:
- Flag specific companies (T- Assos Tourism LLC, T-CR7 Wonders Tourism, etc.)
- Flag bookings longer than 10 days
- Analyze arrival patterns and deposit payments
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Companies to flag
FLAGGED_COMPANIES = [
    "T- Assos Tourism LLC",
    "T-CR7 Wonders Tourism", 
    "T- Neon Reisen GmbH",
    "T- Kurban Tours",
    "T- Kalanit Tours Dubai"
]

def flag_company_no_deposit(deposit_paid):
    """Flag companies based on no deposit paid"""
    if pd.isna(deposit_paid):
        return 1  # Flag if no deposit info
    return 1 if deposit_paid == 0 else 0

def flag_long_bookings(nights):
    """Flag bookings longer than 10 days"""
    if pd.isna(nights):
        return 0
    return 1 if nights > 10 else 0

def calculate_adr(amount, nights):
    """Calculate ADR (Average Daily Rate)"""
    if pd.isna(amount) or pd.isna(nights) or nights == 0:
        return 0
    return amount / nights

def clean_company_name(company_name):
    """Clean and standardize company names"""
    if pd.isna(company_name):
        return "Unknown"
    return str(company_name).strip()

def determine_season(arrival_date):
    """Determine if arrival is in summer or winter season"""
    if pd.isna(arrival_date):
        return "Unknown"
    
    month = arrival_date.month
    # Summer: May to September (5-9), Winter: October to April (10-12, 1-4)
    if month in [5, 6, 7, 8, 9]:
        return 'Summer'
    else:
        return 'Winter'

def process_arrival_report(file_path, output_csv_path=None):
    """
    Process the Arrival Excel file and create enhanced CSV.
    
    Args:
        file_path: Path to the Excel file
        output_csv_path: Path for output CSV (optional)
        
    Returns:
        Tuple of (DataFrame, csv_path)
    """
    try:
        logger.info(f"Processing Arrival report from: {file_path}")
        
        # Read the ARRIVAL CHECK sheet
        df = pd.read_excel(file_path, sheet_name='ARRIVAL CHECK')
        logger.info(f"Loaded {len(df)} records from ARRIVAL CHECK sheet")
        
        # Remove rows with no confirmation number in column A (HCN)
        initial_count = len(df)
        # Remove rows where HCN is null, 0, 0.0, empty string, or any invalid value
        mask = (df[df.columns[0]].notna()) & (df[df.columns[0]] != 0) & (df[df.columns[0]] != 0.0) & (df[df.columns[0]] != '') & (df[df.columns[0]].astype(str).str.strip() != '')
        df = df[mask]
        removed_count = initial_count - len(df)
        logger.info(f"Removed {removed_count} rows without valid confirmation numbers. {len(df)} records remaining.")
        
        # Convert date columns
        df['ARRIVAL'] = pd.to_datetime(df['ARRIVAL'], errors='coerce')
        df['DEPARTURE'] = pd.to_datetime(df['DEPARTURE'], errors='coerce')
        
        # Clean and enhance data
        df['COMPANY_NAME_CLEAN'] = df['COMPANY_NAME'].apply(clean_company_name)
        df['COMPANY_FLAGGED'] = df['DEPOSIT_PAID'].apply(flag_company_no_deposit)
        df['LONG_BOOKING_FLAG'] = df['NIGHTS'].apply(flag_long_bookings)
        df['SEASON'] = df['ARRIVAL'].apply(determine_season)
        
        # Calculate ADR
        df['CALCULATED_ADR'] = df.apply(lambda row: calculate_adr(row['AMOUNT'], row['NIGHTS']), axis=1)
        
        # Handle deposit payments
        df['DEPOSIT_PAID_CLEAN'] = df['DEPOSIT_PAID'].fillna(0)
        df['HAS_DEPOSIT'] = (df['DEPOSIT_PAID_CLEAN'] > 0).astype(int)
        
        # Calculate arrival count (for company analysis)
        df['ARRIVAL_COUNT'] = 1
        
        # Add booking lead time if arrival time is available
        df['ARRIVAL_TIME_CLEAN'] = df['ARRIVAL_TIME1'].fillna('')
        
        # Handle rate codes
        df['RATE_CODE_CLEAN'] = df['RATE_CODE'].fillna('Unknown')
        
        # Clean numerical columns
        numerical_cols = ['NIGHTS', 'PERSONS', 'DEPOSIT_PAID', 'AMOUNT', 'TDF', 'APPROX NET ', 'APPROX TOTAL']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Add year and month for analysis
        df['ARRIVAL_YEAR'] = df['ARRIVAL'].dt.year
        df['ARRIVAL_MONTH'] = df['ARRIVAL'].dt.month
        df['ARRIVAL_MONTH_NAME'] = df['ARRIVAL'].dt.strftime('%B')
        
        # Add day of week
        df['ARRIVAL_DOW'] = df['ARRIVAL'].dt.day_name()
        
        logger.info(f"Enhanced data processing completed")
        
        # Log flagged companies
        flagged_count = df['COMPANY_FLAGGED'].sum()
        long_booking_count = df['LONG_BOOKING_FLAG'].sum()
        logger.info(f"Flagged companies: {flagged_count} records")
        logger.info(f"Long bookings (>10 days): {long_booking_count} records")
        
        # Set output path if not provided
        if output_csv_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_csv_path = f"data/processed/arrival_check_{timestamp}.csv"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved processed data to: {output_csv_path}")
        
        # Also save canonical version
        canonical_path = "data/processed/arrival_check.csv"
        df.to_csv(canonical_path, index=False)
        logger.info(f"Saved canonical version to: {canonical_path}")
        
        return df, canonical_path
        
    except Exception as e:
        logger.error(f"Error processing Arrival report: {e}")
        raise

def get_arrival_summary_stats(df):
    """Get summary statistics for the arrival data"""
    stats = {
        'total_arrivals': len(df),
        'unique_companies': df['COMPANY_NAME_CLEAN'].nunique(),
        'total_amount': df['AMOUNT'].sum(),
        'total_deposit_paid': df['DEPOSIT_PAID_CLEAN'].sum(),
        'flagged_companies': df['COMPANY_FLAGGED'].sum(),
        'long_bookings': df['LONG_BOOKING_FLAG'].sum(),
        'average_adr': df['CALCULATED_ADR'].mean(),
        'arrivals_with_deposit': df['HAS_DEPOSIT'].sum(),
        'top_companies': df.groupby('COMPANY_NAME_CLEAN')['ARRIVAL_COUNT'].sum().sort_values(ascending=False).head(5).to_dict(),
        'seasonal_breakdown': df.groupby('SEASON')['ARRIVAL_COUNT'].sum().to_dict(),
        'monthly_breakdown': df.groupby('ARRIVAL_MONTH_NAME')['ARRIVAL_COUNT'].sum().to_dict()
    }
    return stats

def get_flagged_companies_report(df):
    """Get detailed report on flagged companies"""
    flagged_df = df[df['COMPANY_FLAGGED'] == 1]
    
    if len(flagged_df) == 0:
        return "No flagged companies found"
    
    report = flagged_df.groupby('COMPANY_NAME_CLEAN').agg({
        'ARRIVAL_COUNT': 'sum',
        'AMOUNT': 'sum',
        'DEPOSIT_PAID_CLEAN': 'sum',
        'NIGHTS': 'sum'
    }).round(2)
    
    return report

if __name__ == "__main__":
    # Test the converter
    file_path = "19-08  res_arrivals 2025.xlsm"
    if os.path.exists(file_path):
        df, csv_path = process_arrival_report(file_path)
        stats = get_arrival_summary_stats(df)
        print("Arrival Summary Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\nFlagged Companies Report:")
        flagged_report = get_flagged_companies_report(df)
        print(flagged_report)
    else:
        print(f"File not found: {file_path}")