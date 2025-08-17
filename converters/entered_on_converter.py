"""
Entered On Report Converter
Processes the ENTERED ON sheet from Excel files and splits stay periods across months.

Key Logic:
- If a guest checks in on 30 August and checks out on 2 September:
  * 2 days in August, 1 day in September (checkout day not counted)
  * Total stays = 3 nights
- Split AMOUNT and room nights proportionally across months
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_stay_across_months(arrival_date, departure_date, total_amount, total_nights):
    """
    Split a stay period across months, calculating nights and amount per month.
    
    Args:
        arrival_date: Check-in date
        departure_date: Check-out date  
        total_amount: Total booking amount
        total_nights: Total nights stayed
        
    Returns:
        List of dictionaries with month splits
    """
    splits = []
    current_date = arrival_date
    
    while current_date < departure_date:
        # Calculate end of current month or departure date, whichever is earlier
        end_of_month = (current_date + relativedelta(months=1)).replace(day=1) - timedelta(days=1)
        period_end = min(end_of_month, departure_date - timedelta(days=1))  # Don't count checkout day
        
        # Calculate nights in this month
        if period_end >= current_date:
            nights_in_month = (period_end - current_date).days + 1
            
            # Calculate proportional amount for this month
            amount_for_month = (nights_in_month / total_nights) * total_amount if total_nights > 0 else 0
            
            splits.append({
                'month': current_date.strftime('%Y-%m'),
                'year': current_date.year,
                'month_num': current_date.month,
                'nights_in_month': nights_in_month,
                'amount_in_month': amount_for_month,
                'period_start': current_date,
                'period_end': period_end
            })
        
        # Move to first day of next month
        current_date = (current_date + relativedelta(months=1)).replace(day=1)
    
    return splits

def flag_long_bookings(nights):
    """Flag bookings longer than 10 days"""
    return 1 if nights > 10 else 0

def determine_season(arrival_date):
    """Determine if booking is in summer or winter season"""
    month = arrival_date.month
    # Summer: May to September (5-9), Winter: October to April (10-12, 1-4)
    if month in [5, 6, 7, 8, 9]:
        return 'Summer'
    else:
        return 'Winter'

def create_monthly_matrix(df):
    """
    Create monthly matrix format as shown in the screenshot.
    Converts split data into columns from current month till December 2026.
    """
    # Generate monthly columns from current month till December 2026
    current_date = datetime.now()
    end_date = datetime(2026, 12, 31)
    
    months_list = []
    temp_date = current_date.replace(day=1)  # Start from beginning of current month
    
    while temp_date <= end_date:
        month_abbr = temp_date.strftime('%b').upper()
        year = temp_date.year
        if year == current_date.year:
            # For current year, just use month abbreviation
            column_name = month_abbr
        else:
            # For future years, include year
            column_name = f"{month_abbr}{year}"
        months_list.append((column_name, temp_date))
        temp_date = (temp_date + relativedelta(months=1))
    
    # Group by booking ID (RESV ID or FULL_NAME) to aggregate monthly data
    if 'RESV ID' in df.columns:
        booking_groups = df.groupby('RESV ID')
    else:
        booking_groups = df.groupby('FULL_NAME')
    
    matrix_rows = []
    
    for booking_id, group in booking_groups:
        # Get the first row for base information
        base_row = group.iloc[0].copy()
        
        # Initialize all monthly columns to 0
        for month_col, _ in months_list:
            base_row[month_col] = 0
        
        # Fill in the actual monthly values
        for _, split_row in group.iterrows():
            split_date = split_row['PERIOD_START']
            
            # Find the matching column for this date
            for month_col, col_date in months_list:
                if (split_date.year == col_date.year and 
                    split_date.month == col_date.month):
                    base_row[month_col] = split_row['NIGHTS_IN_MONTH']
                    break
        
        matrix_rows.append(base_row)
    
    logger.info(f"Created monthly matrix with columns: {[col for col, _ in months_list]}")
    return pd.DataFrame(matrix_rows)

def process_entered_on_report(file_path, output_csv_path=None):
    """
    Process the Entered On Excel file and create expanded CSV with month splits and monthly matrix.
    
    Args:
        file_path: Path to the Excel file
        output_csv_path: Path for output CSV (optional)
        
    Returns:
        Tuple of (DataFrame, csv_path)
    """
    try:
        logger.info(f"Processing Entered On report from: {file_path}")
        
        # Read the ENTERED ON sheet
        df = pd.read_excel(file_path, sheet_name='ENTERED ON')
        logger.info(f"Loaded {len(df)} records from ENTERED ON sheet")
        
        # Convert date columns
        df['ARRIVAL'] = pd.to_datetime(df['ARRIVAL'])
        df['DEPARTURE'] = pd.to_datetime(df['DEPARTURE'])
        
        # Multiply AMOUNT by 1.1 as requested
        df['AMOUNT'] = df['AMOUNT'] * 1.1
        
        # Calculate additional fields (check if Season column already exists)
        if 'Season' not in df.columns:
            df['SEASON'] = df['ARRIVAL'].apply(determine_season)
        else:
            # Use existing Season column and clean it
            df['SEASON'] = df['Season'].fillna('Unknown')
        df['LONG_BOOKING_FLAG'] = df['NIGHTS'].apply(flag_long_bookings)
        
        # Create expanded dataframe with month splits
        expanded_rows = []
        
        for _, row in df.iterrows():
            arrival = row['ARRIVAL']
            departure = row['DEPARTURE']
            total_amount = row['AMOUNT'] if pd.notna(row['AMOUNT']) else 0
            total_nights = row['NIGHTS'] if pd.notna(row['NIGHTS']) else 0
            
            # Get month splits for this booking
            month_splits = split_stay_across_months(arrival, departure, total_amount, total_nights)
            
            # Create a row for each month split
            for split in month_splits:
                new_row = row.copy()
                
                # Add split-specific columns
                new_row['SPLIT_MONTH'] = split['month']
                new_row['SPLIT_YEAR'] = split['year'] 
                new_row['SPLIT_MONTH_NUM'] = split['month_num']
                new_row['NIGHTS_IN_MONTH'] = split['nights_in_month']
                new_row['AMOUNT_IN_MONTH'] = split['amount_in_month']
                new_row['PERIOD_START'] = split['period_start']
                new_row['PERIOD_END'] = split['period_end']
                
                # Calculate ADR for this month split
                if split['nights_in_month'] > 0:
                    new_row['ADR_IN_MONTH'] = split['amount_in_month'] / split['nights_in_month']
                else:
                    new_row['ADR_IN_MONTH'] = 0
                
                expanded_rows.append(new_row)
        
        # Create expanded DataFrame
        expanded_df = pd.DataFrame(expanded_rows)
        
        # Add derived columns for analysis (avoid duplicates)
        if 'BOOKING_LEAD_TIME' not in expanded_df.columns:
            expanded_df['BOOKING_LEAD_TIME'] = expanded_df.get('Booking Lead Time ', 0)
        if 'EVENTS_DATES' not in expanded_df.columns:
            expanded_df['EVENTS_DATES'] = expanded_df.get('Events Dates ', '')
        
        # Clean company name for analysis
        expanded_df['COMPANY_CLEAN'] = expanded_df['C_T_S_NAME'].fillna('Unknown')
        
        # Drop duplicate columns if they exist
        columns_to_drop = ['Season', 'Booking Lead Time ', 'Events Dates ']
        for col in columns_to_drop:
            if col in expanded_df.columns:
                expanded_df = expanded_df.drop(columns=[col])
        
        logger.info(f"Expanded to {len(expanded_df)} records after month splitting")
        
        # Create monthly matrix format
        matrix_df = create_monthly_matrix(expanded_df)
        logger.info(f"Created monthly matrix with {len(matrix_df)} bookings")
        
        # Set output path if not provided
        if output_csv_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_csv_path = f"data/processed/entered_on_{timestamp}.csv"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # Save the monthly matrix format (this is what user wants to see)
        matrix_df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved monthly matrix data to: {output_csv_path}")
        
        # Also save canonical version
        canonical_path = "data/processed/entered_on.csv"
        matrix_df.to_csv(canonical_path, index=False)
        logger.info(f"Saved canonical version to: {canonical_path}")
        
        return matrix_df, canonical_path
        
    except Exception as e:
        logger.error(f"Error processing Entered On report: {e}")
        raise

def get_summary_stats(df):
    """Get summary statistics for the processed data"""
    stats = {
        'total_bookings': len(df['RESV ID'].unique()) if 'RESV ID' in df.columns else len(df),
        'total_amount': df['AMOUNT_IN_MONTH'].sum(),
        'total_room_nights': df['NIGHTS_IN_MONTH'].sum(),
        'average_adr': df['ADR_IN_MONTH'].mean(),
        'long_bookings': df['LONG_BOOKING_FLAG'].sum(),
        'months_covered': df['SPLIT_MONTH'].nunique(),
        'companies_count': df['COMPANY_CLEAN'].nunique(),
        'seasons_breakdown': df.groupby('SEASON')['NIGHTS_IN_MONTH'].sum().to_dict()
    }
    return stats

if __name__ == "__main__":
    # Test the converter
    file_path = "16-08-2025 Entered On.xlsm"
    if os.path.exists(file_path):
        df, csv_path = process_entered_on_report(file_path)
        stats = get_summary_stats(df)
        print("Summary Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print(f"File not found: {file_path}")