"""
Last Year Block Data Converter
Converts last year's block data from TXT format to CSV and loads it into the SQL database
"""

import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def convert_last_year_block_data(file_path: str, output_dir: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """
    Convert last year's block data from TXT to CSV format with proper data cleaning
    
    Args:
        file_path: Path to the TXT file containing last year's block data
        output_dir: Directory to save the converted CSV file
        
    Returns:
        Tuple of (DataFrame, output_path)
    """
    try:
        logger.info(f"Starting last year block data conversion for: {file_path}")
        
        # Read the TXT file (tab-separated) with error handling
        try:
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8', on_bad_lines='skip')
        except Exception as e:
            logger.warning(f"First attempt failed: {e}, trying with different parameters...")
            # Try with different parameters if initial read fails
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8', 
                           on_bad_lines='skip', quoting=3, engine='python')
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Clean and standardize the data
        df_cleaned = clean_last_year_block_data(df)
        
        # Set output path
        if output_dir is None:
            project_root = Path(__file__).parent.parent
            output_dir = project_root / 'data' / 'processed'
            output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"last_year_block_data_{timestamp}.csv"
        
        # Save to CSV
        df_cleaned.to_csv(output_path, index=False)
        logger.info(f"Last year block data converted and saved to: {output_path}")
        
        return df_cleaned, str(output_path)
        
    except Exception as e:
        logger.error(f"Error converting last year block data: {e}")
        raise

def clean_last_year_block_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize last year's block data
    
    Args:
        df: Raw last year block data DataFrame
        
    Returns:
        Cleaned DataFrame with standardized columns and last year flag
    """
    try:
        logger.info("Starting last year block data cleaning...")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Select and rename key columns based on the specification
        key_columns = {
            'BLOCKSIZE': 'BlockSize',
            'ALLOTMENT_DATE': 'AllotmentDate', 
            'SREP_CODE': 'SrepCode',
            'BOOKING_STATUS': 'BookingStatus',
            'DESCRIPTION': 'CompanyName',
            'BEGIN_DATE': 'BeginDate',
            'ALLOTMENT_CODE': 'AllotmentCode',
            'RATE_CODE': 'RateCode',
            'MONTH_DESC': 'MonthDesc',
            'WEEK_DAY': 'WeekDay',
            'DAY_OF_MONTH': 'DayOfMonth'
        }
        
        # Select only available columns
        available_columns = {k: v for k, v in key_columns.items() if k in df_clean.columns}
        df_clean = df_clean[list(available_columns.keys())].rename(columns=available_columns)
        
        # Convert date columns
        date_columns = ['AllotmentDate', 'BeginDate']
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        # Clean BlockSize - ensure it's numeric
        if 'BlockSize' in df_clean.columns:
            df_clean['BlockSize'] = pd.to_numeric(df_clean['BlockSize'], errors='coerce')
            df_clean['BlockSize'] = df_clean['BlockSize'].fillna(0).astype(int)
        
        # Clean booking status
        if 'BookingStatus' in df_clean.columns:
            df_clean['BookingStatus'] = df_clean['BookingStatus'].str.upper().str.strip()
            # Map any variations to standard codes
            status_mapping = {
                'ACTUAL': 'ACT',
                'DEFINITE': 'DEF', 
                'PROSPECT': 'PSP',
                'TENTATIVE': 'TEN'
            }
            df_clean['BookingStatus'] = df_clean['BookingStatus'].map(status_mapping).fillna(df_clean['BookingStatus'])
        
        # Clean company names
        if 'CompanyName' in df_clean.columns:
            df_clean['CompanyName'] = df_clean['CompanyName'].str.strip()
        
        # Clean sales rep codes
        if 'SrepCode' in df_clean.columns:
            df_clean['SrepCode'] = df_clean['SrepCode'].str.strip()
        
        # Add derived columns
        if 'AllotmentDate' in df_clean.columns:
            df_clean['Year'] = df_clean['AllotmentDate'].dt.year
            df_clean['Month'] = df_clean['AllotmentDate'].dt.month
            df_clean['Quarter'] = df_clean['AllotmentDate'].dt.quarter
        
        # Add last year flag
        df_clean['IsLastYear'] = True
        df_clean['DataSource'] = 'LastYear'
        
        # Remove rows with missing critical data
        critical_columns = ['BlockSize', 'BookingStatus']
        for col in critical_columns:
            if col in df_clean.columns:
                initial_count = len(df_clean)
                df_clean = df_clean.dropna(subset=[col])
                removed_count = initial_count - len(df_clean)
                if removed_count > 0:
                    logger.warning(f"Removed {removed_count} rows with missing {col}")
        
        # Sort by AllotmentDate
        if 'AllotmentDate' in df_clean.columns:
            df_clean = df_clean.sort_values('AllotmentDate')
        
        logger.info(f"Last year block data cleaning completed. Final shape: {df_clean.shape}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error cleaning last year block data: {e}")
        raise

def load_last_year_block_to_database(df: pd.DataFrame, db_connection=None) -> bool:
    """
    Load last year's block data into the SQL database
    
    Args:
        df: Cleaned last year block data DataFrame
        db_connection: Database connection object
        
    Returns:
        Boolean indicating success
    """
    try:
        logger.info("Loading last year block data to database...")
        
        if db_connection is None:
            from app.database import get_database
            db = get_database()
        else:
            db = db_connection
        
        # Insert the data into the database
        success = db.ingest_block_data(df, table_suffix='_last_year')
        
        if success:
            logger.info(f"Successfully loaded {len(df)} last year block records to database")
        else:
            logger.error("Failed to load last year block data to database")
        
        return success
        
    except Exception as e:
        logger.error(f"Error loading last year block data to database: {e}")
        return False

def get_last_year_block_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for last year's block data
    
    Args:
        df: Last year block data DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    try:
        summary = {}
        
        if 'BlockSize' in df.columns:
            summary['total_blocks'] = df['BlockSize'].sum()
            summary['avg_block_size'] = df['BlockSize'].mean()
            summary['max_block_size'] = df['BlockSize'].max()
        
        if 'BookingStatus' in df.columns:
            summary['booking_status_distribution'] = df['BookingStatus'].value_counts().to_dict()
        
        if 'CompanyName' in df.columns:
            summary['unique_companies'] = df['CompanyName'].nunique()
            summary['top_companies'] = df.groupby('CompanyName')['BlockSize'].sum().nlargest(5).to_dict()
        
        if 'SrepCode' in df.columns:
            summary['unique_sales_reps'] = df['SrepCode'].nunique()
        
        summary['total_records'] = len(df)
        summary['date_range'] = {
            'start': df['AllotmentDate'].min() if 'AllotmentDate' in df.columns else None,
            'end': df['AllotmentDate'].max() if 'AllotmentDate' in df.columns else None
        }
        
        # Add last year specific info
        if 'Year' in df.columns:
            summary['years_covered'] = sorted(df['Year'].unique().tolist())
            summary['is_last_year_data'] = True
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating last year block data summary: {e}")
        return {}

def run_last_year_block_conversion(file_path: str, load_to_db: bool = True) -> Tuple[pd.DataFrame, str, bool]:
    """
    Main function to run last year block data conversion and database loading
    
    Args:
        file_path: Path to the last year block data TXT file
        load_to_db: Whether to load the data to database
        
    Returns:
        Tuple of (DataFrame, output_path, db_success)
    """
    try:
        logger.info(f"=== STARTING LAST YEAR BLOCK DATA CONVERSION ===")
        logger.info(f"Input file: {file_path}")
        
        # Convert the data
        df, output_path = convert_last_year_block_data(file_path)
        
        # Generate summary
        summary = get_last_year_block_summary(df)
        logger.info(f"Last year conversion summary: {summary}")
        
        # Load to database if requested
        db_success = False
        if load_to_db:
            db_success = load_last_year_block_to_database(df)
        
        logger.info(f"=== LAST YEAR BLOCK DATA CONVERSION COMPLETED ===")
        
        return df, output_path, db_success
        
    except Exception as e:
        logger.error(f"Last year block data conversion failed: {e}")
        raise