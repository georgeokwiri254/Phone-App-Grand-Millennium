"""
Block Data Converter
Converts block data from TXT format to CSV with data cleaning and standardization
"""

import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def convert_block_data(file_path: str, output_dir: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """
    Convert block data from TXT to CSV format with proper data cleaning
    
    Args:
        file_path: Path to the TXT file containing block data
        output_dir: Directory to save the converted CSV file
        
    Returns:
        Tuple of (DataFrame, output_path)
    """
    try:
        logger.info(f"Starting block data conversion for: {file_path}")
        
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
        df_cleaned = clean_block_data(df)
        
        # Set output path
        if output_dir is None:
            project_root = Path(__file__).parent.parent
            output_dir = project_root / 'data' / 'processed'
            output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"block_data_{timestamp}.csv"
        
        # Save to CSV
        df_cleaned.to_csv(output_path, index=False)
        logger.info(f"Block data converted and saved to: {output_path}")
        
        return df_cleaned, str(output_path)
        
    except Exception as e:
        logger.error(f"Error converting block data: {e}")
        raise

def clean_block_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize block data
    
    Args:
        df: Raw block data DataFrame
        
    Returns:
        Cleaned DataFrame with standardized columns
    """
    try:
        logger.info("Starting block data cleaning...")
        
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
        
        logger.info(f"Block data cleaning completed. Final shape: {df_clean.shape}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error cleaning block data: {e}")
        raise

def get_block_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for block data
    
    Args:
        df: Block data DataFrame
        
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
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating block data summary: {e}")
        return {}

def run_block_conversion(file_path: str) -> Tuple[pd.DataFrame, str]:
    """
    Main function to run block data conversion
    
    Args:
        file_path: Path to the block data TXT file
        
    Returns:
        Tuple of (DataFrame, output_path)
    """
    try:
        logger.info(f"=== STARTING BLOCK DATA CONVERSION ===")
        logger.info(f"Input file: {file_path}")
        
        # Convert the data
        df, output_path = convert_block_data(file_path)
        
        # Generate summary
        summary = get_block_data_summary(df)
        logger.info(f"Conversion summary: {summary}")
        
        logger.info(f"=== BLOCK DATA CONVERSION COMPLETED ===")
        
        return df, output_path
        
    except Exception as e:
        logger.error(f"Block data conversion failed: {e}")
        raise