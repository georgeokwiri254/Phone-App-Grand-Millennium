"""
STR (Smith Travel Research) Data Converter
Processes STR Excel files according to specifications:
1. Delete first 7 rows
2. Delete columns A, D, K, R
3. Delete any text below the table format
4. Rename columns to standard STR format
5. Save as STR CSV
6. Load to database
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Tuple, Optional

def process_str_file(file_path: str, output_dir: str = None) -> Tuple[pd.DataFrame, str]:
    """
    Process STR Excel file according to specifications.
    
    Args:
        file_path: Path to the STR Excel file
        output_dir: Directory to save processed CSV (defaults to data/processed)
    
    Returns:
        Tuple of (processed_dataframe, csv_file_path)
    """
    
    logging.info(f"Starting STR file processing: {file_path}")
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(file_path)), "data", "processed")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Read Excel file
        logging.info("Reading STR Excel file...")
        df = pd.read_excel(file_path, sheet_name=0)
        
        # Step 1: Delete first 7 rows
        logging.info("Deleting first 7 rows...")
        df = df.iloc[7:].reset_index(drop=True)
        
        # Step 2: Delete columns A, D, K, R (0, 3, 10, 17 in 0-based indexing)
        logging.info("Deleting specified columns (A, D, K, R)...")
        columns_to_drop = []
        if len(df.columns) > 0:
            columns_to_drop.append(0)  # Column A
        if len(df.columns) > 3:
            columns_to_drop.append(3)  # Column D
        if len(df.columns) > 10:
            columns_to_drop.append(10)  # Column K
        if len(df.columns) > 17:
            columns_to_drop.append(17)  # Column R
        
        # Drop columns by index (in reverse order to maintain indices)
        for col_idx in sorted(columns_to_drop, reverse=True):
            df = df.drop(df.columns[col_idx], axis=1)
        
        # Step 3: Delete any text below the table format (remove rows with all NaN or non-numeric data)
        logging.info("Cleaning data below table format...")
        
        # Find the last row with valid date data
        valid_rows = []
        for idx, row in df.iterrows():
            # Check if first column contains date-like data
            first_val = row.iloc[0] if len(row) > 0 else None
            if pd.isna(first_val):
                break
            
            # Try to parse as date or check if it looks like date data
            try:
                if isinstance(first_val, (pd.Timestamp, datetime)):
                    valid_rows.append(idx)
                elif isinstance(first_val, str) and ('/' in first_val or '-' in first_val):
                    valid_rows.append(idx)
                else:
                    # If it's numeric, it might be Excel serial date
                    try:
                        if isinstance(first_val, (int, float)) and first_val > 40000:  # Approximate Excel date range
                            valid_rows.append(idx)
                        else:
                            break
                    except:
                        break
            except:
                break
        
        if valid_rows:
            df = df.iloc[:max(valid_rows)+1]
        
        # Step 4: Rename columns to standard STR format
        logging.info("Renaming columns to STR standard format...")
        
        # Expected 20 columns after deletions (A, D, K, R removed from original 24)
        # Based on actual data structure mapping observed
        str_columns = [
            'Date',                      # Col 0: Date
            'DOW',                       # Col 1: Day of week  
            'My_Prop_Occ',              # Col 2: My property occupancy
            'Comp_Set_Occ',             # Col 3: Comp set occupancy
            'My_Prop_Occ_Change',       # Col 4: My prop occ change
            'Comp_Set_Occ_Change',      # Col 5: Comp set occ change
            'MPI',                      # Col 6: Market Penetration Index
            'Rank_Occ_Raw',             # Col 7: Occupancy rank (e.g. "5 of 6")
            'My_Prop_ADR',              # Col 8: My property ADR
            'Comp_Set_ADR',             # Col 9: Comp set ADR
            'My_Prop_ADR_Change',       # Col 10: My prop ADR change
            'ARI',                      # Col 11: Average Rate Index  
            'My_Prop_RevPar',           # Col 12: My property RevPAR
            'Rank_ADR_Raw',             # Col 13: ADR rank (e.g. "1 of 6")
            'Comp_Set_RevPar',          # Col 14: Comp set RevPAR
            'My_Prop_RevPar_Change',    # Col 15: My prop RevPAR change
            'RGI',                      # Col 16: Revenue Generation Index
            'Comp_Set_RevPar_Change',   # Col 17: Comp set RevPAR change
            'Extra_Col_18',             # Col 18: Additional data
            'Rank_RevPar_Raw'           # Col 19: RevPAR rank (e.g. "2 of 6") - LAST COLUMN
        ]
        # Add extra columns if needed
        while len(str_columns) < len(df.columns):
            str_columns.append(f'Extra_Col_{len(str_columns)}')
        
        # Assign column names (up to available columns)
        num_cols = min(len(df.columns), len(str_columns))
        df.columns = str_columns[:num_cols] + [f'Extra_Col_{i}' for i in range(num_cols, len(df.columns))]
        
        # Extract rank numbers from rank columns if they contain "X of Y" format
        def extract_rank(rank_str):
            try:
                if pd.isna(rank_str) or not isinstance(rank_str, str):
                    return None
                if ' of ' in str(rank_str):
                    rank_part = str(rank_str).split(' of ')[0].strip()
                    return int(rank_part) if rank_part.isdigit() else None
                return None
            except:
                return None
        
        # Process each rank column
        rank_columns = ['Rank_Occ_Raw', 'Rank_ADR_Raw', 'Rank_RevPar_Raw']
        for rank_col in rank_columns:
            if rank_col in df.columns:
                # Create clean rank column
                clean_col = rank_col.replace('_Raw', '')
                df[clean_col] = df[rank_col].apply(extract_rank)
                # Keep original for reference
                df[f'{clean_col}_Original'] = df[rank_col]
                # Remove raw column
                df = df.drop(rank_col, axis=1)
        
        # Convert Date column to proper datetime format
        if 'Date' in df.columns:
            logging.info("Converting Date column to datetime format...")
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = [col for col in df.columns if col not in ['Date', 'DOW']]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows where Date is NaN (invalid dates)
        df = df.dropna(subset=['Date'])
        
        # Step 5: Save as STR CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"str_data_{timestamp}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        canonical_path = os.path.join(output_dir, "str_data.csv")
        
        logging.info(f"Saving STR data to: {csv_path}")
        df.to_csv(csv_path, index=False)
        
        # Also save as canonical filename
        logging.info(f"Saving canonical STR data to: {canonical_path}")
        df.to_csv(canonical_path, index=False)
        
        logging.info(f"STR processing completed successfully. Processed {len(df)} rows and {len(df.columns)} columns")
        
        return df, canonical_path
        
    except Exception as e:
        logging.error(f"Error processing STR file: {str(e)}")
        raise


def validate_str_data(df: pd.DataFrame) -> bool:
    """
    Validate processed STR data.
    
    Args:
        df: Processed STR DataFrame
    
    Returns:
        True if validation passes, False otherwise
    """
    
    logging.info("Validating STR data...")
    
    required_columns = ['Date', 'DOW', 'My_Prop_Occ', 'Comp_Set_Occ', 'MPI', 'ARI', 'RGI']
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing required columns: {missing_cols}")
        return False
    
    # Check date range
    if 'Date' in df.columns:
        date_range = df['Date'].max() - df['Date'].min()
        logging.info(f"STR data date range: {df['Date'].min()} to {df['Date'].max()} ({date_range.days} days)")
    
    # Check for reasonable occupancy values (0-100%)
    occ_columns = ['My_Prop_Occ', 'Comp_Set_Occ']
    for col in occ_columns:
        if col in df.columns:
            invalid_occ = df[(df[col] < 0) | (df[col] > 100)]
            if len(invalid_occ) > 0:
                logging.warning(f"Found {len(invalid_occ)} rows with invalid occupancy in {col}")
    
    # Check for reasonable index values (typically around 100)
    index_columns = ['MPI', 'ARI', 'RGI']
    for col in index_columns:
        if col in df.columns:
            extreme_values = df[(df[col] < 0) | (df[col] > 500)]
            if len(extreme_values) > 0:
                logging.warning(f"Found {len(extreme_values)} rows with extreme index values in {col}")
    
    logging.info("STR data validation completed")
    return True


def get_str_summary_stats(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for STR data.
    
    Args:
        df: Processed STR DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    
    stats = {
        'total_rows': len(df),
        'date_range': {
            'start': df['Date'].min() if 'Date' in df.columns else None,
            'end': df['Date'].max() if 'Date' in df.columns else None
        },
        'occupancy_stats': {},
        'adr_stats': {},
        'revpar_stats': {},
        'index_stats': {}
    }
    
    # Occupancy statistics
    if 'My_Prop_Occ' in df.columns:
        stats['occupancy_stats']['my_property'] = {
            'mean': df['My_Prop_Occ'].mean(),
            'median': df['My_Prop_Occ'].median(),
            'std': df['My_Prop_Occ'].std(),
            'min': df['My_Prop_Occ'].min(),
            'max': df['My_Prop_Occ'].max()
        }
    
    if 'Comp_Set_Occ' in df.columns:
        stats['occupancy_stats']['comp_set'] = {
            'mean': df['Comp_Set_Occ'].mean(),
            'median': df['Comp_Set_Occ'].median(),
            'std': df['Comp_Set_Occ'].std(),
            'min': df['Comp_Set_Occ'].min(),
            'max': df['Comp_Set_Occ'].max()
        }
    
    # ADR statistics
    for col_prefix in ['My_Prop_ADR', 'Comp_Set_ADR']:
        if col_prefix in df.columns:
            key = 'my_property' if 'My_Prop' in col_prefix else 'comp_set'
            stats['adr_stats'][key] = {
                'mean': df[col_prefix].mean(),
                'median': df[col_prefix].median(),
                'std': df[col_prefix].std(),
                'min': df[col_prefix].min(),
                'max': df[col_prefix].max()
            }
    
    # RevPAR statistics
    for col_prefix in ['My_Prop_RevPar', 'Comp_Set_RevPar']:
        if col_prefix in df.columns:
            key = 'my_property' if 'My_Prop' in col_prefix else 'comp_set'
            stats['revpar_stats'][key] = {
                'mean': df[col_prefix].mean(),
                'median': df[col_prefix].median(),
                'std': df[col_prefix].std(),
                'min': df[col_prefix].min(),
                'max': df[col_prefix].max()
            }
    
    # Index statistics (MPI, ARI, RGI)
    for index_col in ['MPI', 'ARI', 'RGI']:
        if index_col in df.columns:
            stats['index_stats'][index_col] = {
                'mean': df[index_col].mean(),
                'median': df[index_col].median(),
                'std': df[index_col].std(),
                'min': df[index_col].min(),
                'max': df[index_col].max()
            }
    
    return stats


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        df, csv_path = process_str_file(file_path)
        print(f"STR data processed successfully: {csv_path}")
        print(f"Shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Validation
        if validate_str_data(df):
            print("\nValidation: PASSED")
        else:
            print("\nValidation: FAILED")
        
        # Summary stats
        stats = get_str_summary_stats(df)
        print(f"\nSummary Statistics:")
        print(f"Total rows: {stats['total_rows']}")
        print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    else:
        print("Usage: python str_converter.py <path_to_str_excel_file>")