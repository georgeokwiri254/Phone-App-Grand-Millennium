"""
Occupancy Converter Wrapper
Wraps the existing mhr_occupancy_converter.py to ensure canonical output paths
"""

import os
import sys
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path
import io

# Fix encoding issues on Windows
if sys.platform == "win32" and not hasattr(sys.stdout, '_wrapped'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stdout._wrapped = True
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        sys.stderr._wrapped = True
    except (AttributeError, ValueError):
        # stdout/stderr might already be wrapped or unavailable
        pass

# Add current directory to path to import the existing processor
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from mhr_occupancy_converter import MHROccupancyConverter

def normalize_occupancy_values(df):
    """
    Normalize occupancy values to ensure they are in 0-100 range
    """
    if 'Occ%' in df.columns:
        # Detect if values are in 0-1 range (like 0.85) and convert to percentage
        max_occ = df['Occ%'].max()
        if max_occ <= 1.0:
            df['Occ%'] = df['Occ%'] * 100
            print("[OK] Converted occupancy from decimal to percentage format")
    
    return df


def validate_occupancy_data(df):
    """
    Validate occupancy data for consistency and realistic values
    """
    validation_results = {
        'total_rows': len(df),
        'date_range': None,
        'negative_values': {},
        'high_occupancy': 0,
        'missing_days': 0
    }
    
    # Check date range
    if 'Date' in df.columns:
        validation_results['date_range'] = f"{df['Date'].min()} to {df['Date'].max()}"
        
        # Check for missing days (should be 365 for 2025)
        expected_days = 365
        actual_days = len(df)
        validation_results['missing_days'] = max(0, expected_days - actual_days)
    
    # Check for negative values in numeric columns
    numeric_cols = ['Rms', 'Daily Revenue', 'Occ%', 'Rm Sold', 'Revenue', 'ADR', 'RevPar']
    for col in numeric_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                validation_results['negative_values'][col] = negative_count
    
    # Check for unrealistic occupancy percentages
    if 'Occ%' in df.columns:
        validation_results['high_occupancy'] = (df['Occ%'] > 100).sum()
    
    return validation_results

def run_occupancy_conversion(excel_file_path, output_dir=None):
    """
    Run occupancy conversion using existing processor with canonical output paths
    
    Args:
        excel_file_path (str): Path to input Excel file
        output_dir (str): Base directory for output (defaults to project root)
        
    Returns:
        tuple: (dataframe, canonical_csv_path)
    """
    if output_dir is None:
        # Get project root directory (assuming this script is in converters/)
        project_root = Path(__file__).parent.parent
        output_dir = project_root
    
    # Define canonical paths
    processed_dir = Path(output_dir) / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    canonical_csv_path = processed_dir / 'occupancy.csv'
    
    # Create timestamped backup path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_csv_path = processed_dir / f'occupancy_{timestamp}.csv'
    
    try:
        print(f"Starting occupancy conversion for: {excel_file_path}")
        
        # Use existing converter
        converter = MHROccupancyConverter()
        temp_output = f"temp_occupancy_output_{timestamp}.csv"
        
        df, temp_path = converter.convert_mhr_to_occupancy_csv(excel_file_path, temp_output)
        
        if df is None:
            raise ValueError("Occupancy converter returned None - conversion failed")
        
        print(f"[OK] Occupancy processor completed. Shape: {df.shape}")
        
        # Normalize occupancy values (0-100 range)
        df = normalize_occupancy_values(df)
        
        # Validate data
        validation_results = validate_occupancy_data(df)
        print(f"[OK] Validation completed:")
        print(f"  - Total rows: {validation_results['total_rows']}")
        print(f"  - Date range: {validation_results['date_range']}")
        print(f"  - Missing days: {validation_results['missing_days']}")
        print(f"  - High occupancy (>100%): {validation_results['high_occupancy']}")
        if validation_results['negative_values']:
            print(f"  - Negative values found: {validation_results['negative_values']}")
        
        # Save timestamped backup
        df.to_csv(backup_csv_path, index=False)
        print(f"[OK] Saved backup to: {backup_csv_path}")
        
        # Atomic write to canonical path (with fallback for locked files)
        temp_canonical = str(canonical_csv_path) + '.tmp'
        df.to_csv(temp_canonical, index=False)
        
        try:
            shutil.move(temp_canonical, canonical_csv_path)
            print(f"[OK] Saved canonical output to: {canonical_csv_path}")
            final_path = str(canonical_csv_path)
        except (PermissionError, FileExistsError) as e:
            # File is locked - use backup as the return path
            print(f"[WARNING] Cannot update canonical file (locked): {e}")
            print(f"[OK] Using backup file as output: {backup_csv_path}")
            # Clean up temp file
            if os.path.exists(temp_canonical):
                os.remove(temp_canonical)
            final_path = str(backup_csv_path)
        
        # Clean up temporary file
        if os.path.exists(temp_output):
            os.remove(temp_output)
        
        return df, final_path
        
    except Exception as e:
        print(f"[ERROR] Occupancy conversion failed: {str(e)}")
        raise