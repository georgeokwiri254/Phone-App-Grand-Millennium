"""
Segment Converter Wrapper
Wraps the existing corrected_final_mhr_processor.py to ensure canonical output paths
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

from corrected_final_mhr_processor import process_mhr_pickup_report

def apply_segment_mapping(df):
    """
    Apply segment mapping to create MergedSegment column
    
    Mapping:
    - Unmanaged/ Leisure - Pre → Retail
    - Unmanaged/ Leisure - Dis → Retail
    - Package → Retail
    - Third Party Intermediary → Retail
    - Managed Corporate - Global → Corporate
    - Managed Corporate - Local → Corporate
    - Government → Corporate
    - Wholesale Fixed Value → Leisure
    - Corporate group → Groups
    - Convention → Groups
    - Association → Groups
    - AD Hoc Group → Groups
    - Tour Group → Groups
    - Contract → Contract
    - Complimentary → Complimentary
    """
    segment_mapping = {
        'Unmanaged/ Leisure - Pre': 'Retail',
        'Unmanaged/ Leisure - Dis': 'Retail',
        'Package': 'Retail',
        'Third Party Intermediary': 'Retail',
        'Managed Corporate - Global': 'Corporate',
        'Managed Corporate - Local': 'Corporate',
        'Government': 'Corporate',
        'Wholesale Fixed Value': 'Leisure',
        'Corporate group': 'Groups',
        'Covention': 'Groups',  # Note: typo in original data
        'Convention': 'Groups',  # Handle both spellings
        'Association': 'Groups',
        'AD Hoc Group': 'Groups',
        'Tour Group': 'Groups',
        'Contract': 'Contract',
        'Complimentary': 'Complimentary'
    }
    
    # Create MergedSegment column
    if 'Segment' in df.columns:
        df['MergedSegment'] = df['Segment'].map(segment_mapping)
        # Fill any unmapped segments with the original segment name
        df['MergedSegment'] = df['MergedSegment'].fillna(df['Segment'])
    
    return df


def convert_month_to_date(df):
    """
    Convert Month column from string (e.g., 'January') to proper date format (2025-01-01)
    Format: dd/mm/yyyy where January becomes 01/01/2025
    """
    if 'Month' not in df.columns:
        return df
    
    # Month name to date mapping for 2025
    month_mapping = {
        'January': '01/01/2025',
        'February': '01/02/2025', 
        'March': '01/03/2025',
        'April': '01/04/2025',
        'May': '01/05/2025',
        'June': '01/06/2025',
        'July': '01/07/2025',
        'August': '01/08/2025',
        'September': '01/09/2025',
        'October': '01/10/2025',
        'November': '01/11/2025',
        'December': '01/12/2025'
    }
    
    # Convert month names to dates
    df['Month'] = df['Month'].map(month_mapping)
    
    # Convert to datetime format for proper handling
    df['Month'] = pd.to_datetime(df['Month'], format='%d/%m/%Y')
    
    return df

def run_segment_conversion(excel_file_path, output_dir=None):
    """
    Run segment conversion using existing processor with canonical output paths
    
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
    
    canonical_csv_path = processed_dir / 'segment.csv'
    
    # Create timestamped backup path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_csv_path = processed_dir / f'segment_{timestamp}.csv'
    
    try:
        print(f"Starting segment conversion for: {excel_file_path}")
        
        # Use existing processor to get DataFrame
        temp_output = f"temp_segment_output_{timestamp}.csv"
        df = process_mhr_pickup_report(excel_file_path, temp_output)
        
        if df is None:
            raise ValueError("Segment processor returned None - conversion failed")
        
        print(f"[OK] Segment processor completed. Shape: {df.shape}")
        
        # Apply segment mapping
        df = apply_segment_mapping(df)
        print(f"[OK] Applied segment mapping. MergedSegment column added.")
        
        # Convert Month column to proper date format
        df = convert_month_to_date(df)
        print(f"[OK] Converted Month column to proper date format.")
        
        # Save timestamped backup
        df.to_csv(backup_csv_path, index=False)
        print(f"[OK] Saved backup to: {backup_csv_path}")
        
        # Atomic write to canonical path
        temp_canonical = str(canonical_csv_path) + '.tmp'
        df.to_csv(temp_canonical, index=False)
        shutil.move(temp_canonical, canonical_csv_path)
        print(f"[OK] Saved canonical output to: {canonical_csv_path}")
        
        # Clean up temporary file
        if os.path.exists(temp_output):
            os.remove(temp_output)
        
        # Validation
        print(f"[OK] Validation - Rows: {len(df)}, Columns: {len(df.columns)}")
        if 'MergedSegment' in df.columns:
            merged_segments = df['MergedSegment'].value_counts()
            print(f"[OK] Merged segments: {dict(merged_segments)}")
        
        return df, str(canonical_csv_path)
        
    except Exception as e:
        print(f"[ERROR] Segment conversion failed: {str(e)}")
        raise