#!/usr/bin/env python3
"""
Corrected Final MHR Pick Up Report 2025 Processor
Complete processing script for extracting and cleaning MHR pickup report data

This script processes any MHR Pick Up Report 2025 Excel file with DPR tab and:
1. Deletes rows 1-3 from the DPR sheet
2. Deletes columns AY to BZ at the start (columns 50+ in Excel)
3. Removes rows between OTHERS and next month for all 12 months
4. Deletes segment labels (UNMANAGED, MANAGED, GROUPS, OTHERS) from column C
5. Applies proper column mappings with correct headers (50 columns)
6. Filters to keep only the 15 specified segments per month
7. Outputs clean CSV with 180 rows (15 segments × 12 months)

Author: Claude Code
Date: 2025
"""

import pandas as pd
import glob
import sys
import os
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

def process_mhr_pickup_report(file_path, output_path=None):
    """
    Complete processing function for MHR Pick Up Report 2025 DPR sheet
    
    Args:
        file_path (str): Path to the input Excel file
        output_path (str): Path for the output CSV file (auto-generated if None)
    
    Returns:
        pandas.DataFrame: Processed dataframe
    """
    
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = f"final_processed_{base_name}.csv"
    
    print(f"Processing MHR Pick Up Report: {file_path}")
    print("=" * 60)
    
    # Load the DPR sheet from the Excel file
    try:
        # First, check what sheets are available
        excel_file = pd.ExcelFile(file_path)
        available_sheets = excel_file.sheet_names
        print(f"Available sheets: {available_sheets}")
        
        if 'DPR' not in available_sheets:
            print(f"[ERROR] 'DPR' sheet not found. Available sheets: {available_sheets}")
            return None
            
        df = pd.read_excel(file_path, sheet_name='DPR', header=None)
        print(f"Original shape: {df.shape}")
    except Exception as e:
        print(f"[ERROR] Error reading file: {str(e)}")
        try:
            # Try to get sheet names for debugging
            excel_file = pd.ExcelFile(file_path)
            print(f"Available sheets: {excel_file.sheet_names}")
        except:
            print("[ERROR] Could not read Excel file at all")
        return None
    
    # Validate minimum requirements
    if df.shape[1] < 3:
        print(f"[ERROR] File must have at least 3 columns, found {df.shape[1]}")
        return None
    
    if df.shape[0] < 4:
        print(f"[ERROR] File must have at least 4 rows, found {df.shape[0]}")
        return None
    
    # STEP 1: Delete rows 1-3 (index 0-2)
    df = df.drop(index=[0, 1, 2]).reset_index(drop=True)
    print(f"[OK] After deleting rows 1-3: {df.shape}")
    
    # STEP 2: Delete columns AY to BZ at the start (columns 51+ in Excel)
    # AX = column 50 (index 49), AY = column 51 (index 50), AZ = 52 (index 51), ..., BZ = column 78 (index 77)
    # Keep column AX (index 49), delete columns from index 50 onwards (AY to BZ)
    if len(df.columns) > 50:
        cols_to_drop = list(range(50, len(df.columns)))
        df = df.drop(df.columns[cols_to_drop], axis=1)
        print(f"[OK] After deleting columns AY to BZ: {df.shape}")
    
    # STEP 3: Delete rows between OTHERS and next month for all 12 months
    rows_to_delete = []
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    
    # Find OTHERS rows
    others_indices = []
    for idx, row in df.iterrows():
        if pd.notna(row.iloc[2]) and str(row.iloc[2]).strip().upper() == 'OTHERS':
            others_indices.append(idx)
    
    print(f"[OK] Found OTHERS at indices: {others_indices}")
    
    # For each OTHERS occurrence, find the next month and delete rows in between
    for others_idx in others_indices:
        for idx in range(others_idx + 1, len(df)):
            if pd.notna(df.iloc[idx, 0]):  # Column A contains months
                month_val = str(df.iloc[idx, 0]).strip()
                if any(month in month_val for month in months):
                    rows_to_delete.extend(range(others_idx, idx))
                    break
    
    # Remove duplicate indices and sort in reverse order
    rows_to_delete = sorted(list(set(rows_to_delete)), reverse=True)
    for row_idx in rows_to_delete:
        if row_idx < len(df):
            df = df.drop(index=row_idx)
    df = df.reset_index(drop=True)
    print(f"[OK] After deleting rows between OTHERS and next months: {df.shape}")
    
    # STEP 4: Delete segment labels from column C
    segment_labels = ['UNMANAGED', 'MANAGED', 'GROUPS', 'OTHERS']
    rows_to_delete = []
    
    for idx, row in df.iterrows():
        if pd.notna(row.iloc[2]):  # Column C
            cell_value = str(row.iloc[2]).strip().upper()
            if cell_value in segment_labels:
                rows_to_delete.append(idx)
    
    df = df.drop(index=rows_to_delete).reset_index(drop=True)
    print(f"[OK] After deleting segment labels: {df.shape}")
    
    # STEP 5: Apply column mappings as per updated MHR specification (no duplicate column names)
    # Updated mapping with corrected labels to prevent duplicates
    column_mapping = [
        'Month',                                                         # A
        'Index_Month_segment',                                          # B
        'Segment',                                                      # C
        'Daily_Pick_up_Rooms',                                          # D
        'Daily_Pick_up_Revenue',                                        # E
        'Daily_Pick_up_ADR',                                            # F
        'Daily_Pick_up_Share',                                          # G
        'Dis_Daily',                                                    # H
        'Month_to_Date_Rooms',                                          # I
        'Month_to_Date_Revenue',                                        # J
        'Month_to_Date_ADR',                                            # K
        'Month_to_Date_Share',                                          # L
        'Dis_MTD',                                                      # M
        'Business_on_the_Books_Rooms',                                 # N
        'Business_on_the_Books_Revenue',                               # O
        'Business_on_the_Books_ADR',                                   # P
        'Business_on_the_Books_Share_per_Segment',                     # Q
        'Dis_BOB',                                                      # R -> Q (maps to CSV column Q)
        'Business_on_the_Books_Same_Time_Last_Year_Rooms',            # S
        'Business_on_the_Books_Same_Time_Last_Year_Revenue',          # T
        'Business_on_the_Books_Same_Time_Last_Year_ADR',              # U
        'Business_on_the_Books_Same_Time_Last_Year_Share_per_Segment', # V
        'Dis_BOB_STLY',                                                 # W
        'Full_Month_Last_Year_Rooms',                                  # X
        'Full_Month_Last_Year_Revenue',                                # Y
        'Full_Month_Last_Year_ADR',                                    # Z
        'Full_Month_Last_Year_Share_per_Segment',                      # AA
        'Dis_Full_Month',                                               # AB
        'Budget_This_Year_Rooms',                                      # AC
        'Budget_This_Year_Revenue',                                    # AD
        'Budget_This_Year_ADR',                                        # AE
        'Budget_This_Year_Share_per_Segment',                          # AF
        'Dis_Budget',                                                   # AG
        'Forecast_This_Year_Rooms',                                    # AH
        'Forecast_This_Year_Revenue',                                  # AI
        'Forecast_This_Year_ADR',                                      # AJ
        'Forecast_This_Year_Share_per_Segment',                        # AK
        'Dis_Forecast',                                                 # AL
        'Year_On_Year_Pace_for_rooms',                                 # AM
        'Year_On_Year_Pace_Revenue',                                   # AN
        'Year_On_Year_Pace_for_ADR',                                   # AO
        'Year_on_Year_Rooms',                                          # AP
        'Year_on_Year_Revenue',                                        # AQ
        'Year_on_Year_ADR',                                            # AR
        'Delta_to_Budget_Rooms',                                       # AS
        'Delta_to_Budget_Revenue',                                     # AT
        'Delta_to_Budget_ADR',                                         # AU
        'Delta_to_Forecast_Rooms',                                     # AV
        'Delta_to_Forecast_Revenue',                                   # AW
        'Delta_to_Forecast_ADR'                                        # AX
    ]
    
    # Apply column names (ensure we have exactly 50 columns or less)
    new_columns = []
    for i in range(min(len(df.columns), len(column_mapping))):
        new_columns.append(column_mapping[i])
    
    # If we have fewer columns than expected, pad with extra column names
    for i in range(len(new_columns), len(df.columns)):
        new_columns.append(f'Extra_Column_{i}')
    
    df.columns = new_columns[:len(df.columns)]
    print(f"[OK] Applied column mapping: {len(df.columns)} columns")
    
    # STEP 6: Clean and filter data
    df = df.dropna(how='all').reset_index(drop=True)
    
    # Filter to keep only the 15 specified segments
    valid_segments = [
        'Unmanaged/ Leisure - Pre',
        'Unmanaged/ Leisure - Dis',
        'Package',
        'Third Party Intermediary',
        'Managed Corporate - Global',
        'Managed Corporate - Local',
        'Government',
        'Wholesale Fixed Value',
        'Corporate group',
        'Covention',
        'Association',
        'AD Hoc Group',
        'Tour Group',
        'Contract',
        'Complimentary'
    ]
    
    if 'Segment' in df.columns:
        mask = df['Segment'].isin(valid_segments)
        df = df[mask].reset_index(drop=True)
    
    print(f"[OK] Final shape after filtering: {df.shape}")
    
    # STEP 7: Save to CSV
    df.to_csv(output_path, index=False)
    print(f"[OK] Processed report saved to: {output_path}")
    
    return df

def validate_output(df):
    """
    Validate the processed output
    
    Args:
        df (pandas.DataFrame): Processed dataframe to validate
    """
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    if 'Segment' in df.columns and 'Month' in df.columns:
        segments_per_month = df.groupby('Month')['Segment'].count()
        print(f"\nSegments per month:")
        for month, count in segments_per_month.items():
            status = "[OK]" if count == 15 else "[ERROR]"
            print(f"  {status} {month}: {count} segments")
        
        unique_segments = df['Segment'].unique()
        print(f"\nUnique segments found: {len(unique_segments)}")
        for segment in sorted(unique_segments):
            print(f"  • {segment}")
    
    print(f"\nColumn structure:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

def find_xlsm_files():
    """
    Find all XLSM files in the current directory
    
    Returns:
        list: List of XLSM file paths
    """
    xlsm_files = glob.glob("*.xlsm")
    return xlsm_files

def process_single_file(input_file):
    """
    Process a single XLSM file
    
    Args:
        input_file (str): Path to the input XLSM file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"PROCESSING: {input_file}")
        print("=" * 80)
        
        # Process the report
        processed_df = process_mhr_pickup_report(input_file)
        
        if processed_df is None:
            return False
        
        # Validate the output
        validate_output(processed_df)
        
        print(f"\n{'='*80}")
        print("[SUCCESS] PROCESSING COMPLETED SUCCESSFULLY!")
        print(f"[OK] Input file: {input_file}")
        print(f"[OK] Total records: {len(processed_df)}")
        print(f"[OK] Expected: 180 records (15 segments x 12 months)")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error processing file '{input_file}': {str(e)}")
        return False

def main():
    """
    Main execution function - processes any XLSM file with DPR tab
    """
    print("CORRECTED FINAL MHR PICKUP REPORT PROCESSOR")
    print("Processes any XLSM file with DPR tab (50 columns)")
    print("=" * 80)
    
    if len(sys.argv) > 1:
        # Process specific file provided as argument
        input_file = sys.argv[1]
        if not os.path.exists(input_file):
            print(f"[ERROR] Error: File '{input_file}' not found.")
            return
        
        success = process_single_file(input_file)
        if success:
            print("\n[SUCCESS] File processed successfully!")
        else:
            print("\n[ERROR] File processing failed!")
    else:
        # Process all XLSM files in current directory
        xlsm_files = find_xlsm_files()
        
        if not xlsm_files:
            print("[ERROR] No XLSM files found in current directory.")
            print("\nUsage:")
            print("  python corrected_final_mhr_processor.py <filename.xlsm>  # Process specific file")
            print("  python corrected_final_mhr_processor.py                  # Process all XLSM files")
            return
        
        print(f"Found {len(xlsm_files)} XLSM file(s) to process:")
        for file in xlsm_files:
            print(f"  • {file}")
        print()
        
        success_count = 0
        for file in xlsm_files:
            print(f"\nProcessing {file}...")
            success = process_single_file(file)
            if success:
                success_count += 1
        
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING COMPLETED!")
        print(f"[OK] Successfully processed: {success_count}/{len(xlsm_files)} files")
        if success_count < len(xlsm_files):
            print(f"[ERROR] Failed to process: {len(xlsm_files) - success_count} files")
        print("="*80)

if __name__ == "__main__":
    main()