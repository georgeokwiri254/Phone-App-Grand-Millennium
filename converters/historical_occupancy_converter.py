"""
Historical Occupancy Converter
Debug version specifically for historical files with enhanced date parsing and column detection
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, date, timedelta
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoricalOccupancyConverter:
    """Convert historical MHR Excel files to daily occupancy CSV format with debugging"""
    
    def __init__(self):
        # Target CSV column structure for occupancy data
        self.output_columns = [
            'Date',           # AZ
            'DOW',            # BA
            'Rms',            # BB
            'Daily Revenue',  # BC
            'Occ%',           # BG
            'Rm Sold',        # BH
            'Revenue',        # BI
            'ADR',            # BJ
            'RevPar'          # BK
        ]
    
    def excel_column_to_number(self, column_letter):
        """Convert Excel column letter to number (A=0, B=1, etc.)"""
        result = 0
        for char in column_letter.upper():
            result = result * 26 + ord(char) - ord('A') + 1
        return result - 1
    
    def find_dpr_sheet(self, excel_file_path):
        """Find DPR sheet in Excel file"""
        try:
            excel_file = pd.ExcelFile(excel_file_path)
            available_sheets = excel_file.sheet_names
            logger.info(f"Available sheets: {available_sheets}")
            
            # Look for DPR sheet (exact match first, then contains)
            if 'DPR' in available_sheets:
                logger.info(f"Found DPR sheet: DPR")
                return 'DPR'
            
            # Look for sheet containing DPR
            for sheet in available_sheets:
                if 'DPR' in sheet.upper():
                    logger.info(f"Found DPR-like sheet: {sheet}")
                    return sheet
            
            return None
        except Exception as e:
            logger.error(f"Error finding DPR sheet: {e}")
            return None
    
    def detect_year_from_filename(self, file_path):
        """Extract year from filename"""
        filename = os.path.basename(file_path)
        # Look for 4-digit year in filename
        import re
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            year = int(year_match.group())
            logger.info(f"Detected year from filename: {year}")
            return year
        
        # Default to current year if not found
        current_year = datetime.now().year
        logger.warning(f"Could not detect year from filename, using: {current_year}")
        return current_year
    
    def generate_date_range(self, year):
        """Generate list of dates for the specified year"""
        start_date = date(year, 1, 1)
        
        # Check if it's a leap year
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            end_date = date(year, 12, 31)  # 366 days
        else:
            end_date = date(year, 12, 31)  # 365 days
        
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)
        
        logger.info(f"Generated {len(dates)} dates for year {year}")
        return dates
    
    def extract_dates_from_column(self, sheet_df, column_index):
        """Extract and analyze dates from a specific column"""
        if column_index >= len(sheet_df.columns):
            return []
        
        column_data = sheet_df.iloc[:, column_index]
        dates = []
        
        logger.info(f"Analyzing column {column_index} for dates...")
        logger.info(f"Column sample values: {column_data.head(10).tolist()}")
        
        for i, value in enumerate(column_data):
            if pd.isna(value):
                continue
            
            try:
                # Try to parse as datetime
                if isinstance(value, datetime):
                    dates.append(value.date())
                elif isinstance(value, str):
                    # Try different date formats
                    parsed_date = pd.to_datetime(value, errors='coerce')
                    if not pd.isna(parsed_date):
                        dates.append(parsed_date.date())
                elif isinstance(value, (int, float)):
                    # Excel date serial number
                    if 40000 < value < 50000:  # Reasonable range for Excel dates
                        parsed_date = pd.to_datetime(value, origin='1899-12-30', unit='D', errors='coerce')
                        if not pd.isna(parsed_date):
                            dates.append(parsed_date.date())
            except:
                continue
        
        logger.info(f"Found {len(dates)} valid dates in column {column_index}")
        if dates:
            logger.info(f"Date range: {min(dates)} to {max(dates)}")
        
        return dates
    
    def find_date_column(self, sheet_df):
        """Find the column containing dates (typically AZ)"""
        az_column = self.excel_column_to_number('AZ')  # Column 51
        
        # Check AZ column first
        if az_column < len(sheet_df.columns):
            dates = self.extract_dates_from_column(sheet_df, az_column)
            if len(dates) > 300:  # Should have around 365 dates
                logger.info(f"Found date column at AZ (index {az_column})")
                return az_column, dates
        
        # Search nearby columns if AZ doesn't work
        for offset in [-2, -1, 1, 2]:
            col_idx = az_column + offset
            if 0 <= col_idx < len(sheet_df.columns):
                dates = self.extract_dates_from_column(sheet_df, col_idx)
                if len(dates) > 300:
                    logger.info(f"Found date column at index {col_idx} (offset {offset} from AZ)")
                    return col_idx, dates
        
        logger.error("Could not find date column with sufficient dates")
        return None, []
    
    def extract_occupancy_data(self, sheet_df, target_year):
        """Extract occupancy data from DPR sheet with enhanced debugging"""
        logger.info("Extracting occupancy data from DPR sheet columns AZ to BK using date matching...")
        
        # Find date column
        date_col_idx, found_dates = self.find_date_column(sheet_df)
        if date_col_idx is None:
            logger.error("Could not find date column")
            return pd.DataFrame()
        
        # Define column mapping (relative to date column)
        base_col = date_col_idx
        column_mapping = {
            'Date': base_col,        # AZ
            'DOW': base_col + 1,     # BA
            'Rms': base_col + 2,     # BB
            'Daily Revenue': base_col + 3,  # BC
            'Occ%': base_col + 7,    # BG (skip BD, BE, BF)
            'Rm Sold': base_col + 8, # BH
            'Revenue': base_col + 9, # BI
            'ADR': base_col + 10,    # BJ
            'RevPar': base_col + 11  # BK
        }
        
        logger.info(f"Column mapping: {column_mapping}")
        
        # Check if we have all required columns
        max_col_needed = max(column_mapping.values())
        if max_col_needed >= len(sheet_df.columns):
            logger.error(f"Not enough columns in sheet. Need {max_col_needed + 1}, have {len(sheet_df.columns)}")
            return pd.DataFrame()
        
        # Filter dates for target year
        year_dates = [d for d in found_dates if d.year == target_year]
        logger.info(f"Found {len(year_dates)} dates for year {target_year}")
        
        if len(year_dates) == 0:
            logger.warning(f"No dates found for year {target_year}, using all found dates")
            year_dates = found_dates
        
        # Extract data for each date
        occupancy_data = []
        for target_date in year_dates:
            # Find row with this date
            date_col_data = sheet_df.iloc[:, date_col_idx]
            for row_idx, cell_value in enumerate(date_col_data):
                try:
                    if pd.isna(cell_value):
                        continue
                    
                    # Convert cell value to date
                    cell_date = None
                    if isinstance(cell_value, datetime):
                        cell_date = cell_value.date()
                    elif isinstance(cell_value, str):
                        parsed = pd.to_datetime(cell_value, errors='coerce')
                        if not pd.isna(parsed):
                            cell_date = parsed.date()
                    elif isinstance(cell_value, (int, float)):
                        if 40000 < cell_value < 50000:
                            parsed = pd.to_datetime(cell_value, origin='1899-12-30', unit='D', errors='coerce')
                            if not pd.isna(parsed):
                                cell_date = parsed.date()
                    
                    if cell_date == target_date:
                        # Found matching date, extract row data
                        row_data = {'Date': target_date}
                        for col_name, col_idx in column_mapping.items():
                            if col_name != 'Date' and col_idx < len(sheet_df.columns):
                                value = sheet_df.iloc[row_idx, col_idx]
                                # Convert to numeric if possible
                                if isinstance(value, str):
                                    try:
                                        value = float(value.replace(',', ''))
                                    except:
                                        pass
                                row_data[col_name] = value
                        
                        occupancy_data.append(row_data)
                        break
                except:
                    continue
        
        # Create DataFrame
        df = pd.DataFrame(occupancy_data)
        
        # If we have data, add day of week
        if not df.empty and 'Date' in df.columns:
            df['DOW'] = df['Date'].apply(lambda x: x.strftime('%a'))
        
        logger.info(f"Extracted {len(df)} rows of occupancy data")
        
        # Show sample data for debugging
        if not df.empty:
            logger.info(f"Sample data:\\n{df.head()}")
            
            # Show data summary
            numeric_cols = ['Rms', 'Daily Revenue', 'Occ%', 'Rm Sold', 'Revenue', 'ADR', 'RevPar']
            for col in numeric_cols:
                if col in df.columns:
                    non_zero_count = (df[col] != 0).sum()
                    logger.info(f"{col}: {non_zero_count} non-zero values out of {len(df)}")
        
        return df
    
    def convert_historical_to_occupancy_csv(self, excel_file_path, output_csv_path=None):
        """Convert historical MHR Excel file to daily occupancy CSV format"""
        try:
            if not os.path.exists(excel_file_path):
                raise FileNotFoundError(f"Excel file not found: {excel_file_path}")
            
            logger.info(f"Converting historical MHR file: {excel_file_path}")
            
            # Detect year from filename
            target_year = self.detect_year_from_filename(excel_file_path)
            
            # Find DPR sheet
            sheet_name = self.find_dpr_sheet(excel_file_path)
            if not sheet_name:
                raise ValueError("Could not find DPR sheet in Excel file")
            
            # Read the DPR sheet
            logger.info(f"Reading DPR sheet: {sheet_name}")
            sheet_df = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)
            logger.info(f"DPR sheet shape: {sheet_df.shape}")
            
            # Extract occupancy data
            occupancy_df = self.extract_occupancy_data(sheet_df, target_year)
            
            if occupancy_df.empty:
                logger.error("No occupancy data extracted")
                return None, None
            
            # Generate output path if not provided
            if output_csv_path is None:
                base_name = os.path.splitext(os.path.basename(excel_file_path))[0]
                output_csv_path = f"historical_occupancy_{target_year}_{base_name}.csv"
            
            # Save to CSV
            occupancy_df.to_csv(output_csv_path, index=False)
            logger.info(f"Historical occupancy data saved to: {output_csv_path}")
            
            # Summary statistics
            logger.info(f"Summary for {target_year}:")
            if 'Revenue' in occupancy_df.columns:
                total_revenue = occupancy_df['Revenue'].sum()
                logger.info(f"Total revenue: {total_revenue:,.2f}")
            
            if 'Occ%' in occupancy_df.columns:
                avg_occupancy = occupancy_df['Occ%'].mean()
                logger.info(f"Average occupancy: {avg_occupancy:.2f}%")
            
            if 'ADR' in occupancy_df.columns:
                avg_adr = occupancy_df['ADR'].mean()
                logger.info(f"Average ADR: {avg_adr:.2f}")
            
            return occupancy_df, output_csv_path
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

def run_historical_occupancy_conversion(excel_file_path, output_dir=None):
    """
    Run historical occupancy conversion with enhanced debugging
    
    Args:
        excel_file_path (str): Path to input Excel file
        output_dir (str): Base directory for output (defaults to project root)
        
    Returns:
        tuple: (dataframe, csv_path)
    """
    if output_dir is None:
        # Get project root directory
        project_root = Path(__file__).parent.parent
        output_dir = project_root
    
    # Define output paths
    processed_dir = Path(output_dir) / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped output path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv_path = processed_dir / f'historical_occupancy_{timestamp}.csv'
    
    try:
        print(f"Starting historical occupancy conversion for: {excel_file_path}")
        
        # Use historical converter
        converter = HistoricalOccupancyConverter()
        df, temp_path = converter.convert_historical_to_occupancy_csv(excel_file_path, str(output_csv_path))
        
        if df is None:
            raise ValueError("Historical occupancy converter returned None - conversion failed")
        
        print(f"[OK] Historical occupancy conversion completed. Shape: {df.shape}")
        print(f"[OK] Output saved to: {output_csv_path}")
        
        return df, str(output_csv_path)
        
    except Exception as e:
        print(f"[ERROR] Historical occupancy conversion failed: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
        run_historical_occupancy_conversion(excel_file)
    else:
        print("Usage: python historical_occupancy_converter.py <excel_file>")