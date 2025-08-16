"""
MHR Daily Occupancy Converter
Extracts daily occupancy data from MHR Excel file columns AZ to BK
Creates CSV with Date, DOW, Rooms, Daily Revenue, ADR, Occ%, Rm Sold, Revenue, ADR, RevPar
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, date
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MHROccupancyConverter:
    """Convert MHR Excel file to daily occupancy CSV format"""
    
    def __init__(self):
        # Target CSV column structure for occupancy data
        self.occupancy_columns = [
            'Date',           # Column AZ
            'DOW',            # Column BA (Day of Week)
            'Rms',            # Column BB (Rooms)
            'Daily Revenue',  # Column BC
            'ADR',            # Column BD (excluded per requirement)
            'Occ%',           # Column BG (Occupancy %)
            'Rm Sold',        # Column BH (Rooms Sold)
            'Revenue',        # Column BI
            'ADR',            # Column BJ
            'RevPar'          # Column BK
        ]
        
        # Exclude columns BD to BF as requested
        self.excluded_columns = ['BD', 'BE', 'BF']
        
        # Final output columns (excluding BD-BF)
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
    
    def find_dpr_sheet(self, excel_file):
        """Find the DPR sheet containing occupancy data"""
        try:
            xl_file = pd.ExcelFile(excel_file)
            logger.info(f"Available sheets: {xl_file.sheet_names}")
            
            # Look specifically for DPR sheet
            dpr_sheet = None
            
            for sheet in xl_file.sheet_names:
                sheet_upper = sheet.upper()
                if 'DPR' in sheet_upper:
                    dpr_sheet = sheet
                    logger.info(f"Found DPR sheet: {sheet}")
                    break
            
            if not dpr_sheet:
                # Look for alternative names that might contain daily data
                alternatives = ['DAILY', 'PICKUP', 'OCCUPANCY', 'REVENUE']
                for alt in alternatives:
                    for sheet in xl_file.sheet_names:
                        if alt in sheet.upper():
                            dpr_sheet = sheet
                            logger.info(f"Found alternative sheet: {sheet}")
                            break
                    if dpr_sheet:
                        break
            
            if not dpr_sheet:
                # Use first sheet as fallback
                dpr_sheet = xl_file.sheet_names[0]
                logger.warning(f"No DPR sheet found, using first sheet: {dpr_sheet}")
            
            logger.info(f"Using DPR sheet: {dpr_sheet}")
            return dpr_sheet
            
        except Exception as e:
            logger.error(f"Error finding DPR sheet: {e}")
            return None
    
    def generate_date_range(self, year=2025):
        """Generate date range from Jan 1 to Dec 31"""
        from datetime import datetime, timedelta
        
        start_date = datetime(year, 1, 1).date()
        end_date = datetime(year, 12, 31).date()
        
        dates = []
        current_date = start_date
        
        while current_date <= end_date:
            dates.append(current_date)
            # Move to next day using timedelta
            current_date = current_date + timedelta(days=1)
        
        return dates
    
    def get_day_of_week(self, date_obj):
        """Get day of week abbreviation"""
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        return days[date_obj.weekday()]
    
    def excel_column_to_number(self, column_letter):
        """Convert Excel column letter to number (A=1, B=2, etc.)"""
        result = 0
        for char in column_letter:
            result = result * 26 + (ord(char.upper()) - ord('A') + 1)
        return result - 1  # Convert to 0-based index
    
    def extract_occupancy_data(self, sheet_df, dates):
        """Extract occupancy data from DPR sheet columns AZ to BK using SUMIFS logic"""
        logger.info("Extracting occupancy data from DPR sheet columns AZ to BK using date matching...")
        
        # Define column mapping (Excel columns to indices)
        column_mapping = {
            'Date': self.excel_column_to_number('AZ'),      # Column AZ (51)
            'DOW': self.excel_column_to_number('BA'),       # Column BA (52)
            'Rms': self.excel_column_to_number('BB'),       # Column BB (53)
            'Daily Revenue': self.excel_column_to_number('BC'),  # Column BC (54)
            'Occ%': self.excel_column_to_number('BG'),      # Column BG (58)
            'Rm Sold': self.excel_column_to_number('BH'),   # Column BH (59)
            'Revenue': self.excel_column_to_number('BI'),   # Column BI (60)
            'ADR': self.excel_column_to_number('BJ'),       # Column BJ (61)
            'RevPar': self.excel_column_to_number('BK')     # Column BK (62)
        }
        
        logger.info(f"Column mapping: {column_mapping}")
        
        # Get the date column (AZ) to find actual date matches
        date_col_index = column_mapping['Date']
        date_column = sheet_df.iloc[:, date_col_index] if date_col_index < len(sheet_df.columns) else None
        
        if date_column is None:
            logger.error("Could not access date column AZ")
            return []
        
        # Create a mapping of dates to row indices
        date_to_row = {}
        for row_idx, cell_value in enumerate(date_column):
            if pd.isna(cell_value):
                continue
                
            # Try to parse the date
            try:
                if isinstance(cell_value, (datetime, pd.Timestamp)):
                    parsed_date = cell_value.date()
                elif isinstance(cell_value, str):
                    # Try different date formats
                    try:
                        parsed_date = pd.to_datetime(cell_value).date()
                    except:
                        continue
                else:
                    try:
                        parsed_date = pd.to_datetime(str(cell_value)).date()
                    except:
                        continue
                
                # Store the row index for this date
                date_to_row[parsed_date] = row_idx
                
            except Exception as e:
                continue
        
        logger.info(f"Found {len(date_to_row)} valid dates in column AZ")
        logger.info(f"Date range in data: {min(date_to_row.keys()) if date_to_row else 'None'} to {max(date_to_row.keys()) if date_to_row else 'None'}")
        
        extracted_data = []
        matched_dates = 0
        
        for target_date in dates:
            # Create row data
            row_data = {}
            
            # Date
            row_data['Date'] = target_date.strftime('%Y-%m-%d')
            
            # Day of Week
            row_data['DOW'] = self.get_day_of_week(target_date)
            
            # Check if this date exists in the actual data
            if target_date in date_to_row:
                matched_dates += 1
                row_index = date_to_row[target_date]
                
                # Extract data from the matching row using SUMIFS logic
                for col_name, col_index in column_mapping.items():
                    if col_name in ['Date', 'DOW']:
                        continue  # Already handled above
                    
                    try:
                        if col_index < len(sheet_df.columns) and row_index < len(sheet_df):
                            value = sheet_df.iloc[row_index, col_index]
                            
                            # Clean and convert value
                            if pd.isna(value):
                                value = 0
                            elif isinstance(value, str):
                                try:
                                    # Remove common formatting
                                    clean_value = value.replace(',', '').replace('%', '').replace('$', '').strip()
                                    value = float(clean_value)
                                except:
                                    value = 0
                            else:
                                try:
                                    value = float(value)
                                except:
                                    value = 0
                            
                            row_data[col_name] = value
                        else:
                            row_data[col_name] = 0
                            
                    except Exception as e:
                        logger.warning(f"Error extracting {col_name} for {target_date}: {e}")
                        row_data[col_name] = 0
                
                # Log first few matches for verification
                if matched_dates <= 10:
                    logger.info(f"Matched {target_date.strftime('%Y-%m-%d')} to row {row_index}: Revenue={row_data.get('Revenue', 0)}, Rooms={row_data.get('Rm Sold', 0)}")
                    
            else:
                # Date not found in data - set all values to 0
                for col_name in column_mapping.keys():
                    if col_name not in ['Date', 'DOW']:
                        row_data[col_name] = 0
            
            extracted_data.append(row_data)
        
        logger.info(f"Extracted {len(extracted_data)} rows of occupancy data")
        logger.info(f"Successfully matched {matched_dates} dates with actual data")
        return extracted_data
    
    def find_data_start_row(self, sheet_df, date_column_index):
        """Find the row where daily data starts in DPR sheet"""
        logger.info("Finding data start row in DPR sheet...")
        
        # In DPR sheets, daily data usually starts after headers
        # Look for date patterns in the specified column (AZ)
        if date_column_index < len(sheet_df.columns):
            date_column = sheet_df.iloc[:, date_column_index]
            
            for i, value in enumerate(date_column):
                if pd.isna(value):
                    continue
                
                # Convert value to string for pattern matching
                value_str = str(value).strip().lower()
                
                # Look for January 1st patterns
                date_patterns = [
                    '1/1', '01/01', '1-jan', 'jan-1', 'january 1', 
                    '2025-01-01', '01-jan-2025', '1/1/2025'
                ]
                
                if any(pattern in value_str for pattern in date_patterns):
                    logger.info(f"Found January 1st pattern at row {i}: {value}")
                    return i
                
                # Check if it's a datetime object for January 1st
                try:
                    if isinstance(value, (datetime, pd.Timestamp)):
                        if value.month == 1 and value.day == 1:
                            logger.info(f"Found January 1st datetime at row {i}")
                            return i
                    # Try parsing as date
                    elif isinstance(value, str):
                        try:
                            parsed_date = pd.to_datetime(value)
                            if parsed_date.month == 1 and parsed_date.day == 1:
                                logger.info(f"Found parsed January 1st at row {i}")
                                return i
                        except:
                            pass
                except:
                    pass
        
        # Look for common DPR data start indicators
        # Check first 50 rows for data patterns
        for i in range(min(50, len(sheet_df))):
            row_data = sheet_df.iloc[i].astype(str).str.lower()
            
            # Look for date-like content in any column
            if any('jan' in cell or '1/1' in cell or '01/01' in cell for cell in row_data):
                logger.info(f"Found date indicator at row {i}")
                return i
        
        # Default to row 2 (assuming row 1 has headers)
        logger.info("Using default start row 2 for DPR data")
        return 2
    
    def convert_mhr_to_occupancy_csv(self, mhr_file_path, output_csv_path=None):
        """Convert MHR Excel file to daily occupancy CSV format"""
        try:
            if not os.path.exists(mhr_file_path):
                raise FileNotFoundError(f"MHR file not found: {mhr_file_path}")
            
            logger.info(f"Converting MHR file to occupancy CSV: {mhr_file_path}")
            
            # Find DPR sheet specifically
            sheet_name = self.find_dpr_sheet(mhr_file_path)
            if not sheet_name:
                raise ValueError("Could not find DPR sheet in MHR file")
            
            # Read the DPR sheet with all columns (to access AZ-BK range)
            logger.info(f"Reading DPR sheet: {sheet_name}")
            sheet_df = pd.read_excel(mhr_file_path, sheet_name=sheet_name, header=None)
            logger.info(f"DPR sheet shape: {sheet_df.shape}")
            logger.info(f"Total columns available: {len(sheet_df.columns)}")
            
            # Check if we have enough columns to reach BK (column 63)
            bk_column_index = self.excel_column_to_number('BK')
            if len(sheet_df.columns) <= bk_column_index:
                logger.warning(f"Sheet only has {len(sheet_df.columns)} columns, need at least {bk_column_index + 1} to reach column BK")
            
            # Generate date range for the year
            dates = self.generate_date_range(2025)
            logger.info(f"Generated {len(dates)} dates from Jan 1 to Dec 31")
            
            # Extract occupancy data using date matching
            occupancy_data = self.extract_occupancy_data(sheet_df, dates)
            
            # Create output dataframe
            output_df = pd.DataFrame(occupancy_data)
            
            # Reorder columns to match required format
            output_df = output_df[self.output_columns]
            
            # Save to CSV
            if output_csv_path is None:
                output_csv_path = "mhr_daily_occupancy.csv"
            
            output_df.to_csv(output_csv_path, index=False)
            logger.info(f"Occupancy data saved to: {output_csv_path}")
            logger.info(f"Output shape: {output_df.shape}")
            
            # Show sample data
            logger.info("\nFirst 10 rows:")
            print(output_df.head(10).to_string(index=False))
            
            # Show summary statistics
            logger.info(f"\nData summary:")
            logger.info(f"Date range: {output_df['Date'].min()} to {output_df['Date'].max()}")
            logger.info(f"Total revenue: {output_df['Revenue'].sum():,.2f}")
            logger.info(f"Average occupancy: {output_df['Occ%'].mean():.2f}%")
            logger.info(f"Average ADR: {output_df['ADR'].mean():.2f}")
            
            return output_df, output_csv_path
            
        except Exception as e:
            logger.error(f"Error converting MHR to occupancy CSV: {e}")
            raise
    
    def validate_output_data(self, output_df):
        """Validate the output data for consistency"""
        logger.info("Validating output data...")
        
        # Check for required columns
        missing_cols = [col for col in self.output_columns if col not in output_df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
        
        # Check date range
        if len(output_df) != 365:  # 2025 is not a leap year
            logger.warning(f"Expected 365 rows, got {len(output_df)}")
        
        # Check for negative values where they shouldn't exist
        numeric_cols = ['Rms', 'Daily Revenue', 'Occ%', 'Rm Sold', 'Revenue', 'ADR', 'RevPar']
        for col in numeric_cols:
            if col in output_df.columns:
                negative_count = (output_df[col] < 0).sum()
                if negative_count > 0:
                    logger.warning(f"Found {negative_count} negative values in {col}")
        
        # Check for unrealistic occupancy percentages
        if 'Occ%' in output_df.columns:
            high_occ = (output_df['Occ%'] > 100).sum()
            if high_occ > 0:
                logger.warning(f"Found {high_occ} occupancy values over 100%")
        
        logger.info("Validation complete")


def test_occupancy_converter():
    """Test the occupancy converter with the available MHR file"""
    converter = MHROccupancyConverter()
    
    mhr_file = "$$$ MHR Pick Up Report 2025 $$$ -22.07.25.xlsm"
    output_file = "mhr_daily_occupancy.csv"
    
    if os.path.exists(mhr_file):
        try:
            df, output_path = converter.convert_mhr_to_occupancy_csv(mhr_file, output_file)
            print(f"[SUCCESS] Occupancy conversion successful!")
            print(f"Output file: {output_path}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Validate the output
            converter.validate_output_data(df)
            
            return True
        except Exception as e:
            print(f"[ERROR] Occupancy conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"[ERROR] MHR file not found: {mhr_file}")
        return False


if __name__ == "__main__":
    test_occupancy_converter()
