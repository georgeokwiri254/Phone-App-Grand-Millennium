"""
Data Integration and Cleaning for Time Series Forecasting
Combines historical data (2022-2024) with 2025 data, addressing datetime errors,
standardizing columns, and preparing data for forecasting models.
"""

import pandas as pd
import numpy as np
import sqlite3
import glob
import os
from datetime import datetime, timedelta
import warnings
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HotelDataIntegrator:
    """Integrates and cleans hotel occupancy and revenue data from multiple years"""
    
    def __init__(self, data_dir: str, db_path: str):
        self.data_dir = Path(data_dir)
        self.db_path = db_path
        self.total_rooms = 339  # Grand Millennium Dubai total rooms
        self.scaler_standard = StandardScaler()
        self.scaler_minmax = MinMaxScaler()
        
    def load_historical_data(self) -> pd.DataFrame:
        """Load and standardize historical data (2022-2024)"""
        logger.info("Loading historical data (2022-2024)")
        
        historical_files = [
            "historical_occupancy_2022_040906 - historical_occupancy_2022_040906.csv.csv",
            "historical_occupancy_2023_040911 - historical_occupancy_2023_040911.csv.csv", 
            "historical_occupancy_2024_040917 - historical_occupancy_2024_040917.csv.csv"
        ]
        
        all_data = []
        
        for file in historical_files:
            file_path = self.data_dir / file
            if file_path.exists():
                logger.info(f"Processing {file}")
                df = pd.read_csv(file_path)
                
                # Standardize column names
                df.columns = ['Date', 'DOW', 'Rooms_Sold', 'Revenue', 'ADR', 'RevPAR']
                
                # Calculate missing Occupancy %
                df['Occupancy_Pct'] = (df['Rooms_Sold'] / self.total_rooms) * 100
                
                # Ensure proper data types
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
                df['ADR'] = pd.to_numeric(df['ADR'], errors='coerce')
                df['RevPAR'] = pd.to_numeric(df['RevPAR'], errors='coerce')
                df['Rooms_Sold'] = pd.to_numeric(df['Rooms_Sold'], errors='coerce')
                df['Occupancy_Pct'] = pd.to_numeric(df['Occupancy_Pct'], errors='coerce')
                
                # Extract year for validation
                year = df['Date'].dt.year.iloc[0] if not df['Date'].isna().all() else None
                logger.info(f"Loaded {len(df)} records for year {year}")
                
                all_data.append(df)
            else:
                logger.warning(f"File not found: {file}")
        
        if all_data:
            historical_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined historical data: {len(historical_df)} records")
            return historical_df
        else:
            logger.error("No historical data files found")
            return pd.DataFrame()
    
    def load_2025_data(self) -> pd.DataFrame:
        """Load and standardize 2025 data"""
        logger.info("Loading 2025 data")
        
        # Get the most recent 2025 file
        pattern = str(self.data_dir / "occupancy_20250817_*.csv")
        files_2025 = glob.glob(pattern)
        
        if not files_2025:
            # Try broader pattern
            pattern = str(self.data_dir / "occupancy_2025*.csv")
            files_2025 = glob.glob(pattern)
        
        if files_2025:
            # Use the most recent file
            latest_file = max(files_2025, key=os.path.getctime)
            logger.info(f"Processing 2025 data: {latest_file}")
            
            df = pd.read_csv(latest_file)
            
            # Map columns to standard format
            column_mapping = {
                'Date': 'Date',
                'DOW': 'DOW', 
                'Occ%': 'Occupancy_Pct',
                'Rm Sold': 'Rooms_Sold',
                'Revenue': 'Revenue',
                'ADR': 'ADR',
                'RevPar': 'RevPAR'
            }
            
            # Rename columns that exist
            existing_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df.rename(columns=existing_cols)
            
            # Ensure we have all required columns
            required_cols = ['Date', 'DOW', 'Occupancy_Pct', 'Rooms_Sold', 'Revenue', 'ADR', 'RevPAR']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'Occupancy_Pct' and 'Rooms_Sold' in df.columns:
                        df['Occupancy_Pct'] = (df['Rooms_Sold'] / self.total_rooms) * 100
                    else:
                        df[col] = np.nan
            
            # Clean data types
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            numeric_cols = ['Revenue', 'ADR', 'RevPAR', 'Rooms_Sold', 'Occupancy_Pct']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Filter out rows with invalid dates
            df = df.dropna(subset=['Date'])
            
            logger.info(f"Loaded {len(df)} records for 2025")
            return df[required_cols]
        else:
            logger.warning("No 2025 data files found")
            return pd.DataFrame()
    
    def validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the combined dataset"""
        logger.info("Validating and cleaning combined data")
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['Date'], keep='last')
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} duplicate dates")
        
        # Validate business rules
        df.loc[df['Occupancy_Pct'] > 100, 'Occupancy_Pct'] = 100
        df.loc[df['Occupancy_Pct'] < 0, 'Occupancy_Pct'] = 0
        df.loc[df['Revenue'] < 0, 'Revenue'] = 0
        df.loc[df['ADR'] < 0, 'ADR'] = 0
        df.loc[df['RevPAR'] < 0, 'RevPAR'] = 0
        
        # Recalculate metrics for consistency
        # RevPAR = ADR * Occupancy% / 100
        df['RevPAR_Calculated'] = df['ADR'] * df['Occupancy_Pct'] / 100
        
        # Use calculated RevPAR if original is missing or inconsistent
        mask = df['RevPAR'].isna() | (abs(df['RevPAR'] - df['RevPAR_Calculated']) > df['RevPAR'] * 0.1)
        df.loc[mask, 'RevPAR'] = df.loc[mask, 'RevPAR_Calculated']
        df = df.drop('RevPAR_Calculated', axis=1)
        
        # Fill missing dates
        df = self.fill_missing_dates(df)
        
        logger.info(f"Cleaned data: {len(df)} records from {df['Date'].min()} to {df['Date'].max()}")
        return df
    
    def fill_missing_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing dates in the time series"""
        if df.empty:
            return df
            
        # Create complete date range
        start_date = df['Date'].min()
        end_date = df['Date'].max()
        complete_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create complete dataframe
        complete_df = pd.DataFrame({'Date': complete_dates})
        
        # Merge with existing data
        df_filled = complete_df.merge(df, on='Date', how='left')
        
        # Fill missing values
        # For DOW, calculate from date
        df_filled['DOW'] = df_filled['Date'].dt.strftime('%a')
        
        # Forward fill other metrics
        numeric_cols = ['Revenue', 'ADR', 'RevPAR', 'Rooms_Sold', 'Occupancy_Pct']
        for col in numeric_cols:
            df_filled[col] = df_filled[col].fillna(method='ffill')
            
        # For any remaining NaNs at the beginning, use 0 for Revenue and mean for others
        df_filled['Revenue'] = df_filled['Revenue'].fillna(0)
        for col in ['ADR', 'RevPAR', 'Occupancy_Pct']:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
        df_filled['Rooms_Sold'] = df_filled['Rooms_Sold'].fillna(
            df_filled['Occupancy_Pct'] * self.total_rooms / 100
        )
        
        missing_count = len(complete_dates) - len(df)
        if missing_count > 0:
            logger.info(f"Filled {missing_count} missing dates")
        
        return df_filled
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based and rolling features"""
        logger.info("Adding time-based and rolling features")
        
        # Time-based features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
        
        # Rolling statistics (7, 14, 30 days)
        for window in [7, 14, 30]:
            df[f'Revenue_MA_{window}'] = df['Revenue'].rolling(window=window, min_periods=1).mean()
            df[f'Occupancy_MA_{window}'] = df['Occupancy_Pct'].rolling(window=window, min_periods=1).mean()
            df[f'ADR_MA_{window}'] = df['ADR'].rolling(window=window, min_periods=1).mean()
            df[f'RevPAR_MA_{window}'] = df['RevPAR'].rolling(window=window, min_periods=1).mean()
        
        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f'Revenue_Lag_{lag}'] = df['Revenue'].shift(lag)
            df[f'Occupancy_Lag_{lag}'] = df['Occupancy_Pct'].shift(lag)
            df[f'ADR_Lag_{lag}'] = df['ADR'].shift(lag)
        
        return df
    
    def normalize_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Apply standardization and normalization"""
        logger.info("Applying standardization and normalization")
        
        # Identify numeric columns for scaling
        base_metrics = ['Revenue', 'Occupancy_Pct', 'ADR', 'RevPAR', 'Rooms_Sold']
        rolling_cols = [col for col in df.columns if any(metric in col for metric in ['_MA_', '_Lag_'])]
        numeric_cols = base_metrics + rolling_cols
        
        # Only scale columns that exist and have numeric data
        cols_to_scale = [col for col in numeric_cols if col in df.columns]
        
        scalers = {}
        df_scaled = df.copy()
        
        # Standard scaling for base metrics
        if cols_to_scale:
            # Handle missing values before scaling
            df_for_scaling = df[cols_to_scale].fillna(df[cols_to_scale].mean())
            
            # Standard scaling
            scaled_data = self.scaler_standard.fit_transform(df_for_scaling)
            scaled_df = pd.DataFrame(scaled_data, columns=cols_to_scale, index=df.index)
            
            # Add scaled columns with suffix
            for col in cols_to_scale:
                df_scaled[f'{col}_Scaled'] = scaled_df[col]
            
            scalers['standard'] = self.scaler_standard
            
            # Min-Max scaling for some metrics (0-1 range)
            minmax_cols = ['Occupancy_Pct', 'Revenue', 'RevPAR']
            minmax_cols = [col for col in minmax_cols if col in df.columns]
            
            if minmax_cols:
                minmax_data = self.scaler_minmax.fit_transform(df_for_scaling[minmax_cols])
                minmax_df = pd.DataFrame(minmax_data, columns=minmax_cols, index=df.index)
                
                for col in minmax_cols:
                    df_scaled[f'{col}_Normalized'] = minmax_df[col]
                
                scalers['minmax'] = self.scaler_minmax
        
        logger.info(f"Applied scaling to {len(cols_to_scale)} columns")
        return df_scaled, scalers
    
    def save_to_sql(self, df: pd.DataFrame, table_name: str = 'hotel_data_combined'):
        """Save the combined dataset to SQLite database"""
        logger.info(f"Saving data to SQL database: {table_name}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Drop existing table if it exists
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Save to SQL
            df.to_sql(table_name, conn, index=False, if_exists='replace')
            
            # Create indexes for better performance
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_date ON {table_name}(Date)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_year_month ON {table_name}(Year, Month)")
            
            conn.commit()
            
            # Verify the save
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            result = conn.execute(count_query).fetchone()
            logger.info(f"Successfully saved {result[0]} records to {table_name}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving to SQL: {str(e)}")
            raise
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = None):
        """Save the combined dataset to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hotel_data_combined_{timestamp}.csv"
        
        output_path = self.data_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved combined data to: {output_path}")
        return output_path
    
    def get_prediction_period(self) -> Tuple[datetime, List[Tuple[str, datetime]]]:
        """Determine prediction periods based on current date"""
        current_date = datetime.now()
        current_month = current_date.replace(day=1)  # First day of current month
        
        prediction_periods = [
            ("3_months", current_month + timedelta(days=90)),
            ("6_months", current_month + timedelta(days=180)), 
            ("12_months", current_month + timedelta(days=365))
        ]
        
        logger.info(f"Current date: {current_date.strftime('%Y-%m-%d')}")
        logger.info("Prediction periods:")
        for period, end_date in prediction_periods:
            logger.info(f"  {period}: until {end_date.strftime('%Y-%m-%d')}")
        
        return current_date, prediction_periods
    
    def run_integration(self) -> Tuple[pd.DataFrame, Dict, str]:
        """Run the complete data integration pipeline"""
        logger.info("Starting hotel data integration pipeline")
        
        # Load data
        historical_df = self.load_historical_data()
        data_2025_df = self.load_2025_data()
        
        # Combine datasets
        if not historical_df.empty and not data_2025_df.empty:
            combined_df = pd.concat([historical_df, data_2025_df], ignore_index=True)
        elif not historical_df.empty:
            combined_df = historical_df
        elif not data_2025_df.empty:
            combined_df = data_2025_df
        else:
            raise ValueError("No data files found to process")
        
        # Clean and validate
        cleaned_df = self.validate_and_clean_data(combined_df)
        
        # Add features
        featured_df = self.add_features(cleaned_df)
        
        # Normalize data
        final_df, scalers = self.normalize_data(featured_df)
        
        # Save to SQL and CSV
        self.save_to_sql(final_df)
        csv_path = self.save_to_csv(final_df)
        
        # Get prediction periods
        current_date, prediction_periods = self.get_prediction_period()
        
        logger.info("Data integration pipeline completed successfully")
        logger.info(f"Final dataset shape: {final_df.shape}")
        logger.info(f"Date range: {final_df['Date'].min()} to {final_df['Date'].max()}")
        
        return final_df, scalers, str(csv_path)


def main():
    """Main execution function"""
    # Configuration
    base_dir = "/home/gee_devops254/Downloads/Revenue Architecture"
    data_dir = os.path.join(base_dir, "data", "processed")
    db_path = os.path.join(base_dir, "db", "revenue.db")
    
    # Run integration
    integrator = HotelDataIntegrator(data_dir, db_path)
    try:
        df, scalers, csv_path = integrator.run_integration()
        print(f"✅ Data integration completed successfully!")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"💾 Saved to: {csv_path}")
        print(f"🗄️ Saved to database: {db_path}")
        
        return df, scalers, csv_path
        
    except Exception as e:
        logger.error(f"Integration failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()