"""
SQLite Database Management for Revenue Analytics
Handles database schema creation, data ingestion, and querying
"""

import sqlite3
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class RevenueDatabase:
    """Manages SQLite database operations for revenue analytics"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file. Defaults to db/revenue.db
        """
        if db_path is None:
            # Get project root and create db path
            project_root = Path(__file__).parent.parent
            db_dir = project_root / 'db'
            db_dir.mkdir(exist_ok=True)
            db_path = db_dir / 'revenue.db'
        
        self.db_path = str(db_path)
        self.connection = None
        self._connect()
        self._create_schema()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _create_schema(self):
        """Create database schema with tables and indexes"""
        try:
            cursor = self.connection.cursor()
            
            # Create segment_analysis table (matches updated MHR column specification)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS segment_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Month DATE,
                    Index_Month_segment TEXT,
                    Segment TEXT,
                    Daily_Pick_up_Rooms REAL,
                    Daily_Pick_up_ADR REAL,
                    Daily_Pick_up_Revenue REAL,
                    Daily_Pick_up_Share REAL,
                    Dis_Daily REAL,
                    Month_to_Date_Rooms REAL,
                    Month_to_Date_ADR REAL,
                    Month_to_Date_Revenue REAL,
                    Month_to_Date_Share REAL,
                    Dis_MTD REAL,
                    Business_on_the_Books_Rooms REAL,
                    Business_on_the_Books_Revenue REAL,
                    Business_on_the_Books_ADR REAL,
                    Business_on_the_Books_Share_per_Segment REAL,
                    Dis_BOB REAL,
                    Business_on_the_Books_Same_Time_Last_Year_Rooms REAL,
                    Business_on_the_Books_Same_Time_Last_Year_Revenue REAL,
                    Business_on_the_Books_Same_Time_Last_Year_ADR REAL,
                    Business_on_the_Books_Same_Time_Last_Year_Share_per_Segment REAL,
                    Dis_BOB_STLY REAL,
                    Full_Month_Last_Year_Rooms REAL,
                    Full_Month_Last_Year_Revenue REAL,
                    Full_Month_Last_Year_ADR REAL,
                    Full_Month_Last_Year_Share_per_Segment REAL,
                    Dis_Full_Month REAL,
                    Budget_This_Year_Rooms REAL,
                    Budget_This_Year_Revenue REAL,
                    Budget_This_Year_ADR REAL,
                    Budget_This_Year_Share_per_Segment REAL,
                    Dis_Budget REAL,
                    Forecast_This_Year_Rooms REAL,
                    Forecast_This_Year_Revenue REAL,
                    Forecast_This_Year_ADR REAL,
                    Forecast_This_Year_Share_per_Segment REAL,
                    Dis_Forecast REAL,
                    Year_On_Year_Pace_for_rooms REAL,
                    Year_On_Year_Pace_Revenue REAL,
                    Year_On_Year_Pace_for_ADR REAL,
                    Year_on_Year_Rooms REAL,
                    Year_on_Year_Revenue REAL,
                    Year_on_Year_ADR REAL,
                    Delta_to_Budget_Rooms REAL,
                    Delta_to_Budget_Revenue REAL,
                    Delta_to_Budget_ADR REAL,
                    Delta_to_Forecast_Rooms REAL,
                    Delta_to_Forecast_Revenue REAL,
                    Delta_to_Forecast_ADR REAL,
                    MergedSegment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create occupancy_analysis table (matches CSV structure exactly)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS occupancy_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Date TEXT,
                    DOW TEXT,
                    Rms REAL,
                    Daily_Revenue REAL,
                    Occ_Pct REAL,
                    Rm_Sold REAL,
                    Revenue REAL,
                    ADR REAL,
                    RevPar REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create forecasts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_ts TIMESTAMP,
                    target TEXT,
                    horizon INTEGER,
                    forecast_date DATE,
                    forecast_value REAL,
                    confidence_lower REAL,
                    confidence_upper REAL,
                    model_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create block data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS block_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    BlockSize INTEGER,
                    AllotmentDate DATE,
                    SrepCode TEXT,
                    BookingStatus TEXT,
                    CompanyName TEXT,
                    BeginDate DATE,
                    AllotmentCode TEXT,
                    RateCode TEXT,
                    MonthDesc TEXT,
                    WeekDay TEXT,
                    DayOfMonth INTEGER,
                    Year INTEGER,
                    Month INTEGER,
                    Quarter INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            self._create_indexes(cursor)
            
            self.connection.commit()
            logger.info("Database schema created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database schema: {e}")
            raise
    
    def _create_indexes(self, cursor):
        """Create database indexes for query performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_segment_month ON segment_analysis(Month)",
            "CREATE INDEX IF NOT EXISTS idx_segment_merged ON segment_analysis(MergedSegment)",
            "CREATE INDEX IF NOT EXISTS idx_occ_date ON occupancy_analysis(Date)",
            "CREATE INDEX IF NOT EXISTS idx_occ_dow ON occupancy_analysis(DOW)",
            "CREATE INDEX IF NOT EXISTS idx_forecast_target ON forecasts(target)",
            "CREATE INDEX IF NOT EXISTS idx_forecast_date ON forecasts(forecast_date)",
            "CREATE INDEX IF NOT EXISTS idx_block_allotment_date ON block_analysis(AllotmentDate)",
            "CREATE INDEX IF NOT EXISTS idx_block_booking_status ON block_analysis(BookingStatus)",
            "CREATE INDEX IF NOT EXISTS idx_block_srep_code ON block_analysis(SrepCode)",
            "CREATE INDEX IF NOT EXISTS idx_block_company ON block_analysis(CompanyName)",
            "CREATE INDEX IF NOT EXISTS idx_block_year_month ON block_analysis(Year, Month)"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                logger.debug(f"Created index: {index_sql}")
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")
    
    def ingest_segment_data(self, df: pd.DataFrame) -> bool:
        """
        Ingest segment analysis data into database
        
        Args:
            df: DataFrame with segment data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Ingesting segment data: {len(df)} rows")
            
            # CSV now has correct column names from MHR processor - no mapping needed
            df_clean = df.copy()
            
            # Use pandas to_sql for efficient bulk insert
            df_clean.to_sql('segment_analysis', self.connection, if_exists='replace', index=False)
            
            # Update metadata
            self._update_metadata('segment_last_updated', datetime.now().isoformat())
            self._update_metadata('segment_rows', str(len(df)))
            
            self.connection.commit()
            logger.info("Segment data ingested successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest segment data: {e}")
            logger.error(f"DataFrame columns: {list(df.columns)}")
            return False
    
    def ingest_occupancy_data(self, df: pd.DataFrame) -> bool:
        """
        Ingest occupancy analysis data into database
        
        Args:
            df: DataFrame with occupancy data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Ingesting occupancy data: {len(df)} rows")
            
            # Clean column names to match database schema
            df_clean = df.copy()
            
            # Map CSV columns to database columns to handle spaces and special characters
            column_mapping = {
                'Occ%': 'Occ_Pct',
                'Rm Sold': 'Rm_Sold',
                'Daily Revenue': 'Daily_Revenue'
            }
            
            df_clean = df_clean.rename(columns=column_mapping)
            
            # Use pandas to_sql for efficient bulk insert
            df_clean.to_sql('occupancy_analysis', self.connection, if_exists='replace', index=False)
            
            # Update metadata
            self._update_metadata('occupancy_last_updated', datetime.now().isoformat())
            self._update_metadata('occupancy_rows', str(len(df)))
            
            self.connection.commit()
            logger.info("Occupancy data ingested successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest occupancy data: {e}")
            logger.error(f"DataFrame columns: {list(df.columns)}")
            return False
    
    def get_segment_data(self, merged_segment: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve segment analysis data
        
        Args:
            merged_segment: Filter by specific merged segment
            
        Returns:
            DataFrame with segment data
        """
        try:
            query = "SELECT * FROM segment_analysis"
            params = []
            
            if merged_segment:
                query += " WHERE MergedSegment = ?"
                params.append(merged_segment)
            
            query += " ORDER BY Month, MergedSegment"
            
            df = pd.read_sql_query(query, self.connection, params=params)
            return df
            
        except Exception as e:
            logger.error(f"Failed to retrieve segment data: {e}")
            return pd.DataFrame()
    
    def get_occupancy_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve occupancy analysis data
        
        Args:
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            DataFrame with occupancy data
        """
        try:
            query = "SELECT * FROM occupancy_analysis"
            params = []
            conditions = []
            
            if start_date:
                conditions.append("Date >= ?")
                params.append(start_date)
            
            if end_date:
                conditions.append("Date <= ?")
                params.append(end_date)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY Date"
            
            df = pd.read_sql_query(query, self.connection, params=params)
            return df
            
        except Exception as e:
            logger.error(f"Failed to retrieve occupancy data: {e}")
            return pd.DataFrame()
    
    def save_forecast(self, target: str, forecast_data: pd.DataFrame, model_type: str = "ExponentialSmoothing") -> bool:
        """
        Save forecast results to database
        
        Args:
            target: Forecast target (e.g., 'occupancy', 'revenue')
            forecast_data: DataFrame with forecast results
            model_type: Type of forecasting model used
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            run_ts = datetime.now().isoformat()
            
            # Prepare forecast data for insertion
            forecast_records = []
            for _, row in forecast_data.iterrows():
                record = {
                    'run_ts': run_ts,
                    'target': target,
                    'forecast_date': row.get('date'),
                    'forecast_value': row.get('forecast'),
                    'confidence_lower': row.get('lower_ci'),
                    'confidence_upper': row.get('upper_ci'),
                    'model_type': model_type
                }
                forecast_records.append(record)
            
            # Insert forecast data
            forecast_df = pd.DataFrame(forecast_records)
            forecast_df.to_sql('forecasts', self.connection, if_exists='append', index=False)
            
            self.connection.commit()
            logger.info(f"Saved {len(forecast_records)} forecast records for {target}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save forecast: {e}")
            return False
    
    def get_latest_forecast(self, target: str) -> pd.DataFrame:
        """
        Retrieve latest forecast for a target
        
        Args:
            target: Forecast target
            
        Returns:
            DataFrame with latest forecast data
        """
        try:
            query = """
                SELECT * FROM forecasts 
                WHERE target = ? AND run_ts = (
                    SELECT MAX(run_ts) FROM forecasts WHERE target = ?
                )
                ORDER BY forecast_date
            """
            
            df = pd.read_sql_query(query, self.connection, params=[target, target])
            return df
            
        except Exception as e:
            logger.error(f"Failed to retrieve forecast: {e}")
            return pd.DataFrame()
    
    def _update_metadata(self, key: str, value: str):
        """Update metadata key-value pair"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO metadata (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (key, value))
        except Exception as e:
            logger.warning(f"Failed to update metadata {key}: {e}")
    
    def get_metadata(self, key: str) -> Optional[str]:
        """Get metadata value by key"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.warning(f"Failed to get metadata {key}: {e}")
            return None
    
    def get_database_stats(self) -> dict:
        """Get database statistics"""
        try:
            stats = {}
            cursor = self.connection.cursor()
            
            # Table row counts
            tables = ['segment_analysis', 'occupancy_analysis', 'forecasts']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f'{table}_rows'] = cursor.fetchone()[0]
            
            # Last updated times
            stats['segment_last_updated'] = self.get_metadata('segment_last_updated')
            stats['occupancy_last_updated'] = self.get_metadata('occupancy_last_updated')
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def ingest_block_data(self, df: pd.DataFrame) -> bool:
        """
        Ingest block analysis data into database
        
        Args:
            df: DataFrame with block data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Ingesting block data: {len(df)} rows")
            
            # Clean data for database insertion
            df_clean = df.copy()
            
            # Convert date columns to string format for SQLite
            date_columns = ['AllotmentDate', 'BeginDate']
            for col in date_columns:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_datetime(df_clean[col]).dt.strftime('%Y-%m-%d')
            
            # Use pandas to_sql for efficient bulk insert
            df_clean.to_sql('block_analysis', self.connection, if_exists='replace', index=False)
            
            # Update metadata
            self._update_metadata('block_last_updated', datetime.now().isoformat())
            self._update_metadata('block_rows', str(len(df)))
            
            logger.info("Block data ingestion completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest block data: {e}")
            return False
    
    def get_block_data(self, booking_status: Optional[str] = None, 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve block analysis data
        
        Args:
            booking_status: Filter by booking status (ACT/DEF/PSP/TEN)
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            DataFrame with block data
        """
        try:
            query = "SELECT * FROM block_analysis"
            params = []
            conditions = []
            
            if booking_status:
                conditions.append("BookingStatus = ?")
                params.append(booking_status)
            
            if start_date:
                conditions.append("AllotmentDate >= ?")
                params.append(start_date)
            
            if end_date:
                conditions.append("AllotmentDate <= ?")
                params.append(end_date)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY AllotmentDate, CompanyName"
            
            df = pd.read_sql_query(query, self.connection, params=params)
            
            # Convert date columns back to datetime
            date_columns = ['AllotmentDate', 'BeginDate']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to retrieve block data: {e}")
            return pd.DataFrame()
    
    def get_block_summary_stats(self) -> dict:
        """
        Get summary statistics for block data
        
        Returns:
            Dictionary with block data statistics
        """
        try:
            stats = {}
            
            # Total blocks and size
            query = "SELECT SUM(BlockSize) as total_blocks, COUNT(*) as total_records FROM block_analysis"
            result = self.connection.execute(query).fetchone()
            stats['total_blocks'] = result[0] if result[0] else 0
            stats['total_records'] = result[1] if result[1] else 0
            
            # Booking status distribution
            query = "SELECT BookingStatus, SUM(BlockSize) as blocks FROM block_analysis GROUP BY BookingStatus"
            result = self.connection.execute(query).fetchall()
            stats['booking_status_distribution'] = {row[0]: row[1] for row in result}
            
            # Top companies by block size
            query = """SELECT CompanyName, SUM(BlockSize) as total_blocks 
                      FROM block_analysis 
                      GROUP BY CompanyName 
                      ORDER BY total_blocks DESC 
                      LIMIT 10"""
            result = self.connection.execute(query).fetchall()
            stats['top_companies'] = {row[0]: row[1] for row in result}
            
            # Monthly distribution
            query = """SELECT Year, Month, SUM(BlockSize) as blocks 
                      FROM block_analysis 
                      GROUP BY Year, Month 
                      ORDER BY Year, Month"""
            result = self.connection.execute(query).fetchall()
            stats['monthly_distribution'] = [{'year': row[0], 'month': row[1], 'blocks': row[2]} 
                                           for row in result]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get block summary stats: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

def get_database() -> RevenueDatabase:
    """Get database instance - singleton pattern"""
    if not hasattr(get_database, '_instance'):
        get_database._instance = RevenueDatabase()
    return get_database._instance