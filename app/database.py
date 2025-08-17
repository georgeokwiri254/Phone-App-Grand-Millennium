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
            
            # Create entered_on table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entered_on (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    FULL_NAME TEXT,
                    FIRST_NAME TEXT,
                    ARRIVAL DATE,
                    DEPARTURE DATE,
                    NIGHTS INTEGER,
                    PERSONS INTEGER,
                    ROOM TEXT,
                    TDF REAL,
                    NET REAL,
                    TOTAL REAL,
                    RATE_CODE TEXT,
                    INSERT_USER TEXT,
                    C_T_S_NAME TEXT,
                    SHORT_RESV_STATUS TEXT,
                    ADR REAL,
                    AMOUNT REAL,
                    COMMENT TEXT,
                    C_CHECK TEXT,
                    RESV_ID TEXT,
                    SEASON TEXT,
                    LONG_BOOKING_FLAG INTEGER,
                    SPLIT_MONTH TEXT,
                    SPLIT_YEAR INTEGER,
                    SPLIT_MONTH_NUM INTEGER,
                    NIGHTS_IN_MONTH INTEGER,
                    AMOUNT_IN_MONTH REAL,
                    PERIOD_START DATE,
                    PERIOD_END DATE,
                    ADR_IN_MONTH REAL,
                    BOOKING_LEAD_TIME INTEGER,
                    EVENTS_DATES TEXT,
                    COMPANY_CLEAN TEXT,
                    AUG INTEGER DEFAULT 0,
                    SEP INTEGER DEFAULT 0,
                    OCT INTEGER DEFAULT 0,
                    NOV INTEGER DEFAULT 0,
                    DEC INTEGER DEFAULT 0,
                    JAN2026 INTEGER DEFAULT 0,
                    FEB2026 INTEGER DEFAULT 0,
                    MAR2026 INTEGER DEFAULT 0,
                    APR2026 INTEGER DEFAULT 0,
                    MAY2026 INTEGER DEFAULT 0,
                    JUN2026 INTEGER DEFAULT 0,
                    JUL2026 INTEGER DEFAULT 0,
                    AUG2026 INTEGER DEFAULT 0,
                    SEP2026 INTEGER DEFAULT 0,
                    OCT2026 INTEGER DEFAULT 0,
                    NOV2026 INTEGER DEFAULT 0,
                    DEC2026 INTEGER DEFAULT 0,
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
            "CREATE INDEX IF NOT EXISTS idx_block_year_month ON block_analysis(Year, Month)",
            "CREATE INDEX IF NOT EXISTS idx_entered_on_arrival ON entered_on(ARRIVAL)",
            "CREATE INDEX IF NOT EXISTS idx_entered_on_company ON entered_on(COMPANY_CLEAN)",
            "CREATE INDEX IF NOT EXISTS idx_entered_on_resv_id ON entered_on(RESV_ID)",
            "CREATE INDEX IF NOT EXISTS idx_entered_on_split_month ON entered_on(SPLIT_MONTH)",
            "CREATE INDEX IF NOT EXISTS idx_entered_on_season ON entered_on(SEASON)"
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
    
    def ingest_entered_on_data(self, df: pd.DataFrame) -> bool:
        """
        Ingest entered on data into database
        
        Args:
            df: DataFrame with entered on data (from converter)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Ingesting entered on data: {len(df)} rows")
            
            # Clean data for database insertion
            df_clean = df.copy()
            
            # Convert date columns to string format for SQLite
            date_columns = ['ARRIVAL', 'DEPARTURE', 'PERIOD_START', 'PERIOD_END']
            for col in date_columns:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_datetime(df_clean[col]).dt.strftime('%Y-%m-%d')
            
            # Handle any column name mismatches
            if 'C=CHECK' in df_clean.columns:
                df_clean = df_clean.rename(columns={'C=CHECK': 'C_CHECK'})
            if 'RESV ID' in df_clean.columns:
                df_clean = df_clean.rename(columns={'RESV ID': 'RESV_ID'})
                
            # Use pandas to_sql for efficient bulk insert
            df_clean.to_sql('entered_on', self.connection, if_exists='replace', index=False)
            
            # Update metadata
            self._update_metadata('entered_on_last_updated', datetime.now().isoformat())
            self._update_metadata('entered_on_rows', str(len(df)))
            
            self.connection.commit()
            logger.info("Entered on data ingested successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest entered on data: {e}")
            logger.error(f"DataFrame columns: {list(df.columns)}")
            return False
    
    def get_entered_on_data(self, company_name: Optional[str] = None, 
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve entered on data
        
        Args:
            company_name: Filter by company name
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            DataFrame with entered on data
        """
        try:
            query = "SELECT * FROM entered_on"
            params = []
            conditions = []
            
            if company_name:
                conditions.append("COMPANY_CLEAN = ?")
                params.append(company_name)
            
            if start_date:
                conditions.append("ARRIVAL >= ?")
                params.append(start_date)
            
            if end_date:
                conditions.append("ARRIVAL <= ?")
                params.append(end_date)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY ARRIVAL, COMPANY_CLEAN"
            
            df = pd.read_sql_query(query, self.connection, params=params)
            
            # Convert date columns back to datetime
            date_columns = ['ARRIVAL', 'DEPARTURE', 'PERIOD_START', 'PERIOD_END']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to retrieve entered on data: {e}")
            return pd.DataFrame()
    
    def get_entered_on_summary_stats(self) -> dict:
        """
        Get summary statistics for entered on data
        
        Returns:
            Dictionary with entered on data statistics
        """
        try:
            stats = {}
            
            # Total bookings and amounts
            query = """SELECT 
                        COUNT(DISTINCT RESV_ID) as unique_bookings,
                        COUNT(*) as total_records,
                        SUM(AMOUNT_IN_MONTH) as total_amount,
                        SUM(NIGHTS_IN_MONTH) as total_nights,
                        AVG(ADR_IN_MONTH) as avg_adr
                       FROM entered_on"""
            result = self.connection.execute(query).fetchone()
            stats['unique_bookings'] = result[0] if result[0] else 0
            stats['total_records'] = result[1] if result[1] else 0
            stats['total_amount'] = result[2] if result[2] else 0
            stats['total_nights'] = result[3] if result[3] else 0
            stats['avg_adr'] = result[4] if result[4] else 0
            
            # Company distribution
            query = """SELECT COMPANY_CLEAN, COUNT(*) as bookings, SUM(AMOUNT_IN_MONTH) as amount 
                      FROM entered_on 
                      GROUP BY COMPANY_CLEAN 
                      ORDER BY amount DESC 
                      LIMIT 10"""
            result = self.connection.execute(query).fetchall()
            stats['top_companies'] = [{'company': row[0], 'bookings': row[1], 'amount': row[2]} 
                                    for row in result]
            
            # Season distribution
            query = """SELECT SEASON, COUNT(*) as bookings, SUM(NIGHTS_IN_MONTH) as nights 
                      FROM entered_on 
                      GROUP BY SEASON"""
            result = self.connection.execute(query).fetchall()
            stats['season_distribution'] = {row[0]: {'bookings': row[1], 'nights': row[2]} 
                                          for row in result}
            
            # Monthly distribution
            query = """SELECT SPLIT_MONTH, SUM(NIGHTS_IN_MONTH) as nights, SUM(AMOUNT_IN_MONTH) as amount 
                      FROM entered_on 
                      GROUP BY SPLIT_MONTH 
                      ORDER BY SPLIT_MONTH"""
            result = self.connection.execute(query).fetchall()
            stats['monthly_distribution'] = [{'month': row[0], 'nights': row[1], 'amount': row[2]} 
                                           for row in result]
            
            # Long bookings
            query = "SELECT COUNT(*) FROM entered_on WHERE LONG_BOOKING_FLAG = 1"
            result = self.connection.execute(query).fetchone()
            stats['long_bookings'] = result[0] if result[0] else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get entered on summary stats: {e}")
            return {}
    
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
            tables = ['segment_analysis', 'occupancy_analysis', 'forecasts', 'entered_on']
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
    
    def save_historical_occupancy_data(self, df: pd.DataFrame, table_name: str) -> bool:
        """
        Save historical occupancy data to database
        
        Args:
            df: DataFrame with historical occupancy data
            table_name: Name of the table to create/update
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            # Create table if it doesn't exist
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Date TEXT,
                    DOW TEXT,
                    "Rm Sold" REAL,
                    Revenue REAL,
                    ADR REAL,
                    RevPar REAL,
                    Occupancy_Pct REAL,
                    Year INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Clear existing data for this table
            cursor.execute(f"DELETE FROM {table_name}")
            
            # Insert data
            df.to_sql(table_name, self.connection, if_exists='append', index=False)
            self.connection.commit()
            
            logger.info(f"Successfully saved {len(df)} historical occupancy records to {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save historical occupancy data to {table_name}: {e}")
            return False
    
    def save_historical_segment_data(self, df: pd.DataFrame, table_name: str) -> bool:
        """
        Save historical segment data to database
        
        Args:
            df: DataFrame with historical segment data
            table_name: Name of the table to create/update
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            # Create table if it doesn't exist
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Month TEXT,
                    Segment TEXT,
                    Business_on_the_Books_Rooms REAL,
                    Business_on_the_Books_Revenue REAL,
                    Business_on_the_Books_ADR REAL,
                    Business_on_the_Books_Share_per_Segment REAL,
                    Dis_BOB REAL,
                    MergedSegment TEXT,
                    Year INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Clear existing data for this table
            cursor.execute(f"DELETE FROM {table_name}")
            
            # Insert data
            df.to_sql(table_name, self.connection, if_exists='append', index=False)
            self.connection.commit()
            
            logger.info(f"Successfully saved {len(df)} historical segment records to {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save historical segment data to {table_name}: {e}")
            return False
    
    def get_historical_occupancy_data(self, table_name: str) -> pd.DataFrame:
        """
        Get historical occupancy data from specified table
        
        Args:
            table_name: Name of the historical table
            
        Returns:
            DataFrame with historical occupancy data
        """
        try:
            query = f"SELECT * FROM {table_name} ORDER BY Date"
            df = pd.read_sql_query(query, self.connection)
            logger.info(f"Retrieved {len(df)} historical occupancy records from {table_name}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical occupancy data from {table_name}: {e}")
            return pd.DataFrame()
    
    def get_historical_segment_data(self, table_name: str) -> pd.DataFrame:
        """
        Get historical segment data from specified table
        
        Args:
            table_name: Name of the historical table
            
        Returns:
            DataFrame with historical segment data
        """
        try:
            query = f"SELECT * FROM {table_name} ORDER BY Month"
            df = pd.read_sql_query(query, self.connection)
            logger.info(f"Retrieved {len(df)} historical segment records from {table_name}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical segment data from {table_name}: {e}")
            return pd.DataFrame()
    
    def save_combined_forecast_data(self, df: pd.DataFrame) -> bool:
        """
        Save combined forecast data to database
        
        Args:
            df: DataFrame with combined forecast data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS combined_forecast_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Date TEXT,
                    "Rm Sold" REAL,
                    Revenue REAL,
                    ADR REAL,
                    RevPar REAL,
                    Occupancy_Pct REAL,
                    Year INTEGER,
                    Month INTEGER,
                    DayOfWeek INTEGER,
                    DayOfYear INTEGER,
                    Quarter INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Clear existing data
            cursor.execute("DELETE FROM combined_forecast_data")
            
            # Insert data
            df.to_sql('combined_forecast_data', self.connection, if_exists='append', index=False)
            self.connection.commit()
            
            logger.info(f"Successfully saved {len(df)} combined forecast records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save combined forecast data: {e}")
            return False
    
    def get_combined_forecast_data(self) -> pd.DataFrame:
        """
        Get combined forecast data from database
        
        Returns:
            DataFrame with combined forecast data
        """
        try:
            query = "SELECT * FROM combined_forecast_data ORDER BY Date"
            df = pd.read_sql_query(query, self.connection)
            if not df.empty:
                df['Date'] = pd.to_datetime(df['Date'])
                logger.info(f"Retrieved {len(df)} combined forecast records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get combined forecast data: {e}")
            return pd.DataFrame()

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