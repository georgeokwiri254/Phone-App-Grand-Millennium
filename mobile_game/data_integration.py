"""
Grand Millennium Revenue Analytics - Data Integration Layer

Connects mobile game with existing Streamlit app data sources,
providing seamless AED currency integration and real-time updates.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
import csv

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhanced_analytics_engine import get_analytics_engine
from aed_currency_handler import AEDCurrencyHandler

class DataIntegrationManager:
    """Manages data integration between mobile game and existing systems"""
    
    def __init__(self):
        """Initialize data integration manager"""
        self.app_root = Path(__file__).parent.parent
        self.analytics_engine = get_analytics_engine()
        self.aed_handler = AEDCurrencyHandler()
        
        # Data paths
        self.data_paths = {
            'processed': self.app_root / 'data' / 'processed',
            'db': self.app_root / 'db',
            'forecasts': self.app_root / 'forecasts'
        }
        
        # Ensure directories exist
        for path in self.data_paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def sync_with_streamlit_data(self) -> Dict[str, bool]:
        """Sync mobile game data with existing Streamlit app data"""
        sync_results = {
            'occupancy_data': False,
            'segment_data': False,
            'forecast_data': False,
            'block_data': False
        }
        
        try:
            # Sync occupancy data
            if self.sync_occupancy_data():
                sync_results['occupancy_data'] = True
            
            # Sync segment data  
            if self.sync_segment_data():
                sync_results['segment_data'] = True
            
            # Sync forecast data
            if self.sync_forecast_data():
                sync_results['forecast_data'] = True
            
            # Sync block data
            if self.sync_block_data():
                sync_results['block_data'] = True
                
        except Exception as e:
            print(f"Data sync error: {e}")
        
        return sync_results
    
    def sync_occupancy_data(self) -> bool:
        """Sync occupancy data from processed CSV files"""
        try:
            occupancy_file = self.data_paths['processed'] / 'occupancy.csv'
            
            if not occupancy_file.exists():
                print("No occupancy.csv found, creating sample data")
                return self.create_sample_occupancy_data()
            
            # Read existing occupancy data
            df = pd.read_csv(occupancy_file)
            
            # Ensure required columns exist
            required_columns = ['Date', 'Total_Rooms_Available', 'Total_Rooms_Occupied', 
                              'Occupancy_Percentage', 'ADR', 'RevPAR']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing columns in occupancy data: {missing_columns}")
                return self.create_sample_occupancy_data()
            
            # Convert AED values and ensure proper formatting
            df = self.process_occupancy_aed_data(df)
            
            # Save to game database
            with sqlite3.connect(str(self.analytics_engine.db_path)) as conn:
                df.to_sql('occupancy_analysis', conn, if_exists='replace', index=False)
            
            print(f"✅ Synced {len(df)} occupancy records")
            return True
            
        except Exception as e:
            print(f"Error syncing occupancy data: {e}")
            return False
    
    def process_occupancy_aed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process occupancy data to ensure proper AED formatting"""
        # Ensure date format
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        # Ensure numeric columns
        numeric_columns = ['Total_Rooms_Available', 'Total_Rooms_Occupied', 
                          'Occupancy_Percentage', 'ADR', 'RevPAR']
        
        for col in numeric_columns:
            if col in df.columns:
                # Handle potential currency symbols or formatting
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calculate Total_Revenue if not present
        if 'Total_Revenue' not in df.columns:
            df['Total_Revenue'] = df['Total_Rooms_Occupied'] * df['ADR']
        
        # Ensure reasonable bounds for game mechanics
        df['Occupancy_Percentage'] = df['Occupancy_Percentage'].clip(0, 100)
        df['ADR'] = df['ADR'].clip(100, 2000)  # Reasonable AED ADR range
        df['RevPAR'] = df['RevPAR'].clip(0, 2000)  # Reasonable AED RevPAR range
        
        return df
    
    def sync_segment_data(self) -> bool:
        """Sync segment analysis data from processed CSV files"""
        try:
            segment_file = self.data_paths['processed'] / 'segment.csv'
            
            if not segment_file.exists():
                print("No segment.csv found, creating sample data")
                return self.create_sample_segment_data()
            
            # Read existing segment data
            df = pd.read_csv(segment_file)
            
            # Process segment data for mobile game
            df = self.process_segment_aed_data(df)
            
            # Save to game database
            with sqlite3.connect(str(self.analytics_engine.db_path)) as conn:
                df.to_sql('segment_analysis', conn, if_exists='replace', index=False)
            
            print(f"✅ Synced {len(df)} segment records")
            return True
            
        except Exception as e:
            print(f"Error syncing segment data: {e}")
            return False
    
    def process_segment_aed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process segment data to ensure proper AED formatting and game compatibility"""
        # Standardize column names
        column_mapping = {
            'market_segment': 'Market_Segment',
            'Market Segment': 'Market_Segment',
            'revenue': 'Revenue',
            'Revenue (AED)': 'Revenue',
            'room_nights': 'Room_Nights',
            'Room Nights': 'Room_Nights',
            'adr': 'ADR',
            'Average Rate': 'ADR'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure date format
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        else:
            # Add current date if missing
            df['Date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Process AED revenue values
        if 'Revenue' in df.columns:
            df['Revenue'] = self.clean_aed_column(df['Revenue'])
        
        if 'ADR' in df.columns:
            df['ADR'] = self.clean_aed_column(df['ADR'])
        
        # Add booking count if missing
        if 'Booking_Count' not in df.columns and 'Room_Nights' in df.columns:
            # Estimate booking count (rooms per booking varies by segment)
            segment_booking_ratios = {
                'Corporate': 1.2,
                'Leisure': 2.5,
                'Group': 8.0,
                'Government': 3.0,
                'Online Travel Agent': 1.8,
                'Walk-in': 1.0
            }
            
            df['Booking_Count'] = df.apply(
                lambda row: max(1, int(row.get('Room_Nights', 0) / 
                segment_booking_ratios.get(row.get('Market_Segment', 'Corporate'), 2.0))), 
                axis=1
            )
        
        # Ensure required columns exist with defaults
        required_columns = {
            'Market_Segment': 'Unknown',
            'Revenue': 0,
            'Room_Nights': 0,
            'ADR': 0,
            'Booking_Count': 0
        }
        
        for col, default in required_columns.items():
            if col not in df.columns:
                df[col] = default
        
        return df
    
    def clean_aed_column(self, series: pd.Series) -> pd.Series:
        """Clean AED currency column removing symbols and converting to numeric"""
        if series.dtype == 'object':
            # Remove currency symbols and formatting
            cleaned = series.astype(str).str.replace(r'[^\d.-]', '', regex=True)
            return pd.to_numeric(cleaned, errors='coerce').fillna(0)
        else:
            return pd.to_numeric(series, errors='coerce').fillna(0)
    
    def sync_forecast_data(self) -> bool:
        """Sync forecast data from forecast files"""
        try:
            forecast_dir = self.data_paths['forecasts']
            
            if not forecast_dir.exists():
                print("No forecasts directory found")
                return False
            
            # Look for recent forecast files
            forecast_files = list(forecast_dir.glob('*forecast*.csv'))
            
            if not forecast_files:
                print("No forecast files found")
                return False
            
            # Process most recent forecast file
            latest_file = max(forecast_files, key=os.path.getctime)
            df = pd.read_csv(latest_file)
            
            # Process forecast data
            df = self.process_forecast_aed_data(df)
            
            # Save to game database
            with sqlite3.connect(str(self.analytics_engine.db_path)) as conn:
                df.to_sql('forecast_analysis', conn, if_exists='replace', index=False)
            
            print(f"✅ Synced forecast data from {latest_file.name}")
            return True
            
        except Exception as e:
            print(f"Error syncing forecast data: {e}")
            return False
    
    def process_forecast_aed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process forecast data for game integration"""
        # Ensure date column
        date_columns = ['Date', 'date', 'Forecast_Date', 'Period']
        date_col = None
        
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df['Date'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
        
        # Process AED columns
        aed_columns = ['Forecasted_Revenue', 'Revenue_Forecast', 'Predicted_ADR', 'Forecasted_RevPAR']
        
        for col in aed_columns:
            if col in df.columns:
                df[col] = self.clean_aed_column(df[col])
        
        return df
    
    def sync_block_data(self) -> bool:
        """Sync block booking data if available"""
        try:
            # Look for block data in processed directory
            block_files = list(self.data_paths['processed'].glob('*block*.csv'))
            
            if not block_files:
                print("No block data files found, creating sample data")
                return self.create_sample_block_data()
            
            # Use the most recent block file
            latest_file = max(block_files, key=os.path.getctime)
            df = pd.read_csv(latest_file)
            
            # Process block data
            df = self.process_block_aed_data(df)
            
            # Save to game database
            with sqlite3.connect(str(self.analytics_engine.db_path)) as conn:
                df.to_sql('block_analysis', conn, if_exists='replace', index=False)
            
            print(f"✅ Synced block data from {latest_file.name}")
            return True
            
        except Exception as e:
            print(f"Error syncing block data: {e}")
            return False
    
    def process_block_aed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process block booking data for game integration"""
        # Standardize column names
        column_mapping = {
            'group_name': 'Group_Name',
            'Group Name': 'Group_Name',
            'arrival_date': 'Arrival_Date',
            'departure_date': 'Departure_Date',
            'block_size': 'Block_Size',
            'quoted_rate': 'Quoted_Rate',
            'status': 'Block_Status'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Process dates
        date_columns = ['Arrival_Date', 'Departure_Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
        
        # Process AED rates
        if 'Quoted_Rate' in df.columns:
            df['Quoted_Rate'] = self.clean_aed_column(df['Quoted_Rate'])
        
        # Calculate revenue potential
        if 'Revenue_Potential' not in df.columns:
            if all(col in df.columns for col in ['Block_Size', 'Quoted_Rate', 'Arrival_Date', 'Departure_Date']):
                df['Revenue_Potential'] = df.apply(self.calculate_block_revenue, axis=1)
        
        return df
    
    def calculate_block_revenue(self, row) -> float:
        """Calculate potential revenue for block booking"""
        try:
            arrival = datetime.strptime(row['Arrival_Date'], '%Y-%m-%d')
            departure = datetime.strptime(row['Departure_Date'], '%Y-%m-%d')
            nights = (departure - arrival).days
            
            return float(row['Block_Size']) * nights * float(row['Quoted_Rate'])
        except:
            return 0.0
    
    def create_sample_occupancy_data(self) -> bool:
        """Create sample occupancy data for testing"""
        try:
            sample_data = self.analytics_engine.generate_sample_hotel_data()
            occupancy_df = sample_data['occupancy_analysis']
            
            # Save to CSV
            occupancy_file = self.data_paths['processed'] / 'occupancy.csv'
            occupancy_df.to_csv(occupancy_file, index=False)
            
            # Save to database
            with sqlite3.connect(str(self.analytics_engine.db_path)) as conn:
                occupancy_df.to_sql('occupancy_analysis', conn, if_exists='replace', index=False)
            
            print("✅ Created sample occupancy data")
            return True
            
        except Exception as e:
            print(f"Error creating sample occupancy data: {e}")
            return False
    
    def create_sample_segment_data(self) -> bool:
        """Create sample segment data for testing"""
        try:
            sample_data = self.analytics_engine.generate_sample_hotel_data()
            segment_df = sample_data['segment_analysis']
            
            # Save to CSV
            segment_file = self.data_paths['processed'] / 'segment.csv'
            segment_df.to_csv(segment_file, index=False)
            
            # Save to database  
            with sqlite3.connect(str(self.analytics_engine.db_path)) as conn:
                segment_df.to_sql('segment_analysis', conn, if_exists='replace', index=False)
            
            print("✅ Created sample segment data")
            return True
            
        except Exception as e:
            print(f"Error creating sample segment data: {e}")
            return False
    
    def create_sample_block_data(self) -> bool:
        """Create sample block booking data for testing"""
        try:
            sample_data = self.analytics_engine.generate_sample_hotel_data()
            block_df = sample_data['block_analysis']
            
            # Save to CSV
            block_file = self.data_paths['processed'] / 'block_data.csv'
            block_df.to_csv(block_file, index=False)
            
            # Save to database
            with sqlite3.connect(str(self.analytics_engine.db_path)) as conn:
                block_df.to_sql('block_analysis', conn, if_exists='replace', index=False)
            
            print("✅ Created sample block data")
            return True
            
        except Exception as e:
            print(f"Error creating sample block data: {e}")
            return False
    
    def get_data_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all data sources"""
        status = {}
        
        # Check processed files
        for data_type in ['occupancy', 'segment']:
            file_path = self.data_paths['processed'] / f'{data_type}.csv'
            status[f'{data_type}_file'] = {
                'exists': file_path.exists(),
                'path': str(file_path),
                'size': file_path.stat().st_size if file_path.exists() else 0,
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None
            }
        
        # Check database
        db_path = self.analytics_engine.db_path
        status['database'] = {
            'exists': db_path.exists(),
            'path': str(db_path),
            'size': db_path.stat().st_size if db_path.exists() else 0
        }
        
        # Check database tables
        if db_path.exists():
            try:
                with sqlite3.connect(str(db_path)) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    status['database']['tables'] = {}
                    for table in tables:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        status['database']['tables'][table] = count
                        
            except Exception as e:
                status['database']['error'] = str(e)
        
        return status
    
    def export_game_data_summary(self) -> Dict[str, Any]:
        """Export summary of game data for mobile interface"""
        try:
            kpis = self.analytics_engine.get_real_time_kpis()
            segments = self.analytics_engine.get_segment_performance_advanced()
            market_conditions = self.analytics_engine.get_current_market_conditions()
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'kpis': {
                    'occupancy': {
                        'value': kpis['occupancy_pct'],
                        'display': kpis['occupancy_display'],
                        'status': 'excellent' if kpis['occupancy_pct'] >= 90 else 'good' if kpis['occupancy_pct'] >= 80 else 'warning'
                    },
                    'adr': {
                        'value': kpis['adr_aed'],
                        'display': kpis['adr_display'],
                        'currency': 'AED'
                    },
                    'revpar': {
                        'value': kpis['revpar_aed'],
                        'display': kpis['revpar_display'],
                        'currency': 'AED'
                    },
                    'revenue': {
                        'value': kpis['revenue_aed'],
                        'display': kpis['revenue_display'],
                        'currency': 'AED'
                    }
                },
                'segments': {
                    'top_performers': segments[:3],
                    'total_segments': len(segments),
                    'total_revenue': sum(s['revenue_aed'] for s in segments)
                },
                'market': {
                    'condition': market_conditions['condition'],
                    'description': market_conditions['description'],
                    'demand_multiplier': market_conditions['demand_multiplier']
                },
                'data_freshness': {
                    'last_occupancy_date': kpis['date'],
                    'segments_period': '30 days',
                    'market_condition_updated': datetime.now().isoformat()
                }
            }
            
            return summary
            
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Global data integration manager
_data_manager = None

def get_data_integration_manager() -> DataIntegrationManager:
    """Get or create global data integration manager instance"""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataIntegrationManager()
    return _data_manager

if __name__ == "__main__":
    # Test data integration
    print("🔄 Data Integration Manager Test")
    print("=" * 40)
    
    manager = DataIntegrationManager()
    
    # Test data sync
    print("📊 Syncing data with Streamlit app...")
    sync_results = manager.sync_with_streamlit_data()
    
    for data_type, success in sync_results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {data_type}")
    
    # Test data status
    print("\n📋 Data Status:")
    status = manager.get_data_status()
    
    for source, info in status.items():
        print(f"  {source}: {'✅' if info.get('exists', False) else '❌'}")
        if 'tables' in info:
            for table, count in info['tables'].items():
                print(f"    {table}: {count} records")
    
    # Test game data summary
    print("\n🎮 Game Data Summary:")
    summary = manager.export_game_data_summary()
    
    if 'error' not in summary:
        print(f"  Current Occupancy: {summary['kpis']['occupancy']['display']}")
        print(f"  Current ADR: {summary['kpis']['adr']['display']}")
        print(f"  Market Condition: {summary['market']['condition']}")
        print(f"  Top Segment: {summary['segments']['top_performers'][0]['segment'] if summary['segments']['top_performers'] else 'None'}")
    else:
        print(f"  Error: {summary['error']}")
    
    print("\n✅ Data Integration Ready for Mobile Game!")