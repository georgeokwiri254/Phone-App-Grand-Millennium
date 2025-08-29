"""
Grand Millennium Revenue Analytics - Enhanced Analytics Engine

Advanced analytics engine that connects mobile game with real hotel data,
providing AED-focused insights and game mechanics integration.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

from aed_currency_handler import AEDCurrencyHandler, aed_to_points

class EnhancedAnalyticsEngine:
    """Enhanced analytics engine for mobile game integration"""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize enhanced analytics engine"""
        if db_path is None:
            db_path = Path(__file__).parent.parent / "db" / "revenue.db"
        
        self.db_path = db_path
        self.aed_handler = AEDCurrencyHandler()
        
        # Cache for performance
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes
        
        # Initialize database connection
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Ensure database and tables exist"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Check if main tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                if not tables:
                    self.create_sample_data()
                    
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data for testing mobile game"""
        sample_data = self.generate_sample_hotel_data()
        
        with sqlite3.connect(str(self.db_path)) as conn:
            # Create and populate sample tables
            sample_data['occupancy_analysis'].to_sql('occupancy_analysis', conn, if_exists='replace', index=False)
            sample_data['segment_analysis'].to_sql('segment_analysis', conn, if_exists='replace', index=False)
            sample_data['block_analysis'].to_sql('block_analysis', conn, if_exists='replace', index=False)
    
    def generate_sample_hotel_data(self) -> Dict[str, pd.DataFrame]:
        """Generate realistic sample hotel data for testing"""
        # Date range for sample data
        start_date = datetime.now() - timedelta(days=90)
        date_range = pd.date_range(start=start_date, periods=90, freq='D')
        
        # Occupancy data
        occupancy_data = []
        base_occupancy = 75.0
        
        for i, date in enumerate(date_range):
            # Seasonal variation
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * i / 365)
            # Weekend effect
            weekend_factor = 1.1 if date.weekday() >= 5 else 1.0
            # Random variation
            random_factor = np.random.normal(1.0, 0.1)
            
            occupancy_pct = min(100, max(30, base_occupancy * seasonal_factor * weekend_factor * random_factor))
            
            rooms_available = 200  # Hotel has 200 rooms
            rooms_occupied = int(rooms_available * occupancy_pct / 100)
            
            # Calculate rates and revenue
            base_adr = 450  # Base ADR in AED
            adr = base_adr * seasonal_factor * np.random.normal(1.0, 0.05)
            revpar = adr * (occupancy_pct / 100)
            total_revenue = rooms_occupied * adr
            
            occupancy_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Total_Rooms_Available': rooms_available,
                'Total_Rooms_Occupied': rooms_occupied,
                'Occupancy_Percentage': round(occupancy_pct, 1),
                'ADR': round(adr, 2),
                'RevPAR': round(revpar, 2),
                'Total_Revenue': round(total_revenue, 2)
            })
        
        # Segment data
        segments = ['Corporate', 'Leisure', 'Group', 'Government', 'Online Travel Agent', 'Walk-in']
        segment_data = []
        
        for date in date_range[-30:]:  # Last 30 days
            daily_revenue = np.random.uniform(80000, 120000)  # Daily revenue in AED
            
            for segment in segments:
                # Different segments have different market shares
                segment_weights = {
                    'Corporate': 0.25,
                    'Leisure': 0.30,
                    'Group': 0.15,
                    'Government': 0.10,
                    'Online Travel Agent': 0.15,
                    'Walk-in': 0.05
                }
                
                base_weight = segment_weights.get(segment, 0.1)
                actual_weight = base_weight * np.random.normal(1.0, 0.2)
                
                segment_revenue = daily_revenue * actual_weight
                room_nights = int(segment_revenue / np.random.uniform(400, 600))  # Random ADR per segment
                segment_adr = segment_revenue / room_nights if room_nights > 0 else 0
                
                segment_data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Market_Segment': segment,
                    'Revenue': round(segment_revenue, 2),
                    'Room_Nights': room_nights,
                    'ADR': round(segment_adr, 2),
                    'Booking_Count': max(1, int(room_nights * np.random.uniform(0.8, 1.2)))
                })
        
        # Block booking data
        block_data = []
        group_names = [
            'Dubai Business Conference 2025',
            'Al-Rashid Wedding Celebration',
            'Emirates Corporate Retreat',
            'International Medical Summit',
            'UAE Government Training',
            'Tourism Board Delegation'
        ]
        
        future_dates = pd.date_range(start=datetime.now() + timedelta(days=7), periods=60, freq='D')
        
        for i, group_name in enumerate(group_names):
            arrival_date = future_dates[i * 10]
            nights = np.random.randint(2, 5)
            departure_date = arrival_date + timedelta(days=nights)
            block_size = np.random.randint(15, 50)
            quoted_rate = np.random.uniform(380, 580)
            
            status = np.random.choice(['Inquiry', 'Tentative', 'Confirmed', 'Cancelled'], 
                                    p=[0.3, 0.4, 0.2, 0.1])
            
            block_data.append({
                'Group_Name': group_name,
                'Arrival_Date': arrival_date.strftime('%Y-%m-%d'),
                'Departure_Date': departure_date.strftime('%Y-%m-%d'),
                'Block_Size': block_size,
                'Quoted_Rate': round(quoted_rate, 2),
                'Block_Status': status,
                'Revenue_Potential': round(block_size * nights * quoted_rate, 2)
            })
        
        return {
            'occupancy_analysis': pd.DataFrame(occupancy_data),
            'segment_analysis': pd.DataFrame(segment_data),
            'block_analysis': pd.DataFrame(block_data)
        }
    
    def get_real_time_kpis(self) -> Dict[str, Union[float, str]]:
        """Get real-time KPIs for dashboard display"""
        cache_key = "real_time_kpis"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Get today's performance
                today = datetime.now().strftime('%Y-%m-%d')
                
                query = """
                SELECT 
                    Occupancy_Percentage,
                    ADR,
                    RevPAR,
                    Total_Revenue,
                    Total_Rooms_Occupied,
                    Total_Rooms_Available
                FROM occupancy_analysis 
                WHERE Date = ? 
                ORDER BY Date DESC 
                LIMIT 1
                """
                
                df = pd.read_sql_query(query, conn, params=[today])
                
                if df.empty:
                    # Get latest available data
                    query_latest = """
                    SELECT 
                        Occupancy_Percentage,
                        ADR,
                        RevPAR,
                        Total_Revenue,
                        Total_Rooms_Occupied,
                        Total_Rooms_Available,
                        Date
                    FROM occupancy_analysis 
                    ORDER BY Date DESC 
                    LIMIT 1
                    """
                    df = pd.read_sql_query(query_latest, conn)
                
                if not df.empty:
                    row = df.iloc[0]
                    kpis = {
                        'occupancy_pct': float(row['Occupancy_Percentage']),
                        'adr_aed': float(row['ADR']),
                        'revpar_aed': float(row['RevPAR']),
                        'revenue_aed': float(row['Total_Revenue']),
                        'rooms_occupied': int(row['Total_Rooms_Occupied']),
                        'rooms_available': int(row['Total_Rooms_Available']),
                        'date': row.get('Date', today),
                        
                        # Formatted for mobile display
                        'occupancy_display': f"{row['Occupancy_Percentage']:.1f}%",
                        'adr_display': self.aed_handler.format_mobile_display(row['ADR'], 'dashboard_card'),
                        'revpar_display': self.aed_handler.format_mobile_display(row['RevPAR'], 'dashboard_card'),
                        'revenue_display': self.aed_handler.format_mobile_display(row['Total_Revenue'], 'dashboard_card')
                    }
                else:
                    # Fallback sample data
                    kpis = self._get_fallback_kpis()
                
        except Exception as e:
            print(f"Error getting KPIs: {e}")
            kpis = self._get_fallback_kpis()
        
        self._cache[cache_key] = (kpis, datetime.now())
        return kpis
    
    def _get_fallback_kpis(self) -> Dict[str, Union[float, str]]:
        """Get fallback KPIs when database is unavailable"""
        sample_occupancy = 82.5
        sample_adr = 465.0
        sample_revpar = sample_adr * (sample_occupancy / 100)
        sample_revenue = sample_revpar * 200  # 200 rooms
        
        return {
            'occupancy_pct': sample_occupancy,
            'adr_aed': sample_adr,
            'revpar_aed': sample_revpar,
            'revenue_aed': sample_revenue,
            'rooms_occupied': int(200 * sample_occupancy / 100),
            'rooms_available': 200,
            'date': datetime.now().strftime('%Y-%m-%d'),
            
            'occupancy_display': f"{sample_occupancy:.1f}%",
            'adr_display': self.aed_handler.format_mobile_display(sample_adr, 'dashboard_card'),
            'revpar_display': self.aed_handler.format_mobile_display(sample_revpar, 'dashboard_card'),
            'revenue_display': self.aed_handler.format_mobile_display(sample_revenue, 'dashboard_card')
        }
    
    def get_segment_performance_advanced(self, days: int = 30) -> List[Dict]:
        """Get advanced segment performance for customer conquest game"""
        cache_key = f"segment_performance_{days}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                
                query = """
                SELECT 
                    Market_Segment,
                    SUM(Revenue) as Total_Revenue,
                    SUM(Room_Nights) as Total_Room_Nights,
                    AVG(ADR) as Avg_ADR,
                    COUNT(DISTINCT Date) as Active_Days,
                    SUM(Booking_Count) as Total_Bookings
                FROM segment_analysis 
                WHERE Date >= ?
                GROUP BY Market_Segment
                ORDER BY Total_Revenue DESC
                """
                
                df = pd.read_sql_query(query, conn, params=[cutoff_date])
                
                segments = []
                for _, row in df.iterrows():
                    # Calculate performance metrics
                    revenue_per_day = row['Total_Revenue'] / max(row['Active_Days'], 1)
                    booking_value = row['Total_Revenue'] / max(row['Total_Bookings'], 1)
                    
                    # Market potential scoring (for game)
                    market_score = self._calculate_segment_market_score(row)
                    
                    segments.append({
                        'segment': row['Market_Segment'],
                        'revenue_aed': float(row['Total_Revenue']),
                        'room_nights': int(row['Total_Room_Nights']),
                        'avg_adr': float(row['Avg_ADR']),
                        'total_bookings': int(row['Total_Bookings']),
                        'revenue_per_day': revenue_per_day,
                        'booking_value': booking_value,
                        'market_score': market_score,
                        
                        # Mobile-formatted displays
                        'revenue_display': self.aed_handler.format_mobile_display(row['Total_Revenue'], 'segment_budget'),
                        'adr_display': self.aed_handler.format_mobile_display(row['Avg_ADR'], 'pricing_slider'),
                        'score_display': f"{market_score:.0f} pts"
                    })
                
        except Exception as e:
            print(f"Error getting segment performance: {e}")
            segments = self._get_fallback_segments()
        
        self._cache[cache_key] = (segments, datetime.now())
        return segments
    
    def _calculate_segment_market_score(self, row: pd.Series) -> float:
        """Calculate market potential score for segment"""
        # Weighted scoring based on multiple factors
        revenue_score = min(row['Total_Revenue'] / 10000, 100)  # Max 100 points
        adr_score = min(row['Avg_ADR'] / 10, 50)  # Max 50 points  
        volume_score = min(row['Total_Room_Nights'] / 10, 30)  # Max 30 points
        consistency_score = min(row['Active_Days'], 20)  # Max 20 points
        
        return revenue_score + adr_score + volume_score + consistency_score
    
    def _get_fallback_segments(self) -> List[Dict]:
        """Get fallback segment data"""
        fallback_segments = [
            {'segment': 'Corporate', 'revenue_aed': 750000, 'avg_adr': 520, 'room_nights': 450, 'market_score': 185},
            {'segment': 'Leisure', 'revenue_aed': 680000, 'avg_adr': 465, 'room_nights': 520, 'market_score': 172},
            {'segment': 'Group', 'revenue_aed': 420000, 'avg_adr': 380, 'room_nights': 380, 'market_score': 145},
            {'segment': 'Online Travel Agent', 'revenue_aed': 350000, 'avg_adr': 425, 'room_nights': 290, 'market_score': 125},
            {'segment': 'Government', 'revenue_aed': 280000, 'avg_adr': 400, 'room_nights': 225, 'market_score': 108},
            {'segment': 'Walk-in', 'revenue_aed': 120000, 'avg_adr': 350, 'room_nights': 95, 'market_score': 85}
        ]
        
        # Add mobile formatting
        for segment in fallback_segments:
            segment.update({
                'total_bookings': int(segment['room_nights'] * 0.8),
                'revenue_per_day': segment['revenue_aed'] / 30,
                'booking_value': segment['revenue_aed'] / max(segment['room_nights'], 1),
                'revenue_display': self.aed_handler.format_mobile_display(segment['revenue_aed'], 'segment_budget'),
                'adr_display': self.aed_handler.format_mobile_display(segment['avg_adr'], 'pricing_slider'),
                'score_display': f"{segment['market_score']:.0f} pts"
            })
        
        return fallback_segments
    
    def calculate_pricing_optimization(self, proposed_adr: float, 
                                     market_conditions: Optional[Dict] = None) -> Dict:
        """Advanced pricing optimization with market factors"""
        kpis = self.get_real_time_kpis()
        current_adr = kpis['adr_aed']
        current_occupancy = kpis['occupancy_pct']
        
        # Market conditions impact
        if market_conditions is None:
            market_conditions = self.get_current_market_conditions()
        
        demand_multiplier = market_conditions.get('demand_multiplier', 1.0)
        
        # Price elasticity model
        rate_change_pct = (proposed_adr - current_adr) / current_adr if current_adr > 0 else 0
        
        # Advanced elasticity calculation
        base_elasticity = -1.5  # Base price elasticity
        market_adjustment = (demand_multiplier - 1.0) * 0.5
        adjusted_elasticity = base_elasticity + market_adjustment
        
        occupancy_change = rate_change_pct * adjusted_elasticity * 100
        projected_occupancy = max(20, min(100, current_occupancy + occupancy_change))
        
        # Revenue calculations
        projected_revpar = proposed_adr * (projected_occupancy / 100)
        current_revpar = current_adr * (current_occupancy / 100)
        revpar_change = projected_revpar - current_revpar
        
        # Competitive analysis
        competitive_score = self._calculate_competitive_score(proposed_adr, market_conditions)
        
        # Game scoring
        base_score = aed_to_points(projected_revpar * 200)  # 200 rooms
        
        # Bonus/penalty factors
        if revpar_change > 0:
            bonus_multiplier = min(2.0, 1 + (revpar_change / current_revpar))
        else:
            bonus_multiplier = max(0.5, 1 + (revpar_change / current_revpar))
        
        final_score = int(base_score * bonus_multiplier * competitive_score)
        
        return {
            'proposed_adr': proposed_adr,
            'projected_occupancy': round(projected_occupancy, 1),
            'projected_revpar': round(projected_revpar, 2),
            'revpar_change': round(revpar_change, 2),
            'rate_change_pct': round(rate_change_pct * 100, 1),
            'competitive_score': competitive_score,
            'game_score': max(0, final_score),
            
            # Mobile displays
            'adr_display': self.aed_handler.format_mobile_display(proposed_adr, 'pricing_slider'),
            'revpar_display': self.aed_handler.format_mobile_display(projected_revpar, 'dashboard_card'),
            'change_display': self.aed_handler.format_mobile_display(revpar_change, 'dashboard_card'),
            'score_display': f"{final_score:,} points"
        }
    
    def _calculate_competitive_score(self, proposed_adr: float, market_conditions: Dict) -> float:
        """Calculate competitive positioning score"""
        # Simplified competitive analysis
        market_condition = market_conditions.get('condition', 'Normal')
        
        competitive_rates = {
            'High Demand': (550, 650),  # AED range
            'Normal': (400, 500),
            'Low Demand': (300, 400),
            'Special Event': (600, 800)
        }
        
        rate_range = competitive_rates.get(market_condition, (400, 500))
        optimal_min, optimal_max = rate_range
        
        if optimal_min <= proposed_adr <= optimal_max:
            return 1.2  # Bonus for optimal pricing
        elif proposed_adr < optimal_min:
            return max(0.8, 1.0 - (optimal_min - proposed_adr) / optimal_min)
        else:
            return max(0.7, 1.0 - (proposed_adr - optimal_max) / optimal_max)
    
    def get_current_market_conditions(self) -> Dict:
        """Get current market conditions for pricing game"""
        # This would integrate with real market data in production
        conditions = [
            {
                'condition': 'High Demand',
                'demand_multiplier': 1.4,
                'description': 'Peak tourism season - rates can be increased',
                'icon': '📈',
                'color': (0.0, 0.8, 0.0, 1)
            },
            {
                'condition': 'Normal',
                'demand_multiplier': 1.0,
                'description': 'Regular business conditions',
                'icon': '📊',
                'color': (0.5, 0.5, 1.0, 1)
            },
            {
                'condition': 'Low Demand',
                'demand_multiplier': 0.7,
                'description': 'Off-season - focus on occupancy over rates',
                'icon': '📉',
                'color': (1.0, 0.5, 0.0, 1)
            },
            {
                'condition': 'Special Event',
                'demand_multiplier': 1.8,
                'description': 'Major event in Dubai - premium pricing opportunity',
                'icon': '🎉',
                'color': (1.0, 0.84, 0.0, 1)
            }
        ]
        
        # Simulate market condition based on time of year, day of week, etc.
        now = datetime.now()
        
        # Winter season (Nov-Mar) = High Demand
        if now.month in [11, 12, 1, 2, 3]:
            return conditions[0]
        # Summer (Jun-Aug) = Low Demand  
        elif now.month in [6, 7, 8]:
            return conditions[2]
        # Weekend = Higher demand
        elif now.weekday() >= 5:
            return conditions[0]
        # Special events (simplified)
        elif now.day in [1, 15]:  # Simulate special events on 1st and 15th
            return conditions[3]
        else:
            return conditions[1]
    
    def get_challenge_targets(self, challenge_type: str, player_level: int = 1) -> Dict:
        """Get challenge targets based on current performance and player level"""
        kpis = self.get_real_time_kpis()
        
        targets = {
            'daily_occupancy': {
                'target_value': max(80.0, kpis['occupancy_pct'] * 1.05),  # 5% improvement
                'reward_base': 5000,
                'description': f"Achieve {max(80.0, kpis['occupancy_pct'] * 1.05):.1f}% occupancy"
            },
            'revenue_target': {
                'target_value': kpis['revenue_aed'] * 1.1,  # 10% improvement
                'reward_base': 7500,
                'description': f"Generate {self.aed_handler.format_aed(kpis['revenue_aed'] * 1.1, compact=True)} revenue"
            },
            'adr_optimization': {
                'target_value': kpis['adr_aed'] * 1.08,  # 8% improvement
                'reward_base': 6000,
                'description': f"Achieve {self.aed_handler.format_aed(kpis['adr_aed'] * 1.08)} ADR"
            }
        }
        
        # Adjust targets based on player level
        level_multipliers = {1: 1.0, 2: 1.1, 3: 1.25, 4: 1.4, 5: 1.6}
        multiplier = level_multipliers.get(player_level, 1.0)
        
        if challenge_type in targets:
            target = targets[challenge_type].copy()
            target['reward_base'] = int(target['reward_base'] * multiplier)
            return target
        
        return targets.get('daily_occupancy', {})
    
    def _get_from_cache(self, key: str) -> Optional[any]:
        """Get data from cache if not expired"""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if (datetime.now() - timestamp).seconds < self._cache_timeout:
                return data
        return None
    
    def clear_cache(self):
        """Clear analytics cache"""
        self._cache.clear()

# Global analytics engine instance
_analytics_engine = None

def get_analytics_engine() -> EnhancedAnalyticsEngine:
    """Get or create global analytics engine instance"""
    global _analytics_engine
    if _analytics_engine is None:
        _analytics_engine = EnhancedAnalyticsEngine()
    return _analytics_engine

if __name__ == "__main__":
    # Test enhanced analytics engine
    print("📊 Enhanced Analytics Engine Test")
    print("=" * 45)
    
    engine = EnhancedAnalyticsEngine()
    
    # Test KPIs
    print("📈 Real-time KPIs:")
    kpis = engine.get_real_time_kpis()
    for key, value in kpis.items():
        if isinstance(value, str) and 'display' in key:
            print(f"  {key}: {value}")
    
    # Test segment analysis
    print("\n👥 Segment Performance:")
    segments = engine.get_segment_performance_advanced()
    for segment in segments[:3]:  # Top 3 segments
        print(f"  {segment['segment']}: {segment['revenue_display']} ({segment['score_display']})")
    
    # Test pricing optimization
    print("\n💎 Pricing Optimization:")
    current_adr = kpis['adr_aed']
    proposed_adr = current_adr * 1.1  # 10% increase
    
    pricing_result = engine.calculate_pricing_optimization(proposed_adr)
    print(f"  Current ADR: {engine.aed_handler.format_aed(current_adr)}")
    print(f"  Proposed ADR: {pricing_result['adr_display']}")
    print(f"  Projected RevPAR: {pricing_result['revpar_display']}")
    print(f"  Game Score: {pricing_result['score_display']}")
    
    # Test market conditions
    print("\n🎯 Market Conditions:")
    market = engine.get_current_market_conditions()
    print(f"  Condition: {market['icon']} {market['condition']}")
    print(f"  Description: {market['description']}")
    
    print("\n✅ Enhanced Analytics Engine Ready for Mobile Game!")