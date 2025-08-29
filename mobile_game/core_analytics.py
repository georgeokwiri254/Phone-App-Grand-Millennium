"""
Grand Millennium Revenue Analytics - Core Backend Logic
Extracted from Streamlit app for mobile game integration

This module provides UI-independent analytics functions that can be used
by both the Streamlit app and the Kivy mobile game.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path

class RevenueAnalytics:
    """Core revenue analytics engine - UI independent"""
    
    def __init__(self, db_path=None):
        """Initialize with database connection"""
        if db_path is None:
            db_path = Path(__file__).parent.parent / "db" / "revenue.db"
        self.db_path = db_path
        self.aed_conversion_rate = 1.0  # Base currency is AED
    
    def get_database_connection(self):
        """Get database connection"""
        return sqlite3.connect(str(self.db_path))
    
    def format_aed_currency(self, amount):
        """Format amount as AED currency"""
        if pd.isna(amount):
            return "د.إ 0"
        return f"د.إ {amount:,.0f}"
    
    def get_daily_occupancy_stats(self, date_filter=None):
        """Get daily occupancy statistics for game scoring"""
        with self.get_database_connection() as conn:
            query = """
            SELECT 
                Date,
                Total_Rooms_Available,
                Total_Rooms_Occupied,
                Occupancy_Percentage,
                RevPAR,
                ADR
            FROM occupancy_analysis 
            """
            if date_filter:
                query += f" WHERE Date >= '{date_filter}'"
            query += " ORDER BY Date DESC LIMIT 30"
            
            df = pd.read_sql_query(query, conn)
            
            if df.empty:
                return {}
            
            latest = df.iloc[0]
            return {
                'occupancy_pct': float(latest['Occupancy_Percentage']),
                'revpar_aed': float(latest['RevPAR']) if pd.notna(latest['RevPAR']) else 0,
                'adr_aed': float(latest['ADR']) if pd.notna(latest['ADR']) else 0,
                'rooms_occupied': int(latest['Total_Rooms_Occupied']),
                'rooms_available': int(latest['Total_Rooms_Available']),
                'date': latest['Date']
            }
    
    def calculate_daily_challenge_score(self, occupancy_target=85.0):
        """Calculate score for daily occupancy challenge"""
        stats = self.get_daily_occupancy_stats()
        if not stats:
            return 0
        
        occupancy = stats['occupancy_pct']
        base_score = stats['revpar_aed']
        
        # Bonus multipliers
        if occupancy >= occupancy_target:
            if occupancy >= 95.0:
                multiplier = 3.0  # Exceptional performance
            elif occupancy >= 90.0:
                multiplier = 2.5  # Excellent
            else:
                multiplier = 2.0  # Good
        else:
            multiplier = 1.0  # Base score only
        
        return int(base_score * multiplier)
    
    def get_segment_performance(self, period_days=30):
        """Get segment analysis for customer conquest game"""
        with self.get_database_connection() as conn:
            query = """
            SELECT 
                Market_Segment,
                SUM(Revenue) as Total_Revenue,
                COUNT(*) as Booking_Count,
                AVG(ADR) as Avg_ADR,
                SUM(Room_Nights) as Total_Room_Nights
            FROM segment_analysis 
            WHERE Date >= date('now', '-{} days')
            GROUP BY Market_Segment
            ORDER BY Total_Revenue DESC
            """.format(period_days)
            
            df = pd.read_sql_query(query, conn)
            
            segments = []
            for _, row in df.iterrows():
                segments.append({
                    'segment': row['Market_Segment'],
                    'revenue_aed': float(row['Total_Revenue']) if pd.notna(row['Total_Revenue']) else 0,
                    'bookings': int(row['Booking_Count']),
                    'adr_aed': float(row['Avg_ADR']) if pd.notna(row['Avg_ADR']) else 0,
                    'room_nights': int(row['Total_Room_Nights'])
                })
            
            return segments
    
    def calculate_segment_strategy_score(self, selected_segments, budget_allocation):
        """Calculate score for segment strategy game"""
        segments = self.get_segment_performance()
        total_score = 0
        
        for segment_data in segments:
            segment_name = segment_data['segment']
            if segment_name in selected_segments:
                budget_pct = budget_allocation.get(segment_name, 0)
                revenue_potential = segment_data['revenue_aed']
                
                # Score based on budget allocation efficiency
                efficiency_score = revenue_potential * (budget_pct / 100)
                total_score += efficiency_score
        
        return int(total_score)
    
    def get_adr_optimization_data(self):
        """Get data for ADR pricing game"""
        stats = self.get_daily_occupancy_stats()
        segments = self.get_segment_performance(7)  # Last week
        
        market_conditions = self.get_market_conditions()
        
        return {
            'current_adr': stats.get('adr_aed', 0),
            'current_occupancy': stats.get('occupancy_pct', 0),
            'segment_mix': segments,
            'market_conditions': market_conditions
        }
    
    def get_market_conditions(self):
        """Simulate market conditions for pricing game"""
        # This would integrate with real market data in production
        conditions = ['High Demand', 'Normal', 'Low Demand', 'Special Event']
        demand_multipliers = [1.5, 1.0, 0.7, 2.0]
        
        import random
        condition_idx = random.randint(0, len(conditions) - 1)
        
        return {
            'condition': conditions[condition_idx],
            'demand_multiplier': demand_multipliers[condition_idx],
            'description': self.get_condition_description(conditions[condition_idx])
        }
    
    def get_condition_description(self, condition):
        """Get description for market conditions"""
        descriptions = {
            'High Demand': 'Tourism season peak - rates can be increased',
            'Normal': 'Regular business conditions',
            'Low Demand': 'Off-season - focus on occupancy over rates',
            'Special Event': 'Major event in city - premium pricing opportunity'
        }
        return descriptions.get(condition, 'Standard market conditions')
    
    def calculate_pricing_score(self, proposed_adr, market_conditions):
        """Calculate score for pricing decisions"""
        current_data = self.get_adr_optimization_data()
        current_adr = current_data['current_adr']
        current_occupancy = current_data['current_occupancy']
        
        demand_multiplier = market_conditions['demand_multiplier']
        
        # Estimate occupancy impact of rate change
        rate_change_pct = (proposed_adr - current_adr) / current_adr if current_adr > 0 else 0
        occupancy_impact = -rate_change_pct * 50  # Simplified elasticity
        new_occupancy = max(0, min(100, current_occupancy + occupancy_impact))
        
        # Apply market conditions
        new_occupancy *= demand_multiplier
        new_occupancy = min(100, new_occupancy)
        
        # Calculate RevPAR and score
        revpar = (proposed_adr * new_occupancy / 100)
        score = int(revpar * 10)  # Scale for game points
        
        return {
            'score': score,
            'projected_occupancy': round(new_occupancy, 1),
            'projected_revpar': round(revpar, 2),
            'rate_change_pct': round(rate_change_pct * 100, 1)
        }
    
    def get_block_booking_opportunities(self):
        """Get group booking requests for block analysis game"""
        with self.get_database_connection() as conn:
            query = """
            SELECT 
                Group_Name,
                Block_Size,
                Arrival_Date,
                Departure_Date,
                Quoted_Rate,
                Block_Status
            FROM block_analysis 
            WHERE Block_Status IN ('Inquiry', 'Tentative')
            ORDER BY Arrival_Date 
            LIMIT 10
            """
            
            try:
                df = pd.read_sql_query(query, conn)
                
                blocks = []
                for _, row in df.iterrows():
                    nights = (pd.to_datetime(row['Departure_Date']) - pd.to_datetime(row['Arrival_Date'])).days
                    revenue_potential = row['Block_Size'] * nights * row['Quoted_Rate']
                    
                    blocks.append({
                        'group_name': row['Group_Name'],
                        'block_size': int(row['Block_Size']),
                        'nights': nights,
                        'rate_aed': float(row['Quoted_Rate']),
                        'revenue_potential_aed': float(revenue_potential),
                        'arrival_date': row['Arrival_Date'],
                        'departure_date': row['Departure_Date']
                    })
                
                return blocks
            except:
                # Return sample data if table doesn't exist
                return self.get_sample_block_data()
    
    def get_sample_block_data(self):
        """Generate sample block booking data for testing"""
        import random
        from datetime import datetime, timedelta
        
        sample_blocks = []
        group_names = ['Corporate Conference', 'Wedding Party', 'Travel Group', 'Sports Team', 'Exhibition Visitors']
        
        for i in range(5):
            arrival = datetime.now() + timedelta(days=random.randint(7, 60))
            nights = random.randint(1, 5)
            departure = arrival + timedelta(days=nights)
            block_size = random.randint(10, 50)
            rate = random.randint(300, 800)
            
            sample_blocks.append({
                'group_name': f"{group_names[i]} #{i+1}",
                'block_size': block_size,
                'nights': nights,
                'rate_aed': rate,
                'revenue_potential_aed': block_size * nights * rate,
                'arrival_date': arrival.strftime('%Y-%m-%d'),
                'departure_date': departure.strftime('%Y-%m-%d')
            })
        
        return sample_blocks
    
    def calculate_block_decision_score(self, block_data, decision, current_occupancy_forecast=75):
        """Calculate score for block booking decisions"""
        revenue_potential = block_data['revenue_potential_aed']
        block_impact = (block_data['block_size'] / 100) * 100  # Assuming 100 rooms
        
        if decision == 'accept':
            # Positive: Revenue gained
            score = revenue_potential * 0.1  # Scale down for game points
            
            # Penalty if we're already high occupancy (displacement risk)
            if current_occupancy_forecast > 85:
                displacement_penalty = score * 0.3
                score -= displacement_penalty
        else:
            # Opportunity cost, but avoid displacement risk
            if current_occupancy_forecast > 85:
                score = revenue_potential * 0.05  # Small reward for avoiding risk
            else:
                score = -revenue_potential * 0.02  # Small penalty for missed opportunity
        
        return int(score)
    
    def get_monthly_achievements(self, player_level=1):
        """Get available achievements based on player level"""
        base_achievements = [
            {'name': 'Occupancy Hero', 'target': 85, 'metric': 'occupancy_pct', 'reward': 5000},
            {'name': 'Revenue Champion', 'target': 500000, 'metric': 'monthly_revenue', 'reward': 10000},
            {'name': 'Rate Master', 'target': 450, 'metric': 'adr_aed', 'reward': 7500},
        ]
        
        # Add level-specific achievements
        if player_level >= 2:
            base_achievements.extend([
                {'name': 'Segment Specialist', 'target': 3, 'metric': 'top_segments', 'reward': 12000},
                {'name': 'Forecast Prophet', 'target': 95, 'metric': 'forecast_accuracy', 'reward': 15000},
            ])
        
        if player_level >= 3:
            base_achievements.extend([
                {'name': 'Block Master', 'target': 5, 'metric': 'blocks_accepted', 'reward': 20000},
                {'name': 'RevPAR Legend', 'target': 400, 'metric': 'revpar_aed', 'reward': 25000},
            ])
        
        return base_achievements
    
    def get_leaderboard_data(self, metric='revenue', period='monthly'):
        """Get leaderboard data for social features"""
        # This would query actual player performance data
        # For now, return sample data
        sample_players = [
            {'name': 'Ahmad Al-Rashid', 'score': 750000, 'level': 4},
            {'name': 'Sarah Mohammed', 'score': 690000, 'level': 3},
            {'name': 'Omar Hassan', 'score': 650000, 'level': 3},
            {'name': 'Fatima Ali', 'score': 620000, 'level': 2},
            {'name': 'You', 'score': 580000, 'level': 2},
        ]
        return sample_players

class GameProgressManager:
    """Manage player progress and achievements"""
    
    def __init__(self, player_id="default"):
        self.player_id = player_id
        self.progress_file = Path(__file__).parent / f"progress_{player_id}.json"
    
    def get_player_level(self, total_points):
        """Calculate player level based on total points"""
        if total_points >= 500000:
            return 5
        elif total_points >= 200000:
            return 4
        elif total_points >= 50000:
            return 3
        elif total_points >= 10000:
            return 2
        else:
            return 1
    
    def get_level_name(self, level):
        """Get level name for UI display"""
        level_names = {
            1: "Trainee Manager",
            2: "Assistant Manager", 
            3: "Revenue Manager",
            4: "Director of Revenue",
            5: "Revenue Strategist"
        }
        return level_names.get(level, "Unknown")
    
    def calculate_points_to_next_level(self, current_points):
        """Calculate points needed for next level"""
        thresholds = [10000, 50000, 200000, 500000, float('inf')]
        current_level = self.get_player_level(current_points)
        
        if current_level >= 5:
            return 0  # Max level reached
        
        return thresholds[current_level - 1] - current_points

if __name__ == "__main__":
    # Test the analytics engine
    analytics = RevenueAnalytics()
    
    print("🏨 Grand Millennium Revenue Analytics - Core Engine Test")
    print("=" * 60)
    
    # Test daily stats
    daily_stats = analytics.get_daily_occupancy_stats()
    print(f"📊 Daily Occupancy: {daily_stats.get('occupancy_pct', 0):.1f}%")
    print(f"💰 RevPAR: {analytics.format_aed_currency(daily_stats.get('revpar_aed', 0))}")
    
    # Test challenge scoring
    score = analytics.calculate_daily_challenge_score()
    print(f"🎮 Daily Challenge Score: {score:,} points")
    
    # Test segment data
    segments = analytics.get_segment_performance()
    print(f"🎯 Top Segment: {segments[0]['segment'] if segments else 'N/A'}")
    
    print("\n✅ Core analytics engine ready for mobile game integration!")