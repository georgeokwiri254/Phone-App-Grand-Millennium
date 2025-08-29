"""
Grand Millennium Revenue Analytics - Game State Manager

Manages player progress, achievements, and game state persistence
for the mobile revenue analytics game.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from core_analytics import RevenueAnalytics, GameProgressManager

class GameState:
    """Manages the overall game state and player progress"""
    
    def __init__(self, player_id="default_player"):
        self.player_id = player_id
        self.analytics = RevenueAnalytics()
        self.progress_manager = GameProgressManager(player_id)
        
        # Game state file
        self.state_file = Path(__file__).parent / f"game_state_{player_id}.json"
        
        # Load existing state or create new
        self.state = self.load_game_state()
    
    def load_game_state(self):
        """Load game state from file or create default"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Default game state
        return {
            'player_id': self.player_id,
            'total_points': 0,
            'level': 1,
            'achievements': [],
            'daily_streak': 0,
            'last_login': None,
            'completed_challenges': [],
            'unlocked_features': ['Dashboard', 'Daily Challenges'],
            'settings': {
                'sound_enabled': True,
                'notifications_enabled': True,
                'currency_format': 'AED'
            },
            'statistics': {
                'games_played': 0,
                'total_revenue_managed': 0,
                'best_occupancy': 0,
                'best_revpar': 0,
                'challenges_completed': 0
            }
        }
    
    def save_game_state(self):
        """Save current game state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"Failed to save game state: {e}")
    
    def add_points(self, points, source="general"):
        """Add points to player's total and check for level ups"""
        old_level = self.state['level']
        self.state['total_points'] += points
        
        # Update level
        new_level = self.progress_manager.get_player_level(self.state['total_points'])
        self.state['level'] = new_level
        
        # Check for level up
        level_up = new_level > old_level
        if level_up:
            self.unlock_features_for_level(new_level)
        
        self.save_game_state()
        return level_up
    
    def unlock_features_for_level(self, level):
        """Unlock new features based on player level"""
        feature_unlocks = {
            1: ['Dashboard', 'Daily Challenges'],
            2: ['Segment Analysis', 'Weekly Missions', 'Customer Conquest'],
            3: ['ADR Optimization', 'Forecasting Games', 'Pricing Master'],
            4: ['Block Analysis', 'Advanced Analytics', 'Group Booking Simulator'],
            5: ['Machine Learning', 'AI Assistant', 'Leaderboards', 'All Features']
        }
        
        for unlock_level in range(1, level + 1):
            if unlock_level in feature_unlocks:
                for feature in feature_unlocks[unlock_level]:
                    if feature not in self.state['unlocked_features']:
                        self.state['unlocked_features'].append(feature)
    
    def complete_daily_challenge(self, challenge_type, score):
        """Mark daily challenge as completed and award points"""
        today = datetime.now().strftime('%Y-%m-%d')
        challenge_key = f"{challenge_type}_{today}"
        
        if challenge_key not in self.state['completed_challenges']:
            self.state['completed_challenges'].append(challenge_key)
            self.state['statistics']['challenges_completed'] += 1
            
            # Update streak
            self.update_daily_streak()
            
            # Award points with streak bonus
            streak_multiplier = min(1 + (self.state['daily_streak'] * 0.1), 3.0)
            final_score = int(score * streak_multiplier)
            
            level_up = self.add_points(final_score, f"daily_challenge_{challenge_type}")
            
            return {
                'base_score': score,
                'streak_bonus': final_score - score,
                'final_score': final_score,
                'level_up': level_up,
                'new_level': self.state['level']
            }
        
        return None  # Challenge already completed today
    
    def update_daily_streak(self):
        """Update daily login streak"""
        today = datetime.now().strftime('%Y-%m-%d')
        last_login = self.state.get('last_login')
        
        if last_login:
            last_date = datetime.strptime(last_login, '%Y-%m-%d')
            today_date = datetime.strptime(today, '%Y-%m-%d')
            days_diff = (today_date - last_date).days
            
            if days_diff == 1:
                # Consecutive day - increment streak
                self.state['daily_streak'] += 1
            elif days_diff > 1:
                # Streak broken - reset
                self.state['daily_streak'] = 1
            # Same day - no change to streak
        else:
            # First login
            self.state['daily_streak'] = 1
        
        self.state['last_login'] = today
    
    def get_available_challenges(self):
        """Get challenges available for today based on player level"""
        today = datetime.now().strftime('%Y-%m-%d')
        level = self.state['level']
        
        all_challenges = [
            {
                'id': 'daily_occupancy',
                'title': 'Daily Occupancy Challenge',
                'description': 'Achieve target occupancy percentage',
                'target': 85.0,
                'icon': '🏨',
                'points': 5000,
                'required_level': 1
            },
            {
                'id': 'revenue_target',
                'title': 'Revenue Target',
                'description': 'Beat yesterday\'s revenue performance',
                'target': 'dynamic',
                'icon': '💰',
                'points': 7500,
                'required_level': 1
            },
            {
                'id': 'segment_optimization',
                'title': 'Segment Master',
                'description': 'Optimize customer segment mix',
                'target': 'optimization',
                'icon': '🎯',
                'points': 10000,
                'required_level': 2
            },
            {
                'id': 'pricing_game',
                'title': 'Pricing Expert',
                'description': 'Set optimal room rates',
                'target': 'optimization',
                'icon': '💎',
                'points': 12500,
                'required_level': 3
            },
            {
                'id': 'block_booking',
                'title': 'Group Guru',
                'description': 'Make smart block booking decisions',
                'target': 'decision',
                'icon': '👥',
                'points': 15000,
                'required_level': 4
            }
        ]
        
        # Filter by level and completion status
        available = []
        for challenge in all_challenges:
            if (challenge['required_level'] <= level and 
                f"{challenge['id']}_{today}" not in self.state['completed_challenges']):
                available.append(challenge)
        
        return available
    
    def get_achievements(self):
        """Get player achievements"""
        return self.analytics.get_monthly_achievements(self.state['level'])
    
    def unlock_achievement(self, achievement_id):
        """Unlock a new achievement"""
        if achievement_id not in self.state['achievements']:
            self.state['achievements'].append(achievement_id)
            self.save_game_state()
            return True
        return False
    
    def is_feature_unlocked(self, feature_name):
        """Check if a feature is unlocked for the player"""
        return feature_name in self.state['unlocked_features']
    
    def get_player_stats(self):
        """Get formatted player statistics for UI"""
        level_name = self.progress_manager.get_level_name(self.state['level'])
        points_to_next = self.progress_manager.calculate_points_to_next_level(self.state['total_points'])
        
        return {
            'player_id': self.state['player_id'],
            'level': self.state['level'],
            'level_name': level_name,
            'total_points': self.state['total_points'],
            'points_to_next_level': points_to_next,
            'daily_streak': self.state['daily_streak'],
            'achievements_count': len(self.state['achievements']),
            'challenges_completed': self.state['statistics']['challenges_completed'],
            'unlocked_features': self.state['unlocked_features']
        }
    
    def get_daily_bonus(self):
        """Calculate and award daily login bonus"""
        today = datetime.now().strftime('%Y-%m-%d')
        last_login = self.state.get('last_login')
        
        if last_login != today:
            self.update_daily_streak()
            
            # Base bonus + streak bonus
            base_bonus = 1000
            streak_bonus = min(self.state['daily_streak'] * 500, 5000)
            total_bonus = base_bonus + streak_bonus
            
            level_up = self.add_points(total_bonus, "daily_bonus")
            
            return {
                'bonus_awarded': total_bonus,
                'streak': self.state['daily_streak'],
                'level_up': level_up
            }
        
        return None  # Already claimed today
    
    def reset_player_progress(self):
        """Reset player progress (for testing or new game)"""
        self.state = {
            'player_id': self.player_id,
            'total_points': 0,
            'level': 1,
            'achievements': [],
            'daily_streak': 0,
            'last_login': None,
            'completed_challenges': [],
            'unlocked_features': ['Dashboard', 'Daily Challenges'],
            'settings': {
                'sound_enabled': True,
                'notifications_enabled': True,
                'currency_format': 'AED'
            },
            'statistics': {
                'games_played': 0,
                'total_revenue_managed': 0,
                'best_occupancy': 0,
                'best_revpar': 0,
                'challenges_completed': 0
            }
        }
        self.save_game_state()

# Global game state instance
_game_state = None

def get_game_state(player_id="default_player"):
    """Get or create global game state instance"""
    global _game_state
    if _game_state is None or _game_state.player_id != player_id:
        _game_state = GameState(player_id)
    return _game_state

if __name__ == "__main__":
    # Test game state management
    print("🎮 Testing Game State Manager")
    print("=" * 40)
    
    # Create test game state
    game = GameState("test_player")
    
    # Test daily bonus
    bonus = game.get_daily_bonus()
    if bonus:
        print(f"💰 Daily Bonus: {bonus['bonus_awarded']} points (Streak: {bonus['streak']})")
    
    # Test challenge completion
    challenges = game.get_available_challenges()
    print(f"🎯 Available Challenges: {len(challenges)}")
    for challenge in challenges[:2]:
        print(f"  - {challenge['title']}: {challenge['points']} points")
    
    # Test stats
    stats = game.get_player_stats()
    print(f"📊 Player Level: {stats['level']} ({stats['level_name']})")
    print(f"🏆 Total Points: {stats['total_points']:,}")
    
    print("\n✅ Game state management ready!")