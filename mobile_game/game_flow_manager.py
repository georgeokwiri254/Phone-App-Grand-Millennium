"""
Grand Millennium Revenue Analytics - Game Flow Manager

Manages the complete game flow, level progression, feature unlocking,
and seamless transitions between different game modes and challenges.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

from game_state import get_game_state
from enhanced_analytics_engine import get_analytics_engine
from aed_currency_handler import AEDCurrencyHandler

class GameFlowState(Enum):
    """Game flow states"""
    WELCOME = "welcome"
    MAIN_MENU = "main_menu"
    DASHBOARD = "dashboard"
    CHALLENGE_SELECT = "challenge_select"
    IN_CHALLENGE = "in_challenge"
    CHALLENGE_COMPLETE = "challenge_complete"
    LEVEL_UP = "level_up"
    ACHIEVEMENT_UNLOCK = "achievement_unlock"
    FEATURE_UNLOCK = "feature_unlock"
    LEADERBOARD = "leaderboard"
    SETTINGS = "settings"
    TUTORIAL = "tutorial"

class GameMode(Enum):
    """Different game modes available"""
    DAILY_CHALLENGE = "daily_challenge"
    SEGMENT_CONQUEST = "segment_conquest"
    PRICING_MASTER = "pricing_master"
    BLOCK_BOOKING = "block_booking"
    FORECAST_PROPHET = "forecast_prophet"
    ANALYTICS_ACADEMY = "analytics_academy"
    FREE_PLAY = "free_play"

class GameFlowManager:
    """Manages complete game flow and progression"""
    
    def __init__(self, player_id: str = "default_player"):
        """Initialize game flow manager"""
        self.player_id = player_id
        self.game_state = get_game_state(player_id)
        self.analytics = get_analytics_engine()
        self.aed_handler = AEDCurrencyHandler()
        
        # Current flow state
        self.current_flow_state = GameFlowState.WELCOME
        self.current_game_mode = None
        self.flow_history = []
        
        # Level progression configuration
        self.level_config = self.setup_level_progression()
        
        # Feature unlock configuration
        self.feature_unlocks = self.setup_feature_unlocks()
        
        # Tutorial configuration
        self.tutorial_steps = self.setup_tutorial_steps()
        
        # Session tracking
        self.session_start = datetime.now()
        self.session_stats = {
            'challenges_attempted': 0,
            'challenges_completed': 0,
            'points_earned': 0,
            'screens_visited': [],
            'time_spent': {}
        }
    
    def setup_level_progression(self) -> Dict[int, Dict[str, Any]]:
        """Setup level progression configuration"""
        return {
            1: {
                'name': 'Trainee Manager',
                'points_required': 0,
                'points_to_next': 10000,
                'description': 'Learning the basics of hotel revenue management',
                'rewards': ['Basic Dashboard Access', 'Daily Occupancy Challenge'],
                'badge': '🎓',
                'color': (0.5, 0.7, 1.0, 1),
                'unlocks': ['Dashboard', 'Daily Challenges']
            },
            2: {
                'name': 'Assistant Manager',
                'points_required': 10000,
                'points_to_next': 40000,
                'description': 'Understanding customer segments and market dynamics',
                'rewards': ['Segment Analysis', 'Weekly Missions', 'د.إ 5,000 Bonus'],
                'badge': '📊',
                'color': (0.0, 0.8, 0.0, 1),
                'unlocks': ['Segment Analysis', 'Customer Conquest', 'Weekly Missions']
            },
            3: {
                'name': 'Revenue Manager',
                'points_required': 50000,
                'points_to_next': 150000,
                'description': 'Mastering pricing strategies and revenue optimization',
                'rewards': ['ADR Optimization', 'Forecasting Games', 'د.إ 15,000 Bonus'],
                'badge': '💎',
                'color': (1.0, 0.0, 0.8, 1),
                'unlocks': ['ADR Optimization', 'Pricing Master', 'Forecasting Games']
            },
            4: {
                'name': 'Director of Revenue',
                'points_required': 200000,
                'points_to_next': 300000,
                'description': 'Advanced analytics and strategic decision making',
                'rewards': ['Block Analysis', 'ML Analytics', 'د.إ 30,000 Bonus'],
                'badge': '🏆',
                'color': (1.0, 0.84, 0.0, 1),
                'unlocks': ['Block Analysis', 'Group Booking Simulator', 'Machine Learning', 'Advanced Analytics']
            },
            5: {
                'name': 'Revenue Strategist',
                'points_required': 500000,
                'points_to_next': 0,  # Max level
                'description': 'Elite revenue management expert with full access',
                'rewards': ['All Features', 'Leaderboards', 'Exclusive Content', 'د.إ 50,000 Bonus'],
                'badge': '👑',
                'color': (0.8, 0.0, 1.0, 1),
                'unlocks': ['All Features', 'Leaderboards', 'AI Assistant', 'Exclusive Content']
            }
        }
    
    def setup_feature_unlocks(self) -> Dict[str, Dict[str, Any]]:
        """Setup feature unlock configuration"""
        return {
            'Dashboard': {
                'required_level': 1,
                'description': 'Monitor hotel performance in real-time',
                'icon': '📊',
                'tutorial': True
            },
            'Daily Challenges': {
                'required_level': 1,
                'description': 'Complete daily revenue targets',
                'icon': '🎯',
                'tutorial': True
            },
            'Segment Analysis': {
                'required_level': 2,
                'description': 'Analyze customer segments and optimize marketing',
                'icon': '👥',
                'tutorial': True,
                'unlock_message': 'Customer segments unlocked! Optimize your marketing strategy.'
            },
            'Customer Conquest': {
                'required_level': 2,
                'description': 'Strategic customer acquisition game',
                'icon': '🎮',
                'tutorial': False
            },
            'ADR Optimization': {
                'required_level': 3,
                'description': 'Master room rate pricing strategies',
                'icon': '💎',
                'tutorial': True,
                'unlock_message': 'Pricing mastery unlocked! Optimize your ADR for maximum RevPAR.'
            },
            'Pricing Master': {
                'required_level': 3,
                'description': 'Advanced pricing optimization game',
                'icon': '🎲',
                'tutorial': False
            },
            'Forecasting Games': {
                'required_level': 3,
                'description': 'Predict future revenue performance',
                'icon': '🔮',
                'tutorial': True
            },
            'Block Analysis': {
                'required_level': 4,
                'description': 'Analyze and optimize group bookings',
                'icon': '🏢',
                'tutorial': True,
                'unlock_message': 'Group booking mastery unlocked! Maximize revenue from large bookings.'
            },
            'Machine Learning': {
                'required_level': 4,
                'description': 'Advanced analytics and prediction models',
                'icon': '🤖',
                'tutorial': False
            },
            'Leaderboards': {
                'required_level': 5,
                'description': 'Compete with other revenue managers',
                'icon': '🏆',
                'tutorial': False,
                'unlock_message': 'Elite status achieved! You can now compete on the leaderboards.'
            },
            'AI Assistant': {
                'required_level': 5,
                'description': 'Advanced AI-powered insights',
                'icon': '🧠',
                'tutorial': False
            }
        }
    
    def setup_tutorial_steps(self) -> Dict[str, List[Dict[str, str]]]:
        """Setup tutorial steps for each feature"""
        return {
            'first_time': [
                {
                    'title': 'Welcome to Grand Millennium Revenue Analytics!',
                    'content': 'Transform your hotel revenue management skills into an engaging mobile game experience.',
                    'action': 'tap_to_continue'
                },
                {
                    'title': 'AED Currency System',
                    'content': 'All revenue, rates, and rewards are displayed in Arab Emirates Dirham (د.إ). Earn points by optimizing your hotel\'s performance!',
                    'action': 'tap_to_continue'
                },
                {
                    'title': 'Level Up Your Skills',
                    'content': 'Complete challenges to earn points, unlock new features, and advance from Trainee Manager to Revenue Strategist.',
                    'action': 'tap_to_continue'
                },
                {
                    'title': 'Ready to Start?',
                    'content': 'Let\'s begin with the Dashboard to see your hotel\'s current performance!',
                    'action': 'start_game'
                }
            ],
            'Dashboard': [
                {
                    'title': 'Mission Control Center',
                    'content': 'This is your revenue command center. Monitor key metrics like occupancy, ADR, and RevPAR in real-time.',
                    'highlight': 'kpi_cards'
                },
                {
                    'title': 'AED Performance Metrics',
                    'content': 'All revenue figures are shown in AED. Higher RevPAR means better revenue optimization!',
                    'highlight': 'revpar_card'
                },
                {
                    'title': 'Daily Challenges',
                    'content': 'Complete daily challenges to earn points and improve your revenue management skills.',
                    'highlight': 'challenge_card'
                }
            ],
            'Daily Challenges': [
                {
                    'title': 'Revenue Challenges',
                    'content': 'Each challenge tests different aspects of revenue management. Complete them to earn AED-based points!',
                    'highlight': 'challenge_list'
                },
                {
                    'title': 'Scoring System',
                    'content': 'Your score is based on actual AED revenue performance. 1 AED in revenue = 1 game point!',
                    'highlight': 'scoring_info'
                }
            ],
            'Segment Analysis': [
                {
                    'title': 'Customer Segments',
                    'content': 'Different customer types have different value. Corporate guests typically have higher ADR than leisure guests.',
                    'highlight': 'segment_cards'
                },
                {
                    'title': 'Budget Allocation',
                    'content': 'Allocate your marketing budget across segments to maximize revenue. Use sliders to adjust allocation.',
                    'highlight': 'budget_sliders'
                }
            ],
            'ADR Optimization': [
                {
                    'title': 'Pricing Strategy',
                    'content': 'Balance room rates with occupancy to maximize RevPAR. Higher rates may reduce occupancy, but can increase total revenue.',
                    'highlight': 'pricing_slider'
                },
                {
                    'title': 'Market Conditions',
                    'content': 'Consider current market demand when setting rates. Special events allow for premium pricing!',
                    'highlight': 'market_display'
                }
            ]
        }
    
    def start_game_session(self) -> Dict[str, Any]:
        """Start a new game session"""
        self.session_start = datetime.now()
        
        # Check if this is the first time playing
        stats = self.game_state.get_player_stats()
        is_first_time = stats['total_points'] == 0 and not self.game_state.state.get('tutorial_completed', False)
        
        if is_first_time:
            self.current_flow_state = GameFlowState.TUTORIAL
            return {
                'flow_state': self.current_flow_state.value,
                'next_screen': 'tutorial',
                'tutorial_steps': self.tutorial_steps['first_time'],
                'is_first_time': True
            }
        else:
            # Check for daily bonus
            bonus = self.game_state.get_daily_bonus()
            if bonus:
                self.current_flow_state = GameFlowState.MAIN_MENU
                return {
                    'flow_state': self.current_flow_state.value,
                    'next_screen': 'main_menu',
                    'daily_bonus': bonus,
                    'show_bonus_popup': True
                }
            else:
                self.current_flow_state = GameFlowState.MAIN_MENU
                return {
                    'flow_state': self.current_flow_state.value,
                    'next_screen': 'main_menu',
                    'returning_player': True
                }
    
    def complete_tutorial(self) -> Dict[str, Any]:
        """Mark tutorial as completed"""
        self.game_state.state['tutorial_completed'] = True
        self.game_state.save_game_state()
        
        # Award tutorial completion bonus
        tutorial_bonus = 2500
        level_up = self.game_state.add_points(tutorial_bonus, "tutorial_completion")
        
        self.current_flow_state = GameFlowState.MAIN_MENU
        
        return {
            'flow_state': self.current_flow_state.value,
            'next_screen': 'main_menu',
            'tutorial_complete': True,
            'bonus_points': tutorial_bonus,
            'level_up': level_up
        }
    
    def navigate_to_screen(self, target_screen: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Navigate to a specific screen with flow management"""
        self.flow_history.append({
            'from_state': self.current_flow_state.value,
            'to_screen': target_screen,
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        })
        
        # Track screen visits
        if target_screen not in self.session_stats['screens_visited']:
            self.session_stats['screens_visited'].append(target_screen)
        
        # Update flow state based on target screen
        flow_mapping = {
            'main_menu': GameFlowState.MAIN_MENU,
            'dashboard': GameFlowState.DASHBOARD,
            'daily_challenge': GameFlowState.CHALLENGE_SELECT,
            'segment_game': GameFlowState.IN_CHALLENGE,
            'pricing_game': GameFlowState.IN_CHALLENGE,
            'block_game': GameFlowState.IN_CHALLENGE,
            'leaderboard': GameFlowState.LEADERBOARD,
            'settings': GameFlowState.SETTINGS
        }
        
        self.current_flow_state = flow_mapping.get(target_screen, GameFlowState.MAIN_MENU)
        
        # Check if feature is unlocked
        unlocked_features = self.game_state.state['unlocked_features']
        feature_required = self.get_required_feature_for_screen(target_screen)
        
        if feature_required and feature_required not in unlocked_features:
            return self.handle_locked_feature(feature_required, target_screen)
        
        # Get tutorial if needed
        tutorial_data = self.get_tutorial_for_screen(target_screen)
        
        return {
            'flow_state': self.current_flow_state.value,
            'next_screen': target_screen,
            'context': context or {},
            'tutorial': tutorial_data,
            'navigation_success': True
        }
    
    def get_required_feature_for_screen(self, screen: str) -> Optional[str]:
        """Get the feature required to access a screen"""
        screen_feature_map = {
            'dashboard': 'Dashboard',
            'daily_challenge': 'Daily Challenges',
            'segment_game': 'Segment Analysis',
            'pricing_game': 'ADR Optimization',
            'block_game': 'Block Analysis',
            'leaderboard': 'Leaderboards'
        }
        return screen_feature_map.get(screen)
    
    def handle_locked_feature(self, feature: str, target_screen: str) -> Dict[str, Any]:
        """Handle attempt to access locked feature"""
        feature_config = self.feature_unlocks.get(feature, {})
        required_level = feature_config.get('required_level', 1)
        current_level = self.game_state.state['level']
        
        return {
            'flow_state': self.current_flow_state.value,
            'next_screen': 'main_menu',  # Stay on main menu
            'feature_locked': True,
            'locked_feature': feature,
            'required_level': required_level,
            'current_level': current_level,
            'unlock_message': f"Reach level {required_level} to unlock {feature}!"
        }
    
    def get_tutorial_for_screen(self, screen: str) -> Optional[Dict[str, Any]]:
        """Get tutorial data for screen if needed"""
        tutorial_key = {
            'dashboard': 'Dashboard',
            'daily_challenge': 'Daily Challenges',
            'segment_game': 'Segment Analysis',
            'pricing_game': 'ADR Optimization'
        }.get(screen)
        
        if not tutorial_key:
            return None
        
        # Check if player has seen this tutorial
        tutorials_seen = self.game_state.state.get('tutorials_seen', [])
        if tutorial_key in tutorials_seen:
            return None
        
        # Check if feature has tutorial
        feature_config = self.feature_unlocks.get(tutorial_key, {})
        if not feature_config.get('tutorial', False):
            return None
        
        return {
            'tutorial_key': tutorial_key,
            'steps': self.tutorial_steps.get(tutorial_key, []),
            'skippable': True
        }
    
    def complete_tutorial_for_feature(self, feature: str) -> Dict[str, Any]:
        """Mark feature tutorial as completed"""
        tutorials_seen = self.game_state.state.get('tutorials_seen', [])
        if feature not in tutorials_seen:
            tutorials_seen.append(feature)
            self.game_state.state['tutorials_seen'] = tutorials_seen
            self.game_state.save_game_state()
            
            # Award tutorial points
            tutorial_points = 1000
            level_up = self.game_state.add_points(tutorial_points, f"tutorial_{feature.lower()}")
            
            return {
                'tutorial_completed': True,
                'feature': feature,
                'points_awarded': tutorial_points,
                'level_up': level_up
            }
        
        return {'tutorial_completed': False}
    
    def start_challenge(self, challenge_id: str, game_mode: GameMode) -> Dict[str, Any]:
        """Start a specific challenge"""
        self.current_game_mode = game_mode
        self.current_flow_state = GameFlowState.IN_CHALLENGE
        self.session_stats['challenges_attempted'] += 1
        
        # Get challenge configuration
        challenges = self.game_state.get_available_challenges()
        challenge = next((c for c in challenges if c['id'] == challenge_id), None)
        
        if not challenge:
            return {
                'error': 'Challenge not found',
                'flow_state': self.current_flow_state.value
            }
        
        # Get current performance data for challenge
        kpis = self.analytics.get_real_time_kpis()
        
        challenge_data = {
            'challenge': challenge,
            'current_performance': kpis,
            'game_mode': game_mode.value,
            'flow_state': self.current_flow_state.value,
            'start_time': datetime.now().isoformat()
        }
        
        # Add mode-specific data
        if game_mode == GameMode.SEGMENT_CONQUEST:
            challenge_data['segments'] = self.analytics.get_segment_performance_advanced()
        elif game_mode == GameMode.PRICING_MASTER:
            challenge_data['market_conditions'] = self.analytics.get_current_market_conditions()
        
        return challenge_data
    
    def complete_challenge(self, challenge_id: str, score: int, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Complete a challenge and handle rewards"""
        self.session_stats['challenges_completed'] += 1
        self.session_stats['points_earned'] += score
        
        # Complete the challenge in game state
        result = self.game_state.complete_daily_challenge(challenge_id, score)
        
        if not result:
            return {
                'already_completed': True,
                'flow_state': GameFlowState.MAIN_MENU.value,
                'next_screen': 'main_menu'
            }
        
        # Check for level up
        if result['level_up']:
            self.current_flow_state = GameFlowState.LEVEL_UP
            return self.handle_level_up(result['new_level'], result)
        
        # Check for achievements
        achievement_unlocked = self.check_for_achievements(challenge_id, score, context)
        
        if achievement_unlocked:
            self.current_flow_state = GameFlowState.ACHIEVEMENT_UNLOCK
            return {
                'flow_state': self.current_flow_state.value,
                'challenge_result': result,
                'achievement': achievement_unlocked,
                'next_screen': 'achievement_popup'
            }
        
        # Regular completion
        self.current_flow_state = GameFlowState.CHALLENGE_COMPLETE
        return {
            'flow_state': self.current_flow_state.value,
            'challenge_result': result,
            'next_screen': 'challenge_complete',
            'celebration': self.get_celebration_level(score)
        }
    
    def handle_level_up(self, new_level: int, challenge_result: Dict) -> Dict[str, Any]:
        """Handle level up process"""
        level_config = self.level_config.get(new_level, {})
        
        # Check for new feature unlocks
        new_unlocks = []
        for feature in level_config.get('unlocks', []):
            if feature not in self.game_state.state['unlocked_features']:
                self.game_state.state['unlocked_features'].append(feature)
                new_unlocks.append(feature)
        
        self.game_state.save_game_state()
        
        # Award level up bonus
        level_bonus = self.calculate_level_bonus(new_level)
        if level_bonus > 0:
            self.game_state.add_points(level_bonus, f"level_{new_level}_bonus")
        
        return {
            'flow_state': GameFlowState.LEVEL_UP.value,
            'new_level': new_level,
            'level_config': level_config,
            'new_unlocks': new_unlocks,
            'level_bonus': level_bonus,
            'challenge_result': challenge_result,
            'next_screen': 'level_up_celebration'
        }
    
    def calculate_level_bonus(self, level: int) -> int:
        """Calculate bonus points for level up"""
        level_bonuses = {
            2: 5000,   # د.إ 5,000 equivalent
            3: 15000,  # د.إ 15,000 equivalent
            4: 30000,  # د.إ 30,000 equivalent
            5: 50000   # د.إ 50,000 equivalent
        }
        return level_bonuses.get(level, 0)
    
    def check_for_achievements(self, challenge_id: str, score: int, context: Optional[Dict]) -> Optional[Dict[str, Any]]:
        """Check if any achievements were unlocked"""
        achievements = self.game_state.get_achievements()
        current_achievements = self.game_state.state.get('achievements', [])
        
        # Check achievement conditions
        for achievement in achievements:
            if achievement['name'] in current_achievements:
                continue
            
            unlocked = False
            
            # Score-based achievements
            if achievement['metric'] == 'points' and score >= achievement['target']:
                unlocked = True
            
            # Challenge completion achievements
            elif achievement['metric'] == 'challenges_completed':
                if self.session_stats['challenges_completed'] >= achievement['target']:
                    unlocked = True
            
            # Revenue-based achievements
            elif achievement['metric'] == 'monthly_revenue' and context:
                monthly_revenue = context.get('monthly_revenue', 0)
                if monthly_revenue >= achievement['target']:
                    unlocked = True
            
            if unlocked:
                self.game_state.unlock_achievement(achievement['name'])
                return {
                    'achievement': achievement,
                    'unlocked': True,
                    'points_bonus': achievement.get('reward', 0)
                }
        
        return None
    
    def get_celebration_level(self, score: int) -> str:
        """Get celebration level based on score"""
        if score >= 50000:
            return 'amazing'
        elif score >= 25000:
            return 'excellent'
        elif score >= 10000:
            return 'good'
        else:
            return 'standard'
    
    def get_game_flow_status(self) -> Dict[str, Any]:
        """Get current game flow status"""
        stats = self.game_state.get_player_stats()
        current_level_config = self.level_config.get(stats['level'], {})
        
        # Calculate session time
        session_duration = (datetime.now() - self.session_start).seconds
        
        return {
            'current_flow_state': self.current_flow_state.value,
            'current_game_mode': self.current_game_mode.value if self.current_game_mode else None,
            'player_stats': stats,
            'current_level_config': current_level_config,
            'session_stats': {
                **self.session_stats,
                'session_duration_seconds': session_duration,
                'session_duration_formatted': self.format_duration(session_duration)
            },
            'available_features': self.get_available_features(),
            'next_unlock': self.get_next_unlock_info(stats['level']),
            'flow_history': self.flow_history[-5:],  # Last 5 navigation events
        }
    
    def get_available_features(self) -> List[Dict[str, Any]]:
        """Get list of available features for current player level"""
        unlocked_features = self.game_state.state['unlocked_features']
        current_level = self.game_state.state['level']
        
        available = []
        for feature, config in self.feature_unlocks.items():
            if feature in unlocked_features:
                available.append({
                    'name': feature,
                    'config': config,
                    'status': 'unlocked'
                })
            elif config['required_level'] == current_level + 1:
                available.append({
                    'name': feature,
                    'config': config,
                    'status': 'next_level'
                })
        
        return available
    
    def get_next_unlock_info(self, current_level: int) -> Optional[Dict[str, Any]]:
        """Get information about next unlock"""
        next_level = current_level + 1
        if next_level not in self.level_config:
            return None
        
        next_level_config = self.level_config[next_level]
        current_points = self.game_state.state['total_points']
        points_needed = next_level_config['points_required'] - current_points
        
        return {
            'next_level': next_level,
            'level_name': next_level_config['name'],
            'points_needed': max(0, points_needed),
            'unlocks': next_level_config['unlocks'],
            'rewards': next_level_config['rewards']
        }
    
    def format_duration(self, seconds: int) -> str:
        """Format duration in seconds to readable format"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes}m {seconds % 60}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    def end_game_session(self) -> Dict[str, Any]:
        """End current game session and provide summary"""
        session_duration = (datetime.now() - self.session_start).seconds
        
        # Save session stats
        session_summary = {
            'session_duration': session_duration,
            'challenges_attempted': self.session_stats['challenges_attempted'],
            'challenges_completed': self.session_stats['challenges_completed'],
            'points_earned': self.session_stats['points_earned'],
            'screens_visited': len(self.session_stats['screens_visited']),
            'final_level': self.game_state.state['level'],
            'session_end': datetime.now().isoformat()
        }
        
        # Update lifetime stats
        lifetime_stats = self.game_state.state.get('lifetime_stats', {})
        lifetime_stats['total_sessions'] = lifetime_stats.get('total_sessions', 0) + 1
        lifetime_stats['total_play_time'] = lifetime_stats.get('total_play_time', 0) + session_duration
        lifetime_stats['total_challenges'] = lifetime_stats.get('total_challenges', 0) + self.session_stats['challenges_completed']
        
        self.game_state.state['lifetime_stats'] = lifetime_stats
        self.game_state.save_game_state()
        
        return {
            'session_summary': session_summary,
            'lifetime_stats': lifetime_stats,
            'comeback_message': self.get_comeback_message()
        }
    
    def get_comeback_message(self) -> str:
        """Get personalized comeback message"""
        level = self.game_state.state['level']
        challenges_completed = self.session_stats['challenges_completed']
        
        if challenges_completed == 0:
            return "Come back tomorrow for fresh daily challenges!"
        elif level < 3:
            return "Keep building your revenue management skills! More challenges await."
        elif level < 5:
            return "You're becoming a revenue expert! Continue your journey tomorrow."
        else:
            return "Elite status achieved! Return to maintain your leadership position."

# Global game flow manager
_game_flow_manager = None

def get_game_flow_manager(player_id: str = "default_player") -> GameFlowManager:
    """Get or create global game flow manager instance"""
    global _game_flow_manager
    if _game_flow_manager is None or _game_flow_manager.player_id != player_id:
        _game_flow_manager = GameFlowManager(player_id)
    return _game_flow_manager

if __name__ == "__main__":
    # Test game flow manager
    print("🎮 Game Flow Manager Test")
    print("=" * 35)
    
    flow_manager = GameFlowManager("test_flow_player")
    
    # Test session start
    print("🚀 Starting game session...")
    session_start = flow_manager.start_game_session()
    print(f"   Flow state: {session_start['flow_state']}")
    print(f"   Next screen: {session_start['next_screen']}")
    
    if session_start.get('is_first_time'):
        print("   First time player - tutorial required")
        
        # Complete tutorial
        tutorial_complete = flow_manager.complete_tutorial()
        print(f"   Tutorial completed: +{tutorial_complete['bonus_points']} points")
    
    # Test navigation
    print("\n📱 Testing navigation...")
    nav_result = flow_manager.navigate_to_screen('dashboard')
    print(f"   Navigation to dashboard: {'✅' if nav_result['navigation_success'] else '❌'}")
    
    # Test challenge flow
    print("\n🎯 Testing challenge flow...")
    challenge_start = flow_manager.start_challenge('daily_occupancy', GameMode.DAILY_CHALLENGE)
    print(f"   Challenge started: {challenge_start['challenge']['title']}")
    
    # Complete challenge
    test_score = 15000
    challenge_complete = flow_manager.complete_challenge('daily_occupancy', test_score)
    print(f"   Challenge completed: +{test_score} points")
    
    if challenge_complete.get('level_up'):
        print(f"   🎊 LEVEL UP! New level: {challenge_complete['new_level']}")
    
    # Test flow status
    print("\n📊 Game flow status:")
    status = flow_manager.get_game_flow_status()
    print(f"   Current level: {status['player_stats']['level']} ({status['player_stats']['level_name']})")
    print(f"   Session challenges: {status['session_stats']['challenges_completed']}")
    print(f"   Available features: {len(status['available_features'])}")
    
    # End session
    print("\n🏁 Ending game session...")
    session_end = flow_manager.end_game_session()
    print(f"   Session duration: {session_end['session_summary']['session_duration']}s")
    print(f"   Comeback message: {session_end['comeback_message']}")
    
    print("\n✅ Game Flow Manager Ready!")