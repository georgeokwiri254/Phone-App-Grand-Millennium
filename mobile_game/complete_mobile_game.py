"""
Grand Millennium Revenue Analytics - Complete Mobile Game

Final integration of all game components with complete flow management,
level progression, and seamless navigation between all features.
"""

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, NoTransition, SlideTransition
from kivy.core.window import Window
from kivy.clock import Clock

# Import all game screens
from complete_game_screens import WelcomeScreen, EnhancedMainMenuScreen, LevelUpCelebrationScreen
from main_game import DashboardScreen, DailyChallengeScreen
from game_screens import SegmentGameScreen, PricingGameScreen

# Import flow management
from game_flow_manager import get_game_flow_manager, GameFlowState
from data_integration import get_data_integration_manager

# Set mobile window size for testing
Window.size = (360, 640)

class CompleteMobileGameApp(App):
    """Complete mobile game application with full feature set"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "Grand Millennium Revenue Analytics"
        
        # Initialize managers
        self.flow_manager = get_game_flow_manager()
        self.data_manager = get_data_integration_manager()
        
        # Game state tracking
        self.current_player_level = 1
        self.session_active = False
    
    def build(self):
        """Build the complete mobile game"""
        # Initialize data integration
        self.initialize_data()
        
        # Create screen manager
        sm = ScreenManager(transition=SlideTransition(direction='left'))
        
        # Add all screens
        self.add_all_screens(sm)
        
        # Determine starting screen
        starting_screen = self.determine_starting_screen()
        sm.current = starting_screen
        
        # Schedule periodic updates
        Clock.schedule_interval(self.update_game_state, 10)  # Every 10 seconds
        
        return sm
    
    def initialize_data(self):
        """Initialize data integration and sync"""
        print("🔄 Initializing Grand Millennium Mobile Game...")
        
        # Sync with existing Streamlit data
        sync_results = self.data_manager.sync_with_streamlit_data()
        
        successful_syncs = sum(1 for success in sync_results.values() if success)
        print(f"📊 Data sync: {successful_syncs}/{len(sync_results)} sources successful")
        
        # Clear analytics cache to ensure fresh data
        from enhanced_analytics_engine import get_analytics_engine
        analytics = get_analytics_engine()
        analytics.clear_cache()
    
    def add_all_screens(self, screen_manager):
        """Add all game screens to screen manager"""
        
        # Core flow screens
        screen_manager.add_widget(WelcomeScreen())
        screen_manager.add_widget(EnhancedMainMenuScreen())
        screen_manager.add_widget(LevelUpCelebrationScreen())
        
        # Game feature screens
        screen_manager.add_widget(DashboardScreen())
        screen_manager.add_widget(DailyChallengeScreen())
        screen_manager.add_widget(SegmentGameScreen())
        screen_manager.add_widget(PricingGameScreen())
        
        # Additional screens
        screen_manager.add_widget(LeaderboardScreen())
        screen_manager.add_widget(SettingsScreen())
        screen_manager.add_widget(HelpScreen())
        
        print(f"📱 Added {len(screen_manager.screens)} game screens")
    
    def determine_starting_screen(self):
        """Determine which screen to start with"""
        session_data = self.flow_manager.start_game_session()
        
        if session_data.get('is_first_time'):
            # First time player - start with tutorial
            welcome_screen = None
            for screen in self.root.screens:
                if screen.name == 'welcome':
                    welcome_screen = screen
                    break
            
            if welcome_screen and 'tutorial_steps' in session_data:
                welcome_screen.start_tutorial(session_data['tutorial_steps'])
            
            return 'welcome'
        
        elif session_data.get('show_bonus_popup'):
            # Returning player with daily bonus
            self.schedule_daily_bonus_popup(session_data['daily_bonus'])
            return 'enhanced_main_menu'
        
        else:
            # Regular returning player
            return 'enhanced_main_menu'
    
    def schedule_daily_bonus_popup(self, bonus_data):
        """Schedule daily bonus popup after screen loads"""
        def show_bonus(dt):
            self.show_daily_bonus_popup(bonus_data)
        
        Clock.schedule_once(show_bonus, 1.0)  # Show after 1 second
    
    def show_daily_bonus_popup(self, bonus_data):
        """Show daily bonus popup"""
        from kivy.uix.popup import Popup
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.label import Label
        from kivy.uix.button import Button
        
        content = BoxLayout(orientation='vertical', spacing=20, padding=20)
        
        # Welcome back message
        welcome_label = Label(
            text="Welcome Back!",
            font_size='20sp',
            bold=True,
            color=(1, 0.84, 0, 1),  # Gold
            size_hint_y=0.3
        )
        content.add_widget(welcome_label)
        
        # Bonus info
        bonus_text = (f"🎁 Daily Login Bonus\n"
                     f"Earned: {bonus_data['bonus_awarded']:,} points\n"
                     f"🔥 Streak: {bonus_data['streak']} days")
        
        bonus_label = Label(
            text=bonus_text,
            font_size='16sp',
            halign='center',
            color=(0.9, 0.9, 0.9, 1),
            size_hint_y=0.5
        )
        content.add_widget(bonus_label)
        
        # Level up notification
        if bonus_data.get('level_up'):
            level_label = Label(
                text="🎊 LEVEL UP! 🎊",
                font_size='18sp',
                bold=True,
                color=(0, 1, 0, 1),
                size_hint_y=0.2
            )
            content.add_widget(level_label)
        
        # Continue button
        continue_btn = Button(text="Continue", size_hint_y=0.2)
        content.add_widget(continue_btn)
        
        popup = Popup(
            title="Daily Bonus",
            content=content,
            size_hint=(0.8, 0.6),
            auto_dismiss=False
        )
        
        continue_btn.bind(on_press=popup.dismiss)
        popup.open()
    
    def update_game_state(self, dt):
        """Periodic game state updates"""
        if not self.session_active:
            self.session_active = True
        
        # Check for level changes
        current_level = self.flow_manager.game_state.state['level']
        if current_level > self.current_player_level:
            self.handle_level_up(current_level)
            self.current_player_level = current_level
        
        # Refresh current screen if it has a refresh method
        current_screen = self.root.current_screen
        if hasattr(current_screen, 'refresh_data'):
            current_screen.refresh_data(dt)
    
    def handle_level_up(self, new_level):
        """Handle level up during gameplay"""
        # Navigate to level up celebration screen
        level_config = self.flow_manager.level_config.get(new_level, {})
        
        # Get level up screen
        level_up_screen = None
        for screen in self.root.screens:
            if screen.name == 'level_up_celebration':
                level_up_screen = screen
                break
        
        if level_up_screen:
            # Prepare level up data
            level_data = {
                'new_level': new_level,
                'level_config': level_config,
                'new_unlocks': level_config.get('unlocks', []),
                'level_bonus': self.flow_manager.calculate_level_bonus(new_level)
            }
            
            level_up_screen.show_level_up(level_data)
            self.root.current = 'level_up_celebration'
    
    def on_stop(self):
        """Handle app shutdown"""
        if self.session_active:
            # End game session
            session_summary = self.flow_manager.end_game_session()
            print(f"📱 Session ended: {session_summary['session_summary']['challenges_completed']} challenges completed")
        
        print("👋 Grand Millennium Mobile Game closed")

# Additional screen classes for completeness
class LeaderboardScreen(DashboardScreen):
    """Leaderboard screen (placeholder)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'leaderboard'
    
    def build_ui(self):
        """Build leaderboard UI"""
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.label import Label
        
        main_layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
        header = self.create_header("Leaderboards", "Coming Soon!")
        main_layout.add_widget(header)
        
        content = Label(
            text="🏆 Leaderboards will be available\nwhen you reach Level 5!\n\nKeep completing challenges\nto unlock this feature.",
            font_size='16sp',
            halign='center',
            color=(0.8, 0.8, 0.8, 1)
        )
        main_layout.add_widget(content)
        
        # Back button
        back_btn = self.create_game_button("← Back to Menu", lambda x: setattr(self.manager, 'current', 'enhanced_main_menu'))
        main_layout.add_widget(back_btn)
        
        self.add_widget(main_layout)

class SettingsScreen(DashboardScreen):
    """Settings screen (placeholder)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'settings'
    
    def build_ui(self):
        """Build settings UI"""
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.label import Label
        from kivy.uix.switch import Switch
        
        main_layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
        header = self.create_header("Settings", "Customize your experience")
        main_layout.add_widget(header)
        
        # Settings options
        settings_layout = BoxLayout(orientation='vertical', spacing=10, size_hint_y=0.6)
        
        # Sound setting
        sound_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)
        sound_label = Label(text="🔊 Sound Effects", font_size='16sp', size_hint_x=0.7)
        sound_switch = Switch(active=True, size_hint_x=0.3)
        sound_layout.add_widget(sound_label)
        sound_layout.add_widget(sound_switch)
        settings_layout.add_widget(sound_layout)
        
        # Notifications setting
        notif_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)
        notif_label = Label(text="🔔 Daily Reminders", font_size='16sp', size_hint_x=0.7)
        notif_switch = Switch(active=True, size_hint_x=0.3)
        notif_layout.add_widget(notif_label)
        notif_layout.add_widget(notif_switch)
        settings_layout.add_widget(notif_layout)
        
        # Currency format
        currency_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)
        currency_label = Label(text="💰 Currency: AED (د.إ)", font_size='16sp', size_hint_x=1.0)
        currency_layout.add_widget(currency_label)
        settings_layout.add_widget(currency_layout)
        
        main_layout.add_widget(settings_layout)
        
        # Back button
        back_btn = self.create_game_button("← Back to Menu", lambda x: setattr(self.manager, 'current', 'enhanced_main_menu'))
        main_layout.add_widget(back_btn)
        
        self.add_widget(main_layout)

class HelpScreen(DashboardScreen):
    """Help screen with game instructions"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'help'
    
    def build_ui(self):
        """Build help UI"""
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.label import Label
        from kivy.uix.scrollview import ScrollView
        
        main_layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
        header = self.create_header("Help & Guide", "Learn to play")
        main_layout.add_widget(header)
        
        # Scrollable help content
        scroll = ScrollView(size_hint_y=0.7)
        help_layout = BoxLayout(orientation='vertical', spacing=15, size_hint_y=None)
        help_layout.bind(minimum_height=help_layout.setter('height'))
        
        help_text = """🎮 How to Play

📊 Dashboard
Monitor your hotel's performance with real-time KPIs in AED currency.

🎯 Daily Challenges
Complete challenges to earn points:
• 1 AED revenue = 1 game point
• Meet occupancy targets
• Optimize pricing strategies

👥 Segment Analysis
Allocate marketing budget across customer segments to maximize revenue.

💎 Pricing Master
Set optimal room rates considering market conditions and demand.

🏆 Level System
• Level 1: Trainee Manager
• Level 2: Assistant Manager  
• Level 3: Revenue Manager
• Level 4: Director of Revenue
• Level 5: Revenue Strategist

💰 AED Currency
All amounts displayed in Arab Emirates Dirham (د.إ).

🔥 Daily Streak
Login daily to maintain your streak and earn bonus points!"""
        
        help_label = Label(
            text=help_text,
            font_size='14sp',
            halign='left',
            valign='top',
            color=(0.9, 0.9, 0.9, 1),
            text_size=(None, None)
        )
        help_layout.add_widget(help_label)
        
        scroll.add_widget(help_layout)
        main_layout.add_widget(scroll)
        
        # Back button
        back_btn = self.create_game_button("← Back to Menu", lambda x: setattr(self.manager, 'current', 'enhanced_main_menu'))
        main_layout.add_widget(back_btn)
        
        self.add_widget(main_layout)

def main():
    """Run the complete mobile game"""
    try:
        print("🎮 Starting Grand Millennium Revenue Analytics Mobile Game")
        print("=" * 60)
        
        # Initialize and run app
        app = CompleteMobileGameApp()
        app.run()
        
    except Exception as e:
        print(f"❌ Error starting mobile game: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()