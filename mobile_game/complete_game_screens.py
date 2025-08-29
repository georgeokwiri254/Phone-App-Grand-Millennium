"""
Grand Millennium Revenue Analytics - Complete Game Screens

Complete set of game screens with flow management, level progression,
tutorials, and seamless navigation between all game features.
"""

from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.graphics import Color, Rectangle, RoundedRectangle, Ellipse
from kivy.uix.widget import Widget

from main_game import GameScreen
from game_flow_manager import get_game_flow_manager, GameMode, GameFlowState
from enhanced_analytics_engine import get_analytics_engine
from aed_currency_handler import AEDCurrencyHandler

class WelcomeScreen(GameScreen):
    """Welcome screen with tutorial for first-time players"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'welcome'
        self.flow_manager = get_game_flow_manager()
        self.tutorial_steps = []
        self.current_step = 0
        self.build_ui()
    
    def build_ui(self):
        """Build welcome/tutorial UI"""
        main_layout = BoxLayout(orientation='vertical', padding=dp(30), spacing=dp(20))
        
        # Logo and title
        header_layout = BoxLayout(orientation='vertical', size_hint_y=0.4, spacing=dp(15))
        
        logo = Label(
            text="🏨",
            font_size='64sp',
            size_hint_y=0.6
        )
        header_layout.add_widget(logo)
        
        title = Label(
            text="Grand Millennium\nRevenue Analytics",
            font_size='24sp',
            bold=True,
            color=self.gold_color,
            halign='center',
            size_hint_y=0.4
        )
        header_layout.add_widget(title)
        
        main_layout.add_widget(header_layout)
        
        # Tutorial content area
        self.tutorial_content = BoxLayout(
            orientation='vertical', 
            size_hint_y=0.4, 
            spacing=dp(15)
        )
        
        self.tutorial_title = Label(
            text="Welcome!",
            font_size='20sp',
            bold=True,
            color=self.accent_color,
            size_hint_y=0.3
        )
        self.tutorial_content.add_widget(self.tutorial_title)
        
        self.tutorial_text = Label(
            text="Loading...",
            font_size='16sp',
            color=(0.9, 0.9, 0.9, 1),
            halign='center',
            valign='middle',
            text_size=(None, None),
            size_hint_y=0.7
        )
        self.tutorial_content.add_widget(self.tutorial_text)
        
        main_layout.add_widget(self.tutorial_content)
        
        # Progress indicator
        self.progress_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1, spacing=dp(5))
        main_layout.add_widget(self.progress_layout)
        
        # Action button
        self.action_button = self.create_game_button("Continue", self.next_step, self.gold_color)
        self.action_button.size_hint_y = 0.1
        main_layout.add_widget(self.action_button)
        
        self.add_widget(main_layout)
    
    def start_tutorial(self, tutorial_steps):
        """Start tutorial with given steps"""
        self.tutorial_steps = tutorial_steps
        self.current_step = 0
        self.update_progress_indicator()
        self.show_current_step()
    
    def show_current_step(self):
        """Display current tutorial step"""
        if not self.tutorial_steps or self.current_step >= len(self.tutorial_steps):
            return
        
        step = self.tutorial_steps[self.current_step]
        self.tutorial_title.text = step['title']
        self.tutorial_text.text = step['content']
        
        # Update text size for proper wrapping
        self.tutorial_text.text_size = (self.tutorial_text.width - dp(20), None)
        
        # Update button text
        if self.current_step == len(self.tutorial_steps) - 1:
            self.action_button.text = "Start Game!"
        else:
            self.action_button.text = "Continue"
    
    def update_progress_indicator(self):
        """Update progress dots"""
        self.progress_layout.clear_widgets()
        
        if not self.tutorial_steps:
            return
        
        for i in range(len(self.tutorial_steps)):
            dot = Widget(size_hint=(None, None), size=(dp(12), dp(12)))
            
            with dot.canvas:
                if i == self.current_step:
                    Color(*self.gold_color)
                elif i < self.current_step:
                    Color(0, 0.8, 0, 1)  # Green for completed
                else:
                    Color(0.3, 0.3, 0.3, 1)  # Gray for upcoming
                
                Ellipse(pos=dot.pos, size=dot.size)
            
            dot.bind(pos=self.update_dot_graphics, size=self.update_dot_graphics)
            self.progress_layout.add_widget(dot)
    
    def update_dot_graphics(self, instance, value):
        """Update dot graphics when position changes"""
        instance.canvas.children[-1].pos = instance.pos
        instance.canvas.children[-1].size = instance.size
    
    def next_step(self, instance):
        """Move to next tutorial step"""
        if self.current_step < len(self.tutorial_steps) - 1:
            self.current_step += 1
            self.update_progress_indicator()
            self.show_current_step()
            
            # Animate transition
            self.animate_step_transition()
        else:
            # Complete tutorial
            self.complete_tutorial()
    
    def animate_step_transition(self):
        """Animate step transition"""
        # Fade out and in effect
        anim = Animation(opacity=0.5, duration=0.2) + Animation(opacity=1, duration=0.2)
        anim.start(self.tutorial_content)
    
    def complete_tutorial(self):
        """Complete tutorial and navigate to main menu"""
        result = self.flow_manager.complete_tutorial()
        
        if result.get('level_up'):
            self.show_level_up_popup(result)
        else:
            self.navigate_to_main_menu()
    
    def show_level_up_popup(self, result):
        """Show level up celebration"""
        content = BoxLayout(orientation='vertical', spacing=dp(15), padding=dp(20))
        
        # Celebration
        celebration = Label(
            text="🎊 Congratulations! 🎊",
            font_size='20sp',
            color=self.gold_color,
            bold=True
        )
        content.add_widget(celebration)
        
        # Points earned
        points_label = Label(
            text=f"Tutorial Completed!\nEarned: {result['bonus_points']:,} points",
            font_size='16sp',
            halign='center'
        )
        content.add_widget(points_label)
        
        # Continue button
        continue_btn = Button(text="Enter Game", size_hint_y=0.3)
        content.add_widget(continue_btn)
        
        popup = Popup(
            title="Tutorial Complete!",
            content=content,
            size_hint=(0.8, 0.6),
            auto_dismiss=False
        )
        
        continue_btn.bind(on_press=lambda x: [popup.dismiss(), self.navigate_to_main_menu()])
        popup.open()
    
    def navigate_to_main_menu(self):
        """Navigate to main menu"""
        self.manager.current = 'enhanced_main_menu'

class EnhancedMainMenuScreen(GameScreen):
    """Enhanced main menu with level progression and feature unlocks"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'enhanced_main_menu'
        self.flow_manager = get_game_flow_manager()
        self.build_ui()
        
        # Refresh data every 30 seconds
        Clock.schedule_interval(self.refresh_data, 30)
    
    def build_ui(self):
        """Build enhanced main menu UI"""
        main_layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))
        
        # Header with logo and player info
        header = self.create_player_header()
        main_layout.add_widget(header)
        
        # Level progress section
        level_progress = self.create_level_progress()
        main_layout.add_widget(level_progress)
        
        # Daily challenges preview
        daily_preview = self.create_daily_challenges_preview()
        main_layout.add_widget(daily_preview)
        
        # Feature menu
        feature_menu = self.create_feature_menu()
        main_layout.add_widget(feature_menu)
        
        # Bottom navigation
        bottom_nav = self.create_bottom_navigation()
        main_layout.add_widget(bottom_nav)
        
        self.add_widget(main_layout)
    
    def create_player_header(self):
        """Create player info header"""
        header = BoxLayout(orientation='horizontal', size_hint_y=0.15, spacing=dp(15))
        
        # Avatar/logo
        avatar = Label(
            text="🎖️",
            font_size='32sp',
            size_hint_x=0.2
        )
        header.add_widget(avatar)
        
        # Player info
        info_layout = BoxLayout(orientation='vertical', size_hint_x=0.6)
        
        stats = self.flow_manager.game_state.get_player_stats()
        
        self.player_name = Label(
            text=f"Level {stats['level']} {stats['level_name']}",
            font_size='18sp',
            bold=True,
            color=self.gold_color,
            halign='left'
        )
        info_layout.add_widget(self.player_name)
        
        self.player_points = Label(
            text=f"{stats['total_points']:,} points",
            font_size='14sp',
            color=(0.8, 0.8, 0.8, 1),
            halign='left'
        )
        info_layout.add_widget(self.player_points)
        
        header.add_widget(info_layout)
        
        # Daily streak
        streak_layout = BoxLayout(orientation='vertical', size_hint_x=0.2)
        
        streak_label = Label(
            text="🔥",
            font_size='24sp',
            size_hint_y=0.6
        )
        streak_layout.add_widget(streak_label)
        
        self.streak_count = Label(
            text=f"{stats['daily_streak']}",
            font_size='14sp',
            color=(1.0, 0.5, 0.0, 1),
            size_hint_y=0.4
        )
        streak_layout.add_widget(self.streak_count)
        
        header.add_widget(streak_layout)
        
        return header
    
    def create_level_progress(self):
        """Create level progression display"""
        progress_layout = BoxLayout(
            orientation='vertical', 
            size_hint_y=0.15, 
            spacing=dp(8),
            padding=dp(10)
        )
        
        with progress_layout.canvas.before:
            Color(0.1, 0.15, 0.25, 1)
            RoundedRectangle(size=progress_layout.size, pos=progress_layout.pos, radius=[10])
        
        progress_layout.bind(size=self._update_card_rect, pos=self._update_card_rect)
        
        # Progress header
        stats = self.flow_manager.game_state.get_player_stats()
        next_unlock = self.flow_manager.get_next_unlock_info(stats['level'])
        
        if next_unlock and next_unlock['points_needed'] > 0:
            header_text = f"Next Level: {next_unlock['level_name']}"
            progress_text = f"{next_unlock['points_needed']:,} points needed"
            
            progress_value = 100 - (next_unlock['points_needed'] / 
                                  (self.flow_manager.level_config[next_unlock['next_level']]['points_required'] - 
                                   self.flow_manager.level_config[stats['level']]['points_required']) * 100)
        else:
            header_text = "Max Level Achieved!"
            progress_text = "🎉 Elite Status"
            progress_value = 100
        
        header_label = Label(
            text=header_text,
            font_size='14sp',
            color=self.accent_color,
            size_hint_y=0.4
        )
        progress_layout.add_widget(header_label)
        
        # Progress bar
        self.level_progress_bar = ProgressBar(
            max=100,
            value=progress_value,
            size_hint_y=0.3
        )
        progress_layout.add_widget(self.level_progress_bar)
        
        progress_label = Label(
            text=progress_text,
            font_size='12sp',
            color=(0.7, 0.7, 0.7, 1),
            size_hint_y=0.3
        )
        progress_layout.add_widget(progress_label)
        
        return progress_layout
    
    def create_daily_challenges_preview(self):
        """Create daily challenges preview"""
        challenges_layout = BoxLayout(
            orientation='vertical',
            size_hint_y=0.2,
            spacing=dp(8),
            padding=dp(10)
        )
        
        with challenges_layout.canvas.before:
            Color(0.1, 0.25, 0.1, 1)  # Green tint
            RoundedRectangle(size=challenges_layout.size, pos=challenges_layout.pos, radius=[10])
        
        challenges_layout.bind(size=self._update_card_rect, pos=self._update_card_rect)
        
        # Header
        header_layout = BoxLayout(orientation='horizontal', size_hint_y=0.4)
        
        title = Label(
            text="🎯 Today's Challenges",
            font_size='16sp',
            bold=True,
            color=(1, 1, 1, 1),
            size_hint_x=0.7
        )
        header_layout.add_widget(title)
        
        challenges = self.flow_manager.game_state.get_available_challenges()
        count_label = Label(
            text=f"{len(challenges)} available",
            font_size='12sp',
            color=(0.8, 1, 0.8, 1),
            size_hint_x=0.3
        )
        header_layout.add_widget(count_label)
        
        challenges_layout.add_widget(header_layout)
        
        # Challenge preview
        if challenges:
            challenge = challenges[0]  # Show first challenge
            challenge_text = f"• {challenge['title']}: {challenge['points']:,} points"
        else:
            challenge_text = "All challenges completed! Come back tomorrow."
        
        preview_label = Label(
            text=challenge_text,
            font_size='14sp',
            color=(0.9, 0.9, 0.9, 1),
            halign='left',
            size_hint_y=0.4
        )
        challenges_layout.add_widget(preview_label)
        
        # Action button
        action_btn = self.create_game_button(
            "View Challenges",
            lambda x: self.navigate_to_challenges(),
            (0.2, 0.8, 0.2, 1)
        )
        action_btn.size_hint_y = 0.2
        challenges_layout.add_widget(action_btn)
        
        return challenges_layout
    
    def create_feature_menu(self):
        """Create main feature menu"""
        menu_scroll = ScrollView(size_hint_y=0.35)
        menu_layout = GridLayout(cols=2, spacing=dp(10), size_hint_y=None, padding=dp(5))
        menu_layout.bind(minimum_height=menu_layout.setter('height'))
        
        # Get available features
        available_features = self.flow_manager.get_available_features()
        unlocked_features = [f for f in available_features if f['status'] == 'unlocked']
        
        # Create feature buttons
        feature_buttons = [
            {'name': 'Dashboard', 'icon': '📊', 'screen': 'dashboard'},
            {'name': 'Daily Challenges', 'icon': '🎯', 'screen': 'daily_challenge'},
            {'name': 'Segment Analysis', 'icon': '👥', 'screen': 'segment_game'},
            {'name': 'Pricing Master', 'icon': '💎', 'screen': 'pricing_game'},
            {'name': 'Block Analysis', 'icon': '🏢', 'screen': 'block_game'},
            {'name': 'Leaderboard', 'icon': '🏆', 'screen': 'leaderboard'},
            {'name': 'Settings', 'icon': '⚙️', 'screen': 'settings'},
            {'name': 'Help', 'icon': '❓', 'screen': 'help'}
        ]
        
        for button_info in feature_buttons:
            is_unlocked = any(f['name'] == button_info['name'] for f in unlocked_features)
            btn = self.create_feature_button(button_info, is_unlocked)
            menu_layout.add_widget(btn)
        
        menu_scroll.add_widget(menu_layout)
        return menu_scroll
    
    def create_feature_button(self, button_info, is_unlocked):
        """Create individual feature button"""
        btn_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=dp(90))
        
        # Button styling based on unlock status
        if is_unlocked:
            bg_color = self.accent_color
            text_color = (1, 1, 1, 1)
            opacity = 1.0
        else:
            bg_color = (0.3, 0.3, 0.3, 1)
            text_color = (0.6, 0.6, 0.6, 1)
            opacity = 0.6
        
        btn = Button(
            text=f"{button_info['icon']}\n{button_info['name']}",
            font_size='14sp',
            background_color=bg_color,
            color=text_color,
            opacity=opacity
        )
        
        if is_unlocked:
            btn.bind(on_press=lambda x, screen=button_info['screen']: self.navigate_to_screen(screen))
        else:
            btn.bind(on_press=lambda x, name=button_info['name']: self.show_locked_feature(name))
        
        btn_layout.add_widget(btn)
        return btn_layout
    
    def create_bottom_navigation(self):
        """Create bottom navigation bar"""
        nav_layout = BoxLayout(orientation='horizontal', size_hint_y=0.15, spacing=dp(10))
        
        # Quick stats
        stats_btn = self.create_game_button("📊 Stats", self.show_stats_popup)
        nav_layout.add_widget(stats_btn)
        
        # Achievement progress
        achievements_btn = self.create_game_button("🏆 Achievements", self.show_achievements_popup)
        nav_layout.add_widget(achievements_btn)
        
        # Settings
        settings_btn = self.create_game_button("⚙️ Settings", lambda x: self.navigate_to_screen('settings'))
        nav_layout.add_widget(settings_btn)
        
        return nav_layout
    
    def navigate_to_challenges(self):
        """Navigate to challenges screen"""
        self.navigate_to_screen('daily_challenge')
    
    def navigate_to_screen(self, screen_name):
        """Navigate to specific screen with flow management"""
        result = self.flow_manager.navigate_to_screen(screen_name)
        
        if result.get('feature_locked'):
            self.show_locked_feature(result['locked_feature'])
        elif result.get('tutorial'):
            # Show tutorial first, then navigate
            self.show_feature_tutorial(result['tutorial'], screen_name)
        else:
            self.manager.current = screen_name
    
    def show_locked_feature(self, feature_name):
        """Show locked feature popup"""
        feature_config = self.flow_manager.feature_unlocks.get(feature_name, {})
        required_level = feature_config.get('required_level', 1)
        current_level = self.flow_manager.game_state.state['level']
        
        content = BoxLayout(orientation='vertical', spacing=dp(15), padding=dp(20))
        
        icon_label = Label(
            text="🔒",
            font_size='48sp',
            color=(0.8, 0.8, 0.0, 1)
        )
        content.add_widget(icon_label)
        
        title_label = Label(
            text=f"{feature_name} Locked",
            font_size='18sp',
            bold=True,
            color=(1, 1, 1, 1)
        )
        content.add_widget(title_label)
        
        info_label = Label(
            text=f"Reach Level {required_level} to unlock this feature!\n"
                 f"Current Level: {current_level}\n"
                 f"Complete more challenges to level up.",
            font_size='14sp',
            halign='center',
            color=(0.9, 0.9, 0.9, 1)
        )
        content.add_widget(info_label)
        
        close_btn = Button(text="OK", size_hint_y=0.3)
        content.add_widget(close_btn)
        
        popup = Popup(
            title="Feature Locked",
            content=content,
            size_hint=(0.8, 0.6)
        )
        
        close_btn.bind(on_press=popup.dismiss)
        popup.open()
    
    def show_feature_tutorial(self, tutorial_data, target_screen):
        """Show feature tutorial popup"""
        # For now, mark tutorial as seen and navigate
        # TODO: Implement interactive tutorial overlay
        self.flow_manager.complete_tutorial_for_feature(tutorial_data['tutorial_key'])
        self.manager.current = target_screen
    
    def show_stats_popup(self, instance):
        """Show player statistics popup"""
        status = self.flow_manager.get_game_flow_status()
        stats = status['player_stats']
        session_stats = status['session_stats']
        
        content = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(20))
        
        # Player info
        player_info = f"""👤 Player Statistics
        
Level: {stats['level']} ({stats['level_name']})
Total Points: {stats['total_points']:,}
Daily Streak: {stats['daily_streak']} days
Challenges Completed: {stats['challenges_completed']}
Unlocked Features: {len(stats['unlocked_features'])}

📱 Current Session:
Duration: {session_stats['session_duration_formatted']}
Challenges Attempted: {session_stats['challenges_attempted']}
Challenges Completed: {session_stats['challenges_completed']}
Points Earned: {session_stats['points_earned']:,}"""
        
        stats_label = Label(
            text=player_info,
            font_size='12sp',
            halign='left',
            valign='top',
            color=(0.9, 0.9, 0.9, 1)
        )
        content.add_widget(stats_label)
        
        close_btn = Button(text="Close", size_hint_y=0.2)
        content.add_widget(close_btn)
        
        popup = Popup(
            title="Player Statistics",
            content=content,
            size_hint=(0.85, 0.8)
        )
        
        close_btn.bind(on_press=popup.dismiss)
        popup.open()
    
    def show_achievements_popup(self, instance):
        """Show achievements popup"""
        achievements = self.flow_manager.game_state.get_achievements()
        unlocked = self.flow_manager.game_state.state.get('achievements', [])
        
        content = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(15))
        
        scroll = ScrollView()
        achievements_layout = BoxLayout(orientation='vertical', spacing=dp(10), size_hint_y=None)
        achievements_layout.bind(minimum_height=achievements_layout.setter('height'))
        
        for achievement in achievements[:6]:  # Show first 6 achievements
            is_unlocked = achievement['name'] in unlocked
            
            ach_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(50))
            
            # Status icon
            status_icon = Label(
                text="🏆" if is_unlocked else "🔒",
                font_size='20sp',
                size_hint_x=0.15,
                color=(1, 0.84, 0, 1) if is_unlocked else (0.5, 0.5, 0.5, 1)
            )
            ach_layout.add_widget(status_icon)
            
            # Achievement info
            info_layout = BoxLayout(orientation='vertical', size_hint_x=0.85)
            
            name_label = Label(
                text=achievement['name'],
                font_size='14sp',
                bold=True,
                halign='left',
                color=(1, 1, 1, 1) if is_unlocked else (0.7, 0.7, 0.7, 1)
            )
            info_layout.add_widget(name_label)
            
            reward_text = f"Reward: {achievement['reward']:,} points"
            reward_label = Label(
                text=reward_text,
                font_size='12sp',
                halign='left',
                color=(0.8, 0.8, 0.8, 1)
            )
            info_layout.add_widget(reward_label)
            
            ach_layout.add_widget(info_layout)
            achievements_layout.add_widget(ach_layout)
        
        scroll.add_widget(achievements_layout)
        content.add_widget(scroll)
        
        close_btn = Button(text="Close", size_hint_y=0.15)
        content.add_widget(close_btn)
        
        popup = Popup(
            title=f"Achievements ({len(unlocked)}/{len(achievements)})",
            content=content,
            size_hint=(0.9, 0.8)
        )
        
        close_btn.bind(on_press=popup.dismiss)
        popup.open()
    
    def refresh_data(self, dt):
        """Refresh menu data"""
        # Update player stats
        stats = self.flow_manager.game_state.get_player_stats()
        self.player_name.text = f"Level {stats['level']} {stats['level_name']}"
        self.player_points.text = f"{stats['total_points']:,} points"
        self.streak_count.text = f"{stats['daily_streak']}"
        
        # Update level progress
        next_unlock = self.flow_manager.get_next_unlock_info(stats['level'])
        if next_unlock and next_unlock['points_needed'] > 0:
            progress_value = 100 - (next_unlock['points_needed'] / 
                                  (self.flow_manager.level_config[next_unlock['next_level']]['points_required'] - 
                                   self.flow_manager.level_config[stats['level']]['points_required']) * 100)
            self.level_progress_bar.value = progress_value

class LevelUpCelebrationScreen(GameScreen):
    """Level up celebration screen"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'level_up_celebration'
        self.flow_manager = get_game_flow_manager()
        self.level_data = {}
        self.build_ui()
    
    def build_ui(self):
        """Build level up celebration UI"""
        main_layout = BoxLayout(orientation='vertical', padding=dp(30), spacing=dp(20))
        
        # Celebration header
        celebration_layout = BoxLayout(orientation='vertical', size_hint_y=0.3, spacing=dp(10))
        
        # Animated celebration
        self.celebration_icon = Label(
            text="🎊",
            font_size='64sp',
            color=self.gold_color
        )
        celebration_layout.add_widget(self.celebration_icon)
        
        self.level_up_title = Label(
            text="LEVEL UP!",
            font_size='32sp',
            bold=True,
            color=self.gold_color
        )
        celebration_layout.add_widget(self.level_up_title)
        
        main_layout.add_widget(celebration_layout)
        
        # Level info
        self.level_info_layout = BoxLayout(orientation='vertical', size_hint_y=0.4, spacing=dp(15))
        main_layout.add_widget(self.level_info_layout)
        
        # Rewards section
        self.rewards_layout = BoxLayout(orientation='vertical', size_hint_y=0.2, spacing=dp(10))
        main_layout.add_widget(self.rewards_layout)
        
        # Continue button
        self.continue_button = self.create_game_button("Continue", self.continue_game, self.gold_color)
        self.continue_button.size_hint_y = 0.1
        main_layout.add_widget(self.continue_button)
        
        self.add_widget(main_layout)
        
        # Start celebration animation
        self.animate_celebration()
    
    def show_level_up(self, level_data):
        """Display level up information"""
        self.level_data = level_data
        
        # Update level info
        self.level_info_layout.clear_widgets()
        
        new_level = level_data['new_level']
        level_config = level_data['level_config']
        
        # Badge and level name
        badge_layout = BoxLayout(orientation='horizontal', size_hint_y=0.4, spacing=dp(15))
        
        badge_label = Label(
            text=level_config.get('badge', '🎖️'),
            font_size='48sp'
        )
        badge_layout.add_widget(badge_label)
        
        level_name = Label(
            text=f"Level {new_level}\n{level_config.get('name', 'Revenue Expert')}",
            font_size='20sp',
            bold=True,
            halign='center',
            color=self.accent_color
        )
        badge_layout.add_widget(level_name)
        
        self.level_info_layout.add_widget(badge_layout)
        
        # Description
        description = Label(
            text=level_config.get('description', 'Congratulations on your promotion!'),
            font_size='14sp',
            halign='center',
            color=(0.9, 0.9, 0.9, 1),
            size_hint_y=0.3
        )
        self.level_info_layout.add_widget(description)
        
        # New unlocks
        if level_data.get('new_unlocks'):
            unlocks_text = "🔓 New Features Unlocked:\n" + "\n".join(f"• {unlock}" for unlock in level_data['new_unlocks'])
            unlocks_label = Label(
                text=unlocks_text,
                font_size='12sp',
                halign='center',
                color=(0, 1, 0, 1),
                size_hint_y=0.3
            )
            self.level_info_layout.add_widget(unlocks_label)
        
        # Show rewards
        self.show_rewards(level_data)
    
    def show_rewards(self, level_data):
        """Show level up rewards"""
        self.rewards_layout.clear_widgets()
        
        rewards_title = Label(
            text="🎁 Rewards Earned:",
            font_size='16sp',
            bold=True,
            color=self.gold_color,
            size_hint_y=0.3
        )
        self.rewards_layout.add_widget(rewards_title)
        
        # Points bonus
        if level_data.get('level_bonus', 0) > 0:
            bonus_text = f"💰 Bonus: {self.aed_handler.format_aed(level_data['level_bonus'])} points!"
            bonus_label = Label(
                text=bonus_text,
                font_size='14sp',
                color=self.gold_color,
                size_hint_y=0.35
            )
            self.rewards_layout.add_widget(bonus_label)
        
        # Challenge result
        if 'challenge_result' in level_data:
            result = level_data['challenge_result']
            challenge_text = f"🏆 Challenge: +{result['final_score']:,} points"
            challenge_label = Label(
                text=challenge_text,
                font_size='12sp',
                color=(0.8, 0.8, 0.8, 1),
                size_hint_y=0.35
            )
            self.rewards_layout.add_widget(challenge_label)
    
    def animate_celebration(self):
        """Animate celebration elements"""
        # Bounce animation for icon
        bounce = (Animation(font_size='72sp', duration=0.3) + 
                 Animation(font_size='64sp', duration=0.3))
        bounce.repeat = True
        bounce.start(self.celebration_icon)
        
        # Pulse animation for title
        pulse = (Animation(opacity=0.7, duration=0.5) + 
                Animation(opacity=1.0, duration=0.5))
        pulse.repeat = True
        pulse.start(self.level_up_title)
    
    def continue_game(self, instance):
        """Continue to main menu"""
        self.manager.current = 'enhanced_main_menu'

if __name__ == "__main__":
    print("🎮 Complete Game Screens with Flow Management")
    print("=" * 50)
    print("✅ Welcome/Tutorial Screen")
    print("✅ Enhanced Main Menu with Level Progress")
    print("✅ Level Up Celebration Screen")
    print("✅ Flow Management Integration")
    print("✅ Feature Unlock System")
    print("✅ Achievement Display")
    print("✅ Tutorial System")
    print("\n🚀 Ready for complete game flow implementation!")