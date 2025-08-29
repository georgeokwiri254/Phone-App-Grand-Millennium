"""
Grand Millennium Revenue Analytics - Mobile Optimized Screens

Mobile-optimized game screens with responsive design, touch optimization,
and enhanced user experience for different screen sizes and orientations.
"""

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.progressbar import ProgressBar
from kivy.uix.slider import Slider
from kivy.uix.popup import Popup
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.graphics import Color, RoundedRectangle

from main_game import GameScreen
from mobile_touch_optimizer import mobile_ux_enhancer, responsive_design, touch_optimizer
from game_flow_manager import get_game_flow_manager
from enhanced_analytics_engine import get_analytics_engine
from aed_currency_handler import AEDCurrencyHandler

class MobileOptimizedMainMenu(GameScreen):
    """Mobile-optimized main menu with responsive design"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'mobile_main_menu'
        self.flow_manager = get_game_flow_manager()
        self.aed_handler = AEDCurrencyHandler()
        self.layout_config = responsive_design.get_layout_config()
        self.build_ui()
        
        # Auto-refresh and window size change handling
        Window.bind(size=self.on_window_resize)
        Clock.schedule_interval(self.refresh_data, 30)
    
    def build_ui(self):
        """Build responsive mobile UI"""
        main_layout = BoxLayout(orientation='vertical')
        
        # Responsive sizing based on screen category
        header_height = self.layout_config['header_height']
        content_height = self.layout_config['content_height'] 
        nav_height = self.layout_config['nav_height']
        
        # Header section
        header = self.create_responsive_header()
        header.size_hint_y = header_height
        main_layout.add_widget(header)
        
        # Main content area
        content = self.create_responsive_content()
        content.size_hint_y = content_height
        main_layout.add_widget(content)
        
        # Bottom navigation
        navigation = self.create_responsive_navigation()
        navigation.size_hint_y = nav_height
        main_layout.add_widget(navigation)
        
        self.add_widget(main_layout)
        
        # Add swipe navigation
        mobile_ux_enhancer.add_swipe_navigation(
            self,
            left_callback=self.swipe_left_action,
            right_callback=self.swipe_right_action
        )
    
    def create_responsive_header(self):
        """Create responsive header with player info"""
        header_layout = BoxLayout(
            orientation='horizontal',
            spacing=responsive_design.get_responsive_value(dp(10), 'spacing'),
            padding=responsive_design.get_responsive_value(dp(15), 'padding')
        )
        
        # Player avatar and info
        player_info = BoxLayout(orientation='horizontal', size_hint_x=0.7, spacing=dp(10))
        
        # Avatar
        avatar = Label(
            text="🎖️",
            font_size=responsive_design.create_responsive_font_size(28),
            size_hint_x=0.2
        )
        player_info.add_widget(avatar)
        
        # Player stats
        stats_layout = BoxLayout(orientation='vertical', size_hint_x=0.8)
        
        stats = self.flow_manager.game_state.get_player_stats()
        
        self.player_name = Label(
            text=f"Level {stats['level']} {stats['level_name']}",
            font_size=responsive_design.create_responsive_font_size(16),
            bold=True,
            color=self.gold_color,
            halign='left',
            text_size=(None, None)
        )
        stats_layout.add_widget(self.player_name)
        
        self.player_points = Label(
            text=f"{stats['total_points']:,} points",
            font_size=responsive_design.create_responsive_font_size(12),
            color=(0.8, 0.8, 0.8, 1),
            halign='left'
        )
        stats_layout.add_widget(self.player_points)
        
        player_info.add_widget(stats_layout)
        header_layout.add_widget(player_info)
        
        # Daily streak and actions
        actions_layout = BoxLayout(orientation='vertical', size_hint_x=0.3, spacing=dp(5))
        
        # Streak display
        streak_layout = BoxLayout(orientation='horizontal', size_hint_y=0.6)
        
        streak_icon = Label(
            text="🔥",
            font_size=responsive_design.create_responsive_font_size(18),
            size_hint_x=0.4
        )
        streak_layout.add_widget(streak_icon)
        
        self.streak_count = Label(
            text=f"{stats['daily_streak']}",
            font_size=responsive_design.create_responsive_font_size(14),
            color=(1.0, 0.5, 0.0, 1),
            bold=True,
            size_hint_x=0.6
        )
        streak_layout.add_widget(self.streak_count)
        
        actions_layout.add_widget(streak_layout)
        
        # Quick action button (responsive size)
        if responsive_design.current_breakpoint in ['lg', 'xl']:
            quick_action = mobile_ux_enhancer.create_mobile_optimized_button(
                "Daily Bonus",
                self.claim_daily_bonus,
                size_hint_y=0.4
            )
            actions_layout.add_widget(quick_action)
        
        header_layout.add_widget(actions_layout)
        
        return header_layout
    
    def create_responsive_content(self):
        """Create responsive main content area"""
        content_scroll = ScrollView()
        content_layout = BoxLayout(
            orientation='vertical',
            spacing=responsive_design.get_responsive_value(dp(15), 'spacing'),
            size_hint_y=None,
            padding=responsive_design.get_responsive_value(dp(15), 'padding')
        )
        content_layout.bind(minimum_height=content_layout.setter('height'))
        
        # Level progress card
        level_card = self.create_level_progress_card()
        content_layout.add_widget(level_card)
        
        # Daily challenges preview
        challenges_card = self.create_challenges_preview_card()
        content_layout.add_widget(challenges_card)
        
        # Feature grid - responsive columns
        features_grid = self.create_features_grid()
        content_layout.add_widget(features_grid)
        
        # Quick stats (only show on larger screens)
        if self.layout_config['show_secondary_info']:
            stats_card = self.create_quick_stats_card()
            content_layout.add_widget(stats_card)
        
        content_scroll.add_widget(content_layout)
        
        # Add pull-to-refresh
        mobile_ux_enhancer.add_pull_to_refresh(content_scroll, self.refresh_all_data)
        
        return content_scroll
    
    def create_level_progress_card(self):
        """Create responsive level progress card"""
        card = mobile_ux_enhancer.create_mobile_card(
            "Level Progress",
            "",
            "📈"
        )
        
        # Progress bar
        stats = self.flow_manager.game_state.get_player_stats()
        next_unlock = self.flow_manager.get_next_unlock_info(stats['level'])
        
        progress_layout = BoxLayout(orientation='vertical', spacing=dp(8))
        
        if next_unlock and next_unlock['points_needed'] > 0:
            # Calculate progress percentage
            level_config = self.flow_manager.level_config
            current_level_points = level_config[stats['level']]['points_required']
            next_level_points = level_config[next_unlock['next_level']]['points_required']
            total_for_level = next_level_points - current_level_points
            current_progress = stats['total_points'] - current_level_points
            progress_percentage = (current_progress / total_for_level) * 100 if total_for_level > 0 else 0
            
            progress_label = Label(
                text=f"Next: {next_unlock['level_name']}",
                font_size=responsive_design.create_responsive_font_size(14),
                color=self.accent_color,
                size_hint_y=0.3
            )
            progress_layout.add_widget(progress_label)
            
            self.progress_bar = ProgressBar(
                max=100,
                value=progress_percentage,
                size_hint_y=0.4
            )
            progress_layout.add_widget(self.progress_bar)
            
            points_needed_label = Label(
                text=f"{next_unlock['points_needed']:,} points needed",
                font_size=responsive_design.create_responsive_font_size(12),
                color=(0.7, 0.7, 0.7, 1),
                size_hint_y=0.3
            )
            progress_layout.add_widget(points_needed_label)
        else:
            max_level_label = Label(
                text="🎉 Max Level Achieved!",
                font_size=responsive_design.create_responsive_font_size(16),
                color=self.gold_color,
                bold=True
            )
            progress_layout.add_widget(max_level_label)
        
        card.add_widget(progress_layout)
        return card
    
    def create_challenges_preview_card(self):
        """Create challenges preview card"""
        challenges = self.flow_manager.game_state.get_available_challenges()
        
        if challenges:
            challenge = challenges[0]
            content_text = f"🎯 {challenge['title']}\n{challenge['points']:,} points available"
            action_needed = True
        else:
            content_text = "🎉 All challenges completed!\nCome back tomorrow for new challenges."
            action_needed = False
        
        card = mobile_ux_enhancer.create_mobile_card(
            "Today's Challenge",
            content_text,
            "🎯"
        )
        
        if action_needed:
            # Add action button
            challenge_button = mobile_ux_enhancer.create_mobile_optimized_button(
                "Start Challenge",
                lambda x: self.navigate_to_challenges(),
                size_hint_y=None
            )
            challenge_button.height = responsive_design.get_responsive_value(dp(40), 'button_height')
            card.add_widget(challenge_button)
        
        return card
    
    def create_features_grid(self):
        """Create responsive features grid"""
        grid_container = BoxLayout(orientation='vertical', size_hint_y=None)
        grid_container.bind(minimum_height=grid_container.setter('height'))
        
        # Grid header
        grid_header = Label(
            text="🎮 Game Features",
            font_size=responsive_design.create_responsive_font_size(18),
            bold=True,
            color=self.accent_color,
            size_hint_y=None,
            height=responsive_design.get_responsive_value(dp(40), 'button_height')
        )
        grid_container.add_widget(grid_header)
        
        # Features grid
        features_grid = GridLayout(
            cols=self.layout_config['grid_cols'],
            spacing=responsive_design.get_responsive_value(dp(10), 'spacing'),
            size_hint_y=None,
            height=responsive_design.get_responsive_value(dp(200), 'button_height')
        )
        
        # Get available features
        available_features = self.flow_manager.get_available_features()
        unlocked_features = [f for f in available_features if f['status'] == 'unlocked']
        
        # Create feature buttons
        main_features = [
            {'name': 'Dashboard', 'icon': '📊', 'screen': 'dashboard'},
            {'name': 'Challenges', 'icon': '🎯', 'screen': 'daily_challenge'},
            {'name': 'Segments', 'icon': '👥', 'screen': 'segment_game'},
            {'name': 'Pricing', 'icon': '💎', 'screen': 'pricing_game'},
        ]
        
        # Add additional features for larger screens
        if responsive_design.current_breakpoint in ['lg', 'xl']:
            main_features.extend([
                {'name': 'Analytics', 'icon': '🤖', 'screen': 'analytics'},
                {'name': 'Leaderboard', 'icon': '🏆', 'screen': 'leaderboard'},
            ])
        
        for feature in main_features:
            is_unlocked = any(f['name'] == feature['name'] for f in unlocked_features) or feature['name'] in ['Dashboard', 'Challenges']
            
            feature_button = self.create_responsive_feature_button(feature, is_unlocked)
            features_grid.add_widget(feature_button)
        
        grid_container.add_widget(features_grid)
        
        return grid_container
    
    def create_responsive_feature_button(self, feature_info, is_unlocked):
        """Create responsive feature button"""
        button_layout = BoxLayout(orientation='vertical', spacing=dp(5))
        
        # Button styling based on unlock status and screen size
        if is_unlocked:
            bg_color = self.accent_color
            text_color = (1, 1, 1, 1)
            opacity = 1.0
        else:
            bg_color = (0.3, 0.3, 0.3, 1)
            text_color = (0.6, 0.6, 0.6, 1)
            opacity = 0.6
        
        # Responsive button size
        button_height = responsive_design.get_responsive_value(dp(70), 'button_height')
        font_size = responsive_design.create_responsive_font_size(12)
        
        button = Button(
            text=f"{feature_info['icon']}\n{feature_info['name']}",
            font_size=font_size,
            background_color=bg_color,
            color=text_color,
            opacity=opacity,
            size_hint_y=None,
            height=button_height
        )
        
        # Optimize for touch
        button = touch_optimizer.optimize_touch_targets(button)
        
        if is_unlocked:
            button.bind(on_press=lambda x: self.navigate_to_feature(feature_info['screen']))
        else:
            button.bind(on_press=lambda x: self.show_locked_feature_popup(feature_info['name']))
        
        button_layout.add_widget(button)
        
        return button_layout
    
    def create_quick_stats_card(self):
        """Create quick stats card for larger screens"""
        analytics = get_analytics_engine()
        kpis = analytics.get_real_time_kpis()
        
        stats_text = f"📊 Today's Performance\n"
        stats_text += f"Occupancy: {kpis['occupancy_display']}\n"
        stats_text += f"ADR: {kpis['adr_display']}\n"
        stats_text += f"RevPAR: {kpis['revpar_display']}"
        
        return mobile_ux_enhancer.create_mobile_card(
            "Hotel Performance",
            stats_text,
            "📊"
        )
    
    def create_responsive_navigation(self):
        """Create responsive bottom navigation"""
        nav_layout = BoxLayout(
            orientation='horizontal',
            spacing=responsive_design.get_responsive_value(dp(5), 'spacing'),
            padding=responsive_design.get_responsive_value(dp(10), 'padding')
        )
        
        # Navigation buttons - adjust based on screen size
        if responsive_design.current_breakpoint in ['xs', 'sm']:
            # Fewer buttons on small screens
            nav_buttons = [
                {'text': '📊', 'callback': lambda x: self.navigate_to_feature('dashboard')},
                {'text': '🎯', 'callback': lambda x: self.navigate_to_feature('daily_challenge')},
                {'text': '🏆', 'callback': self.show_achievements},
                {'text': '⚙️', 'callback': lambda x: self.navigate_to_feature('settings')},
            ]
        else:
            # More buttons on larger screens
            nav_buttons = [
                {'text': '📊 Dashboard', 'callback': lambda x: self.navigate_to_feature('dashboard')},
                {'text': '🎯 Challenges', 'callback': lambda x: self.navigate_to_feature('daily_challenge')},
                {'text': '📈 Stats', 'callback': self.show_player_stats},
                {'text': '🏆 Achievements', 'callback': self.show_achievements},
                {'text': '⚙️ Settings', 'callback': lambda x: self.navigate_to_feature('settings')},
            ]
        
        for button_info in nav_buttons:
            nav_button = mobile_ux_enhancer.create_mobile_optimized_button(
                button_info['text'],
                button_info['callback']
            )
            nav_layout.add_widget(nav_button)
        
        return nav_layout
    
    def on_window_resize(self, instance, size):
        """Handle window resize for responsive design"""
        # Update breakpoint
        responsive_design.current_breakpoint = responsive_design.get_current_breakpoint()
        self.layout_config = responsive_design.get_layout_config()
        
        # Rebuild UI with new responsive values
        Clock.schedule_once(lambda dt: self.rebuild_responsive_ui(), 0.1)
    
    def rebuild_responsive_ui(self):
        """Rebuild UI with updated responsive values"""
        self.clear_widgets()
        self.build_ui()
    
    def swipe_left_action(self):
        """Handle swipe left gesture"""
        self.navigate_to_feature('daily_challenge')
    
    def swipe_right_action(self):
        """Handle swipe right gesture"""
        self.navigate_to_feature('dashboard')
    
    def navigate_to_challenges(self):
        """Navigate to challenges screen"""
        self.navigate_to_feature('daily_challenge')
    
    def navigate_to_feature(self, feature_screen):
        """Navigate to specific feature screen"""
        result = self.flow_manager.navigate_to_screen(feature_screen)
        
        if result.get('feature_locked'):
            self.show_locked_feature_popup(result['locked_feature'])
        else:
            self.manager.current = feature_screen
    
    def show_locked_feature_popup(self, feature_name):
        """Show responsive locked feature popup"""
        content = BoxLayout(
            orientation='vertical',
            spacing=responsive_design.get_responsive_value(dp(15), 'spacing'),
            padding=responsive_design.get_responsive_value(dp(20), 'padding')
        )
        
        # Lock icon
        lock_icon = Label(
            text="🔒",
            font_size=responsive_design.create_responsive_font_size(48),
            color=(0.8, 0.8, 0.0, 1),
            size_hint_y=0.3
        )
        content.add_widget(lock_icon)
        
        # Feature info
        feature_config = self.flow_manager.feature_unlocks.get(feature_name, {})
        required_level = feature_config.get('required_level', 1)
        current_level = self.flow_manager.game_state.state['level']
        
        info_text = f"{feature_name} is locked!\n\nReach Level {required_level} to unlock.\nCurrent Level: {current_level}"
        
        info_label = Label(
            text=info_text,
            font_size=responsive_design.create_responsive_font_size(14),
            halign='center',
            color=(0.9, 0.9, 0.9, 1),
            size_hint_y=0.5
        )
        content.add_widget(info_label)
        
        # Close button
        close_btn = mobile_ux_enhancer.create_mobile_optimized_button(
            "OK",
            lambda x: popup.dismiss(),
            size_hint_y=0.2
        )
        content.add_widget(close_btn)
        
        popup = Popup(
            title="Feature Locked",
            content=content,
            size_hint=(0.85, 0.6)
        )
        popup.open()
    
    def claim_daily_bonus(self, instance):
        """Claim daily bonus"""
        bonus = self.flow_manager.game_state.get_daily_bonus()
        if bonus:
            self.show_bonus_popup(bonus)
        else:
            # Show already claimed message
            self.show_info_popup("Daily bonus already claimed today!")
    
    def show_bonus_popup(self, bonus_data):
        """Show responsive daily bonus popup"""
        content = BoxLayout(
            orientation='vertical',
            spacing=responsive_design.get_responsive_value(dp(15), 'spacing'),
            padding=responsive_design.get_responsive_value(dp(20), 'padding')
        )
        
        # Bonus animation
        bonus_icon = Label(
            text="🎁",
            font_size=responsive_design.create_responsive_font_size(48),
            color=self.gold_color,
            size_hint_y=0.3
        )
        content.add_widget(bonus_icon)
        
        # Bonus info
        bonus_text = f"Daily Login Bonus!\nEarned: {bonus_data['bonus_awarded']:,} points\n🔥 Streak: {bonus_data['streak']} days"
        
        bonus_label = Label(
            text=bonus_text,
            font_size=responsive_design.create_responsive_font_size(16),
            halign='center',
            color=(0.9, 0.9, 0.9, 1),
            size_hint_y=0.5
        )
        content.add_widget(bonus_label)
        
        # Continue button
        continue_btn = mobile_ux_enhancer.create_mobile_optimized_button(
            "Awesome!",
            lambda x: popup.dismiss(),
            size_hint_y=0.2
        )
        content.add_widget(continue_btn)
        
        popup = Popup(
            title="Daily Bonus",
            content=content,
            size_hint=(0.8, 0.6),
            auto_dismiss=False
        )
        popup.open()
        
        # Animate bonus icon
        bounce = Animation(font_size=responsive_design.create_responsive_font_size(52), duration=0.3) + \
                Animation(font_size=responsive_design.create_responsive_font_size(48), duration=0.3)
        bounce.repeat = True
        bounce.start(bonus_icon)
    
    def show_info_popup(self, message):
        """Show responsive info popup"""
        content = BoxLayout(orientation='vertical', spacing=dp(15), padding=dp(20))
        
        info_label = Label(
            text=message,
            font_size=responsive_design.create_responsive_font_size(14),
            halign='center',
            color=(0.9, 0.9, 0.9, 1)
        )
        content.add_widget(info_label)
        
        ok_btn = mobile_ux_enhancer.create_mobile_optimized_button("OK", lambda x: popup.dismiss())
        content.add_widget(ok_btn)
        
        popup = Popup(title="Info", content=content, size_hint=(0.7, 0.4))
        popup.open()
    
    def show_player_stats(self, instance):
        """Show responsive player stats popup"""
        stats = self.flow_manager.game_state.get_player_stats()
        status = self.flow_manager.get_game_flow_status()
        session_stats = status['session_stats']
        
        content = BoxLayout(
            orientation='vertical', 
            spacing=responsive_design.get_responsive_value(dp(10), 'spacing'),
            padding=responsive_design.get_responsive_value(dp(20), 'padding')
        )
        
        # Stats text
        stats_text = f"""📊 Player Statistics
        
Level: {stats['level']} ({stats['level_name']})
Total Points: {stats['total_points']:,}
Daily Streak: {stats['daily_streak']} days
Challenges Completed: {stats['challenges_completed']}

📱 Current Session:
Duration: {session_stats['session_duration_formatted']}
Challenges: {session_stats['challenges_completed']}
Points Earned: {session_stats['points_earned']:,}"""
        
        stats_label = Label(
            text=stats_text,
            font_size=responsive_design.create_responsive_font_size(12),
            halign='left',
            valign='top',
            color=(0.9, 0.9, 0.9, 1)
        )
        content.add_widget(stats_label)
        
        close_btn = mobile_ux_enhancer.create_mobile_optimized_button("Close", lambda x: popup.dismiss())
        content.add_widget(close_btn)
        
        popup = Popup(
            title="Player Stats",
            content=content,
            size_hint=(0.9, 0.8)
        )
        popup.open()
    
    def show_achievements(self, instance):
        """Show responsive achievements popup"""
        achievements = self.flow_manager.game_state.get_achievements()
        unlocked = self.flow_manager.game_state.state.get('achievements', [])
        
        content = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(15))
        
        # Achievements scroll
        scroll = ScrollView()
        achievements_layout = BoxLayout(
            orientation='vertical', 
            spacing=responsive_design.get_responsive_value(dp(10), 'spacing'),
            size_hint_y=None
        )
        achievements_layout.bind(minimum_height=achievements_layout.setter('height'))
        
        for achievement in achievements[:6]:  # Show first 6
            is_unlocked = achievement['name'] in unlocked
            
            ach_layout = BoxLayout(
                orientation='horizontal',
                size_hint_y=None,
                height=responsive_design.get_responsive_value(dp(50), 'button_height'),
                spacing=dp(10)
            )
            
            # Status icon
            status_icon = Label(
                text="🏆" if is_unlocked else "🔒",
                font_size=responsive_design.create_responsive_font_size(20),
                size_hint_x=0.15,
                color=(1, 0.84, 0, 1) if is_unlocked else (0.5, 0.5, 0.5, 1)
            )
            ach_layout.add_widget(status_icon)
            
            # Achievement info
            info_layout = BoxLayout(orientation='vertical', size_hint_x=0.85)
            
            name_label = Label(
                text=achievement['name'],
                font_size=responsive_design.create_responsive_font_size(14),
                bold=True,
                halign='left',
                color=(1, 1, 1, 1) if is_unlocked else (0.7, 0.7, 0.7, 1)
            )
            info_layout.add_widget(name_label)
            
            reward_label = Label(
                text=f"Reward: {achievement['reward']:,} points",
                font_size=responsive_design.create_responsive_font_size(11),
                halign='left',
                color=(0.8, 0.8, 0.8, 1)
            )
            info_layout.add_widget(reward_label)
            
            ach_layout.add_widget(info_layout)
            achievements_layout.add_widget(ach_layout)
        
        scroll.add_widget(achievements_layout)
        content.add_widget(scroll)
        
        close_btn = mobile_ux_enhancer.create_mobile_optimized_button("Close", lambda x: popup.dismiss())
        content.add_widget(close_btn)
        
        popup = Popup(
            title=f"Achievements ({len(unlocked)}/{len(achievements)})",
            content=content,
            size_hint=(0.9, 0.8)
        )
        popup.open()
    
    def refresh_data(self, dt):
        """Refresh UI data"""
        # Update player stats
        stats = self.flow_manager.game_state.get_player_stats()
        self.player_name.text = f"Level {stats['level']} {stats['level_name']}"
        self.player_points.text = f"{stats['total_points']:,} points"
        self.streak_count.text = f"{stats['daily_streak']}"
        
        # Update progress bar if it exists
        if hasattr(self, 'progress_bar'):
            next_unlock = self.flow_manager.get_next_unlock_info(stats['level'])
            if next_unlock and next_unlock['points_needed'] > 0:
                level_config = self.flow_manager.level_config
                current_level_points = level_config[stats['level']]['points_required']
                next_level_points = level_config[next_unlock['next_level']]['points_required']
                total_for_level = next_level_points - current_level_points
                current_progress = stats['total_points'] - current_level_points
                progress_percentage = (current_progress / total_for_level) * 100 if total_for_level > 0 else 0
                self.progress_bar.value = progress_percentage
    
    def refresh_all_data(self):
        """Refresh all data (triggered by pull-to-refresh)"""
        # Clear analytics cache for fresh data
        analytics = get_analytics_engine()
        analytics.clear_cache()
        
        # Trigger UI refresh
        self.refresh_data(None)
        
        # Haptic feedback
        touch_optimizer.trigger_haptic_feedback('medium')

if __name__ == "__main__":
    print("📱 Mobile Optimized Screens Test")
    print("=" * 40)
    
    # Test responsive design detection
    layout_config = responsive_design.get_layout_config()
    print(f"Screen category: {responsive_design.current_breakpoint}")
    print(f"Grid columns: {layout_config['grid_cols']}")
    print(f"Show secondary info: {layout_config['show_secondary_info']}")
    
    # Test touch optimization
    print(f"Min touch target: {touch_optimizer.MIN_TOUCH_TARGET_SIZE}px")
    print(f"Recommended touch: {touch_optimizer.RECOMMENDED_TOUCH_SIZE}px")
    print(f"Is mobile platform: {touch_optimizer.is_mobile}")
    
    print("\n✅ Mobile Optimized Screens Ready!")