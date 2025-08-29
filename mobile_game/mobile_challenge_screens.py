"""
Grand Millennium Revenue Analytics - Mobile Optimized Challenge Screens

Mobile-optimized challenge and game screens with touch-friendly controls,
responsive design, and enhanced mobile user experience.
"""

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.progressbar import ProgressBar
from kivy.uix.popup import Popup
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.graphics import Color, RoundedRectangle

from main_game import GameScreen
from mobile_touch_optimizer import mobile_ux_enhancer, responsive_design, touch_optimizer
from game_flow_manager import get_game_flow_manager, GameMode
from enhanced_analytics_engine import get_analytics_engine
from aed_currency_handler import AEDCurrencyHandler

class MobileOptimizedDailyChallenges(GameScreen):
    """Mobile-optimized daily challenges screen"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'mobile_daily_challenge'
        self.flow_manager = get_game_flow_manager()
        self.analytics = get_analytics_engine()
        self.aed_handler = AEDCurrencyHandler()
        self.layout_config = responsive_design.get_layout_config()
        self.build_ui()
    
    def build_ui(self):
        """Build mobile-optimized challenge UI"""
        main_layout = BoxLayout(orientation='vertical')
        
        # Header
        header = self.create_mobile_header()
        header.size_hint_y = 0.15
        main_layout.add_widget(header)
        
        # Challenge content
        content = self.create_challenge_content()
        content.size_hint_y = 0.7
        main_layout.add_widget(content)
        
        # Navigation
        navigation = self.create_challenge_navigation()
        navigation.size_hint_y = 0.15
        main_layout.add_widget(navigation)
        
        self.add_widget(main_layout)
        
        # Add swipe navigation for challenge switching
        mobile_ux_enhancer.add_swipe_navigation(
            self,
            left_callback=self.next_challenge,
            right_callback=self.previous_challenge
        )
    
    def create_mobile_header(self):
        """Create mobile-optimized header"""
        header_layout = BoxLayout(
            orientation='vertical',
            spacing=responsive_design.get_responsive_value(dp(8), 'spacing'),
            padding=responsive_design.get_responsive_value(dp(15), 'padding')
        )
        
        # Title and subtitle
        title_layout = BoxLayout(orientation='horizontal')
        
        title = Label(
            text="🎯 Daily Challenges",
            font_size=responsive_design.create_responsive_font_size(20),
            bold=True,
            color=self.gold_color,
            size_hint_x=0.7,
            halign='left'
        )
        title_layout.add_widget(title)
        
        # Challenge counter
        challenges = self.flow_manager.game_state.get_available_challenges()
        counter = Label(
            text=f"{len(challenges)} available",
            font_size=responsive_design.create_responsive_font_size(12),
            color=(0.8, 0.8, 0.8, 1),
            size_hint_x=0.3,
            halign='right'
        )
        title_layout.add_widget(counter)
        
        header_layout.add_widget(title_layout)
        
        # Progress indicator for small screens
        if responsive_design.current_breakpoint in ['xs', 'sm']:
            progress_text = f"Complete challenges to earn AED-based points"
            progress_label = Label(
                text=progress_text,
                font_size=responsive_design.create_responsive_font_size(12),
                color=(0.7, 0.7, 0.7, 1),
                size_hint_y=0.3
            )
            header_layout.add_widget(progress_label)
        
        return header_layout
    
    def create_challenge_content(self):
        """Create mobile-optimized challenge content"""
        content_scroll = ScrollView()
        content_layout = BoxLayout(
            orientation='vertical',
            spacing=responsive_design.get_responsive_value(dp(15), 'spacing'),
            size_hint_y=None,
            padding=responsive_design.get_responsive_value(dp(15), 'padding')
        )
        content_layout.bind(minimum_height=content_layout.setter('height'))
        
        # Get available challenges
        challenges = self.flow_manager.game_state.get_available_challenges()
        
        if challenges:
            for i, challenge in enumerate(challenges):
                challenge_card = self.create_mobile_challenge_card(challenge, i)
                content_layout.add_widget(challenge_card)
        else:
            # No challenges available
            no_challenges_card = self.create_no_challenges_card()
            content_layout.add_widget(no_challenges_card)
        
        # Performance summary (for larger screens)
        if self.layout_config['show_secondary_info']:
            performance_card = self.create_performance_summary_card()
            content_layout.add_widget(performance_card)
        
        content_scroll.add_widget(content_layout)
        
        # Add pull-to-refresh
        mobile_ux_enhancer.add_pull_to_refresh(content_scroll, self.refresh_challenges)
        
        return content_scroll
    
    def create_mobile_challenge_card(self, challenge, index):
        """Create mobile-optimized challenge card"""
        # Create card container
        card_container = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=responsive_design.get_responsive_value(dp(140), 'card_height'),
            spacing=responsive_design.get_responsive_value(dp(10), 'spacing'),
            padding=responsive_design.get_responsive_value(dp(15), 'padding')
        )
        
        # Card background with gradient effect
        with card_container.canvas.before:
            Color(0.15 + (index * 0.05), 0.25 + (index * 0.03), 0.35 + (index * 0.02), 1)
            RoundedRectangle(size=card_container.size, pos=card_container.pos, radius=[dp(12)])
        
        card_container.bind(size=self.update_card_background, pos=self.update_card_background)
        
        # Challenge header
        header_layout = BoxLayout(orientation='horizontal', size_hint_y=0.3, spacing=dp(10))
        
        # Icon and priority indicator
        icon_layout = BoxLayout(orientation='horizontal', size_hint_x=0.3)
        
        challenge_icon = Label(
            text=challenge['icon'],
            font_size=responsive_design.create_responsive_font_size(24),
            size_hint_x=0.6
        )
        icon_layout.add_widget(challenge_icon)
        
        # Priority/difficulty indicator
        difficulty_dots = "●" * min(challenge['required_level'], 3)
        difficulty_label = Label(
            text=difficulty_dots,
            font_size=responsive_design.create_responsive_font_size(12),
            color=self.get_difficulty_color(challenge['required_level']),
            size_hint_x=0.4
        )
        icon_layout.add_widget(difficulty_label)
        
        header_layout.add_widget(icon_layout)
        
        # Challenge title and points
        title_layout = BoxLayout(orientation='vertical', size_hint_x=0.7)
        
        title_label = Label(
            text=challenge['title'],
            font_size=responsive_design.create_responsive_font_size(16),
            bold=True,
            color=(1, 1, 1, 1),
            halign='left',
            text_size=(None, None)
        )
        title_layout.add_widget(title_label)
        
        points_label = Label(
            text=f"{self.aed_handler.format_aed(challenge['points'], compact=True)} points",
            font_size=responsive_design.create_responsive_font_size(14),
            color=self.gold_color,
            halign='left'
        )
        title_layout.add_widget(points_label)
        
        header_layout.add_widget(title_layout)
        
        card_container.add_widget(header_layout)
        
        # Challenge description
        description_label = Label(
            text=challenge['description'],
            font_size=responsive_design.create_responsive_font_size(12),
            color=(0.9, 0.9, 0.9, 1),
            size_hint_y=0.3,
            halign='left',
            valign='middle',
            text_size=(None, None)
        )
        card_container.add_widget(description_label)
        
        # Action buttons
        action_layout = BoxLayout(orientation='horizontal', size_hint_y=0.4, spacing=dp(10))
        
        # Preview/Info button (for smaller screens)
        if responsive_design.current_breakpoint in ['xs', 'sm']:
            info_button = mobile_ux_enhancer.create_mobile_optimized_button(
                "ℹ️",
                lambda x: self.show_challenge_info(challenge),
                size_hint_y=None
            )
            info_button.height = dp(35)
            info_button.size_hint_x = 0.2
            action_layout.add_widget(info_button)
        
        # Main action button
        action_button = mobile_ux_enhancer.create_mobile_optimized_button(
            "Start Challenge",
            lambda x: self.start_challenge(challenge),
            size_hint_y=None
        )
        action_button.height = dp(35)
        action_button.background_color = (0.2, 0.8, 0.2, 1)
        action_layout.add_widget(action_button)
        
        card_container.add_widget(action_layout)
        
        return card_container
    
    def get_difficulty_color(self, level):
        """Get color for difficulty indicator"""
        colors = {
            1: (0.0, 1.0, 0.0, 1),  # Green - Easy
            2: (1.0, 1.0, 0.0, 1),  # Yellow - Medium
            3: (1.0, 0.5, 0.0, 1),  # Orange - Hard
            4: (1.0, 0.0, 0.0, 1),  # Red - Very Hard
            5: (0.8, 0.0, 1.0, 1),  # Purple - Expert
        }
        return colors.get(level, (0.5, 0.5, 0.5, 1))
    
    def create_no_challenges_card(self):
        """Create card for when no challenges are available"""
        return mobile_ux_enhancer.create_mobile_card(
            "All Done! 🎉",
            "You've completed all today's challenges!\nCome back tomorrow for new challenges.\n\nKeep your streak alive!",
            "✅"
        )
    
    def create_performance_summary_card(self):
        """Create performance summary card for larger screens"""
        kpis = self.analytics.get_real_time_kpis()
        
        summary_text = f"📊 Current Performance\n"
        summary_text += f"Occupancy: {kpis['occupancy_display']}\n"
        summary_text += f"ADR: {kpis['adr_display']}\n"
        summary_text += f"RevPAR: {kpis['revpar_display']}"
        
        return mobile_ux_enhancer.create_mobile_card(
            "Hotel Performance",
            summary_text,
            "🏨"
        )
    
    def create_challenge_navigation(self):
        """Create mobile-optimized navigation"""
        nav_layout = BoxLayout(
            orientation='horizontal',
            spacing=responsive_design.get_responsive_value(dp(10), 'spacing'),
            padding=responsive_design.get_responsive_value(dp(15), 'padding')
        )
        
        # Back button
        back_button = mobile_ux_enhancer.create_mobile_optimized_button(
            "← Back",
            self.go_back
        )
        nav_layout.add_widget(back_button)
        
        # Quick stats button (for larger screens)
        if self.layout_config['show_secondary_info']:
            stats_button = mobile_ux_enhancer.create_mobile_optimized_button(
                "📊 Stats",
                self.show_quick_stats
            )
            nav_layout.add_widget(stats_button)
        
        # Refresh button
        refresh_button = mobile_ux_enhancer.create_mobile_optimized_button(
            "🔄 Refresh",
            lambda x: self.refresh_challenges()
        )
        nav_layout.add_widget(refresh_button)
        
        return nav_layout
    
    def show_challenge_info(self, challenge):
        """Show detailed challenge information"""
        content = BoxLayout(
            orientation='vertical',
            spacing=responsive_design.get_responsive_value(dp(15), 'spacing'),
            padding=responsive_design.get_responsive_value(dp(20), 'padding')
        )
        
        # Challenge header
        header_layout = BoxLayout(orientation='horizontal', size_hint_y=0.2, spacing=dp(10))
        
        icon_label = Label(
            text=challenge['icon'],
            font_size=responsive_design.create_responsive_font_size(32),
            size_hint_x=0.3
        )
        header_layout.add_widget(icon_label)
        
        title_layout = BoxLayout(orientation='vertical', size_hint_x=0.7)
        
        title_label = Label(
            text=challenge['title'],
            font_size=responsive_design.create_responsive_font_size(18),
            bold=True,
            color=(1, 1, 1, 1),
            halign='left'
        )
        title_layout.add_widget(title_label)
        
        points_label = Label(
            text=f"Reward: {self.aed_handler.format_aed(challenge['points'], compact=True)} points",
            font_size=responsive_design.create_responsive_font_size(14),
            color=self.gold_color,
            halign='left'
        )
        title_layout.add_widget(points_label)
        
        header_layout.add_widget(title_layout)
        content.add_widget(header_layout)
        
        # Challenge details
        details_text = f"{challenge['description']}\n\n"
        
        # Get current performance for context
        if challenge['id'] == 'daily_occupancy':
            kpis = self.analytics.get_real_time_kpis()
            current_occupancy = kpis['occupancy_pct']
            target = challenge.get('target', 85)
            
            details_text += f"Current Occupancy: {current_occupancy:.1f}%\n"
            details_text += f"Target: {target:.1f}%\n"
            
            if current_occupancy >= target:
                details_text += "🎉 Target achieved! Tap Start to claim points."
            else:
                details_text += f"Need {target - current_occupancy:.1f}% more to complete."
        
        details_label = Label(
            text=details_text,
            font_size=responsive_design.create_responsive_font_size(12),
            halign='center',
            valign='middle',
            color=(0.9, 0.9, 0.9, 1),
            size_hint_y=0.6
        )
        content.add_widget(details_label)
        
        # Action buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=0.2, spacing=dp(10))
        
        close_btn = mobile_ux_enhancer.create_mobile_optimized_button(
            "Close",
            lambda x: popup.dismiss()
        )
        button_layout.add_widget(close_btn)
        
        start_btn = mobile_ux_enhancer.create_mobile_optimized_button(
            "Start Challenge",
            lambda x: [popup.dismiss(), self.start_challenge(challenge)]
        )
        start_btn.background_color = (0.2, 0.8, 0.2, 1)
        button_layout.add_widget(start_btn)
        
        content.add_widget(button_layout)
        
        popup = Popup(
            title="Challenge Details",
            content=content,
            size_hint=(0.9, 0.7)
        )
        popup.open()
    
    def start_challenge(self, challenge):
        """Start a specific challenge"""
        # Determine game mode based on challenge type
        game_mode_map = {
            'daily_occupancy': GameMode.DAILY_CHALLENGE,
            'revenue_target': GameMode.DAILY_CHALLENGE,
            'segment_optimization': GameMode.SEGMENT_CONQUEST,
            'pricing_game': GameMode.PRICING_MASTER,
            'block_booking': GameMode.BLOCK_BOOKING
        }
        
        game_mode = game_mode_map.get(challenge['id'], GameMode.DAILY_CHALLENGE)
        
        # Start challenge through flow manager
        challenge_data = self.flow_manager.start_challenge(challenge['id'], game_mode)
        
        if 'error' in challenge_data:
            self.show_error_popup(challenge_data['error'])
            return
        
        # Navigate to appropriate screen based on challenge type
        screen_map = {
            'daily_occupancy': self.show_occupancy_challenge,
            'revenue_target': self.show_revenue_challenge,
            'segment_optimization': lambda: setattr(self.manager, 'current', 'segment_game'),
            'pricing_game': lambda: setattr(self.manager, 'current', 'pricing_game'),
            'block_booking': lambda: setattr(self.manager, 'current', 'block_game')
        }
        
        action = screen_map.get(challenge['id'], self.show_generic_challenge)
        if callable(action):
            action()
        else:
            action(challenge_data)
    
    def show_occupancy_challenge(self):
        """Show occupancy challenge popup"""
        kpis = self.analytics.get_real_time_kpis()
        current_occupancy = kpis['occupancy_pct']
        target = 85.0  # Default target
        
        content = BoxLayout(
            orientation='vertical',
            spacing=responsive_design.get_responsive_value(dp(15), 'spacing'),
            padding=responsive_design.get_responsive_value(dp(20), 'padding')
        )
        
        # Challenge header
        header_label = Label(
            text="🏨 Occupancy Challenge",
            font_size=responsive_design.create_responsive_font_size(18),
            bold=True,
            color=self.gold_color,
            size_hint_y=0.2
        )
        content.add_widget(header_label)
        
        # Progress visualization
        progress_layout = BoxLayout(orientation='vertical', size_hint_y=0.4, spacing=dp(10))
        
        stats_text = f"Current Occupancy: {current_occupancy:.1f}%\nTarget: {target:.1f}%"
        stats_label = Label(
            text=stats_text,
            font_size=responsive_design.create_responsive_font_size(14),
            halign='center'
        )
        progress_layout.add_widget(stats_label)
        
        # Progress bar
        progress_bar = ProgressBar(
            max=100,
            value=min(current_occupancy, 100),
            size_hint_y=0.3
        )
        progress_layout.add_widget(progress_bar)
        
        content.add_widget(progress_layout)
        
        # Result and action
        result_layout = BoxLayout(orientation='vertical', size_hint_y=0.3, spacing=dp(10))
        
        if current_occupancy >= target:
            # Challenge completed
            result_label = Label(
                text="🎉 Challenge Complete!",
                font_size=responsive_design.create_responsive_font_size(16),
                color=(0, 1, 0, 1),
                bold=True
            )
            result_layout.add_widget(result_label)
            
            # Award points
            score = self.analytics.calculate_daily_challenge_score(target)
            result = self.flow_manager.game_state.complete_daily_challenge('daily_occupancy', score)
            
            if result:
                points_label = Label(
                    text=f"Earned: {result['final_score']:,} points!",
                    font_size=responsive_design.create_responsive_font_size(14),
                    color=self.gold_color
                )
                result_layout.add_widget(points_label)
                
                # Trigger haptic feedback
                touch_optimizer.trigger_haptic_feedback('heavy')
        else:
            # Challenge not complete
            remaining = target - current_occupancy
            result_label = Label(
                text=f"Need {remaining:.1f}% more!\nKeep optimizing your hotel's performance.",
                font_size=responsive_design.create_responsive_font_size(14),
                color=(1, 0.5, 0, 1),
                halign='center'
            )
            result_layout.add_widget(result_label)
        
        content.add_widget(result_layout)
        
        # Close button
        close_btn = mobile_ux_enhancer.create_mobile_optimized_button(
            "Close",
            lambda x: popup.dismiss(),
            size_hint_y=0.1
        )
        content.add_widget(close_btn)
        
        popup = Popup(
            title="Occupancy Challenge",
            content=content,
            size_hint=(0.85, 0.6)
        )
        popup.open()
    
    def show_revenue_challenge(self):
        """Show revenue challenge (placeholder)"""
        self.show_info_popup("Revenue challenge coming soon!")
    
    def show_generic_challenge(self):
        """Show generic challenge (placeholder)"""
        self.show_info_popup("Challenge feature coming soon!")
    
    def show_error_popup(self, error_message):
        """Show error popup"""
        content = BoxLayout(orientation='vertical', spacing=dp(15), padding=dp(20))
        
        error_label = Label(
            text=f"❌ Error\n\n{error_message}",
            font_size=responsive_design.create_responsive_font_size(14),
            halign='center',
            color=(1, 0.5, 0.5, 1)
        )
        content.add_widget(error_label)
        
        ok_btn = mobile_ux_enhancer.create_mobile_optimized_button("OK", lambda x: popup.dismiss())
        content.add_widget(ok_btn)
        
        popup = Popup(title="Error", content=content, size_hint=(0.7, 0.4))
        popup.open()
    
    def show_info_popup(self, message):
        """Show info popup"""
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
    
    def show_quick_stats(self, instance):
        """Show quick performance stats"""
        kpis = self.analytics.get_real_time_kpis()
        
        stats_text = f"""📊 Quick Stats
        
🏨 Occupancy: {kpis['occupancy_display']}
💎 ADR: {kpis['adr_display']}
📈 RevPAR: {kpis['revpar_display']}
💰 Revenue: {kpis['revenue_display']}"""
        
        content = BoxLayout(orientation='vertical', spacing=dp(15), padding=dp(20))
        
        stats_label = Label(
            text=stats_text,
            font_size=responsive_design.create_responsive_font_size(12),
            halign='left',
            color=(0.9, 0.9, 0.9, 1)
        )
        content.add_widget(stats_label)
        
        close_btn = mobile_ux_enhancer.create_mobile_optimized_button("Close", lambda x: popup.dismiss())
        content.add_widget(close_btn)
        
        popup = Popup(title="Performance Stats", content=content, size_hint=(0.8, 0.6))
        popup.open()
    
    def next_challenge(self):
        """Navigate to next challenge (swipe gesture)"""
        # This would cycle through challenges if multiple are shown
        touch_optimizer.trigger_haptic_feedback('light')
    
    def previous_challenge(self):
        """Navigate to previous challenge (swipe gesture)"""
        # This would cycle through challenges if multiple are shown  
        touch_optimizer.trigger_haptic_feedback('light')
    
    def refresh_challenges(self):
        """Refresh challenge data"""
        # Clear cache and rebuild UI
        self.analytics.clear_cache()
        self.clear_widgets()
        self.build_ui()
        
        # Haptic feedback
        touch_optimizer.trigger_haptic_feedback('medium')
    
    def go_back(self, instance):
        """Navigate back to main menu"""
        self.manager.current = 'mobile_main_menu'
    
    def update_card_background(self, instance, value):
        """Update card background when size changes"""
        instance.canvas.before.children[-1].size = instance.size
        instance.canvas.before.children[-1].pos = instance.pos

if __name__ == "__main__":
    print("🎯 Mobile Optimized Challenge Screens Test")
    print("=" * 50)
    
    # Test responsive design
    layout_config = responsive_design.get_layout_config()
    print(f"Screen category: {responsive_design.current_breakpoint}")
    print(f"Card height: {layout_config['card_height']}px")
    print(f"Show secondary info: {layout_config['show_secondary_info']}")
    
    # Test challenge system integration
    flow_manager = get_game_flow_manager()
    challenges = flow_manager.game_state.get_available_challenges()
    print(f"Available challenges: {len(challenges)}")
    
    # Test mobile optimizations
    print(f"Touch target size: {touch_optimizer.RECOMMENDED_TOUCH_SIZE}px")
    print(f"Haptic feedback available: {touch_optimizer.is_mobile}")
    
    print("\n✅ Mobile Challenge Screens Ready!")