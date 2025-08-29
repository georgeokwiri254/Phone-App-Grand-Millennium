"""
Grand Millennium Revenue Analytics - Main Kivy Game Application

Mobile game interface that transforms revenue analytics into an engaging
touch-friendly experience with AED currency and gamification elements.
"""

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.progressbar import ProgressBar
from kivy.uix.slider import Slider
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.core.window import Window
from datetime import timedelta, datetime
from converters.block_converter import run_block_conversion

from core_analytics import RevenueAnalytics
from game_state import get_game_state

# Set mobile-optimized window size for testing
Window.size = (360, 640)  # Standard mobile resolution

class GameScreen(Screen):
    """Base class for all game screens with common styling"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analytics = RevenueAnalytics()
        self.game_state = get_game_state()
        
        # Common styling
        self.bg_color = (0.05, 0.1, 0.2, 1)  # Dark blue
        self.accent_color = (0.2, 0.6, 1.0, 1)  # Light blue
        self.gold_color = (1.0, 0.84, 0.0, 1)  # Gold for AED
        
        with self.canvas.before:
            Color(*self.bg_color)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        
        self.bind(size=self._update_rect, pos=self._update_rect)
    
    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos
    
    def create_header(self, title, subtitle=None):
        """Create styled header for screens"""
        header = BoxLayout(orientation='vertical', size_hint_y=0.2, spacing=dp(5))
        
        # Title
        title_label = Label(
            text=title,
            font_size='24sp',
            bold=True,
            color=self.gold_color,
            size_hint_y=0.7
        )
        header.add_widget(title_label)
        
        # Subtitle
        if subtitle:
            subtitle_label = Label(
                text=subtitle,
                font_size='16sp',
                color=(0.8, 0.8, 0.8, 1),
                size_hint_y=0.3
            )
            header.add_widget(subtitle_label)
        
        return header
    
    def create_aed_display(self, amount, label="Revenue"):
        """Create styled AED amount display"""
        container = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(60))
        
        # Currency icon
        icon_label = Label(
            text="د.إ",
            font_size='20sp',
            color=self.gold_color,
            size_hint_x=0.2
        )
        container.add_widget(icon_label)
        
        # Amount
        amount_layout = BoxLayout(orientation='vertical')
        
        amount_label = Label(
            text=f"{amount:,.0f}",
            font_size='18sp',
            bold=True,
            color=self.gold_color
        )
        amount_layout.add_widget(amount_label)
        
        label_text = Label(
            text=label,
            font_size='12sp',
            color=(0.7, 0.7, 0.7, 1)
        )
        amount_layout.add_widget(label_text)
        
        container.add_widget(amount_layout)
        
        return container
    
    def create_game_button(self, text, callback, color=None):
        """Create styled game button"""
        if color is None:
            color = self.accent_color
        
        btn = Button(
            text=text,
            size_hint_y=None,
            height=dp(50),
            background_color=color,
            font_size='16sp'
        )
        btn.bind(on_press=callback)
        return btn

class MainMenuScreen(GameScreen):
    """Main menu with navigation to game features"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'main_menu'
        self.build_ui()
    
    def build_ui(self):
        root_layout = FloatLayout()

        # Background Image
        background_image = Image(
            source='/home/gee_devops254/Downloads/Revenue App/Pictures/Grand Millennium Dubai Hotel Exterior 2.jpg',
            allow_stretch=True,
            keep_ratio=False,
            size_hint=(1, 1)
        )
        root_layout.add_widget(background_image)

        # Overlay with a semi-transparent background for readability
        overlay = BoxLayout(
            size_hint=(1, 1),
            orientation='vertical',
            padding=dp(20),
            spacing=dp(15)
        )
        with overlay.canvas.before:
            Color(0, 0, 0, 0.5) # Semi-transparent black
            self.rect = Rectangle(size=overlay.size, pos=overlay.pos)
        overlay.bind(size=self._update_rect, pos=self._update_rect)
        
        main_layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))
        
        # Header with logo
        header = BoxLayout(orientation='vertical', size_hint_y=0.3, spacing=dp(10))
        
        # Logo
        logo = Image(
            source='/home/gee_devops254/Downloads/Revenue App/Pictures/xkrIXuKn_400x400.jpg',
            size_hint_y=0.6
        )
        header.add_widget(logo)
        
        # Title
        title = Label(
            text="Grand Millennium\nRevenue Analytics",
            font_size='20sp',
            bold=True,
            color=self.gold_color,
            halign='center',
            size_hint_y=0.4
        )
        header.add_widget(title)
        
        main_layout.add_widget(header)
        
        # Player stats
        stats = self.create_player_stats()
        main_layout.add_widget(stats)
        
        # Menu options
        menu_scroll = ScrollView(size_hint_y=0.5)
        menu_layout = GridLayout(cols=2, spacing=dp(10), size_hint_y=None)
        menu_layout.bind(minimum_height=menu_layout.setter('height'))
        
        # Menu buttons based on unlocked features
        menu_items = self.get_menu_items()
        for item in menu_items:
            btn = self.create_menu_button(item['title'], item['icon'], item['callback'])
            menu_layout.add_widget(btn)
        
        menu_scroll.add_widget(menu_layout)
        main_layout.add_widget(menu_scroll)

        overlay.add_widget(main_layout)
        root_layout.add_widget(overlay)
        
        self.add_widget(root_layout)
    
    def create_player_stats(self):
        """Create player statistics display"""
        stats_data = self.game_state.get_player_stats()
        
        stats_layout = BoxLayout(orientation='vertical', size_hint_y=0.2, spacing=dp(5))
        
        # Level and points
        level_layout = BoxLayout(orientation='horizontal')
        
        level_label = Label(
            text=f"Level {stats_data['level']}: {stats_data['level_name']}",
            font_size='16sp',
            color=self.accent_color,
            size_hint_x=0.7
        )
        level_layout.add_widget(level_label)
        
        points_label = Label(
            text=f"{stats_data['total_points']:,} pts",
            font_size='14sp',
            color=self.gold_color,
            size_hint_x=0.3
        )
        level_layout.add_widget(points_label)
        
        stats_layout.add_widget(level_layout)
        
        # Progress bar to next level
        if stats_data['points_to_next_level'] > 0:
            progress_layout = BoxLayout(orientation='vertical', size_hint_y=0.4)
            
            progress_label = Label(
                text=f"{stats_data['points_to_next_level']:,} points to next level",
                font_size='12sp',
                color=(0.7, 0.7, 0.7, 1),
                size_hint_y=0.3
            )
            progress_layout.add_widget(progress_label)
            
            progress = ProgressBar(
                max=100,
                value=((stats_data['total_points'] % 50000) / 50000) * 100,
                size_hint_y=0.7
            )
            progress_layout.add_widget(progress)
            
            stats_layout.add_widget(progress_layout)
        
        # Daily streak
        streak_label = Label(
            text=f"🔥 {stats_data['daily_streak']} day streak",
            font_size='12sp',
            color=(1.0, 0.5, 0.0, 1),
            size_hint_y=0.3
        )
        stats_layout.add_widget(streak_label)
        
        return stats_layout
    
    def get_menu_items(self):
        """Get available menu items based on unlocked features"""
        unlocked = self.game_state.state['unlocked_features']
        
        all_items = [
            {'title': 'Game Dashboard', 'icon': '🕹️', 'callback': self.go_to_game_dashboard, 'feature': 'Dashboard'},
            {'title': 'Dashboard', 'icon': '📊', 'callback': self.go_to_dashboard, 'feature': 'Dashboard'},
            {'title': 'Daily Challenge', 'icon': '🎯', 'callback': self.go_to_daily_challenge, 'feature': 'Daily Challenges'},
            {'title': 'Segment Game', 'icon': '👥', 'callback': self.go_to_segment_game, 'feature': 'Segment Analysis'},
            {'title': 'Pricing Master', 'icon': '💎', 'callback': self.go_to_pricing_game, 'feature': 'ADR Optimization'},
            {'title': 'Block Bookings', 'icon': '🏢', 'callback': self.go_to_block_game, 'feature': 'Block Analysis'},
            {'title': 'Forecasting', 'icon': '🔮', 'callback': self.go_to_forecast_game, 'feature': 'Forecasting Games'},
            {'title': 'Leaderboard', 'icon': '🏆', 'callback': self.go_to_leaderboard, 'feature': 'Leaderboards'},
            {'title': 'Settings', 'icon': '⚙️', 'callback': self.go_to_settings, 'feature': 'Dashboard'},
        ]
        
        return [item for item in all_items if item['feature'] in unlocked]
    
    def create_menu_button(self, title, icon, callback):
        """Create menu button with icon and text"""
        btn_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=dp(80))
        
        btn = Button(
            text=f"{icon}\n{title}",
            font_size='14sp',
            background_color=self.accent_color,
            size_hint_y=1
        )
        btn.bind(on_press=lambda x: callback())
        
        btn_layout.add_widget(btn)
        return btn_layout
    
    def go_to_game_dashboard(self):
        self.manager.current = 'game_dashboard'

    def go_to_dashboard(self):
        self.manager.current = 'dashboard'
    
    def go_to_daily_challenge(self):
        self.manager.current = 'daily_challenge'
    
    def go_to_segment_game(self):
        self.manager.current = 'segment_game'
    
    def go_to_pricing_game(self):
        self.manager.current = 'pricing_game'
    
    def go_to_block_game(self):
        self.manager.current = 'block_game'
    
    def go_to_forecast_game(self):
        self.manager.current = 'forecast_game'
    
    def go_to_leaderboard(self):
        self.manager.current = 'leaderboard'
    
    def go_to_settings(self):
        self.manager.current = 'settings'

class DashboardScreen(GameScreen):
    """Mission Control Center - Real-time KPI dashboard"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'dashboard'
        self.build_ui()
        
        # Auto-refresh every 30 seconds
        Clock.schedule_interval(self.refresh_data, 30)
    
    def build_ui(self):
        main_layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))
        
        # Header
        header = self.create_header("Mission Control", "Today's Performance")
        main_layout.add_widget(header)
        
        # KPI Cards
        kpi_scroll = ScrollView(size_hint_y=0.7)
        kpi_layout = BoxLayout(orientation='vertical', spacing=dp(15), size_hint_y=None)
        kpi_layout.bind(minimum_height=kpi_layout.setter('height'))
        
        # Get today's stats
        daily_stats = self.analytics.get_daily_occupancy_stats()
        
        # Occupancy Card
        occupancy_card = self.create_kpi_card(
            "Hotel Occupancy",
            f"{daily_stats.get('occupancy_pct', 0):.1f}%",
            "🏨",
            self.get_occupancy_status(daily_stats.get('occupancy_pct', 0))
        )
        kpi_layout.add_widget(occupancy_card)
        
        # Revenue Card
        revpar = daily_stats.get('revpar_aed', 0)
        revenue_card = self.create_kpi_card(
            "RevPAR Today",
            self.analytics.format_aed_currency(revpar),
            "💰",
            self.get_revenue_status(revpar)
        )
        kpi_layout.add_widget(revenue_card)
        
        # ADR Card
        adr = daily_stats.get('adr_aed', 0)
        adr_card = self.create_kpi_card(
            "Average Rate",
            self.analytics.format_aed_currency(adr),
            "💎",
            "tracking"
        )
        kpi_layout.add_widget(adr_card)
        
        # Daily Challenge Progress
        challenge_card = self.create_challenge_progress_card()
        kpi_layout.add_widget(challenge_card)
        
        kpi_scroll.add_widget(kpi_layout)
        main_layout.add_widget(kpi_scroll)
        
        # Navigation
        nav_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1, spacing=dp(10))
        
        back_btn = self.create_game_button("← Menu", self.go_back)
        nav_layout.add_widget(back_btn)
        
        challenge_btn = self.create_game_button("Start Challenge", self.start_challenge, self.gold_color)
        nav_layout.add_widget(challenge_btn)
        
        main_layout.add_widget(nav_layout)
        
        self.add_widget(main_layout)
    
    def create_kpi_card(self, title, value, icon, status):
        """Create KPI card with animation potential"""
        card_layout = BoxLayout(
            orientation='horizontal', 
            size_hint_y=None, 
            height=dp(80),
            spacing=dp(15)
        )
        
        # Add rounded rectangle background
        with card_layout.canvas.before:
            Color(0.15, 0.2, 0.35, 1)  # Card background
            RoundedRectangle(size=card_layout.size, pos=card_layout.pos, radius=[10])
        
        card_layout.bind(size=self._update_card_rect, pos=self._update_card_rect)
        
        # Icon
        icon_label = Label(
            text=icon,
            font_size='32sp',
            size_hint_x=0.2
        )
        card_layout.add_widget(icon_label)
        
        # Content
        content_layout = BoxLayout(orientation='vertical', size_hint_x=0.6)
        
        title_label = Label(
            text=title,
            font_size='14sp',
            color=(0.8, 0.8, 0.8, 1),
            halign='left'
        )
        content_layout.add_widget(title_label)
        
        value_label = Label(
            text=value,
            font_size='20sp',
            bold=True,
            color=self.gold_color,
            halign='left'
        )
        content_layout.add_widget(value_label)
        
        card_layout.add_widget(content_layout)
        
        # Status indicator
        status_colors = {
            'excellent': (0.0, 1.0, 0.0, 1),
            'good': (1.0, 1.0, 0.0, 1),
            'warning': (1.0, 0.5, 0.0, 1),
            'tracking': (0.5, 0.5, 1.0, 1)
        }
        
        status_label = Label(
            text="●",
            font_size='20sp',
            color=status_colors.get(status, (0.5, 0.5, 0.5, 1)),
            size_hint_x=0.2
        )
        card_layout.add_widget(status_label)
        
        return card_layout
    
    def _update_card_rect(self, instance, value):
        instance.canvas.before.children[-1].size = instance.size
        instance.canvas.before.children[-1].pos = instance.pos
    
    def get_occupancy_status(self, occupancy):
        """Get occupancy status for color coding"""
        if occupancy >= 90:
            return 'excellent'
        elif occupancy >= 80:
            return 'good'
        elif occupancy >= 70:
            return 'warning'
        else:
            return 'tracking'
    
    def get_revenue_status(self, revpar):
        """Get revenue status for color coding"""
        if revpar >= 400:
            return 'excellent'
        elif revpar >= 300:
            return 'good'
        elif revpar >= 200:
            return 'warning'
        else:
            return 'tracking'
    
    def create_challenge_progress_card(self):
        """Create daily challenge progress display"""
        challenges = self.game_state.get_available_challenges()
        
        card_layout = BoxLayout(
            orientation='vertical', 
            size_hint_y=None, 
            height=dp(100),
            spacing=dp(10),
            padding=dp(15)
        )
        
        with card_layout.canvas.before:
            Color(0.1, 0.3, 0.1, 1)  # Green card background
            RoundedRectangle(size=card_layout.size, pos=card_layout.pos, radius=[10])
        
        card_layout.bind(size=self._update_card_rect, pos=self._update_card_rect)
        
        title_layout = BoxLayout(orientation='horizontal')
        
        title_label = Label(
            text="🎯 Today's Challenges",
            font_size='16sp',
            bold=True,
            color=(1, 1, 1, 1),
            size_hint_x=0.8
        )
        title_layout.add_widget(title_label)
        
        count_label = Label(
            text=f"{len(challenges)} available",
            font_size='12sp',
            color=(0.8, 1, 0.8, 1),
            size_hint_x=0.2
        )
        title_layout.add_widget(count_label)
        
        card_layout.add_widget(title_layout)
        
        if challenges:
            challenge = challenges[0]  # Show first available challenge
            challenge_label = Label(
                text=f"• {challenge['title']}: {challenge['points']:,} points",
                font_size='14sp',
                color=(0.9, 0.9, 0.9, 1),
                halign='left'
            )
            card_layout.add_widget(challenge_label)
        
        return card_layout
    
    def refresh_data(self, dt):
        """Refresh dashboard data"""
        # This would trigger UI updates with new data
        pass
    
    def go_back(self, instance):
        self.manager.current = 'main_menu'
    
    def start_challenge(self, instance):
        self.manager.current = 'daily_challenge'

class DailyChallengeScreen(GameScreen):
    """Daily occupancy and revenue challenges"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'daily_challenge'
        self.current_challenge = None
        self.build_ui()
    
    def build_ui(self):
        main_layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))
        
        # Header
        header = self.create_header("Daily Challenge", "Achieve today's targets")
        main_layout.add_widget(header)
        
        # Challenge selection
        challenges_scroll = ScrollView(size_hint_y=0.6)
        challenges_layout = BoxLayout(orientation='vertical', spacing=dp(10), size_hint_y=None)
        challenges_layout.bind(minimum_height=challenges_layout.setter('height'))
        
        available_challenges = self.game_state.get_available_challenges()
        
        for challenge in available_challenges:
            challenge_card = self.create_challenge_card(challenge)
            challenges_layout.add_widget(challenge_card)
        
        challenges_scroll.add_widget(challenges_layout)
        main_layout.add_widget(challenges_scroll)
        
        # Navigation
        nav_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1, spacing=dp(10))
        
        back_btn = self.create_game_button("← Dashboard", self.go_back)
        nav_layout.add_widget(back_btn)
        
        refresh_btn = self.create_game_button("Refresh", self.refresh_challenges)
        nav_layout.add_widget(refresh_btn)
        
        main_layout.add_widget(nav_layout)
        
        self.add_widget(main_layout)
    
    def create_challenge_card(self, challenge):
        """Create interactive challenge card"""
        card_layout = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=dp(120),
            spacing=dp(10),
            padding=dp(15)
        )
        
        with card_layout.canvas.before:
            Color(0.2, 0.25, 0.4, 1)
            RoundedRectangle(size=card_layout.size, pos=card_layout.pos, radius=[15])
        
        card_layout.bind(size=self._update_card_rect, pos=self._update_card_rect)
        
        # Header
        header_layout = BoxLayout(orientation='horizontal')
        
        icon_label = Label(
            text=challenge['icon'],
            font_size='24sp',
            size_hint_x=0.15
        )
        header_layout.add_widget(icon_label)
        
        title_label = Label(
            text=challenge['title'],
            font_size='16sp',
            bold=True,
            color=(1, 1, 1, 1),
            halign='left',
            size_hint_x=0.6
        )
        header_layout.add_widget(title_label)
        
        points_label = Label(
            text=f"{challenge['points']:,} pts",
            font_size='14sp',
            color=self.gold_color,
            size_hint_x=0.25
        )
        header_layout.add_widget(points_label)
        
        card_layout.add_widget(header_layout)
        
        # Description
        desc_label = Label(
            text=challenge['description'],
            font_size='12sp',
            color=(0.8, 0.8, 0.8, 1),
            halign='left',
            size_hint_y=0.4
        )
        card_layout.add_widget(desc_label)
        
        # Action button
        play_btn = self.create_game_button(
            "Play Challenge",
            lambda x: self.start_challenge(challenge),
            (0.2, 0.8, 0.2, 1)
        )
        play_btn.size_hint_y = 0.4
        card_layout.add_widget(play_btn)
        
        return card_layout

    def _update_card_rect(self, instance, value):
        instance.canvas.before.children[-1].size = instance.size
        instance.canvas.before.children[-1].pos = instance.pos
    
    def start_challenge(self, challenge):
        """Start specific challenge"""
        self.current_challenge = challenge
        
        if challenge['id'] == 'daily_occupancy':
            self.show_occupancy_challenge()
        elif challenge['id'] == 'revenue_target':
            self.show_revenue_challenge()
        elif challenge['id'] == 'segment_optimization':
            self.manager.current = 'segment_game'
        elif challenge['id'] == 'pricing_game':
            self.manager.current = 'pricing_game'
        elif challenge['id'] == 'block_booking':
            self.manager.current = 'block_game'
    
    def show_occupancy_challenge(self):
        """Show occupancy challenge popup"""
        daily_stats = self.analytics.get_daily_occupancy_stats()
        current_occupancy = daily_stats.get('occupancy_pct', 0)
        target = self.current_challenge['target']
        
        content = BoxLayout(orientation='vertical', spacing=dp(15), padding=dp(20))
        
        title_label = Label(
            text="🏨 Occupancy Challenge",
            font_size='18sp',
            bold=True,
            color=self.gold_color
        )
        content.add_widget(title_label)
        
        stats_label = Label(
            text=f"Current Occupancy: {current_occupancy:.1f}%\nTarget: {target}%",
            font_size='14sp',
            halign='center'
        )
        content.add_widget(stats_label)
        
        if current_occupancy >= target:
            result_label = Label(
                text="🎉 Challenge Complete!",
                font_size='16sp',
                color=(0, 1, 0, 1)
            )
            content.add_widget(result_label)
            
            # Award points
            score = self.analytics.calculate_daily_challenge_score(target)
            result = self.game_state.complete_daily_challenge('daily_occupancy', score)
            
            if result:
                points_label = Label(
                    text=f"Earned: {result['final_score']:,} points!",
                    font_size='14sp',
                    color=self.gold_color
                )
                content.add_widget(points_label)
        else:
            result_label = Label(
                text=f"Need {target - current_occupancy:.1f}% more!",
                font_size='14sp',
                color=(1, 0.5, 0, 1)
            )
            content.add_widget(result_label)
        
        close_btn = Button(text="Close", size_hint_y=0.3)
        content.add_widget(close_btn)
        
        popup = Popup(
            title="Daily Challenge",
            content=content,
            size_hint=(0.8, 0.6)
        )
        
        close_btn.bind(on_press=popup.dismiss)
        popup.open()
    
    def show_revenue_challenge(self):
        """Show revenue challenge popup"""
        # Implementation for revenue challenge
        pass
    
    def refresh_challenges(self, instance):
        """Refresh available challenges"""
        self.clear_widgets()
        self.build_ui()
    
    def go_back(self, instance):
        self.manager.current = 'dashboard'

class SettingsScreen(GameScreen):
    """Placeholder for game settings"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'settings'
        self.build_ui()

    def build_ui(self):
        main_layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))

        # Header
        header = self.create_header("Settings", "Configure your game")
        main_layout.add_widget(header)

        # Coming soon message
        coming_soon_label = Label(
            text="Settings screen is under construction.\nComing soon!",
            font_size='18sp',
            color=(0.8, 0.8, 0.8, 1),
            halign='center'
        )
        main_layout.add_widget(coming_soon_label)

        # Navigation
        nav_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1, spacing=dp(10))
        back_btn = self.create_game_button("← Back to Menu", self.go_to_main_menu)
        nav_layout.add_widget(back_btn)
        main_layout.add_widget(nav_layout)

        self.add_widget(main_layout)

    def go_to_main_menu(self, instance):
        self.manager.current = 'main_menu'

class GameDashboardScreen(GameScreen):
    """The main interactive game screen"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'game_dashboard'
        self.current_date = datetime(2025, 8, 10) # Start date of the simulation
        self.build_ui()

    def build_ui(self):
        main_layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))

        # Header
        self.header_date_label = Label(text=self.current_date.strftime("%d %B %Y"), font_size='20sp', bold=True)
        header = self.create_header("Game Dashboard", "")
        header.add_widget(self.header_date_label)
        main_layout.add_widget(header)

        # Hotel Stats
        stats_layout = GridLayout(cols=2, size_hint_y=0.2)
        self.occupancy_label = Label(text="Occupancy: 0/339", font_size='18sp')
        self.revenue_label = Label(text="Revenue (Aug): 0 AED", font_size='18sp')
        stats_layout.add_widget(self.occupancy_label)
        stats_layout.add_widget(self.revenue_label)
        main_layout.add_widget(stats_layout)

        # Simulation Controls
        controls_layout = BoxLayout(size_hint_y=0.1, spacing=dp(10))
        next_day_btn = self.create_game_button("Next Day", self.next_day)
        promotions_btn = self.create_game_button("Promotions", self.go_to_promotions, color=(0.8, 0.2, 0.2, 1))
        controls_layout.add_widget(next_day_btn)
        controls_layout.add_widget(promotions_btn)
        main_layout.add_widget(controls_layout)
        
        # Navigation
        nav_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1, spacing=dp(10))
        back_btn = self.create_game_button("← Back to Menu", self.go_to_main_menu)
        nav_layout.add_widget(back_btn)
        main_layout.add_widget(nav_layout)

        self.add_widget(main_layout)
        self.update_stats()

    def next_day(self, instance):
        self.current_date += timedelta(days=1)
        self.update_stats()

    def update_stats(self):
        # Update header with new date
        self.header_date_label.text = self.current_date.strftime("%d %B %Y")

        app = App.get_running_app()
        if app.block_data is not None:
            # Filter data for the current day
            daily_bookings = app.block_data[app.block_data['AllotmentDate'].dt.date == self.current_date.date()]
            
            # Calculate occupancy
            rooms_booked = daily_bookings['BlockSize'].sum()
            self.occupancy_label.text = f"Occupancy: {int(rooms_booked)}/339"

            # Calculate revenue for the current month
            monthly_bookings = app.block_data[app.block_data['AllotmentDate'].dt.month == self.current_date.month]
            monthly_revenue = monthly_bookings['Revenue'].sum()
            self.revenue_label.text = f"Revenue ({self.current_date.strftime('%b')}): {monthly_revenue:,.0f} AED"

    def go_to_promotions(self, instance):
        # This will eventually go to a new promotions screen
        pass

    def go_to_main_menu(self, instance):
        self.manager.current = 'main_menu'

class RevenueGameApp(App):
    """Main Kivy application"""

    def build(self):
        self.title = "Grand Millennium Revenue Game"
        self.load_data()
        
        # Screen manager
        sm = ScreenManager()
        
        # Add screens
        sm.add_widget(MainMenuScreen())
        sm.add_widget(DashboardScreen())
        sm.add_widget(DailyChallengeScreen())
        sm.add_widget(SettingsScreen())
        sm.add_widget(GameDashboardScreen())
        
        # Set initial screen
        sm.current = 'main_menu'
        
        return sm

    def load_data(self):
        try:
            file_path = "/home/gee_devops254/Downloads/Revenue App/Block Data.txt"
            self.block_data, _ = run_block_conversion(file_path)
            print("Block data loaded successfully.")
        except Exception as e:
            print(f"Error loading block data: {e}")


if __name__ == '__main__':
    RevenueGameApp().run()