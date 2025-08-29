"""
Grand Millennium Revenue Analytics - Additional Game Screens

Specialized game screens for segment analysis, pricing optimization,
and block booking challenges with AED currency integration.
"""

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.progressbar import ProgressBar
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.graphics import Color, Rectangle, RoundedRectangle

from main_game import GameScreen

class SegmentGameScreen(GameScreen):
    """Customer Conquest - Segment Analysis Game"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'segment_game'
        self.budget_sliders = {}
        self.total_budget = 100000  # AED budget for marketing
        self.segments_data = []
        self.build_ui()
    
    def build_ui(self):
        main_layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))
        
        # Header
        header = self.create_header(
            "Customer Conquest",
            f"Allocate د.إ {self.total_budget:,} marketing budget"
        )
        main_layout.add_widget(header)
        
        # Budget display
        budget_display = self.create_budget_display()
        main_layout.add_widget(budget_display)
        
        # Segment allocation sliders
        segments_scroll = ScrollView(size_hint_y=0.5)
        segments_layout = BoxLayout(orientation='vertical', spacing=dp(15), size_hint_y=None)
        segments_layout.bind(minimum_height=segments_layout.setter('height'))
        
        # Get segment data
        self.segments_data = self.analytics.get_segment_performance()
        
        for segment_data in self.segments_data[:6]:  # Top 6 segments
            segment_card = self.create_segment_slider(segment_data)
            segments_layout.add_widget(segment_card)
        
        segments_scroll.add_widget(segments_layout)
        main_layout.add_widget(segments_scroll)
        
        # Action buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1, spacing=dp(10))
        
        back_btn = self.create_game_button("← Back", self.go_back)
        button_layout.add_widget(back_btn)
        
        calculate_btn = self.create_game_button("Calculate ROI", self.calculate_roi, self.gold_color)
        button_layout.add_widget(calculate_btn)
        
        play_btn = self.create_game_button("Execute Strategy", self.execute_strategy, (0, 0.8, 0, 1))
        button_layout.add_widget(play_btn)
        
        main_layout.add_widget(button_layout)
        
        self.add_widget(main_layout)
    
    def create_budget_display(self):
        """Create budget allocation display"""
        budget_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1, spacing=dp(10))
        
        # Total budget
        total_label = Label(
            text=f"Total Budget: د.إ {self.total_budget:,}",
            font_size='16sp',
            color=self.gold_color,
            size_hint_x=0.5
        )
        budget_layout.add_widget(total_label)
        
        # Remaining budget (will be updated dynamically)
        self.remaining_label = Label(
            text="Remaining: د.إ 100,000",
            font_size='14sp',
            color=(0.8, 0.8, 0.8, 1),
            size_hint_x=0.5
        )
        budget_layout.add_widget(self.remaining_label)
        
        return budget_layout
    
    def create_segment_slider(self, segment_data):
        """Create segment allocation slider with performance data"""
        card_layout = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=dp(120),
            spacing=dp(8),
            padding=dp(10)
        )
        
        with card_layout.canvas.before:
            Color(0.15, 0.25, 0.35, 1)
            RoundedRectangle(size=card_layout.size, pos=card_layout.pos, radius=[10])
        
        card_layout.bind(size=self._update_card_rect, pos=self._update_card_rect)
        
        # Segment header
        header_layout = BoxLayout(orientation='horizontal', size_hint_y=0.3)
        
        segment_label = Label(
            text=segment_data['segment'],
            font_size='14sp',
            bold=True,
            color=(1, 1, 1, 1),
            size_hint_x=0.6,
            halign='left'
        )
        header_layout.add_widget(segment_label)
        
        revenue_label = Label(
            text=f"Rev: د.إ {segment_data['revenue_aed']:,.0f}",
            font_size='12sp',
            color=self.gold_color,
            size_hint_x=0.4
        )
        header_layout.add_widget(revenue_label)
        
        card_layout.add_widget(header_layout)
        
        # Performance metrics
        metrics_layout = BoxLayout(orientation='horizontal', size_hint_y=0.25)
        
        bookings_label = Label(
            text=f"{segment_data['bookings']} bookings",
            font_size='10sp',
            color=(0.8, 0.8, 0.8, 1),
            size_hint_x=0.5
        )
        metrics_layout.add_widget(bookings_label)
        
        adr_label = Label(
            text=f"ADR: د.إ {segment_data['adr_aed']:.0f}",
            font_size='10sp',
            color=(0.8, 0.8, 0.8, 1),
            size_hint_x=0.5
        )
        metrics_layout.add_widget(adr_label)
        
        card_layout.add_widget(metrics_layout)
        
        # Budget allocation slider
        slider_layout = BoxLayout(orientation='horizontal', size_hint_y=0.35, spacing=dp(10))
        
        slider = Slider(
            min=0,
            max=50,  # Max 50% of budget per segment
            value=15,  # Default allocation
            step=5,
            size_hint_x=0.7
        )
        
        # Store slider reference
        self.budget_sliders[segment_data['segment']] = slider
        
        slider.bind(value=self.update_budget_display)
        slider_layout.add_widget(slider)
        
        # Percentage display
        self.slider_value_label = Label(
            text="15%",
            font_size='12sp',
            color=self.accent_color,
            size_hint_x=0.3
        )
        slider_layout.add_widget(self.slider_value_label)
        
        card_layout.add_widget(slider_layout)
        
        return card_layout
    
    def update_budget_display(self, instance, value):
        """Update budget display when sliders change"""
        total_allocated = sum(slider.value for slider in self.budget_sliders.values())
        remaining_pct = 100 - total_allocated
        remaining_amount = (remaining_pct / 100) * self.total_budget
        
        self.remaining_label.text = f"Remaining: د.إ {remaining_amount:,.0f} ({remaining_pct:.0f}%)"
        
        # Update individual slider label (need to track which slider changed)
        for segment, slider in self.budget_sliders.items():
            if slider == instance:
                # Find and update the corresponding label
                slider.parent.children[0].text = f"{value:.0f}%"
                break
    
    def calculate_roi(self, instance):
        """Calculate ROI for current budget allocation"""
        allocation = {}
        for segment, slider in self.budget_sliders.items():
            allocation[segment] = slider.value
        
        # Calculate expected return using analytics
        selected_segments = list(allocation.keys())
        score = self.analytics.calculate_segment_strategy_score(selected_segments, allocation)
        
        # Show ROI popup
        self.show_roi_popup(score, allocation)
    
    def show_roi_popup(self, score, allocation):
        """Show ROI calculation results"""
        content = BoxLayout(orientation='vertical', spacing=dp(15), padding=dp(20))
        
        title_label = Label(
            text="📊 ROI Calculation",
            font_size='18sp',
            bold=True,
            color=self.gold_color
        )
        content.add_widget(title_label)
        
        # ROI metrics
        projected_revenue = score * 4  # Simplified multiplier
        total_allocated = sum(allocation.values())
        roi_percentage = (projected_revenue / (total_allocated * 1000)) * 100 if total_allocated > 0 else 0
        
        metrics_label = Label(
            text=f"Projected Revenue: د.إ {projected_revenue:,.0f}\n"
                 f"Budget Allocated: {total_allocated:.0f}%\n"
                 f"ROI: {roi_percentage:.1f}%",
            font_size='14sp',
            halign='center'
        )
        content.add_widget(metrics_label)
        
        # Strategy assessment
        if roi_percentage > 300:
            assessment = "🏆 Excellent strategy!"
            color = (0, 1, 0, 1)
        elif roi_percentage > 200:
            assessment = "👍 Good allocation"
            color = (0.8, 0.8, 0, 1)
        else:
            assessment = "⚠️ Consider rebalancing"
            color = (1, 0.5, 0, 1)
        
        assessment_label = Label(
            text=assessment,
            font_size='16sp',
            color=color
        )
        content.add_widget(assessment_label)
        
        close_btn = Button(text="Close", size_hint_y=0.3)
        content.add_widget(close_btn)
        
        popup = Popup(
            title="ROI Analysis",
            content=content,
            size_hint=(0.85, 0.7)
        )
        
        close_btn.bind(on_press=popup.dismiss)
        popup.open()
    
    def execute_strategy(self, instance):
        """Execute the marketing strategy and award points"""
        allocation = {}
        for segment, slider in self.budget_sliders.items():
            allocation[segment] = slider.value
        
        selected_segments = list(allocation.keys())
        final_score = self.analytics.calculate_segment_strategy_score(selected_segments, allocation)
        
        # Complete challenge and award points
        result = self.game_state.complete_daily_challenge('segment_optimization', final_score)
        
        if result:
            self.show_success_popup(result)
        else:
            self.show_already_completed_popup()
    
    def show_success_popup(self, result):
        """Show successful strategy execution"""
        content = BoxLayout(orientation='vertical', spacing=dp(15), padding=dp(20))
        
        title_label = Label(
            text="🎉 Strategy Executed!",
            font_size='18sp',
            bold=True,
            color=(0, 1, 0, 1)
        )
        content.add_widget(title_label)
        
        points_label = Label(
            text=f"Base Score: {result['base_score']:,}\n"
                 f"Streak Bonus: +{result['streak_bonus']:,}\n"
                 f"Final Score: {result['final_score']:,} points!",
            font_size='14sp',
            halign='center',
            color=self.gold_color
        )
        content.add_widget(points_label)
        
        if result['level_up']:
            level_label = Label(
                text=f"🎊 LEVEL UP! You're now level {result['new_level']}!",
                font_size='16sp',
                color=(1, 0.84, 0, 1)
            )
            content.add_widget(level_label)
        
        continue_btn = Button(text="Continue", size_hint_y=0.3)
        content.add_widget(continue_btn)
        
        popup = Popup(
            title="Success!",
            content=content,
            size_hint=(0.8, 0.6)
        )
        
        continue_btn.bind(on_press=lambda x: [popup.dismiss(), self.go_back(None)])
        popup.open()
    
    def show_already_completed_popup(self):
        """Show message when challenge already completed today"""
        content = BoxLayout(orientation='vertical', spacing=dp(15), padding=dp(20))
        
        title_label = Label(
            text="Already Completed",
            font_size='16sp',
            color=(1, 0.5, 0, 1)
        )
        content.add_widget(title_label)
        
        message_label = Label(
            text="You've already completed this challenge today!\nCome back tomorrow for new challenges.",
            font_size='14sp',
            halign='center'
        )
        content.add_widget(message_label)
        
        ok_btn = Button(text="OK", size_hint_y=0.4)
        content.add_widget(ok_btn)
        
        popup = Popup(
            title="Challenge Status",
            content=content,
            size_hint=(0.7, 0.5)
        )
        
        ok_btn.bind(on_press=popup.dismiss)
        popup.open()
    
    def go_back(self, instance):
        self.manager.current = 'main_menu'

class PricingGameScreen(GameScreen):
    """Pricing Master - ADR Optimization Game"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'pricing_game'
        self.current_adr = 0
        self.proposed_adr = 0
        self.market_conditions = {}
        self.build_ui()
    
    def build_ui(self):
        main_layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))
        
        # Header
        header = self.create_header(
            "Pricing Master",
            "Optimize room rates for maximum RevPAR"
        )
        main_layout.add_widget(header)
        
        # Current market data
        self.market_display = self.create_market_display()
        main_layout.add_widget(self.market_display)
        
        # Pricing slider
        pricing_section = self.create_pricing_section()
        main_layout.add_widget(pricing_section)
        
        # Results display
        self.results_display = self.create_results_display()
        main_layout.add_widget(self.results_display)
        
        # Action buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1, spacing=dp(10))
        
        back_btn = self.create_game_button("← Back", self.go_back)
        button_layout.add_widget(back_btn)
        
        simulate_btn = self.create_game_button("Simulate", self.simulate_pricing, self.accent_color)
        button_layout.add_widget(simulate_btn)
        
        execute_btn = self.create_game_button("Execute", self.execute_pricing, (0, 0.8, 0, 1))
        button_layout.add_widget(execute_btn)
        
        main_layout.add_widget(button_layout)
        
        self.add_widget(main_layout)
        
        # Initialize data
        self.refresh_market_data()
    
    def create_market_display(self):
        """Create market conditions display"""
        market_layout = BoxLayout(
            orientation='vertical',
            size_hint_y=0.25,
            spacing=dp(10),
            padding=dp(15)
        )
        
        with market_layout.canvas.before:
            Color(0.1, 0.2, 0.3, 1)
            RoundedRectangle(size=market_layout.size, pos=market_layout.pos, radius=[10])
        
        market_layout.bind(size=self._update_card_rect, pos=self._update_card_rect)
        
        # Market conditions header
        self.market_title = Label(
            text="📊 Market Conditions: Loading...",
            font_size='16sp',
            bold=True,
            color=self.gold_color,
            size_hint_y=0.3
        )
        market_layout.add_widget(self.market_title)
        
        # Current performance
        self.performance_label = Label(
            text="Current ADR: Loading...\nOccupancy: Loading...",
            font_size='14sp',
            halign='center',
            size_hint_y=0.4
        )
        market_layout.add_widget(self.performance_label)
        
        # Market description
        self.market_desc = Label(
            text="Market analysis loading...",
            font_size='12sp',
            color=(0.8, 0.8, 0.8, 1),
            halign='center',
            size_hint_y=0.3
        )
        market_layout.add_widget(self.market_desc)
        
        return market_layout
    
    def create_pricing_section(self):
        """Create ADR pricing controls"""
        pricing_layout = BoxLayout(
            orientation='vertical',
            size_hint_y=0.2,
            spacing=dp(10),
            padding=dp(15)
        )
        
        # Pricing slider
        slider_layout = BoxLayout(orientation='horizontal', spacing=dp(15))
        
        min_label = Label(text="د.إ 200", font_size='12sp', size_hint_x=0.15)
        slider_layout.add_widget(min_label)
        
        self.adr_slider = Slider(
            min=200,
            max=800,
            value=400,
            step=25,
            size_hint_x=0.6
        )
        self.adr_slider.bind(value=self.update_proposed_adr)
        slider_layout.add_widget(self.adr_slider)
        
        max_label = Label(text="د.إ 800", font_size='12sp', size_hint_x=0.15)
        slider_layout.add_widget(max_label)
        
        pricing_layout.add_widget(slider_layout)
        
        # Current selection display
        self.adr_display = Label(
            text="Proposed ADR: د.إ 400",
            font_size='16sp',
            color=self.gold_color,
            bold=True
        )
        pricing_layout.add_widget(self.adr_display)
        
        return pricing_layout
    
    def create_results_display(self):
        """Create pricing simulation results display"""
        results_layout = BoxLayout(
            orientation='vertical',
            size_hint_y=0.2,
            spacing=dp(10),
            padding=dp(15)
        )
        
        with results_layout.canvas.before:
            Color(0.15, 0.25, 0.15, 1)  # Green tint
            RoundedRectangle(size=results_layout.size, pos=results_layout.pos, radius=[10])
        
        results_layout.bind(size=self._update_card_rect, pos=self._update_card_rect)
        
        self.results_title = Label(
            text="📈 Simulation Results",
            font_size='16sp',
            bold=True,
            color=(0.8, 1, 0.8, 1)
        )
        results_layout.add_widget(self.results_title)
        
        self.results_content = Label(
            text="Adjust pricing and click 'Simulate' to see projected results",
            font_size='12sp',
            halign='center',
            color=(0.9, 0.9, 0.9, 1)
        )
        results_layout.add_widget(self.results_content)
        
        return results_layout
    
    def refresh_market_data(self):
        """Load current market data"""
        # Get real data from analytics
        adr_data = self.analytics.get_adr_optimization_data()
        self.current_adr = adr_data['current_adr']
        current_occupancy = adr_data['current_occupancy']
        self.market_conditions = adr_data['market_conditions']
        
        # Update displays
        self.market_title.text = f"📊 Market: {self.market_conditions['condition']}"
        self.performance_label.text = (
            f"Current ADR: د.إ {self.current_adr:.0f}\n"
            f"Occupancy: {current_occupancy:.1f}%"
        )
        self.market_desc.text = self.market_conditions['description']
        
        # Set slider to current ADR
        self.adr_slider.value = max(200, min(800, self.current_adr))
        self.proposed_adr = self.adr_slider.value
    
    def update_proposed_adr(self, instance, value):
        """Update proposed ADR display"""
        self.proposed_adr = value
        self.adr_display.text = f"Proposed ADR: د.إ {value:.0f}"
        
        # Show rate change
        if self.current_adr > 0:
            change_pct = ((value - self.current_adr) / self.current_adr) * 100
            if abs(change_pct) > 1:
                change_text = f" ({change_pct:+.1f}%)"
                if change_pct > 0:
                    change_color = (0, 1, 0, 1)  # Green for increase
                else:
                    change_color = (1, 0.5, 0, 1)  # Orange for decrease
                self.adr_display.color = change_color
            else:
                self.adr_display.color = self.gold_color
    
    def simulate_pricing(self, instance):
        """Simulate pricing impact"""
        results = self.analytics.calculate_pricing_score(self.proposed_adr, self.market_conditions)
        
        self.results_content.text = (
            f"Projected Occupancy: {results['projected_occupancy']:.1f}%\n"
            f"Projected RevPAR: د.إ {results['projected_revpar']:.0f}\n"
            f"Estimated Score: {results['score']:,} points"
        )
        
        # Color code results
        if results['score'] > 4000:
            self.results_content.color = (0, 1, 0, 1)  # Green - excellent
        elif results['score'] > 3000:
            self.results_content.color = (0.8, 0.8, 0, 1)  # Yellow - good
        else:
            self.results_content.color = (1, 0.5, 0, 1)  # Orange - caution
    
    def execute_pricing(self, instance):
        """Execute pricing strategy"""
        results = self.analytics.calculate_pricing_score(self.proposed_adr, self.market_conditions)
        final_score = results['score']
        
        # Complete challenge
        challenge_result = self.game_state.complete_daily_challenge('pricing_game', final_score)
        
        if challenge_result:
            self.show_pricing_success(challenge_result, results)
        else:
            self.show_already_completed_popup()
    
    def show_pricing_success(self, challenge_result, pricing_results):
        """Show successful pricing execution"""
        content = BoxLayout(orientation='vertical', spacing=dp(15), padding=dp(20))
        
        title_label = Label(
            text="💎 Pricing Executed!",
            font_size='18sp',
            bold=True,
            color=self.gold_color
        )
        content.add_widget(title_label)
        
        # Results summary
        results_label = Label(
            text=f"New ADR: د.إ {self.proposed_adr:.0f}\n"
                 f"Projected RevPAR: د.إ {pricing_results['projected_revpar']:.0f}\n"
                 f"Occupancy Impact: {pricing_results['rate_change_pct']:+.1f}%",
            font_size='14sp',
            halign='center'
        )
        content.add_widget(results_label)
        
        # Points earned
        points_label = Label(
            text=f"Points Earned: {challenge_result['final_score']:,}!",
            font_size='16sp',
            color=(0, 1, 0, 1),
            bold=True
        )
        content.add_widget(points_label)
        
        continue_btn = Button(text="Continue", size_hint_y=0.3)
        content.add_widget(continue_btn)
        
        popup = Popup(
            title="Pricing Success!",
            content=content,
            size_hint=(0.8, 0.6)
        )
        
        continue_btn.bind(on_press=lambda x: [popup.dismiss(), self.go_back(None)])
        popup.open()
    
    def show_already_completed_popup(self):
        """Show message when pricing challenge already completed"""
        content = BoxLayout(orientation='vertical', spacing=dp(15), padding=dp(20))
        
        message_label = Label(
            text="You've already completed the pricing challenge today!\nTry again tomorrow.",
            font_size='14sp',
            halign='center'
        )
        content.add_widget(message_label)
        
        ok_btn = Button(text="OK", size_hint_y=0.4)
        content.add_widget(ok_btn)
        
        popup = Popup(
            title="Already Complete",
            content=content,
            size_hint=(0.7, 0.4)
        )
        
        ok_btn.bind(on_press=popup.dismiss)
        popup.open()
    
    def go_back(self, instance):
        self.manager.current = 'main_menu'