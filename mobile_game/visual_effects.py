"""
Grand Millennium Revenue Analytics - Visual Effects System

Advanced visual effects for mobile game including progress animations,
chart transitions, and UI enhancements with AED currency display.
"""

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

try:
    from kivy.graphics import Color, Rectangle, Ellipse, Line, Triangle, Quad
    from kivy.graphics import PushMatrix, PopMatrix, Rotate, Scale, Translate
    from kivy.animation import Animation
    from kivy.clock import Clock
    from kivy.metrics import dp, sp
    from kivy.utils import get_color_from_hex
    from kivy.uix.widget import Widget
    from kivy.uix.label import Label
    KIVY_AVAILABLE = True
except ImportError:
    KIVY_AVAILABLE = False

class VisualEffectsManager:
    """Manages visual effects and animations for revenue analytics game"""
    
    def __init__(self):
        """Initialize visual effects manager"""
        self.project_root = Path(__file__).parent
        
        # Dubai/UAE themed color palette
        self.color_palette = {
            'gold': get_color_from_hex('#FFD700') if KIVY_AVAILABLE else (1, 0.84, 0, 1),
            'royal_blue': get_color_from_hex('#1E3A8A') if KIVY_AVAILABLE else (0.12, 0.23, 0.54, 1),
            'emerald': get_color_from_hex('#10B981') if KIVY_AVAILABLE else (0.06, 0.73, 0.51, 1),
            'ruby': get_color_from_hex('#EF4444') if KIVY_AVAILABLE else (0.94, 0.27, 0.27, 1),
            'pearl': get_color_from_hex('#F8FAFC') if KIVY_AVAILABLE else (0.97, 0.98, 0.99, 1),
            'sand': get_color_from_hex('#F59E0B') if KIVY_AVAILABLE else (0.96, 0.62, 0.04, 1),
            'ocean': get_color_from_hex('#0EA5E9') if KIVY_AVAILABLE else (0.05, 0.65, 0.91, 1),
            'sunset': get_color_from_hex('#F97316') if KIVY_AVAILABLE else (0.98, 0.45, 0.09, 1)
        }
        
        # Animation configurations
        self.animation_configs = {
            'aed_counter': {
                'duration': 1.5,
                'transition': 'out_cubic',
                'steps': 30
            },
            'progress_bar': {
                'duration': 1.0,
                'transition': 'out_quart',
                'bounce': True
            },
            'chart_reveal': {
                'duration': 2.0,
                'transition': 'out_expo',
                'stagger_delay': 0.1
            },
            'level_transition': {
                'duration': 3.0,
                'transition': 'out_elastic',
                'particle_count': 20
            }
        }
        
        print("✨ Visual Effects Manager Initialized")
    
    def create_aed_counter_animation(self, label_widget, start_amount: float, end_amount: float, 
                                   format_compact: bool = False):
        """Create animated AED currency counter"""
        if not KIVY_AVAILABLE:
            print(f"💰 AED Counter: د.إ {start_amount:,.0f} → د.إ {end_amount:,.0f}")
            return
        
        config = self.animation_configs['aed_counter']
        steps = config['steps']
        duration_per_step = config['duration'] / steps
        
        def format_aed_amount(amount):
            """Format AED amount for display"""
            if format_compact and amount >= 1000:
                if amount >= 1_000_000_000:
                    return f"د.إ {amount/1_000_000_000:.1f}B"
                elif amount >= 1_000_000:
                    return f"د.إ {amount/1_000_000:.1f}M"
                elif amount >= 1_000:
                    return f"د.إ {amount/1_000:.1f}K"
            return f"د.إ {amount:,.0f}"
        
        def update_counter(step):
            """Update counter for each animation step"""
            if step >= steps:
                # Final value
                label_widget.text = format_aed_amount(end_amount)
                return False
            
            # Calculate current value with easing
            progress = step / steps
            # Apply cubic ease-out
            eased_progress = 1 - pow(1 - progress, 3)
            current_amount = start_amount + (end_amount - start_amount) * eased_progress
            
            label_widget.text = format_aed_amount(current_amount)
            return True
        
        # Schedule counter updates
        step_counter = [0]  # Use list to make it mutable in nested function
        
        def animation_step(dt):
            if update_counter(step_counter[0]):
                step_counter[0] += 1
                return True
            return False
        
        Clock.schedule_interval(animation_step, duration_per_step)
        print(f"💰 Started AED counter animation: {format_aed_amount(start_amount)} → {format_aed_amount(end_amount)}")
    
    def create_progress_bar_animation(self, progress_widget, target_progress: float,
                                    show_particles: bool = True):
        """Create animated progress bar with particles"""
        if not KIVY_AVAILABLE:
            print(f"📊 Progress Bar: {target_progress:.1%}")
            return
        
        config = self.animation_configs['progress_bar']
        
        # Create progress animation
        progress_anim = Animation(
            value=target_progress,
            duration=config['duration'],
            transition=config['transition']
        )
        
        # Add bounce effect at the end
        if config['bounce'] and target_progress >= 1.0:
            bounce_anim = Animation(
                value=1.1,
                duration=0.2,
                transition='out_back'
            ) + Animation(
                value=1.0,
                duration=0.3,
                transition='out_elastic'
            )
            progress_anim = progress_anim + bounce_anim
        
        # Start progress animation
        progress_anim.start(progress_widget)
        
        # Add particle effects for milestones
        if show_particles:
            self._add_progress_particles(progress_widget, target_progress)
        
        print(f"📊 Started progress animation to {target_progress:.1%}")
    
    def create_chart_reveal_animation(self, chart_elements: List[Widget]):
        """Create staggered reveal animation for chart elements"""
        if not KIVY_AVAILABLE:
            print("📈 Chart Reveal Animation")
            return
        
        config = self.animation_configs['chart_reveal']
        
        # Initially hide all elements
        for element in chart_elements:
            element.opacity = 0
            element.scale = 0.8
        
        # Staggered reveal
        for i, element in enumerate(chart_elements):
            delay = i * config['stagger_delay']
            
            reveal_anim = Animation(
                opacity=1.0,
                scale=1.0,
                duration=config['duration'],
                transition=config['transition']
            )
            
            Clock.schedule_once(lambda dt, elem=element: reveal_anim.start(elem), delay)
        
        print(f"📈 Started chart reveal for {len(chart_elements)} elements")
    
    def create_level_up_celebration(self, container_widget, level_name: str):
        """Create spectacular level up celebration"""
        if not KIVY_AVAILABLE:
            print(f"🎉 Level Up Celebration: {level_name}")
            return
        
        config = self.animation_configs['level_transition']
        
        # Create celebration label
        celebration_label = Label(
            text=f"Level Up!\n{level_name}",
            font_size=sp(32),
            halign='center',
            color=self.color_palette['gold']
        )
        celebration_label.opacity = 0
        celebration_label.scale = 0.5
        
        container_widget.add_widget(celebration_label)
        
        # Dramatic entrance
        entrance_anim = Animation(
            opacity=1.0,
            scale=1.2,
            duration=0.5,
            transition='out_back'
        ) + Animation(
            scale=1.0,
            duration=0.5,
            transition='out_elastic'
        )
        
        # Hold and exit
        hold_duration = 2.0
        exit_anim = Animation(
            opacity=0,
            scale=0.8,
            duration=0.5,
            transition='in_cubic'
        )
        
        def remove_label(animation, widget):
            try:
                container_widget.remove_widget(widget)
            except:
                pass
        
        full_animation = entrance_anim + Animation(duration=hold_duration) + exit_anim
        full_animation.bind(on_complete=remove_label)
        full_animation.start(celebration_label)
        
        # Add particle burst
        self._create_level_up_particles(container_widget, config['particle_count'])
        
        print(f"🎉 Level up celebration started for: {level_name}")
    
    def create_revenue_wave_effect(self, canvas, wave_color: str = 'ocean'):
        """Create animated wave effect representing revenue flow"""
        if not KIVY_AVAILABLE:
            print("🌊 Revenue Wave Effect")
            return
        
        color = self.color_palette.get(wave_color, self.color_palette['ocean'])
        
        # Wave parameters
        amplitude = dp(30)
        frequency = 0.02
        speed = 2.0
        
        wave_points = []
        width = dp(400)  # Widget width
        
        def update_wave(dt):
            """Update wave animation"""
            nonlocal wave_points
            
            # Generate wave points
            current_time = Clock.get_time() * speed
            points = []
            
            for x in range(0, int(width), 5):
                y = amplitude * math.sin(frequency * x + current_time)
                points.extend([x, y])
            
            # Update canvas (simplified - would need actual Canvas implementation)
            wave_points = points
        
        Clock.schedule_interval(update_wave, 1/60)  # 60 FPS
        print("🌊 Revenue wave effect started")
    
    def create_achievement_badge_animation(self, badge_widget, badge_type: str = 'gold'):
        """Create achievement badge animation"""
        if not KIVY_AVAILABLE:
            print(f"🏆 Achievement Badge: {badge_type}")
            return
        
        badge_color = self.color_palette.get(badge_type, self.color_palette['gold'])
        
        # Initial state
        badge_widget.opacity = 0
        badge_widget.scale = 0
        badge_widget.rotation = -180
        
        # Multi-stage animation
        appear_anim = Animation(
            opacity=1.0,
            scale=1.3,
            rotation=0,
            duration=0.6,
            transition='out_back'
        )
        
        settle_anim = Animation(
            scale=1.0,
            duration=0.4,
            transition='out_elastic'
        )
        
        # Subtle pulsing
        pulse_anim = Animation(
            scale=1.1,
            duration=0.8,
            transition='in_out_sine'
        ) + Animation(
            scale=1.0,
            duration=0.8,
            transition='in_out_sine'
        )
        pulse_anim.repeat = True
        
        # Chain animations
        full_anim = appear_anim + settle_anim
        
        def start_pulse(animation, widget):
            pulse_anim.start(widget)
        
        full_anim.bind(on_complete=start_pulse)
        full_anim.start(badge_widget)
        
        print(f"🏆 Achievement badge animation started: {badge_type}")
    
    def create_data_loading_animation(self, loading_widget):
        """Create elegant data loading animation"""
        if not KIVY_AVAILABLE:
            print("⏳ Data Loading Animation")
            return
        
        # Rotating arc animation
        rotation_anim = Animation(
            rotation=360,
            duration=1.5,
            transition='linear'
        )
        rotation_anim.repeat = True
        
        # Pulsing opacity
        pulse_anim = Animation(
            opacity=0.3,
            duration=1.0,
            transition='in_out_cubic'
        ) + Animation(
            opacity=1.0,
            duration=1.0,
            transition='in_out_cubic'
        )
        pulse_anim.repeat = True
        
        # Start animations
        rotation_anim.start(loading_widget)
        pulse_anim.start(loading_widget)
        
        print("⏳ Data loading animation started")
        
        return {'rotation': rotation_anim, 'pulse': pulse_anim}
    
    def stop_loading_animation(self, loading_widget, animations: Dict):
        """Stop loading animation smoothly"""
        if not KIVY_AVAILABLE or not animations:
            return
        
        # Stop all animations
        for anim in animations.values():
            anim.stop(loading_widget)
        
        # Return to normal state
        reset_anim = Animation(
            opacity=1.0,
            rotation=0,
            duration=0.3,
            transition='out_cubic'
        )
        reset_anim.start(loading_widget)
        
        print("⏳ Loading animation stopped")
    
    def _add_progress_particles(self, progress_widget, target_progress: float):
        """Add particle effects at progress milestones"""
        milestone_points = [0.25, 0.5, 0.75, 1.0]
        
        for milestone in milestone_points:
            if target_progress >= milestone:
                delay = milestone * self.animation_configs['progress_bar']['duration']
                Clock.schedule_once(
                    lambda dt, mp=milestone: self._create_milestone_particles(progress_widget, mp),
                    delay
                )
    
    def _create_milestone_particles(self, widget, milestone: float):
        """Create particles for progress milestone"""
        if not KIVY_AVAILABLE:
            print(f"✨ Milestone particles: {milestone:.0%}")
            return
        
        particle_count = int(milestone * 10) + 5
        colors = [
            self.color_palette['gold'],
            self.color_palette['emerald'],
            self.color_palette['ocean']
        ]
        
        for i in range(particle_count):
            self._create_single_particle(widget, random.choice(colors))
    
    def _create_level_up_particles(self, container, particle_count: int):
        """Create particles for level up celebration"""
        if not KIVY_AVAILABLE:
            return
        
        celebration_colors = [
            self.color_palette['gold'],
            self.color_palette['pearl'],
            self.color_palette['sunset']
        ]
        
        for i in range(particle_count):
            delay = i * 0.05  # Stagger particle creation
            color = random.choice(celebration_colors)
            Clock.schedule_once(
                lambda dt, c=color: self._create_single_particle(container, c),
                delay
            )
    
    def _create_single_particle(self, container, color):
        """Create individual particle with physics"""
        if not KIVY_AVAILABLE:
            return
        
        # Create particle widget
        particle = Widget()
        particle.size = (dp(6), dp(6))
        
        # Random position and velocity
        start_x = container.center_x + random.uniform(-dp(20), dp(20))
        start_y = container.center_y + random.uniform(-dp(20), dp(20))
        
        particle.pos = (start_x, start_y)
        
        # Physics simulation
        velocity_x = random.uniform(-100, 100)
        velocity_y = random.uniform(50, 150)
        gravity = -200
        
        end_x = start_x + velocity_x * 1.5
        end_y = start_y + velocity_y * 1.5 + 0.5 * gravity * 1.5**2
        
        # Animation
        physics_anim = Animation(
            x=end_x,
            y=end_y,
            opacity=0,
            duration=1.5,
            transition='out_cubic'
        )
        
        # Add to container and animate
        container.add_widget(particle)
        
        def remove_particle(animation, widget):
            try:
                container.remove_widget(widget)
            except:
                pass
        
        physics_anim.bind(on_complete=remove_particle)
        physics_anim.start(particle)
    
    def create_segment_transition_effect(self, old_segment: str, new_segment: str, container):
        """Create transition effect between market segments"""
        if not KIVY_AVAILABLE:
            print(f"🔄 Segment Transition: {old_segment} → {new_segment}")
            return
        
        # Create transition overlay
        transition_label = Label(
            text=f"{old_segment}\n↓\n{new_segment}",
            font_size=sp(24),
            halign='center',
            color=self.color_palette['royal_blue']
        )
        
        transition_label.opacity = 0
        container.add_widget(transition_label)
        
        # Slide and fade animation
        slide_anim = Animation(
            opacity=1.0,
            y=container.center_y + dp(50),
            duration=0.5,
            transition='out_cubic'
        ) + Animation(
            y=container.center_y,
            duration=0.5,
            transition='out_back'
        ) + Animation(
            duration=1.0  # Hold
        ) + Animation(
            opacity=0,
            y=container.center_y - dp(50),
            duration=0.5,
            transition='in_cubic'
        )
        
        def cleanup(animation, widget):
            try:
                container.remove_widget(widget)
            except:
                pass
        
        slide_anim.bind(on_complete=cleanup)
        slide_anim.start(transition_label)
        
        print(f"🔄 Segment transition effect: {old_segment} → {new_segment}")
    
    def get_color_palette(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Get the Dubai-themed color palette"""
        return self.color_palette.copy()
    
    def test_visual_effects(self):
        """Test all visual effects"""
        print("\n✨ Testing Visual Effects...")
        print("=" * 40)
        
        if not KIVY_AVAILABLE:
            print("⚠️  Kivy not available - effects will be simulated")
        
        # Test color palette
        print("\n🎨 Color Palette:")
        for color_name, color_value in self.color_palette.items():
            print(f"   {color_name}: {color_value}")
        
        # Test animation configurations
        print("\n⚙️  Animation Configurations:")
        for anim_name, config in self.animation_configs.items():
            print(f"   {anim_name}: {config['duration']}s {config['transition']}")
        
        print("\n✅ Visual effects system ready!")

def get_visual_effects_manager():
    """Get singleton instance of visual effects manager"""
    if not hasattr(get_visual_effects_manager, '_instance'):
        get_visual_effects_manager._instance = VisualEffectsManager()
    return get_visual_effects_manager._instance

# Quick helper functions
def animate_aed_counter(label, start_amount, end_amount, compact=False):
    """Quick helper for AED counter animation"""
    manager = get_visual_effects_manager()
    manager.create_aed_counter_animation(label, start_amount, end_amount, compact)

def animate_progress_bar(progress_widget, target_progress):
    """Quick helper for progress bar animation"""
    manager = get_visual_effects_manager()
    manager.create_progress_bar_animation(progress_widget, target_progress)

def celebrate_level_up(container, level_name):
    """Quick helper for level up celebration"""
    manager = get_visual_effects_manager()
    manager.create_level_up_celebration(container, level_name)

if __name__ == "__main__":
    # Initialize and test visual effects
    effects_manager = VisualEffectsManager()
    effects_manager.test_visual_effects()