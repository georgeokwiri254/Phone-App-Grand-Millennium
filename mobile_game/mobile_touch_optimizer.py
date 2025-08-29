"""
Grand Millennium Revenue Analytics - Mobile Touch Optimizer

Optimizes touch interactions, responsive design, and mobile-specific
features for the revenue analytics mobile game.
"""

from kivy.core.window import Window
from kivy.metrics import dp, sp
from kivy.clock import Clock
from kivy.animation import Animation
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle, RoundedRectangle, Line
from kivy.utils import platform
from typing import Dict, Tuple, Optional, List
import math

class TouchOptimizer:
    """Mobile touch interaction optimizer"""
    
    # Touch target size guidelines (Material Design)
    MIN_TOUCH_TARGET_SIZE = dp(48)  # Minimum 48dp
    RECOMMENDED_TOUCH_SIZE = dp(56)  # Recommended 56dp
    LARGE_TOUCH_SIZE = dp(72)  # For important actions
    
    # Screen size categories
    SCREEN_SIZES = {
        'small': (320, 480),    # Small phones
        'normal': (360, 640),   # Standard phones  
        'large': (414, 896),    # Large phones
        'xlarge': (768, 1024)   # Tablets
    }
    
    def __init__(self):
        """Initialize touch optimizer"""
        self.current_screen_category = self.detect_screen_category()
        self.touch_feedback_widgets = {}
        self.gesture_recognizers = {}
        
        # Platform-specific optimizations
        self.is_mobile = platform in ('android', 'ios')
        self.setup_platform_optimizations()
    
    def detect_screen_category(self) -> str:
        """Detect current screen size category"""
        width, height = Window.size
        
        # Normalize to portrait orientation
        if width > height:
            width, height = height, width
        
        if width <= 360 and height <= 640:
            return 'small'
        elif width <= 414 and height <= 896:
            return 'normal'
        elif width <= 480 and height <= 1024:
            return 'large'
        else:
            return 'xlarge'
    
    def setup_platform_optimizations(self):
        """Setup platform-specific optimizations"""
        if self.is_mobile:
            # Mobile-specific settings
            Window.softinput_mode = 'below_target'
            Window.allow_screensaver = False
            
            # Adjust for different densities
            self.density_scale = self.get_density_scale()
        else:
            # Desktop testing mode
            self.density_scale = 1.0
    
    def get_density_scale(self) -> float:
        """Get screen density scale factor"""
        # This would use platform-specific APIs in production
        screen_category_scales = {
            'small': 1.0,
            'normal': 1.25,
            'large': 1.5,
            'xlarge': 2.0
        }
        return screen_category_scales.get(self.current_screen_category, 1.0)
    
    def optimize_touch_targets(self, widget: Widget) -> Widget:
        """Optimize widget for touch interaction"""
        # Ensure minimum touch target size
        if hasattr(widget, 'size_hint'):
            if widget.size_hint == (None, None):
                # Fixed size - ensure minimum
                min_width = max(widget.width, self.MIN_TOUCH_TARGET_SIZE)
                min_height = max(widget.height, self.MIN_TOUCH_TARGET_SIZE)
                widget.size = (min_width, min_height)
        
        # Add touch feedback if it's a button
        if isinstance(widget, Button):
            self.add_touch_feedback(widget)
        
        # Add accessibility improvements
        self.add_accessibility_features(widget)
        
        return widget
    
    def add_touch_feedback(self, button: Button):
        """Add visual and haptic touch feedback"""
        original_background = button.background_color[:]
        
        def on_press(instance):
            """Touch down feedback"""
            # Visual feedback - darken button
            darker_color = [c * 0.8 if i < 3 else c for i, c in enumerate(original_background)]
            instance.background_color = darker_color
            
            # Scale animation
            anim = Animation(size=(instance.width * 0.95, instance.height * 0.95), duration=0.1)
            anim.start(instance)
            
            # Haptic feedback (mobile only)
            if self.is_mobile:
                self.trigger_haptic_feedback('light')
        
        def on_release(instance):
            """Touch up feedback"""
            # Restore original color
            instance.background_color = original_background
            
            # Restore size
            anim = Animation(size=(instance.width / 0.95, instance.height / 0.95), duration=0.1)
            anim.start(instance)
        
        button.bind(on_press=on_press)
        button.bind(on_release=on_release)
        
        # Store for cleanup
        self.touch_feedback_widgets[id(button)] = button
    
    def trigger_haptic_feedback(self, intensity: str = 'medium'):
        """Trigger haptic feedback on mobile devices"""
        if not self.is_mobile:
            return
        
        try:
            if platform == 'android':
                # Android haptic feedback
                from jnius import autoclass
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                activity = PythonActivity.mActivity
                
                if intensity == 'light':
                    # Light tap
                    activity.getWindow().getDecorView().performHapticFeedback(3)
                elif intensity == 'medium':
                    # Medium click
                    activity.getWindow().getDecorView().performHapticFeedback(1)
                elif intensity == 'heavy':
                    # Heavy long press
                    activity.getWindow().getDecorView().performHapticFeedback(0)
                    
            elif platform == 'ios':
                # iOS haptic feedback
                from pyobjus import autoclass
                UIImpactFeedbackGenerator = autoclass('UIImpactFeedbackGenerator')
                
                if intensity == 'light':
                    generator = UIImpactFeedbackGenerator.alloc().initWithStyle_(0)  # Light
                elif intensity == 'medium':
                    generator = UIImpactFeedbackGenerator.alloc().initWithStyle_(1)  # Medium
                else:
                    generator = UIImpactFeedbackGenerator.alloc().initWithStyle_(2)  # Heavy
                
                generator.impactOccurred()
                
        except ImportError:
            # Haptic feedback not available
            pass
    
    def add_accessibility_features(self, widget: Widget):
        """Add accessibility features for better usability"""
        # Ensure sufficient contrast
        if isinstance(widget, (Button, Label)) and hasattr(widget, 'color'):
            widget.color = self.ensure_sufficient_contrast(widget.color)
        
        # Add semantic labels for screen readers (future enhancement)
        if hasattr(widget, 'text') and widget.text:
            # This would integrate with platform accessibility APIs
            pass
    
    def ensure_sufficient_contrast(self, color: tuple) -> tuple:
        """Ensure sufficient color contrast for accessibility"""
        # Calculate luminance
        def luminance(rgb):
            def channel_luminance(c):
                if c <= 0.03928:
                    return c / 12.92
                else:
                    return pow((c + 0.055) / 1.055, 2.4)
            
            return 0.2126 * channel_luminance(rgb[0]) + 0.7152 * channel_luminance(rgb[1]) + 0.0722 * channel_luminance(rgb[2])
        
        # If contrast is too low, adjust
        current_luminance = luminance(color[:3])
        if current_luminance < 0.5:  # Dark text, ensure it's dark enough
            return tuple(min(c, 0.2) for c in color[:3]) + (color[3] if len(color) > 3 else (1.0,))
        else:  # Light text, ensure it's light enough
            return tuple(max(c, 0.8) for c in color[:3]) + (color[3] if len(color) > 3 else (1.0,))

class ResponsiveDesign:
    """Responsive design system for different screen sizes"""
    
    def __init__(self):
        """Initialize responsive design system"""
        self.touch_optimizer = TouchOptimizer()
        self.breakpoints = self.setup_breakpoints()
        self.current_breakpoint = self.get_current_breakpoint()
        
        # Responsive scaling factors
        self.scaling_factors = self.setup_scaling_factors()
        
        # Layout configurations
        self.layout_configs = self.setup_layout_configs()
    
    def setup_breakpoints(self) -> Dict[str, int]:
        """Setup responsive design breakpoints"""
        return {
            'xs': 320,   # Extra small phones
            'sm': 360,   # Small phones
            'md': 414,   # Medium phones
            'lg': 768,   # Large phones/small tablets
            'xl': 1024   # Tablets
        }
    
    def get_current_breakpoint(self) -> str:
        """Get current breakpoint based on screen width"""
        width = min(Window.size)  # Use smaller dimension
        
        if width >= self.breakpoints['xl']:
            return 'xl'
        elif width >= self.breakpoints['lg']:
            return 'lg'
        elif width >= self.breakpoints['md']:
            return 'md'
        elif width >= self.breakpoints['sm']:
            return 'sm'
        else:
            return 'xs'
    
    def setup_scaling_factors(self) -> Dict[str, Dict[str, float]]:
        """Setup scaling factors for different screen sizes"""
        return {
            'xs': {
                'font_size': 0.8,
                'padding': 0.8,
                'spacing': 0.8,
                'button_height': 0.9
            },
            'sm': {
                'font_size': 0.9,
                'padding': 0.9,
                'spacing': 0.9,
                'button_height': 0.95
            },
            'md': {
                'font_size': 1.0,
                'padding': 1.0,
                'spacing': 1.0,
                'button_height': 1.0
            },
            'lg': {
                'font_size': 1.1,
                'padding': 1.1,
                'spacing': 1.1,
                'button_height': 1.05
            },
            'xl': {
                'font_size': 1.2,
                'padding': 1.2,
                'spacing': 1.2,
                'button_height': 1.1
            }
        }
    
    def setup_layout_configs(self) -> Dict[str, Dict[str, any]]:
        """Setup layout configurations for different screen sizes"""
        return {
            'xs': {
                'grid_cols': 1,
                'card_height': dp(80),
                'header_height': 0.12,
                'content_height': 0.75,
                'nav_height': 0.13,
                'show_secondary_info': False
            },
            'sm': {
                'grid_cols': 2,
                'card_height': dp(90),
                'header_height': 0.15,
                'content_height': 0.70,
                'nav_height': 0.15,
                'show_secondary_info': True
            },
            'md': {
                'grid_cols': 2,
                'card_height': dp(100),
                'header_height': 0.15,
                'content_height': 0.70,
                'nav_height': 0.15,
                'show_secondary_info': True
            },
            'lg': {
                'grid_cols': 3,
                'card_height': dp(110),
                'header_height': 0.12,
                'content_height': 0.73,
                'nav_height': 0.15,
                'show_secondary_info': True
            },
            'xl': {
                'grid_cols': 3,
                'card_height': dp(120),
                'header_height': 0.10,
                'content_height': 0.75,
                'nav_height': 0.15,
                'show_secondary_info': True
            }
        }
    
    def get_responsive_value(self, base_value: float, property_type: str) -> float:
        """Get responsive value based on current breakpoint"""
        scale_factor = self.scaling_factors[self.current_breakpoint].get(property_type, 1.0)
        return base_value * scale_factor
    
    def get_layout_config(self) -> Dict[str, any]:
        """Get layout configuration for current screen size"""
        return self.layout_configs[self.current_breakpoint]
    
    def apply_responsive_sizing(self, widget: Widget, config: Optional[Dict] = None):
        """Apply responsive sizing to widget"""
        if not config:
            config = self.get_layout_config()
        
        # Apply touch optimization
        widget = self.touch_optimizer.optimize_touch_targets(widget)
        
        # Apply responsive padding and spacing
        if hasattr(widget, 'padding'):
            base_padding = widget.padding[0] if widget.padding else dp(10)
            responsive_padding = self.get_responsive_value(base_padding, 'padding')
            widget.padding = [responsive_padding] * 4
        
        if hasattr(widget, 'spacing'):
            base_spacing = widget.spacing if hasattr(widget, 'spacing') else dp(10)
            responsive_spacing = self.get_responsive_value(base_spacing, 'spacing')
            widget.spacing = responsive_spacing
        
        return widget
    
    def create_responsive_font_size(self, base_size: float) -> str:
        """Create responsive font size"""
        responsive_size = self.get_responsive_value(base_size, 'font_size')
        return f"{responsive_size:.0f}sp"

class GestureRecognizer:
    """Advanced gesture recognition for mobile interactions"""
    
    def __init__(self):
        """Initialize gesture recognizer"""
        self.gesture_threshold = dp(20)  # Minimum distance for gesture
        self.velocity_threshold = dp(100)  # Minimum velocity for swipe
        self.tap_timeout = 0.3  # Maximum time for tap
        
        # Gesture state tracking
        self.touch_start_pos = None
        self.touch_start_time = None
        self.gesture_callbacks = {}
    
    def add_gesture_listener(self, widget: Widget, gesture_type: str, callback):
        """Add gesture listener to widget"""
        if id(widget) not in self.gesture_callbacks:
            self.gesture_callbacks[id(widget)] = {}
        
        self.gesture_callbacks[id(widget)][gesture_type] = callback
        
        # Bind touch events
        widget.bind(on_touch_down=self.on_touch_down)
        widget.bind(on_touch_move=self.on_touch_move)
        widget.bind(on_touch_up=self.on_touch_up)
    
    def on_touch_down(self, widget: Widget, touch):
        """Handle touch down event"""
        if widget.collide_point(*touch.pos):
            self.touch_start_pos = touch.pos
            self.touch_start_time = touch.time_start
            return True
        return False
    
    def on_touch_move(self, widget: Widget, touch):
        """Handle touch move event"""
        if not self.touch_start_pos:
            return False
        
        if widget.collide_point(*touch.pos):
            # Calculate drag distance
            distance = Vector(touch.pos).distance(Vector(self.touch_start_pos))
            
            if distance > self.gesture_threshold:
                # This is a drag gesture
                self.trigger_gesture(widget, 'drag', {
                    'start_pos': self.touch_start_pos,
                    'current_pos': touch.pos,
                    'distance': distance
                })
            
            return True
        return False
    
    def on_touch_up(self, widget: Widget, touch):
        """Handle touch up event"""
        if not self.touch_start_pos:
            return False
        
        if widget.collide_point(*touch.pos):
            touch_duration = touch.time_start - self.touch_start_time if self.touch_start_time else 0
            distance = Vector(touch.pos).distance(Vector(self.touch_start_pos))
            
            if touch_duration < self.tap_timeout and distance < self.gesture_threshold:
                # This is a tap
                self.trigger_gesture(widget, 'tap', {
                    'pos': touch.pos,
                    'duration': touch_duration
                })
            
            elif distance > self.gesture_threshold:
                # This is a swipe
                direction = self.get_swipe_direction(self.touch_start_pos, touch.pos)
                velocity = distance / max(touch_duration, 0.01)
                
                if velocity > self.velocity_threshold:
                    self.trigger_gesture(widget, 'swipe', {
                        'direction': direction,
                        'distance': distance,
                        'velocity': velocity,
                        'start_pos': self.touch_start_pos,
                        'end_pos': touch.pos
                    })
            
            # Reset state
            self.touch_start_pos = None
            self.touch_start_time = None
            
            return True
        return False
    
    def get_swipe_direction(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float]) -> str:
        """Determine swipe direction"""
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'up' if dy > 0 else 'down'
    
    def trigger_gesture(self, widget: Widget, gesture_type: str, gesture_data: Dict):
        """Trigger gesture callback"""
        widget_callbacks = self.gesture_callbacks.get(id(widget), {})
        callback = widget_callbacks.get(gesture_type)
        
        if callback:
            callback(widget, gesture_data)

class MobileUXEnhancer:
    """Mobile user experience enhancements"""
    
    def __init__(self):
        """Initialize mobile UX enhancer"""
        self.responsive_design = ResponsiveDesign()
        self.gesture_recognizer = GestureRecognizer()
        self.loading_indicators = {}
        
    def create_mobile_optimized_button(self, text: str, callback, size_hint_y=None) -> Button:
        """Create mobile-optimized button"""
        layout_config = self.responsive_design.get_layout_config()
        
        # Calculate responsive button height
        base_height = dp(50)
        if size_hint_y is None:
            button_height = self.responsive_design.get_responsive_value(base_height, 'button_height')
            size_hint_y = None
            height = button_height
        else:
            height = None
        
        # Create button with responsive font size
        font_size = self.responsive_design.create_responsive_font_size(16)
        
        button = Button(
            text=text,
            font_size=font_size,
            size_hint_y=size_hint_y,
            height=height,
            background_color=(0.2, 0.6, 1.0, 1),
            color=(1, 1, 1, 1)
        )
        
        # Add touch optimization
        button = self.responsive_design.touch_optimizer.optimize_touch_targets(button)
        
        # Bind callback
        button.bind(on_press=callback)
        
        return button
    
    def create_mobile_card(self, title: str, content: str, icon: str = None) -> BoxLayout:
        """Create mobile-optimized card layout"""
        layout_config = self.responsive_design.get_layout_config()
        
        card_layout = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=layout_config['card_height'],
            spacing=self.responsive_design.get_responsive_value(dp(8), 'spacing'),
            padding=self.responsive_design.get_responsive_value(dp(12), 'padding')
        )
        
        # Card background
        with card_layout.canvas.before:
            Color(0.15, 0.2, 0.3, 1)
            RoundedRectangle(size=card_layout.size, pos=card_layout.pos, radius=[dp(10)])
        
        card_layout.bind(size=self.update_card_background, pos=self.update_card_background)
        
        # Header with icon
        if icon:
            header_layout = BoxLayout(orientation='horizontal', size_hint_y=0.4)
            
            icon_label = Label(
                text=icon,
                font_size=self.responsive_design.create_responsive_font_size(20),
                size_hint_x=0.2
            )
            header_layout.add_widget(icon_label)
            
            title_label = Label(
                text=title,
                font_size=self.responsive_design.create_responsive_font_size(16),
                bold=True,
                color=(1, 1, 1, 1),
                halign='left',
                size_hint_x=0.8
            )
            header_layout.add_widget(title_label)
            
            card_layout.add_widget(header_layout)
        else:
            title_label = Label(
                text=title,
                font_size=self.responsive_design.create_responsive_font_size(16),
                bold=True,
                color=(1, 1, 1, 1),
                size_hint_y=0.4
            )
            card_layout.add_widget(title_label)
        
        # Content
        content_label = Label(
            text=content,
            font_size=self.responsive_design.create_responsive_font_size(14),
            color=(0.9, 0.9, 0.9, 1),
            size_hint_y=0.6,
            halign='left',
            valign='top'
        )
        card_layout.add_widget(content_label)
        
        return card_layout
    
    def update_card_background(self, instance, value):
        """Update card background when size changes"""
        instance.canvas.before.children[-1].size = instance.size
        instance.canvas.before.children[-1].pos = instance.pos
    
    def create_responsive_slider(self, min_val: float, max_val: float, initial_val: float, 
                                callback, label_text: str = None) -> BoxLayout:
        """Create mobile-optimized slider with labels"""
        layout_config = self.responsive_design.get_layout_config()
        
        slider_layout = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=dp(80),
            spacing=self.responsive_design.get_responsive_value(dp(5), 'spacing')
        )
        
        # Label if provided
        if label_text:
            label = Label(
                text=label_text,
                font_size=self.responsive_design.create_responsive_font_size(14),
                size_hint_y=0.3,
                color=(0.9, 0.9, 0.9, 1)
            )
            slider_layout.add_widget(label)
        
        # Slider with value labels
        slider_container = BoxLayout(orientation='horizontal', size_hint_y=0.4, spacing=dp(10))
        
        # Min label
        min_label = Label(
            text=str(int(min_val)),
            font_size=self.responsive_design.create_responsive_font_size(12),
            size_hint_x=0.15,
            color=(0.7, 0.7, 0.7, 1)
        )
        slider_container.add_widget(min_label)
        
        # Slider
        slider = Slider(
            min=min_val,
            max=max_val,
            value=initial_val,
            size_hint_x=0.7,
            cursor_size=(dp(20), dp(20)),  # Larger cursor for touch
            step=max(1, (max_val - min_val) / 100)
        )
        slider.bind(value=callback)
        slider_container.add_widget(slider)
        
        # Max label
        max_label = Label(
            text=str(int(max_val)),
            font_size=self.responsive_design.create_responsive_font_size(12),
            size_hint_x=0.15,
            color=(0.7, 0.7, 0.7, 1)
        )
        slider_container.add_widget(max_label)
        
        slider_layout.add_widget(slider_container)
        
        # Current value display
        value_label = Label(
            text=f"Current: {initial_val:.0f}",
            font_size=self.responsive_design.create_responsive_font_size(14),
            size_hint_y=0.3,
            color=(0.2, 0.6, 1.0, 1)
        )
        
        def update_value_label(instance, value):
            value_label.text = f"Current: {value:.0f}"
        
        slider.bind(value=update_value_label)
        slider_layout.add_widget(value_label)
        
        return slider_layout
    
    def add_pull_to_refresh(self, scroll_widget, refresh_callback):
        """Add pull-to-refresh functionality"""
        def on_scroll_start(instance, touch):
            if instance.scroll_y >= 1.0:  # At top
                # Store initial touch position
                instance.refresh_start_y = touch.pos[1]
                return True
            return False
        
        def on_scroll_move(instance, touch):
            if hasattr(instance, 'refresh_start_y'):
                pull_distance = touch.pos[1] - instance.refresh_start_y
                
                if pull_distance > dp(100):  # 100dp pull threshold
                    # Trigger refresh
                    self.show_loading_indicator("Refreshing data...")
                    Clock.schedule_once(lambda dt: self.trigger_refresh(refresh_callback), 0.5)
                    del instance.refresh_start_y
        
        scroll_widget.bind(on_touch_down=on_scroll_start)
        scroll_widget.bind(on_touch_move=on_scroll_move)
    
    def trigger_refresh(self, refresh_callback):
        """Trigger refresh callback and hide loading"""
        refresh_callback()
        self.hide_loading_indicator()
    
    def show_loading_indicator(self, message: str = "Loading..."):
        """Show loading indicator"""
        # This would create a loading overlay
        print(f"Loading: {message}")
    
    def hide_loading_indicator(self):
        """Hide loading indicator"""
        # This would hide the loading overlay
        print("Loading complete")
    
    def add_swipe_navigation(self, widget, left_callback=None, right_callback=None):
        """Add swipe navigation to widget"""
        def handle_swipe(widget_instance, gesture_data):
            direction = gesture_data['direction']
            
            if direction == 'left' and left_callback:
                # Haptic feedback
                self.responsive_design.touch_optimizer.trigger_haptic_feedback('light')
                left_callback()
            elif direction == 'right' and right_callback:
                # Haptic feedback
                self.responsive_design.touch_optimizer.trigger_haptic_feedback('light')
                right_callback()
        
        self.gesture_recognizer.add_gesture_listener(widget, 'swipe', handle_swipe)

# Global instances
mobile_ux_enhancer = MobileUXEnhancer()
responsive_design = ResponsiveDesign()
touch_optimizer = TouchOptimizer()

def optimize_for_mobile(widget: Widget) -> Widget:
    """Quick function to optimize any widget for mobile"""
    return responsive_design.apply_responsive_sizing(
        touch_optimizer.optimize_touch_targets(widget)
    )

if __name__ == "__main__":
    # Test mobile optimization
    print("📱 Mobile Touch Optimizer Test")
    print("=" * 40)
    
    # Test screen detection
    print(f"Screen category: {touch_optimizer.current_screen_category}")
    print(f"Breakpoint: {responsive_design.current_breakpoint}")
    print(f"Is mobile: {touch_optimizer.is_mobile}")
    
    # Test responsive values
    layout_config = responsive_design.get_layout_config()
    print(f"Grid columns: {layout_config['grid_cols']}")
    print(f"Card height: {layout_config['card_height']}px")
    
    # Test font scaling
    font_16 = responsive_design.create_responsive_font_size(16)
    font_20 = responsive_design.create_responsive_font_size(20)
    print(f"Font 16sp responsive: {font_16}")
    print(f"Font 20sp responsive: {font_20}")
    
    print("\n✅ Mobile Touch Optimization Ready!")