"""
Grand Millennium Revenue Analytics - Enhancement Integration

Integration layer that connects game enhancements with mobile game components.
Provides unified interface for sounds, animations, and visual effects.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json

try:
    from game_enhancements import get_game_enhancement_manager
    from visual_effects import get_visual_effects_manager
    from mobile_touch_optimizer import mobile_ux_enhancer
    from aed_currency_handler import AEDCurrencyHandler
    ENHANCEMENT_MODULES_AVAILABLE = True
except ImportError:
    ENHANCEMENT_MODULES_AVAILABLE = False

try:
    from kivy.uix.widget import Widget
    from kivy.clock import Clock
    KIVY_AVAILABLE = True
except ImportError:
    KIVY_AVAILABLE = False

class EnhancementIntegration:
    """Unified enhancement integration for mobile revenue analytics game"""
    
    def __init__(self):
        """Initialize enhancement integration"""
        self.project_root = Path(__file__).parent
        
        # Initialize managers
        if ENHANCEMENT_MODULES_AVAILABLE:
            self.enhancement_manager = get_game_enhancement_manager()
            self.visual_manager = get_visual_effects_manager()
        else:
            self.enhancement_manager = None
            self.visual_manager = None
        
        # Enhancement profiles for different game states
        self.enhancement_profiles = {
            'dashboard_entry': {
                'sound': 'dashboard_refresh',
                'haptic': 'light',
                'animation': 'card_reveal',
                'visual_effect': 'revenue_wave'
            },
            'button_interaction': {
                'sound': 'button_tap',
                'haptic': 'light',
                'animation': 'button_press',
                'visual_effect': None
            },
            'challenge_complete': {
                'sound': 'challenge_complete',
                'haptic': 'success',
                'animation': 'level_up',
                'visual_effect': 'success_confetti',
                'aed_animation': True
            },
            'challenge_fail': {
                'sound': 'challenge_fail',
                'haptic': 'error',
                'animation': None,
                'visual_effect': 'error_sparks'
            },
            'level_progression': {
                'sound': 'level_up',
                'haptic': 'level_up',
                'animation': 'level_up',
                'visual_effect': 'level_up_celebration',
                'celebration': True
            },
            'aed_reward': {
                'sound': 'coin_collect',
                'haptic': 'medium',
                'animation': 'coin_collect',
                'visual_effect': 'coin_burst',
                'aed_animation': True
            },
            'achievement_unlock': {
                'sound': 'achievement_unlock',
                'haptic': 'success',
                'animation': 'achievement_badge',
                'visual_effect': 'achievement_celebration',
                'celebration': True
            },
            'swipe_navigation': {
                'sound': 'swipe',
                'haptic': 'light',
                'animation': 'swipe_response',
                'visual_effect': None
            },
            'data_loading': {
                'sound': None,
                'haptic': None,
                'animation': 'loading_spinner',
                'visual_effect': 'data_loading'
            },
            'segment_transition': {
                'sound': 'notification',
                'haptic': 'medium',
                'animation': 'segment_transition',
                'visual_effect': 'segment_transition'
            }
        }
        
        # AED currency handler
        self.aed_handler = AEDCurrencyHandler() if ENHANCEMENT_MODULES_AVAILABLE else None
        
        print("🎮 Enhancement Integration System Initialized")
    
    def trigger_enhancement_profile(self, profile_name: str, widget: Optional[Widget] = None, **params):
        """Trigger a complete enhancement profile"""
        if profile_name not in self.enhancement_profiles:
            print(f"⚠️  Enhancement profile not found: {profile_name}")
            return
        
        profile = self.enhancement_profiles[profile_name]
        
        # Play sound effect
        if profile.get('sound') and self.enhancement_manager:
            self.enhancement_manager.play_sound(profile['sound'])
        
        # Trigger haptic feedback
        if profile.get('haptic') and self.enhancement_manager:
            self.enhancement_manager.trigger_haptic_feedback(profile['haptic'])
        
        # Apply animation
        if profile.get('animation') and widget and self.enhancement_manager:
            self.enhancement_manager.animate_widget(widget, profile['animation'])
        
        # Create visual effect
        if profile.get('visual_effect') and widget and self.enhancement_manager:
            self.enhancement_manager.create_particle_effect(widget, profile['visual_effect'])
        
        # Special handling for celebrations
        if profile.get('celebration') and widget and self.visual_manager:
            celebration_type = params.get('celebration_type', 'general')
            if profile_name == 'level_progression':
                level_name = params.get('level_name', 'New Level')
                self.visual_manager.create_level_up_celebration(widget, level_name)
            elif profile_name == 'achievement_unlock':
                badge_type = params.get('badge_type', 'gold')
                self.visual_manager.create_achievement_badge_animation(widget, badge_type)
        
        # AED animation
        if profile.get('aed_animation') and self.visual_manager:
            start_amount = params.get('start_amount', 0)
            end_amount = params.get('end_amount', 0)
            if start_amount != end_amount and widget:
                self.visual_manager.create_aed_counter_animation(
                    widget, start_amount, end_amount, params.get('compact', False)
                )
        
        print(f"🎯 Triggered enhancement profile: {profile_name}")
    
    def enhance_button_interaction(self, button_widget, button_type: str = 'primary'):
        """Enhanced button interaction with type-specific effects"""
        base_profile = 'button_interaction'
        
        # Customize based on button type
        if button_type == 'primary':
            # Standard button
            self.trigger_enhancement_profile(base_profile, button_widget)
        elif button_type == 'success':
            # Success button (e.g., complete challenge)
            profile = self.enhancement_profiles[base_profile].copy()
            profile['haptic'] = 'medium'
            profile['sound'] = 'challenge_complete'
            self._trigger_custom_profile(profile, button_widget)
        elif button_type == 'danger':
            # Danger button
            profile = self.enhancement_profiles[base_profile].copy()
            profile['haptic'] = 'heavy'
            profile['sound'] = 'notification'
            self._trigger_custom_profile(profile, button_widget)
        elif button_type == 'aed_action':
            # AED-related action
            self.trigger_enhancement_profile('aed_reward', button_widget)
    
    def enhance_challenge_completion(self, widget, success: bool, aed_earned: float = 0, 
                                   previous_aed: float = 0):
        """Enhanced challenge completion with AED animation"""
        if success:
            self.trigger_enhancement_profile('challenge_complete', widget,
                                           start_amount=previous_aed,
                                           end_amount=previous_aed + aed_earned)
            
            # Additional celebration for large amounts
            if aed_earned >= 10000:  # Major reward threshold
                Clock.schedule_once(
                    lambda dt: self.trigger_enhancement_profile('achievement_unlock', widget,
                                                              badge_type='gold'),
                    1.0
                ) if KIVY_AVAILABLE else None
        else:
            self.trigger_enhancement_profile('challenge_fail', widget)
    
    def enhance_level_progression(self, widget, old_level: int, new_level: int, level_name: str):
        """Enhanced level progression with celebration"""
        self.trigger_enhancement_profile('level_progression', widget, level_name=level_name)
        
        # Progressive celebration intensity
        celebration_intensity = min(new_level, 5)  # Cap at level 5
        
        for i in range(celebration_intensity):
            delay = i * 0.5
            Clock.schedule_once(
                lambda dt, intensity=i: self._create_delayed_celebration(widget, intensity),
                delay
            ) if KIVY_AVAILABLE else None
    
    def enhance_dashboard_entry(self, dashboard_widgets: List[Widget]):
        """Enhanced dashboard entry with staggered reveals"""
        if self.visual_manager:
            self.visual_manager.create_chart_reveal_animation(dashboard_widgets)
        
        # Background sound
        if self.enhancement_manager:
            self.enhancement_manager.play_sound('dashboard_refresh')
            
        print("📊 Enhanced dashboard entry")
    
    def enhance_aed_transaction(self, label_widget, transaction_type: str, 
                              amount: float, previous_total: float = 0):
        """Enhanced AED transaction with contextual effects"""
        new_total = previous_total
        
        if transaction_type == 'earn':
            new_total += amount
            self.trigger_enhancement_profile('aed_reward', label_widget,
                                           start_amount=previous_total,
                                           end_amount=new_total)
        elif transaction_type == 'spend':
            new_total -= amount
            # Spending has different sound/haptic
            profile = self.enhancement_profiles['aed_reward'].copy()
            profile['sound'] = 'button_tap'
            profile['haptic'] = 'light'
            self._trigger_custom_profile(profile, label_widget)
            
            if self.visual_manager:
                self.visual_manager.create_aed_counter_animation(
                    label_widget, previous_total, new_total, compact=True
                )
        elif transaction_type == 'update':
            # Just update display
            if self.visual_manager:
                self.visual_manager.create_aed_counter_animation(
                    label_widget, previous_total, amount
                )
        
        return new_total
    
    def enhance_swipe_navigation(self, widget, direction: str, target_screen: str):
        """Enhanced swipe navigation with directional feedback"""
        self.trigger_enhancement_profile('swipe_navigation', widget)
        
        # Directional haptic patterns
        if self.enhancement_manager:
            direction_haptics = {
                'left': 'light',
                'right': 'light',
                'up': 'medium',
                'down': 'light'
            }
            haptic_type = direction_haptics.get(direction, 'light')
            self.enhancement_manager.trigger_haptic_feedback(haptic_type)
        
        print(f"👆 Enhanced swipe navigation: {direction} → {target_screen}")
    
    def enhance_data_loading(self, loading_widget, loading_text: str = "Loading..."):
        """Enhanced data loading with visual feedback"""
        if self.visual_manager:
            animations = self.visual_manager.create_data_loading_animation(loading_widget)
            
            # Store animations for later cleanup
            if not hasattr(loading_widget, '_loading_animations'):
                loading_widget._loading_animations = animations
        
        print(f"⏳ Enhanced data loading: {loading_text}")
    
    def stop_data_loading(self, loading_widget, success: bool = True):
        """Stop data loading with completion feedback"""
        if self.visual_manager and hasattr(loading_widget, '_loading_animations'):
            self.visual_manager.stop_loading_animation(
                loading_widget, loading_widget._loading_animations
            )
            delattr(loading_widget, '_loading_animations')
        
        # Completion feedback
        if success:
            if self.enhancement_manager:
                self.enhancement_manager.play_sound('dashboard_refresh')
                self.enhancement_manager.trigger_haptic_feedback('light')
        else:
            if self.enhancement_manager:
                self.enhancement_manager.play_sound('challenge_fail')
                self.enhancement_manager.trigger_haptic_feedback('error')
        
        print(f"✅ Data loading completed: {'success' if success else 'failed'}")
    
    def enhance_segment_analysis(self, widget, old_segment: str, new_segment: str, 
                                performance_change: float):
        """Enhanced segment analysis with transition effects"""
        self.trigger_enhancement_profile('segment_transition', widget)
        
        if self.visual_manager:
            self.visual_manager.create_segment_transition_effect(old_segment, new_segment, widget)
        
        # Performance-based feedback
        if performance_change > 0:
            # Positive change
            Clock.schedule_once(
                lambda dt: self.trigger_enhancement_profile('aed_reward', widget),
                1.5
            ) if KIVY_AVAILABLE else None
        elif performance_change < 0:
            # Negative change
            Clock.schedule_once(
                lambda dt: self.trigger_enhancement_profile('challenge_fail', widget),
                1.5
            ) if KIVY_AVAILABLE else None
        
        print(f"📈 Enhanced segment analysis: {old_segment} → {new_segment} ({performance_change:+.1%})")
    
    def _trigger_custom_profile(self, profile: Dict[str, Any], widget: Optional[Widget]):
        """Trigger a custom enhancement profile"""
        if profile.get('sound') and self.enhancement_manager:
            self.enhancement_manager.play_sound(profile['sound'])
        
        if profile.get('haptic') and self.enhancement_manager:
            self.enhancement_manager.trigger_haptic_feedback(profile['haptic'])
        
        if profile.get('animation') and widget and self.enhancement_manager:
            self.enhancement_manager.animate_widget(widget, profile['animation'])
    
    def _create_delayed_celebration(self, widget, intensity: int):
        """Create delayed celebration effect"""
        if self.enhancement_manager:
            if intensity % 2 == 0:  # Even numbers = particles
                effect_type = ['coin_burst', 'level_up_stars', 'success_confetti'][intensity % 3]
                self.enhancement_manager.create_particle_effect(widget, effect_type)
            else:  # Odd numbers = sounds
                sounds = ['coin_collect', 'achievement_unlock'][intensity % 2]
                self.enhancement_manager.play_sound(sounds)
    
    def get_enhancement_settings(self) -> Dict[str, Any]:
        """Get current enhancement settings"""
        if self.enhancement_manager:
            return self.enhancement_manager.get_enhancement_settings()
        return {}
    
    def update_enhancement_settings(self, **settings):
        """Update enhancement settings"""
        if self.enhancement_manager:
            self.enhancement_manager.update_enhancement_settings(**settings)
    
    def create_enhancement_guide(self):
        """Create enhancement integration guide"""
        guide = {
            "enhancement_integration_guide": {
                "overview": "Integration layer for game enhancements in Grand Millennium Revenue Analytics",
                "enhancement_profiles": {
                    profile_name: {
                        "description": f"Enhancement profile for {profile_name.replace('_', ' ')}",
                        "components": profile
                    }
                    for profile_name, profile in self.enhancement_profiles.items()
                },
                "usage_examples": {
                    "button_click": "integration.enhance_button_interaction(button_widget, 'primary')",
                    "challenge_complete": "integration.enhance_challenge_completion(widget, True, 5000, 10000)",
                    "level_up": "integration.enhance_level_progression(widget, 2, 3, 'Revenue Analyst')",
                    "dashboard_load": "integration.enhance_dashboard_entry(dashboard_widgets)",
                    "aed_earn": "integration.enhance_aed_transaction(label, 'earn', 2500, 15000)"
                },
                "customization": {
                    "sound_settings": "Control volume, enable/disable sounds",
                    "haptic_settings": "Control vibration intensity and patterns",
                    "animation_settings": "Control animation speed and effects",
                    "visual_settings": "Control particle effects and visual enhancements"
                },
                "mobile_optimization": {
                    "performance": "All effects optimized for mobile devices",
                    "battery": "Efficient animations to preserve battery life",
                    "accessibility": "Support for reduced motion preferences",
                    "compatibility": "Fallback support for devices without capabilities"
                }
            }
        }
        
        guide_file = self.project_root / "enhancement_integration_guide.json"
        with open(guide_file, 'w') as f:
            json.dump(guide, f, indent=2)
        
        print(f"📋 Enhancement integration guide created: {guide_file}")
    
    def test_integration(self):
        """Test enhancement integration system"""
        print("\n🎮 Testing Enhancement Integration...")
        print("=" * 45)
        
        # Test profile triggers
        print("\n🎯 Testing Enhancement Profiles:")
        test_profiles = ['button_interaction', 'challenge_complete', 'level_progression', 'aed_reward']
        
        for profile in test_profiles:
            print(f"   Testing {profile}...")
            self.trigger_enhancement_profile(profile, None, 
                                           start_amount=1000, end_amount=1500, 
                                           level_name="Test Level")
        
        # Test AED transactions
        print("\n💰 Testing AED Transactions:")
        transactions = [
            ('earn', 2500, 10000),
            ('spend', 1000, 12500),
            ('update', 11500, 11500)
        ]
        
        for transaction_type, amount, previous in transactions:
            result = self.enhance_aed_transaction(None, transaction_type, amount, previous)
            print(f"   {transaction_type}: {amount} AED (total: {result})")
        
        print("\n✅ Enhancement integration test complete!")

def get_enhancement_integration():
    """Get singleton instance of enhancement integration"""
    if not hasattr(get_enhancement_integration, '_instance'):
        get_enhancement_integration._instance = EnhancementIntegration()
    return get_enhancement_integration._instance

# Quick helper functions for common enhancement patterns
def enhance_button_click(button_widget, button_type='primary'):
    """Quick helper for button click enhancement"""
    integration = get_enhancement_integration()
    integration.enhance_button_interaction(button_widget, button_type)

def enhance_aed_earning(label_widget, amount_earned, current_total):
    """Quick helper for AED earning enhancement"""
    integration = get_enhancement_integration()
    return integration.enhance_aed_transaction(label_widget, 'earn', amount_earned, current_total)

def enhance_challenge_success(widget, aed_reward, previous_total):
    """Quick helper for challenge success enhancement"""
    integration = get_enhancement_integration()
    integration.enhance_challenge_completion(widget, True, aed_reward, previous_total)

def enhance_level_up(widget, old_level, new_level, level_name):
    """Quick helper for level up enhancement"""
    integration = get_enhancement_integration()
    integration.enhance_level_progression(widget, old_level, new_level, level_name)

if __name__ == "__main__":
    # Initialize and test integration system
    integration = EnhancementIntegration()
    integration.create_enhancement_guide()
    integration.test_integration()