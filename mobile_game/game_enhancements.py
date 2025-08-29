"""
Grand Millennium Revenue Analytics - Game Enhancements

Sound effects, animations, and visual enhancements for mobile gaming experience.
Optimized for Android/iOS deployment with fallback options.
"""

import os
import sys
from pathlib import Path
import json
from typing import Dict, Any, Optional, List

try:
    from kivy.animation import Animation
    from kivy.clock import Clock
    from kivy.core.audio import SoundLoader
    from kivy.utils import platform
    from kivy.metrics import dp
    from kivy.vector import Vector
    from kivy.graphics import Color, Rectangle, Ellipse, Line
    from kivy.uix.label import Label
    from kivy.uix.widget import Widget
    KIVY_AVAILABLE = True
except ImportError:
    KIVY_AVAILABLE = False

try:
    from plyer import vibrator, audio
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False

class GameEnhancementManager:
    """Manages sound effects, animations, and visual enhancements"""
    
    def __init__(self):
        """Initialize enhancement manager"""
        self.project_root = Path(__file__).parent
        self.assets_dir = self.project_root / "assets"
        self.sounds_dir = self.assets_dir / "sounds"
        self.animations_dir = self.assets_dir / "animations"
        
        # Create directories
        self.assets_dir.mkdir(exist_ok=True)
        self.sounds_dir.mkdir(exist_ok=True)
        self.animations_dir.mkdir(exist_ok=True)
        
        # Enhancement settings
        self.settings = {
            'sound_enabled': True,
            'vibration_enabled': True,
            'animations_enabled': True,
            'particle_effects': True,
            'sound_volume': 0.7,
            'animation_speed': 1.0
        }
        
        # Load sound effects
        self.sounds = {}
        self.load_sound_effects()
        
        # Animation presets
        self.animation_presets = self.setup_animation_presets()
        
        print("🎮 Game Enhancement System Initialized")
    
    def load_sound_effects(self):
        """Load and register sound effects"""
        if not KIVY_AVAILABLE:
            print("⚠️  Kivy not available - sound effects will be simulated")
            return
        
        # Sound effect definitions
        sound_effects = {
            'button_tap': {
                'file': 'button_tap.wav',
                'description': 'Button press sound',
                'volume': 0.5
            },
            'level_up': {
                'file': 'level_up.wav', 
                'description': 'Level progression sound',
                'volume': 0.8
            },
            'achievement_unlock': {
                'file': 'achievement.wav',
                'description': 'Achievement unlocked',
                'volume': 0.7
            },
            'coin_collect': {
                'file': 'coin_collect.wav',
                'description': 'AED currency earned',
                'volume': 0.6
            },
            'challenge_complete': {
                'file': 'success.wav',
                'description': 'Challenge completion',
                'volume': 0.7
            },
            'challenge_fail': {
                'file': 'fail.wav',
                'description': 'Challenge failed',
                'volume': 0.4
            },
            'swipe': {
                'file': 'swipe.wav',
                'description': 'Swipe gesture',
                'volume': 0.3
            },
            'notification': {
                'file': 'notification.wav',
                'description': 'In-app notification',
                'volume': 0.5
            },
            'dashboard_refresh': {
                'file': 'refresh.wav',
                'description': 'Data refresh sound',
                'volume': 0.4
            },
            'prize_win': {
                'file': 'prize_win.wav',
                'description': 'Major prize won',
                'volume': 0.9
            }
        }
        
        # Load actual sound files (if they exist)
        for sound_id, sound_info in sound_effects.items():
            sound_path = self.sounds_dir / sound_info['file']
            
            if sound_path.exists() and KIVY_AVAILABLE:
                try:
                    sound = SoundLoader.load(str(sound_path))
                    if sound:
                        sound.volume = sound_info['volume'] * self.settings['sound_volume']
                        self.sounds[sound_id] = sound
                        print(f"✅ Loaded sound: {sound_id}")
                    else:
                        print(f"⚠️  Could not load: {sound_path}")
                except Exception as e:
                    print(f"❌ Sound loading error for {sound_id}: {e}")
            else:
                # Create placeholder/description
                self.sounds[sound_id] = {
                    'placeholder': True,
                    'description': sound_info['description'],
                    'volume': sound_info['volume']
                }
                print(f"📝 Sound placeholder: {sound_id} ({sound_info['description']})")
    
    def play_sound(self, sound_id: str, volume_override: Optional[float] = None):
        """Play a sound effect"""
        if not self.settings['sound_enabled']:
            return
        
        if sound_id not in self.sounds:
            print(f"⚠️  Sound not found: {sound_id}")
            return
        
        sound = self.sounds[sound_id]
        
        if isinstance(sound, dict) and sound.get('placeholder'):
            # Simulate sound play
            print(f"🔊 Playing sound: {sound['description']}")
            return
        
        try:
            if KIVY_AVAILABLE and hasattr(sound, 'play'):
                if volume_override:
                    original_volume = sound.volume
                    sound.volume = volume_override * self.settings['sound_volume']
                    sound.play()
                    sound.volume = original_volume
                else:
                    sound.play()
                print(f"🔊 Played sound: {sound_id}")
        except Exception as e:
            print(f"❌ Sound play error for {sound_id}: {e}")
    
    def trigger_haptic_feedback(self, feedback_type: str = "light"):
        """Trigger haptic feedback on mobile devices"""
        if not self.settings['vibration_enabled']:
            return
        
        if platform == 'android' and PLYER_AVAILABLE:
            try:
                vibration_patterns = {
                    'light': 0.1,
                    'medium': 0.2,
                    'heavy': 0.3,
                    'success': [0.1, 0.1, 0.2],
                    'error': [0.3, 0.1, 0.3],
                    'level_up': [0.2, 0.1, 0.2, 0.1, 0.3]
                }
                
                pattern = vibration_patterns.get(feedback_type, 0.1)
                
                if isinstance(pattern, list):
                    for duration in pattern:
                        vibrator.vibrate(duration)
                        Clock.schedule_once(lambda dt: None, 0.1)
                else:
                    vibrator.vibrate(pattern)
                    
                print(f"📳 Haptic feedback: {feedback_type}")
            except Exception as e:
                print(f"⚠️  Haptic feedback error: {e}")
        else:
            print(f"📳 Haptic feedback simulated: {feedback_type}")
    
    def setup_animation_presets(self) -> Dict[str, Dict[str, Any]]:
        """Setup animation presets for common UI interactions"""
        if not KIVY_AVAILABLE:
            return {}
        
        return {
            'button_press': {
                'scale_down': {'scale': 0.95, 'duration': 0.1, 'transition': 'out_quad'},
                'scale_up': {'scale': 1.0, 'duration': 0.1, 'transition': 'out_quad'}
            },
            'card_reveal': {
                'fade_in': {'opacity': 1.0, 'duration': 0.3, 'transition': 'out_cubic'},
                'slide_up': {'y': 0, 'duration': 0.4, 'transition': 'out_back'}
            },
            'level_up': {
                'bounce': {'scale': 1.2, 'duration': 0.2, 'transition': 'out_back'},
                'settle': {'scale': 1.0, 'duration': 0.3, 'transition': 'out_elastic'}
            },
            'coin_collect': {
                'collect': {'scale': 1.5, 'opacity': 0, 'duration': 0.5, 'transition': 'out_quad'}
            },
            'swipe_response': {
                'slide': {'x': 50, 'duration': 0.2, 'transition': 'out_cubic'},
                'return': {'x': 0, 'duration': 0.3, 'transition': 'out_back'}
            },
            'notification_popup': {
                'appear': {'scale': 1.0, 'opacity': 1.0, 'duration': 0.3, 'transition': 'out_back'},
                'disappear': {'scale': 0.8, 'opacity': 0, 'duration': 0.2, 'transition': 'in_quad'}
            },
            'progress_fill': {
                'fill': {'width': None, 'duration': 1.0, 'transition': 'out_cubic'}  # width set dynamically
            },
            'dashboard_refresh': {
                'spin': {'angle': 360, 'duration': 0.8, 'transition': 'linear'}
            }
        }
    
    def animate_widget(self, widget, animation_name: str, **kwargs):
        """Apply animation to a widget"""
        if not KIVY_AVAILABLE or not self.settings['animations_enabled']:
            print(f"🎬 Animation simulated: {animation_name}")
            return
        
        if animation_name not in self.animation_presets:
            print(f"⚠️  Animation preset not found: {animation_name}")
            return
        
        preset = self.animation_presets[animation_name]
        
        # Apply animation speed modifier
        speed_modifier = self.settings['animation_speed']
        
        try:
            if animation_name == 'button_press':
                # Two-stage animation
                scale_down = Animation(**preset['scale_down'])
                scale_down.duration *= (1.0 / speed_modifier)
                
                scale_up = Animation(**preset['scale_up'])  
                scale_up.duration *= (1.0 / speed_modifier)
                
                animation = scale_down + scale_up
                animation.start(widget)
                
            elif animation_name == 'card_reveal':
                # Parallel animations
                fade_anim = Animation(**preset['fade_in'])
                fade_anim.duration *= (1.0 / speed_modifier)
                
                slide_anim = Animation(**preset['slide_up'])
                slide_anim.duration *= (1.0 / speed_modifier)
                
                fade_anim.start(widget)
                slide_anim.start(widget)
                
            elif animation_name == 'level_up':
                # Sequential bounce and settle
                bounce = Animation(**preset['bounce'])
                bounce.duration *= (1.0 / speed_modifier)
                
                settle = Animation(**preset['settle'])
                settle.duration *= (1.0 / speed_modifier)
                
                animation = bounce + settle
                animation.start(widget)
                
            else:
                # Single animation
                anim_data = list(preset.values())[0]
                anim_data = anim_data.copy()
                anim_data.update(kwargs)
                anim_data['duration'] *= (1.0 / speed_modifier)
                
                animation = Animation(**anim_data)
                animation.start(widget)
            
            print(f"🎬 Started animation: {animation_name}")
            
        except Exception as e:
            print(f"❌ Animation error for {animation_name}: {e}")
    
    def create_particle_effect(self, widget, effect_type: str, **params):
        """Create particle effects for enhanced visual feedback"""
        if not KIVY_AVAILABLE or not self.settings['particle_effects']:
            print(f"✨ Particle effect simulated: {effect_type}")
            return
        
        particle_effects = {
            'coin_burst': {
                'particles': 8,
                'colors': [(1, 0.84, 0, 1), (0.9, 0.7, 0, 1)],  # Gold colors
                'size_range': (dp(4), dp(8)),
                'velocity_range': (50, 100),
                'duration': 1.0
            },
            'level_up_stars': {
                'particles': 12,
                'colors': [(1, 1, 1, 1), (0.8, 0.9, 1, 1)],  # White/blue
                'size_range': (dp(3), dp(6)),
                'velocity_range': (30, 80),
                'duration': 1.5
            },
            'success_confetti': {
                'particles': 15,
                'colors': [(0, 0.8, 0.4, 1), (0.2, 0.9, 0.2, 1)],  # Green
                'size_range': (dp(2), dp(5)),
                'velocity_range': (40, 90),
                'duration': 2.0
            },
            'error_sparks': {
                'particles': 6,
                'colors': [(1, 0.3, 0.3, 1), (0.9, 0.1, 0.1, 1)],  # Red
                'size_range': (dp(3), dp(6)),
                'velocity_range': (60, 120),
                'duration': 0.8
            }
        }
        
        if effect_type not in particle_effects:
            print(f"⚠️  Particle effect not found: {effect_type}")
            return
        
        effect = particle_effects[effect_type]
        effect.update(params)  # Override with custom params
        
        try:
            # Create particle system (simplified implementation)
            for i in range(effect['particles']):
                particle = self._create_particle(widget, effect, i)
                if particle:
                    self._animate_particle(particle, effect)
            
            print(f"✨ Created particle effect: {effect_type}")
            
        except Exception as e:
            print(f"❌ Particle effect error for {effect_type}: {e}")
    
    def _create_particle(self, parent_widget, effect_config, index):
        """Create individual particle"""
        if not KIVY_AVAILABLE:
            return None
        
        import random
        from kivy.uix.widget import Widget
        
        particle = Widget()
        particle.size = random.uniform(*effect_config['size_range']), random.uniform(*effect_config['size_range'])
        
        # Random position around parent center
        parent_center_x = parent_widget.center_x
        parent_center_y = parent_widget.center_y
        
        particle.center_x = parent_center_x + random.uniform(-dp(10), dp(10))
        particle.center_y = parent_center_y + random.uniform(-dp(10), dp(10))
        
        # Add to parent
        parent_widget.add_widget(particle)
        
        # Set color (simplified - would use Canvas in real implementation)
        color = random.choice(effect_config['colors'])
        setattr(particle, '_particle_color', color)
        
        return particle
    
    def _animate_particle(self, particle, effect_config):
        """Animate individual particle"""
        if not KIVY_AVAILABLE:
            return
        
        import random
        
        # Random direction and velocity
        angle = random.uniform(0, 360)
        velocity = random.uniform(*effect_config['velocity_range'])
        
        # Calculate end position
        import math
        end_x = particle.x + velocity * math.cos(math.radians(angle))
        end_y = particle.y + velocity * math.sin(math.radians(angle))
        
        # Create animation
        animation = Animation(
            x=end_x,
            y=end_y,
            opacity=0,
            duration=effect_config['duration'],
            transition='out_quad'
        )
        
        # Remove particle when animation completes
        def remove_particle(animation, particle):
            try:
                if particle.parent:
                    particle.parent.remove_widget(particle)
            except:
                pass
        
        animation.bind(on_complete=remove_particle)
        animation.start(particle)
    
    def create_sound_pack_info(self):
        """Create sound pack information for users to add their own sounds"""
        sound_pack_info = {
            "sound_pack_info": {
                "format": "WAV or OGG recommended for mobile",
                "sample_rate": "44.1kHz",
                "bit_depth": "16-bit",
                "duration_recommendations": {
                    "button_tap": "0.1-0.2 seconds",
                    "level_up": "1-2 seconds",
                    "achievement_unlock": "1.5-3 seconds",
                    "coin_collect": "0.3-0.8 seconds",
                    "challenge_complete": "1-2 seconds",
                    "challenge_fail": "0.5-1 second",
                    "swipe": "0.1-0.3 seconds",
                    "notification": "0.5-1 second",
                    "dashboard_refresh": "0.3-0.6 seconds",
                    "prize_win": "2-4 seconds"
                },
                "volume_guidelines": {
                    "background": "0.1-0.3",
                    "ui_sounds": "0.3-0.7",
                    "celebrations": "0.7-1.0"
                },
                "file_paths": {
                    "sounds_directory": "mobile_game/assets/sounds/",
                    "required_files": [
                        "button_tap.wav",
                        "level_up.wav",
                        "achievement.wav",
                        "coin_collect.wav",
                        "success.wav",
                        "fail.wav",
                        "swipe.wav",
                        "notification.wav",
                        "refresh.wav",
                        "prize_win.wav"
                    ]
                },
                "licensing": "Ensure all audio files are royalty-free or properly licensed",
                "mobile_optimization": "Keep files under 100KB each for faster loading"
            }
        }
        
        sound_info_file = self.sounds_dir / "sound_pack_guide.json"
        with open(sound_info_file, 'w') as f:
            json.dump(sound_pack_info, f, indent=2)
        
        print(f"📋 Sound pack guide created: {sound_info_file}")
    
    def get_enhancement_settings(self) -> Dict[str, Any]:
        """Get current enhancement settings"""
        return self.settings.copy()
    
    def update_enhancement_settings(self, **settings):
        """Update enhancement settings"""
        for key, value in settings.items():
            if key in self.settings:
                self.settings[key] = value
                print(f"⚙️  Updated {key}: {value}")
            else:
                print(f"⚠️  Unknown setting: {key}")
    
    def test_all_enhancements(self):
        """Test all enhancement features"""
        print("\n🎮 Testing Game Enhancements...")
        print("=" * 40)
        
        # Test sounds
        print("\n🔊 Testing Sound Effects:")
        test_sounds = ['button_tap', 'level_up', 'coin_collect', 'achievement_unlock']
        for sound in test_sounds:
            self.play_sound(sound)
            Clock.schedule_once(lambda dt: None, 0.2) if KIVY_AVAILABLE else None
        
        # Test haptic feedback
        print("\n📳 Testing Haptic Feedback:")
        haptic_types = ['light', 'medium', 'heavy', 'success']
        for haptic in haptic_types:
            self.trigger_haptic_feedback(haptic)
        
        # Test particle effects
        print("\n✨ Testing Particle Effects:")
        particle_types = ['coin_burst', 'level_up_stars', 'success_confetti']
        for particle in particle_types:
            self.create_particle_effect(None, particle)
        
        print("\n✅ Enhancement system test complete!")

def get_game_enhancement_manager():
    """Get singleton instance of game enhancement manager"""
    if not hasattr(get_game_enhancement_manager, '_instance'):
        get_game_enhancement_manager._instance = GameEnhancementManager()
    return get_game_enhancement_manager._instance

# Mobile-optimized enhancement helpers
def play_ui_sound(sound_name: str):
    """Quick helper for UI sound effects"""
    manager = get_game_enhancement_manager()
    manager.play_sound(sound_name)

def haptic_tap():
    """Quick helper for tap haptic feedback"""
    manager = get_game_enhancement_manager()
    manager.trigger_haptic_feedback('light')

def haptic_success():
    """Quick helper for success haptic feedback"""
    manager = get_game_enhancement_manager()
    manager.trigger_haptic_feedback('success')

def animate_button_press(button_widget):
    """Quick helper for button press animation"""
    manager = get_game_enhancement_manager()
    manager.animate_widget(button_widget, 'button_press')

def celebrate_achievement(widget):
    """Quick helper for achievement celebration"""
    manager = get_game_enhancement_manager()
    manager.play_sound('achievement_unlock')
    manager.trigger_haptic_feedback('success')
    manager.animate_widget(widget, 'level_up')
    manager.create_particle_effect(widget, 'success_confetti')

if __name__ == "__main__":
    # Initialize and test enhancement system
    enhancement_manager = GameEnhancementManager()
    enhancement_manager.create_sound_pack_info()
    enhancement_manager.test_all_enhancements()