"""
Grand Millennium Revenue Analytics - Mobile Game Main Entry Point

Main application entry point for Android APK compilation.
Initializes the complete mobile game with all optimizations.
"""

__version__ = '1.0.0'

import os
import sys
from pathlib import Path

# Add parent directory to Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

try:
    # Configure Kivy for Android
    import kivy
    kivy.require('2.1.0')
    
    from kivy.config import Config
    from kivy.utils import platform
    
    # Android-specific configurations
    if platform == 'android':
        # Optimize for mobile performance
        Config.set('graphics', 'multisamples', '0')
        Config.set('kivy', 'window_icon', '')
        Config.set('kivy', 'exit_on_escape', '0')
        
        # Handle Android permissions
        from android.permissions import request_permissions, Permission
        request_permissions([
            Permission.WRITE_EXTERNAL_STORAGE,
            Permission.READ_EXTERNAL_STORAGE,
            Permission.INTERNET,
            Permission.VIBRATE
        ])
        
        # Android-specific imports
        from jnius import autoclass
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        
    from kivy.core.window import Window
    
    # Import mobile game application
    from complete_mobile_game import CompleteMobileGameApp
    from mobile_touch_optimizer import mobile_ux_enhancer
    from game_flow_manager import get_game_flow_manager
    from data_integration import get_data_integration_manager
    
    print("🎮 Grand Millennium Revenue Analytics - Mobile Game v{}".format(__version__))
    print("=" * 60)
    
    def main():
        """Main application entry point"""
        try:
            # Initialize mobile optimizations
            print("📱 Initializing mobile optimizations...")
            
            # Setup window for mobile
            if platform == 'android':
                # Android-specific window setup
                Window.softinput_mode = 'below_target'
                Window.keyboard_anim_args = {'d': 0.2, 't': 'in_out_expo'}
            else:
                # Desktop testing mode
                Window.size = (360, 640)
            
            # Initialize data integration
            print("🔄 Setting up data integration...")
            data_manager = get_data_integration_manager()
            
            # Sync data in background for mobile performance
            import threading
            def background_sync():
                try:
                    sync_results = data_manager.sync_with_streamlit_data()
                    successful_syncs = sum(1 for success in sync_results.values() if success)
                    print(f"📊 Background sync: {successful_syncs}/{len(sync_results)} sources")
                except Exception as e:
                    print(f"⚠️  Background sync error: {e}")
            
            sync_thread = threading.Thread(target=background_sync, daemon=True)
            sync_thread.start()
            
            # Initialize game flow manager
            print("🎯 Initializing game flow...")
            flow_manager = get_game_flow_manager()
            
            # Create and run mobile game app
            print("🚀 Starting Grand Millennium Revenue Analytics Mobile Game...")
            app = CompleteMobileGameApp()
            app.title = "Grand Millennium Revenue Analytics"
            
            # Mobile-specific app settings
            if platform == 'android':
                # Prevent sleep mode during gameplay
                from plyer import wakelock
                wakelock.acquire()
            
            # Run the application
            app.run()
            
            # Cleanup
            if platform == 'android':
                try:
                    wakelock.release()
                except:
                    pass
            
            print("👋 Game session ended")
            
        except Exception as e:
            print(f"❌ Application error: {e}")
            import traceback
            traceback.print_exc()
            
            # Show error dialog on mobile
            if platform == 'android':
                try:
                    from plyer import notification
                    notification.notify(
                        title='Grand Millennium Revenue Game',
                        message=f'Error: {str(e)[:50]}...',
                        timeout=10
                    )
                except:
                    pass
            
            sys.exit(1)
    
    if __name__ == '__main__':
        main()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📦 Please install required dependencies:")
    print("pip install kivy kivymd pandas numpy")
    sys.exit(1)
except Exception as e:
    print(f"❌ Startup error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)