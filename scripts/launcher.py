#!/usr/bin/env python3
"""
Grand Millennium Revenue Analytics Launcher
Launches the Streamlit application with proper configuration
"""

import sys
import subprocess
import os
import webbrowser
import time
from pathlib import Path

def main():
    """Launch the Streamlit application"""
    try:
        # Get the project root directory (parent of scripts)
        project_root = Path(__file__).parent.parent
        app_path = project_root / "app" / "streamlit_app_simple.py"
        
        print("Grand Millennium Revenue Analytics")
        print("=" * 50)
        print(f"Project root: {project_root}")
        print(f"App path: {app_path}")
        
        # Verify the app file exists
        if not app_path.exists():
            print(f"ERROR: Streamlit app not found at {app_path}")
            input("Press Enter to exit...")
            return
        
        # Change to project directory
        os.chdir(project_root)
        
        print("Starting Streamlit server...")
        print("This will open your web browser automatically.")
        print("If it doesn't open, go to: http://localhost:8501")
        print("")
        print("Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Launch Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
        
        # Run the command
        process = subprocess.Popen(cmd, cwd=project_root)
        
        # Wait a moment then try to open browser
        time.sleep(3)
        try:
            webbrowser.open("http://localhost:8501")
        except:
            pass  # Browser opening is optional
        
        # Wait for the process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error launching application: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()