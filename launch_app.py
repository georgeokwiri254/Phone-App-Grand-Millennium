#!/usr/bin/env python3
"""
Double-click launcher for Grand Millennium Revenue Analytics
"""

import sys
import subprocess
from pathlib import Path

def main():
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    launcher_script = script_dir / "scripts" / "launcher.py"
    
    try:
        # Run the launcher script
        subprocess.run([sys.executable, str(launcher_script)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching app: {e}")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"Unexpected error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()