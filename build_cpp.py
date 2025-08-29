#!/usr/bin/env python3
"""
Build script for C++ revenue analytics module
"""

import subprocess
import sys
import os
from pathlib import Path

def build_cpp_module():
    """Build the C++ module using setup.py"""
    try:
        print("Building C++ revenue analytics module...")
        
        # Change to project directory
        project_dir = Path(__file__).parent
        os.chdir(project_dir)
        
        # Build the extension
        result = subprocess.run([
            sys.executable, "setup.py", "build_ext", "--inplace"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ C++ module built successfully!")
            print("Output:", result.stdout)
            return True
        else:
            print("❌ Failed to build C++ module:")
            print("Error:", result.stderr)
            print("Output:", result.stdout)
            return False
            
    except Exception as e:
        print(f"❌ Error building C++ module: {e}")
        return False

def install_requirements():
    """Install required packages"""
    try:
        print("Installing pybind11...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "pybind11"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ pybind11 installed successfully!")
            return True
        else:
            print("❌ Failed to install pybind11:")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return False

if __name__ == "__main__":
    print("Revenue Analytics C++ Module Builder")
    print("=" * 40)
    
    # Install requirements first
    if not install_requirements():
        sys.exit(1)
    
    # Build the module
    if build_cpp_module():
        print("\n🎉 Build complete! The C++ module is ready to use.")
        print("You can now run your Streamlit app with fast C++ computations.")
    else:
        print("\n⚠️  Build failed. The app will use Python fallback implementations.")
        sys.exit(1)