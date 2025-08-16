#!/usr/bin/env python3
"""
Quick test script to verify the Streamlit app can start
"""

import sys
import os
import io
from pathlib import Path

# Fix encoding issues on Windows
if sys.platform == "win32" and not hasattr(sys.stdout, '_wrapped'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stdout._wrapped = True
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        sys.stderr._wrapped = True
    except (AttributeError, ValueError):
        # stdout/stderr might already be wrapped or unavailable
        pass

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    # Test basic imports
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    print("[OK] Core dependencies imported successfully")
    
    # Test app module import
    from app import streamlit_app
    print("[OK] Streamlit app module imported successfully")
    
    # Test database connection
    from app.database import RevenueDatabase
    db = RevenueDatabase()
    print("[OK] Database initialization successful")
    
    print("\n[SUCCESS] All tests passed! App should start successfully.")
    
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("Installing missing dependencies...")
    os.system("pip install -r requirements.txt")
    
except Exception as e:
    print(f"[ERROR] Error: {e}")
    print("Check the error above and fix any issues")