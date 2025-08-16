"""
Converter wrapper functions for Grand Millennium Revenue Analytics
Ensures canonical output paths and consistent interfaces
"""

import sys
import os
from pathlib import Path

# Add the converters directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .segment_converter import run_segment_conversion
from .occupancy_converter import run_occupancy_conversion

__all__ = ['run_segment_conversion', 'run_occupancy_conversion']