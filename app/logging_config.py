"""
Logging Configuration for Revenue Analytics Application
Sets up rotating file handlers for conversion and application logs
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging():
    """Set up logging configuration with rotating file handlers"""
    
    # Create logs directory
    project_root = Path(__file__).parent.parent
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up conversion log (for converter operations)
    conversion_log_path = logs_dir / 'conversion.log'
    conversion_handler = RotatingFileHandler(
        conversion_log_path,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    conversion_handler.setLevel(logging.INFO)
    conversion_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    conversion_handler.setFormatter(conversion_formatter)
    
    # Create conversion logger
    conversion_logger = logging.getLogger('conversion')
    conversion_logger.addHandler(conversion_handler)
    conversion_logger.setLevel(logging.INFO)
    
    # Set up application log (for Streamlit app operations)
    app_log_path = logs_dir / 'app.log'
    app_handler = RotatingFileHandler(
        app_log_path,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    app_handler.setLevel(logging.INFO)
    app_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    app_handler.setFormatter(app_formatter)
    
    # Create app logger
    app_logger = logging.getLogger('app')
    app_logger.addHandler(app_handler)
    app_logger.setLevel(logging.INFO)
    
    return conversion_logger, app_logger

def get_conversion_logger():
    """Get the conversion logger instance"""
    return logging.getLogger('conversion')

def get_app_logger():
    """Get the application logger instance"""
    return logging.getLogger('app')

def get_log_content(log_name: str, lines: int = 100) -> str:
    """
    Get the last N lines from a log file
    
    Args:
        log_name: Name of log file ('conversion' or 'app')
        lines: Number of lines to return from the end
        
    Returns:
        String containing the last N lines of the log
    """
    try:
        project_root = Path(__file__).parent.parent
        log_path = project_root / 'logs' / f'{log_name}.log'
        
        if not log_path.exists():
            return f"Log file {log_name}.log not found"
        
        with open(log_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            
        # Return last N lines
        last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return ''.join(last_lines)
        
    except Exception as e:
        return f"Error reading log file: {str(e)}"

# Initialize logging when module is imported
setup_logging()