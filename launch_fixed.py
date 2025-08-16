#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed launcher for Grand Millennium Revenue Analytics Dashboard
Sets proper encoding environment variables for Windows
"""

import os
import sys
import subprocess
import socket
import webbrowser
import time
import threading
from pathlib import Path

def find_free_port(start_port=8501, max_attempts=10):
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', port))
                s.listen(1)
                return port
        except OSError:
            continue
    return None

def set_encoding_environment():
    """Set proper encoding environment variables for Windows."""
    # Set UTF-8 encoding environment variables
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    
    # Set console code page to UTF-8 on Windows
    if sys.platform == "win32":
        try:
            # Change console code page to UTF-8
            os.system('chcp 65001 > nul')
        except Exception:
            pass

def kill_existing_streamlit():
    """Kill existing Streamlit processes on Windows."""
    if sys.platform == "win32":
        try:
            # Kill any existing streamlit processes
            subprocess.run(['taskkill', '/f', '/im', 'streamlit.exe'], 
                         shell=True, capture_output=True)
            # Kill processes using port 8501
            result = subprocess.run(['netstat', '-ano'], 
                                  capture_output=True, text=True, shell=True)
            for line in result.stdout.split('\n'):
                if ':8501' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) > 4:
                        pid = parts[-1]
                        subprocess.run(['taskkill', '/f', '/pid', pid], 
                                     shell=True, capture_output=True)
        except Exception:
            pass

def wait_for_server(port, timeout=30):
    """Wait for the Streamlit server to start."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                if result == 0:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False

def open_browser_when_ready(port):
    """Open browser when server is ready."""
    if wait_for_server(port):
        print(f"Server is ready! Opening browser...")
        webbrowser.open(f'http://localhost:{port}')
    else:
        print(f"Server didn't start within timeout. Please open manually: http://localhost:{port}")

def main():
    """Main launcher function."""
    print("Setting up Grand Millennium Revenue Analytics Dashboard...")
    
    # Set encoding environment
    set_encoding_environment()
    
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Change to project directory
    os.chdir(project_root)
    
    # Add project root to PYTHONPATH
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Find a free port
    port = find_free_port()
    if port is None:
        print("Could not find a free port. Trying to kill existing processes...")
        kill_existing_streamlit()
        port = find_free_port()
        if port is None:
            port = 8502  # fallback
    
    print(f"Starting dashboard on port {port}...")
    
    # Launch streamlit via subprocess (more reliable)
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "app/streamlit_app.py", 
            f"--server.port={port}",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--server.address=localhost"
        ]
        
        # Set environment
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root)
        
        print(f"Launching: {' '.join(cmd)}")
        print(f"Dashboard starting on port {port}...")
        
        # Start the server process
        process = subprocess.Popen(cmd, env=env)
        
        # Start browser opening in a separate thread
        browser_thread = threading.Thread(target=open_browser_when_ready, args=(port,))
        browser_thread.daemon = True
        browser_thread.start()
        
        print(f"Server starting... Browser will open automatically at:")
        print(f"http://localhost:{port}")
        print("Press Ctrl+C to stop the dashboard")
        
        # Wait for process
        process.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
        try:
            process.terminate()
        except:
            pass
    except Exception as e:
        print(f"Error launching Streamlit: {e}")
        print("Please check if Streamlit is installed: pip install streamlit")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()