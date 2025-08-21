#!/usr/bin/env python3
"""
Grand Millennium Revenue Analytics - Enhanced Launcher with Port Management
This script handles port conflicts automatically
"""

import sys
import subprocess
import os
import webbrowser
import time
import socket
from pathlib import Path

def is_port_in_use(port: int) -> bool:
    """Check if a port is currently in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(('localhost', port))
            return False
        except OSError:
            return True

def find_next_available_port(start_port: int = 8511, max_attempts: int = 50) -> int:
    """Find the next available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts}")

def kill_port_process_windows(port: int) -> bool:
    """Kill process using port on Windows using netstat and taskkill"""
    try:
        # Find process using the port
        result = subprocess.run(
            ['netstat', '-ano'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        lines = result.stdout.split('\n')
        for line in lines:
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    print(f"Found process PID {pid} using port {port}")
                    
                    # Ask for confirmation
                    response = input(f"Kill process {pid} using port {port}? (y/N): ").lower().strip()
                    if response in ['y', 'yes']:
                        # Kill the process
                        kill_result = subprocess.run(
                            ['taskkill', '/PID', pid, '/F'], 
                            capture_output=True, 
                            text=True
                        )
                        if kill_result.returncode == 0:
                            print(f"✅ Successfully killed process {pid}")
                            time.sleep(2)  # Wait for port to be released
                            return True
                        else:
                            print(f"❌ Failed to kill process {pid}: {kill_result.stderr}")
                    else:
                        print("Process not killed by user choice")
                    return False
        
        print(f"No process found using port {port}")
        return False
        
    except Exception as e:
        print(f"Error killing process on port {port}: {e}")
        return False

def get_available_port(preferred_port: int = 8511) -> int:
    """Get an available port, handling conflicts"""
    print(f"Checking port {preferred_port}...")
    
    if not is_port_in_use(preferred_port):
        print(f"✅ Port {preferred_port} is available")
        return preferred_port
    
    print(f"⚠️ Port {preferred_port} is already in use")
    print("\nOptions:")
    print("1. Kill the process using the port")
    print("2. Use the next available port")
    print("3. Exit")
    
    while True:
        try:
            choice = input("Select an option (1-3): ").strip()
            
            if choice == '1':
                if kill_port_process_windows(preferred_port):
                    if not is_port_in_use(preferred_port):
                        print(f"✅ Port {preferred_port} is now available")
                        return preferred_port
                    else:
                        print("Port still in use after killing process")
                        continue
                else:
                    continue
                    
            elif choice == '2':
                try:
                    available_port = find_next_available_port(preferred_port + 1)
                    print(f"✅ Using port {available_port} instead")
                    return available_port
                except RuntimeError as e:
                    print(f"❌ {e}")
                    continue
                    
            elif choice == '3':
                print("Exiting...")
                sys.exit(0)
                
            else:
                print("Invalid choice. Please select 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

def main():
    """Main launcher function"""
    try:
        # Get the project root directory
        project_root = Path(__file__).parent
        app_path = project_root / "app" / "streamlit_app_simple.py"
        
        print("Grand Millennium Revenue Analytics - Enhanced Launcher")
        print("=" * 60)
        print(f"Project root: {project_root}")
        print(f"App path: {app_path}")
        
        # Verify the app file exists
        if not app_path.exists():
            print(f"ERROR: Streamlit app not found at {app_path}")
            input("Press Enter to exit...")
            return
        
        # Change to project directory
        os.chdir(project_root)
        
        # Get available port with conflict resolution
        port = get_available_port(8511)
        
        print("")
        print("🚀 Starting Streamlit server...")
        print("This will open your web browser automatically.")
        print(f"🌐 Server URL: http://localhost:{port}")
        print("")
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Launch Streamlit with the determined port
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        print("")
        
        # Run the command
        process = subprocess.Popen(cmd, cwd=project_root)
        
        # Wait a moment then try to open browser
        time.sleep(4)
        try:
            webbrowser.open(f"http://localhost:{port}")
            print(f"🌍 Browser opened at http://localhost:{port}")
        except Exception as e:
            print(f"⚠️ Could not open browser automatically: {e}")
            print(f"Please manually open: http://localhost:{port}")
        
        # Wait for the process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        try:
            process.terminate()
        except:
            pass
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()