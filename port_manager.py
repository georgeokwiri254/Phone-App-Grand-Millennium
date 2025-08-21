#!/usr/bin/env python3
"""
Port Management Utilities for Grand Millennium Revenue Analytics
Handles port conflicts by killing processes or finding available ports
"""

import socket
import subprocess
import sys
import psutil
import time
from typing import Optional, List


def is_port_in_use(port: int) -> bool:
    """Check if a port is currently in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(('localhost', port))
            return False
        except OSError:
            return True


def find_process_using_port(port: int) -> Optional[dict]:
    """Find the process using a specific port"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                for conn in proc.connections(kind='inet'):
                    if conn.laddr.port == port:
                        return {
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else 'N/A'
                        }
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        print(f"Error finding process using port {port}: {e}")
    return None


def kill_process_on_port(port: int) -> bool:
    """Kill the process using a specific port"""
    process_info = find_process_using_port(port)
    if not process_info:
        print(f"No process found using port {port}")
        return False
    
    try:
        print(f"Found process using port {port}:")
        print(f"  PID: {process_info['pid']}")
        print(f"  Name: {process_info['name']}")
        print(f"  Command: {process_info['cmdline']}")
        
        # Ask user for confirmation
        response = input(f"Do you want to kill this process? (y/N): ").lower().strip()
        if response in ['y', 'yes']:
            # Try to kill the process gracefully first
            try:
                proc = psutil.Process(process_info['pid'])
                proc.terminate()  # Send SIGTERM
                
                # Wait for process to terminate
                try:
                    proc.wait(timeout=5)
                    print(f"✅ Process {process_info['pid']} terminated successfully")
                    return True
                except psutil.TimeoutExpired:
                    # Force kill if it doesn't terminate gracefully
                    proc.kill()  # Send SIGKILL
                    print(f"✅ Process {process_info['pid']} force killed")
                    return True
                    
            except psutil.NoSuchProcess:
                print(f"Process {process_info['pid']} already terminated")
                return True
            except psutil.AccessDenied:
                print(f"❌ Access denied. Cannot kill process {process_info['pid']}")
                print("Try running as administrator or use a different port")
                return False
        else:
            print("Process not killed by user choice")
            return False
            
    except Exception as e:
        print(f"Error killing process: {e}")
        return False


def find_next_available_port(start_port: int = 8501, max_attempts: int = 100) -> int:
    """Find the next available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts}")


def get_available_port(preferred_port: int = 8511, auto_kill: bool = False) -> int:
    """
    Get an available port, handling conflicts appropriately
    
    Args:
        preferred_port: The preferred port to use
        auto_kill: If True, automatically kill processes on the preferred port
        
    Returns:
        An available port number
    """
    print(f"Checking port {preferred_port}...")
    
    if not is_port_in_use(preferred_port):
        print(f"✅ Port {preferred_port} is available")
        return preferred_port
    
    print(f"⚠️ Port {preferred_port} is already in use")
    
    if auto_kill:
        print("Attempting to kill process using the port...")
        if kill_process_on_port(preferred_port):
            time.sleep(1)  # Wait a moment for port to be released
            if not is_port_in_use(preferred_port):
                print(f"✅ Port {preferred_port} is now available")
                return preferred_port
            else:
                print("Port still in use after killing process")
    
    # Ask user what to do
    print("\nOptions:")
    print("1. Kill the process using the port")
    print("2. Use the next available port")
    print("3. Exit")
    
    while True:
        try:
            choice = input("Select an option (1-3): ").strip()
            
            if choice == '1':
                if kill_process_on_port(preferred_port):
                    time.sleep(1)  # Wait for port to be released
                    if not is_port_in_use(preferred_port):
                        print(f"✅ Port {preferred_port} is now available")
                        return preferred_port
                    else:
                        print("Port still in use after killing process")
                        continue
                else:
                    continue  # Try again or choose different option
                    
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


if __name__ == "__main__":
    # Test the port manager
    import argparse
    
    parser = argparse.ArgumentParser(description='Port Management Utility')
    parser.add_argument('--port', type=int, default=8511, help='Port to check')
    parser.add_argument('--kill', action='store_true', help='Automatically kill process on port')
    parser.add_argument('--find-next', action='store_true', help='Find next available port')
    
    args = parser.parse_args()
    
    if args.find_next:
        try:
            port = find_next_available_port(args.port)
            print(f"Next available port: {port}")
        except RuntimeError as e:
            print(f"Error: {e}")
    else:
        port = get_available_port(args.port, args.kill)
        print(f"Available port: {port}")