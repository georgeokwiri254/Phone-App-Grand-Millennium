#!/usr/bin/env python3
"""
PyInstaller build script for Grand Millennium Revenue Analytics
Creates standalone executable with all dependencies
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Project configuration
PROJECT_NAME = "GrandMillenniumAnalytics"
MAIN_SCRIPT = "scripts/launcher.py"
ICON_FILE = None  # Add path to .ico file if available

# Get project root
PROJECT_ROOT = Path(__file__).parent
BUILD_DIR = PROJECT_ROOT / "build"
DIST_DIR = PROJECT_ROOT / "dist"

def clean_build_dirs():
    """Clean previous build directories"""
    print("🧹 Cleaning previous build directories...")
    
    for dir_path in [BUILD_DIR, DIST_DIR]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"   Removed: {dir_path}")
    
    # Also remove .spec files
    for spec_file in PROJECT_ROOT.glob("*.spec"):
        spec_file.unlink()
        print(f"   Removed: {spec_file}")

def get_pyinstaller_command():
    """Build PyInstaller command"""
    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed" if sys.platform == "win32" else "--console",
        "--name", PROJECT_NAME,
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR),
        "--specpath", str(PROJECT_ROOT),
    ]
    
    # Add icon if available
    if ICON_FILE and Path(ICON_FILE).exists():
        cmd.extend(["--icon", ICON_FILE])
    
    # Add data files and directories
    data_additions = [
        # Add the entire app directory
        "--add-data", f"app{os.pathsep}app",
        # Add converters directory
        "--add-data", f"converters{os.pathsep}converters",
        # Add database directory (create if doesn't exist)
        "--add-data", f"db{os.pathsep}db",
        # Add data directories
        "--add-data", f"data{os.pathsep}data",
        # Add logs directory
        "--add-data", f"logs{os.pathsep}logs",
        # Add models directory
        "--add-data", f"models{os.pathsep}models",
        # Add config directory
        "--add-data", f"config{os.pathsep}config",
    ]
    
    cmd.extend(data_additions)
    
    # Hidden imports for packages that might not be detected
    hidden_imports = [
        "--hidden-import", "streamlit",
        "--hidden-import", "plotly",
        "--hidden-import", "pandas",
        "--hidden-import", "numpy",
        "--hidden-import", "sqlite3",
        "--hidden-import", "openpyxl",
        "--hidden-import", "statsmodels",
        "--hidden-import", "sklearn",
        "--hidden-import", "joblib",
    ]
    
    # Add webview if available
    try:
        import webview
        hidden_imports.extend(["--hidden-import", "webview"])
    except ImportError:
        print("⚠️  PyWebView not available, building without webview support")
    
    cmd.extend(hidden_imports)
    
    # Exclude unnecessary modules to reduce size
    excludes = [
        "--exclude-module", "tkinter",
        "--exclude-module", "matplotlib",
        "--exclude-module", "IPython",
        "--exclude-module", "jupyter",
        "--exclude-module", "notebook",
    ]
    
    cmd.extend(excludes)
    
    # Add main script
    cmd.append(str(PROJECT_ROOT / MAIN_SCRIPT))
    
    return cmd

def ensure_directories():
    """Ensure required directories exist"""
    print("📁 Ensuring required directories exist...")
    
    required_dirs = [
        PROJECT_ROOT / "db",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "config",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "raw",
    ]
    
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create empty .gitkeep file to ensure directory is included
        gitkeep = dir_path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    
    print("   ✅ All directories ready")

def create_readme():
    """Create README for distribution"""
    readme_content = """# Grand Millennium Revenue Analytics

## Installation and Usage

1. Extract all files to a directory
2. Run the executable file:
   - Windows: GrandMillenniumAnalytics.exe
   - Linux/Mac: ./GrandMillenniumAnalytics

## First Run

1. The application will create necessary directories and files
2. Go to the "Loading" tab to upload and process Excel files
3. Use other tabs for analysis after data is loaded

## Requirements

- No additional software installation required
- All dependencies are bundled with the executable

## Data Files

- Place Excel files (.xlsm) in the same directory as the executable
- Processed data will be stored in the 'data/processed' directory
- Database files are stored in the 'db' directory

## Logs

- Check 'logs/' directory for application logs
- Use the "Controls & Logs" tab to view recent log entries

## Support

For issues or questions, check the logs or contact support.

Generated with PyInstaller
"""
    
    readme_path = PROJECT_ROOT / "README_EXECUTABLE.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"📝 Created: {readme_path}")

def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'streamlit', 'pandas', 'plotly', 'numpy', 'openpyxl',
        'statsmodels', 'scikit-learn', 'joblib', 'pyinstaller'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def build_executable():
    """Build the executable"""
    print("🏗️  Building executable...")
    
    # Get PyInstaller command
    cmd = get_pyinstaller_command()
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run PyInstaller
    try:
        result = subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        print("✅ Build completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed with exit code {e.returncode}")
        return False

def post_build_tasks():
    """Perform post-build tasks"""
    print("📦 Performing post-build tasks...")
    
    # Check if executable was created
    if sys.platform == "win32":
        exe_name = f"{PROJECT_NAME}.exe"
    else:
        exe_name = PROJECT_NAME
    
    exe_path = DIST_DIR / exe_name
    
    if not exe_path.exists():
        print(f"❌ Executable not found: {exe_path}")
        return False
    
    print(f"✅ Executable created: {exe_path}")
    print(f"📏 Size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Copy additional files to dist directory
    additional_files = [
        "README.md",
        "requirements.txt",
        "claude.md",
        "README_EXECUTABLE.md"
    ]
    
    for file_name in additional_files:
        src = PROJECT_ROOT / file_name
        if src.exists():
            dst = DIST_DIR / file_name
            shutil.copy2(src, dst)
            print(f"📄 Copied: {file_name}")
    
    return True

def main():
    """Main build process"""
    print("🚀 Grand Millennium Revenue Analytics - Build Script")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Clean previous builds
    clean_build_dirs()
    
    # Ensure directories exist
    ensure_directories()
    
    # Create README for distribution
    create_readme()
    
    # Build executable
    if not build_executable():
        sys.exit(1)
    
    # Post-build tasks
    if not post_build_tasks():
        sys.exit(1)
    
    print("\n🎉 Build completed successfully!")
    print(f"📂 Executable location: {DIST_DIR}")
    print("\nTo test the executable:")
    print(f"   cd {DIST_DIR}")
    if sys.platform == "win32":
        print(f"   .\\{PROJECT_NAME}.exe")
    else:
        print(f"   ./{PROJECT_NAME}")

if __name__ == "__main__":
    main()