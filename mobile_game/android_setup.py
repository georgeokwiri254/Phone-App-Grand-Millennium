"""
Grand Millennium Revenue Analytics - Android Setup and Build Script

Automated setup and build script for Android APK generation using Buildozer.
Handles dependencies, configuration, and build process.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import shutil
import json

class AndroidBuilder:
    """Android APK builder using Buildozer"""
    
    def __init__(self):
        """Initialize Android builder"""
        self.project_root = Path(__file__).parent
        self.buildozer_spec = self.project_root / "buildozer.spec"
        self.assets_dir = self.project_root / "assets"
        
        # Build configuration
        self.build_config = {
            'debug': True,
            'release': False,
            'clean_build': False,
            'update_buildozer': True
        }
        
        # System requirements
        self.system_requirements = {
            'python': '3.8+',
            'java': '1.8+',
            'buildozer': '1.4.0+',
            'cython': '0.29+',
            'git': 'latest'
        }
    
    def check_system_requirements(self):
        """Check if system meets requirements for Android building"""
        print("🔍 Checking System Requirements...")
        print("=" * 40)
        
        missing_requirements = []
        
        # Check Python version
        try:
            python_version = sys.version_info
            if python_version < (3, 8):
                missing_requirements.append(f"Python 3.8+ (current: {python_version.major}.{python_version.minor})")
            else:
                print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        except Exception as e:
            missing_requirements.append(f"Python version check failed: {e}")
        
        # Check Java
        try:
            java_result = subprocess.run(['java', '-version'], 
                                       capture_output=True, text=True, timeout=10)
            if java_result.returncode == 0:
                print("✅ Java installed")
            else:
                missing_requirements.append("Java 1.8+")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            missing_requirements.append("Java 1.8+ (OpenJDK recommended)")
        
        # Check Git
        try:
            git_result = subprocess.run(['git', '--version'], 
                                      capture_output=True, text=True, timeout=10)
            if git_result.returncode == 0:
                print("✅ Git installed")
            else:
                missing_requirements.append("Git")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            missing_requirements.append("Git")
        
        # Check Buildozer
        try:
            buildozer_result = subprocess.run(['buildozer', 'version'], 
                                            capture_output=True, text=True, timeout=10)
            if buildozer_result.returncode == 0:
                print("✅ Buildozer installed")
            else:
                missing_requirements.append("Buildozer")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            missing_requirements.append("Buildozer")
        
        # Check Cython
        try:
            import Cython
            print(f"✅ Cython {Cython.__version__}")
        except ImportError:
            missing_requirements.append("Cython")
        
        if missing_requirements:
            print("\n❌ Missing Requirements:")
            for req in missing_requirements:
                print(f"   - {req}")
            return False
        
        print("\n✅ All system requirements met!")
        return True
    
    def install_dependencies(self):
        """Install required Python dependencies"""
        print("\n📦 Installing Python Dependencies...")
        print("=" * 40)
        
        # Core dependencies
        dependencies = [
            'buildozer>=1.4.0',
            'cython>=0.29.0',
            'kivy>=2.1.0',
            'kivymd>=1.0.0',
            'pandas>=1.5.0',
            'numpy>=1.21.0',
            'Pillow>=8.0.0',
            'python-dateutil>=2.8.0',
            'requests>=2.25.0',
            'plyer>=2.0.0'
        ]
        
        for dep in dependencies:
            try:
                print(f"Installing {dep}...")
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', dep
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"✅ {dep} installed")
                else:
                    print(f"⚠️  {dep} installation warning: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print(f"⏰ {dep} installation timed out")
            except Exception as e:
                print(f"❌ {dep} installation failed: {e}")
        
        print("📦 Dependency installation complete")
    
    def create_assets(self):
        """Create required assets for Android build"""
        print("\n🎨 Creating Android Assets...")
        print("=" * 30)
        
        # Create assets directory
        self.assets_dir.mkdir(exist_ok=True)
        
        # Create app icon (placeholder)
        icon_content = '''
        # App Icon Placeholder
        # For production, replace with actual 512x512 PNG icon
        # Path: mobile_game/assets/icon.png
        '''
        
        icon_placeholder = self.assets_dir / "icon_readme.txt"
        with open(icon_placeholder, 'w') as f:
            f.write(icon_content.strip())
        
        # Create splash screen (placeholder)
        splash_content = '''
        # Splash Screen Placeholder
        # For production, replace with actual 1920x1080 PNG splash screen
        # Path: mobile_game/assets/splash.png
        '''
        
        splash_placeholder = self.assets_dir / "splash_readme.txt"
        with open(splash_placeholder, 'w') as f:
            f.write(splash_content.strip())
        
        print("✅ Asset placeholders created")
        print("📝 Note: Replace placeholders with actual PNG files for production")
    
    def configure_buildozer(self):
        """Configure buildozer.spec file"""
        print("\n⚙️  Configuring Buildozer...")
        print("=" * 30)
        
        if not self.buildozer_spec.exists():
            print("❌ buildozer.spec not found!")
            return False
        
        # Read current buildozer.spec
        with open(self.buildozer_spec, 'r') as f:
            spec_content = f.read()
        
        print("✅ Buildozer configuration verified")
        
        # Validate key settings
        required_settings = [
            'title = Grand Millennium Revenue Analytics',
            'package.name = grandmillenniumrevenue',
            'package.domain = com.grandmillenniumdubai',
            'source.main = main.py',
            'requirements = python3,kivy'
        ]
        
        missing_settings = []
        for setting in required_settings:
            key = setting.split('=')[0].strip()
            if key not in spec_content:
                missing_settings.append(setting)
        
        if missing_settings:
            print("⚠️  Missing buildozer settings:")
            for setting in missing_settings:
                print(f"   - {setting}")
        else:
            print("✅ All required buildozer settings present")
        
        return True
    
    def clean_build_environment(self):
        """Clean previous build artifacts"""
        print("\n🧹 Cleaning Build Environment...")
        print("=" * 35)
        
        # Directories to clean
        clean_dirs = [
            '.buildozer',
            'bin',
            '__pycache__',
            '.pytest_cache'
        ]
        
        for clean_dir in clean_dirs:
            dir_path = self.project_root / clean_dir
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    print(f"✅ Cleaned {clean_dir}/")
                except Exception as e:
                    print(f"⚠️  Could not clean {clean_dir}/: {e}")
            else:
                print(f"ℹ️  {clean_dir}/ not found (already clean)")
        
        print("🧹 Build environment cleaned")
    
    def build_apk(self, build_type='debug'):
        """Build Android APK"""
        print(f"\n🔨 Building Android APK ({build_type})...")
        print("=" * 40)
        
        # Change to project directory
        os.chdir(self.project_root)
        
        try:
            # Buildozer command
            if build_type == 'debug':
                cmd = ['buildozer', 'android', 'debug']
            else:
                cmd = ['buildozer', 'android', 'release']
            
            print(f"Running: {' '.join(cmd)}")
            print("⏳ This may take 10-30 minutes for first build...")
            
            # Run buildozer
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            return_code = process.poll()
            
            if return_code == 0:
                print("\n🎉 APK Build Successful!")
                
                # Find generated APK
                bin_dir = self.project_root / "bin"
                if bin_dir.exists():
                    apk_files = list(bin_dir.glob("*.apk"))
                    if apk_files:
                        latest_apk = max(apk_files, key=os.path.getctime)
                        apk_size = latest_apk.stat().st_size / (1024 * 1024)  # MB
                        print(f"📱 APK Location: {latest_apk}")
                        print(f"📏 APK Size: {apk_size:.1f} MB")
                        return str(latest_apk)
                
                return True
            else:
                print(f"\n❌ APK Build Failed (exit code: {return_code})")
                return False
                
        except subprocess.TimeoutExpired:
            print("\n⏰ Build timed out (>30 minutes)")
            return False
        except KeyboardInterrupt:
            print("\n⏹️  Build interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Build error: {e}")
            return False
    
    def create_build_info(self, apk_path=None):
        """Create build information file"""
        build_info = {
            'app_name': 'Grand Millennium Revenue Analytics',
            'version': '1.0.0',
            'build_date': str(Path(__file__).stat().st_mtime),
            'platform': 'Android',
            'build_type': 'debug' if self.build_config['debug'] else 'release',
            'apk_path': str(apk_path) if apk_path else None,
            'features': [
                'Revenue Analytics Dashboard',
                'Daily Challenges with AED Currency',
                'Segment Analysis Games',
                'Pricing Optimization',
                'Level Progression System',
                'Mobile Touch Interface',
                'Responsive Design',
                'Offline Capability'
            ],
            'requirements': {
                'android_version': '5.0+ (API 21)',
                'storage': '50 MB',
                'ram': '2 GB recommended',
                'permissions': [
                    'Internet Access',
                    'Storage Access',
                    'Vibration',
                    'Wake Lock'
                ]
            }
        }
        
        build_info_file = self.project_root / "build_info.json"
        with open(build_info_file, 'w') as f:
            json.dump(build_info, f, indent=2)
        
        print(f"📋 Build info saved to: {build_info_file}")
    
    def run_full_build(self):
        """Run complete Android build process"""
        print("🚀 Grand Millennium Revenue Analytics - Android Build")
        print("=" * 60)
        
        # Step 1: Check system requirements
        if not self.check_system_requirements():
            print("\n❌ System requirements not met. Please install missing components.")
            return False
        
        # Step 2: Install dependencies
        self.install_dependencies()
        
        # Step 3: Create assets
        self.create_assets()
        
        # Step 4: Configure buildozer
        if not self.configure_buildozer():
            print("❌ Buildozer configuration failed")
            return False
        
        # Step 5: Clean build environment (if requested)
        if self.build_config['clean_build']:
            self.clean_build_environment()
        
        # Step 6: Build APK
        build_result = self.build_apk('debug' if self.build_config['debug'] else 'release')
        
        if build_result:
            # Step 7: Create build info
            apk_path = build_result if isinstance(build_result, str) else None
            self.create_build_info(apk_path)
            
            print("\n🎉 Android Build Process Complete!")
            print("=" * 40)
            print("📱 Your Grand Millennium Revenue Analytics APK is ready!")
            if apk_path:
                print(f"📂 APK Location: {apk_path}")
            print("📋 Build info saved to build_info.json")
            
            return True
        else:
            print("\n❌ Android Build Process Failed")
            print("📝 Check the build output above for errors")
            return False

def main():
    """Main build script entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Grand Millennium Revenue Analytics - Android Builder')
    parser.add_argument('--clean', action='store_true', help='Clean build environment before building')
    parser.add_argument('--release', action='store_true', help='Build release APK (default: debug)')
    parser.add_argument('--check-only', action='store_true', help='Only check system requirements')
    
    args = parser.parse_args()
    
    # Create builder
    builder = AndroidBuilder()
    
    # Update build configuration
    builder.build_config['clean_build'] = args.clean
    builder.build_config['debug'] = not args.release
    builder.build_config['release'] = args.release
    
    if args.check_only:
        # Only check requirements
        if builder.check_system_requirements():
            print("✅ System ready for Android building!")
            return 0
        else:
            print("❌ System requirements not met")
            return 1
    else:
        # Run full build
        if builder.run_full_build():
            return 0
        else:
            return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)