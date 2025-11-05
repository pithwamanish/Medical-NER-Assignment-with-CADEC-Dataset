#!/usr/bin/env python3
"""
Project setup script for CADEC NER Project.

This script helps set up the project environment, verify dependencies,
and ensure all directories are properly configured.
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8+."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        "numpy",
        "pandas",
        "torch",
        "transformers",
        "sentence_transformers",
        "seqeval",
        "sklearn",
        "matplotlib",
        "seaborn",
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} not found")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create output directories if they don't exist."""
    directories = [
        Path("outputs"),
        Path("outputs/models"),
        Path("outputs/results"),
        Path("outputs/cache"),
        Path("outputs/logs"),
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory ready: {directory}")

def verify_dataset():
    """Verify CADEC dataset structure."""
    required_dirs = [
        Path("cadec/text"),
        Path("cadec/original"),
        Path("cadec/sct"),
        Path("cadec/meddra"),
    ]
    
    all_exist = True
    for directory in required_dirs:
        if directory.exists():
            file_count = len(list(directory.glob("*")))
            print(f"✅ {directory}: {file_count} files")
        else:
            print(f"⚠️  {directory}: not found (optional)")
            all_exist = False
    
    return all_exist

def test_imports():
    """Test if project modules can be imported."""
    try:
        import config
        print("✅ config module imported")
        
        from src.data_loader import load_ground_truth
        print("✅ data_loader module imported")
        
        from src.evaluation import calculate_metrics
        print("✅ evaluation module imported")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Run setup checks."""
    print("=" * 60)
    print("CADEC NER Project Setup")
    print("=" * 60)
    print()
    
    checks_passed = True
    
    print("1. Checking Python version...")
    if not check_python_version():
        checks_passed = False
    print()
    
    print("2. Checking dependencies...")
    if not check_dependencies():
        checks_passed = False
    print()
    
    print("3. Creating output directories...")
    create_directories()
    print()
    
    print("4. Verifying dataset structure...")
    verify_dataset()
    print()
    
    print("5. Testing module imports...")
    if not test_imports():
        checks_passed = False
    print()
    
    print("=" * 60)
    if checks_passed:
        print("✅ Setup complete! Project is ready to use.")
    else:
        print("⚠️  Setup incomplete. Please address the issues above.")
    print("=" * 60)

if __name__ == "__main__":
    main()

