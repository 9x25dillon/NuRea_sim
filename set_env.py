#!/usr/bin/env python3
"""
Environment Setup Script for HRM Project

This script sets the PYTHONPATH and other environment variables
since .env files are blocked by global ignore.
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Set up environment variables for the HRM project."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    
    # Set PYTHONPATH
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if str(project_root) not in current_pythonpath:
        if current_pythonpath:
            new_pythonpath = f"{project_root}{os.pathsep}{current_pythonpath}"
        else:
            new_pythonpath = str(project_root)
        
        os.environ['PYTHONPATH'] = new_pythonpath
        print(f"✅ PYTHONPATH set to: {new_pythonpath}")
    else:
        print(f"✅ PYTHONPATH already contains: {project_root}")
    
    # Set CUDA environment variables (if CUDA is available)
    cuda_paths = [
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7",
    ]
    
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            os.environ['CUDA_HOME'] = cuda_path
            os.environ['CUDA_PATH'] = cuda_path
            print(f"✅ CUDA_HOME set to: {cuda_path}")
            break
    else:
        print("⚠️  CUDA not found in common locations")
    
    return True

def print_environment_info():
    """Print current environment information."""
    print("\n🔍 Current Environment:")
    print(f"   PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"   CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    print(f"   CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
    print(f"   Project Root: {Path(__file__).parent.absolute()}")

if __name__ == "__main__":
    print("🚀 Setting up HRM Project Environment...")
    setup_environment()
    print_environment_info()
    
    print("\n💡 To use this in your scripts:")
    print("   from set_env import setup_environment")
    print("   setup_environment()")
    print("\n   Or run this script before starting your main application.")
