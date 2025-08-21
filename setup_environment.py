#!/usr/bin/env python3
"""
Environment Setup Script for HRM Project

This script helps set up the Python environment, install dependencies,
and configure CUDA settings for the HRM project.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def check_cuda_installation():
    """Check CUDA installation and set environment variables."""
    print("\n🔍 Checking CUDA installation...")
    
    # Common CUDA installation paths
    cuda_paths = [
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7",
        "/usr/local/cuda-12.0",
        "/usr/local/cuda-11.8",
        "/usr/local/cuda-11.7",
    ]
    
    cuda_home = None
    for path in cuda_paths:
        if os.path.exists(path):
            cuda_home = path
            break
    
    if cuda_home:
        print(f"✅ CUDA found at: {cuda_home}")
        os.environ['CUDA_HOME'] = cuda_home
        os.environ['CUDA_PATH'] = cuda_home
        
        # Add CUDA to PATH if not already there
        cuda_bin = os.path.join(cuda_home, 'bin')
        if cuda_bin not in os.environ.get('PATH', ''):
            os.environ['PATH'] = cuda_bin + os.pathsep + os.environ.get('PATH', '')
        
        return True
    else:
        print("⚠️  CUDA not found in common locations")
        print("   You may need to install CUDA or set CUDA_HOME manually")
        return False


def check_pytorch_cuda():
    """Check if PyTorch can access CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  PyTorch CUDA not available")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def create_env_file():
    """Create .env file with environment variables."""
    print("\n📝 Creating .env file...")
    
    env_content = """# HRM Project Environment Variables
PYTHONPATH=${PYTHONPATH}:${PWD}

# CUDA Configuration
CUDA_HOME=${CUDA_HOME}
CUDA_PATH=${CUDA_PATH}

# Python path for the project
PYTHONPATH=${PYTHONPATH}:${workspaceFolder}
"""
    
    env_file = Path(".env")
    if env_file.exists():
        print("   .env file already exists")
    else:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("   ✅ .env file created")
    
    return True


def install_dependencies():
    """Install project dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Check if virtual environment exists
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("   Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
            print("   ✅ Virtual environment created")
        except subprocess.CalledProcessError:
            print("   ❌ Failed to create virtual environment")
            return False
    
    # Determine pip path
    if platform.system() == "Windows":
        pip_path = ".venv/Scripts/pip.exe"
    else:
        pip_path = ".venv/bin/pip"
    
    # Upgrade pip
    try:
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        print("   ✅ pip upgraded")
    except subprocess.CalledProcessError:
        print("   ⚠️  Failed to upgrade pip, continuing...")
    
    # Install requirements
    requirements_file = "requirements-clean.txt"
    if Path(requirements_file).exists():
        try:
            subprocess.run([pip_path, "install", "-r", requirements_file], check=True)
            print("   ✅ Dependencies installed from requirements-clean.txt")
        except subprocess.CalledProcessError:
            print("   ❌ Failed to install dependencies")
            return False
    else:
        print(f"   ⚠️  {requirements_file} not found, skipping dependency installation")
    
    return True


def setup_vscode():
    """Set up VS Code configuration."""
    print("\n⚙️  Setting up VS Code configuration...")
    
    vscode_dir = Path(".vscode")
    vscode_dir.mkdir(exist_ok=True)
    
    # Settings.json
    settings = {
        "python.defaultInterpreterPath": "./.venv/Scripts/python.exe" if platform.system() == "Windows" else "./.venv/bin/python",
        "python.terminal.activateEnvironment": True,
        "python.envFile": "${workspaceFolder}/.env",
        "python.analysis.extraPaths": [
            "${workspaceFolder}",
            "${workspaceFolder}/models",
            "${workspaceFolder}/utils",
            "${workspaceFolder}/dataset"
        ],
        "python.linting.enabled": True,
        "python.linting.flake8Enabled": True,
        "python.formatting.provider": "black",
        "editor.formatOnSave": True,
        "editor.codeActionsOnSave": {
            "source.organizeImports": True
        }
    }
    
    import json
    settings_file = vscode_dir / "settings.json"
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=4)
    
    print("   ✅ VS Code settings configured")
    return True


def main():
    """Main setup function."""
    print("🚀 HRM Project Environment Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check CUDA
    cuda_available = check_cuda_installation()
    
    # Check PyTorch CUDA
    if cuda_available:
        check_pytorch_cuda()
    
    # Create environment file
    create_env_file()
    
    # Install dependencies
    install_dependencies()
    
    # Setup VS Code
    setup_vscode()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. In VS Code, press Ctrl+Shift+P and select 'Python: Select Interpreter'")
    print("2. Choose the interpreter from .venv/Scripts/python.exe (Windows) or .venv/bin/python (Linux/Mac)")
    print("3. The .env file will automatically set PYTHONPATH")
    print("4. You can now run the HRM project!")
    
    if not cuda_available:
        print("\n⚠️  Note: CUDA not detected. Training will use CPU only.")
        print("   To enable GPU training, install CUDA and set CUDA_HOME")


if __name__ == "__main__":
    main()
