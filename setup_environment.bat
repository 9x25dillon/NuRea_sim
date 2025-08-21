@echo off
REM HRM Project Environment Setup Script for Windows
REM This script helps set up the Python environment and install dependencies

echo.
echo ========================================
echo    HRM Project Environment Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python detected
python --version

REM Check if we're in the right directory
if not exist "models\hrm\hrm_act_v1.py" (
    echo ❌ Please run this script from the HRM project directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo ✅ HRM project directory detected

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo.
    echo 📦 Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo.
echo 🔄 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies from clean requirements
echo.
echo 📦 Installing dependencies...
if exist "requirements-clean.txt" (
    pip install -r requirements-clean.txt
    if errorlevel 1 (
        echo ⚠️  Some dependencies failed to install, but continuing...
    ) else (
        echo ✅ Dependencies installed successfully
    )
) else (
    echo ⚠️  requirements-clean.txt not found, skipping dependency installation
)

REM Test the setup
echo.
echo 🧪 Testing the setup...
python test_adapter.py

echo.
echo ========================================
echo           Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. In VS Code, press Ctrl+Shift+P and select "Python: Select Interpreter"
echo 2. Choose the interpreter from .venv\Scripts\python.exe
echo 3. The .env file will automatically set PYTHONPATH
echo 4. You can now run the HRM project!
echo.
echo To activate the environment in a new terminal:
echo   .venv\Scripts\activate.bat
echo.
pause
