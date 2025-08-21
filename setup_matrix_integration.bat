@echo off
echo 🚀 Setting up HRM Matrix Orchestrator Integration...
echo.

echo 📁 Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo.
echo 🔧 Creating virtual environment...
python -m venv .venv
if %errorlevel% neq 0 (
    echo ❌ Failed to create virtual environment!
    pause
    exit /b 1
)

echo.
echo 📦 Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ❌ Failed to activate virtual environment!
    pause
    exit /b 1
)

echo.
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

echo.
echo 📚 Installing dependencies...
pip install numpy torch httpx pydantic

echo.
echo 🔄 Renaming env file...
if exist env_file.txt (
    ren env_file.txt .env
    echo ✅ Environment file created
) else (
    echo ⚠️  env_file.txt not found - you may need to create .env manually
)

echo.
echo 🧪 Running test...
python test_matrix_integration.py

echo.
echo 🎉 Setup complete! 
echo.
echo 💡 Next steps:
echo    1. In Cursor: Ctrl+Shift+P → "Python: Select Interpreter" → .venv\Scripts\python.exe
echo    2. Press F5 to run the test with debugging
echo    3. Or run: python test_matrix_integration.py
echo.
pause
