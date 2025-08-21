@echo off
echo 🚀 Setting up HRM Matrix Orchestrator Integration on Windows...
echo.

echo 📁 Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.10+ first.
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
echo 📚 Installing dependencies (editable mode)...
pip install -e .[ml,dev]
if %errorlevel% neq 0 (
    echo ⚠️  Editable install failed, trying regular install...
    pip install -r requirements.txt
)

echo.
echo 🔄 Renaming env file...
if exist env_file.txt (
    ren env_file.txt .env
    echo ✅ Environment file created
) else (
    echo ⚠️  env_file.txt not found - you may need to create .env manually
)

echo.
echo 🧪 Testing Matrix Orchestrator CLI...
python -c "import matrix_orchestrator as mo; print('Backend:', mo.SET.backend)"
if %errorlevel% neq 0 (
    echo ❌ Matrix Orchestrator test failed!
    pause
    exit /b 1
)

echo.
echo 🧪 Testing HRM model import...
python -c "from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1; print('HRM model import OK')"
if %errorlevel% neq 0 (
    echo ❌ HRM model import failed!
    pause
    exit /b 1
)

echo.
echo 🧪 Running orchestrator demo...
python matrix_orchestrator.py --plan plan.json
if %errorlevel% neq 0 (
    echo ⚠️  Orchestrator demo failed, but setup may still be OK
)

echo.
echo 🎉 Setup complete! 
echo.
echo 💡 Next steps in Cursor IDE:
echo    1. Ctrl+Shift+P → "Python: Select Interpreter" → .venv\Scripts\python.exe
echo    2. Press F5 to run the integration test
echo    3. Or run: python run_opt.py
echo.
echo 📁 Files created:
echo    - .venv/ (virtual environment)
echo    - .env (environment variables)
echo    - runs/ (orchestrator output)
echo.
pause
