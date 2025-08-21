#!/bin/bash
# CarryOn MVP Setup Script

set -e

echo "🚀 Setting up CarryOn MVP..."

# Check if Python 3.8+ is available
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install server dependencies
echo "📚 Installing server dependencies..."
cd server
pip install -r requirements.txt

# Install development dependencies
echo "🔧 Installing development dependencies..."
pip install pytest black flake8

# Create data directory
echo "📁 Creating data directories..."
mkdir -p data
mkdir -p logs

# Initialize database
echo "🗄️  Initializing database..."
python -c "
from app.db import create_db_and_tables
try:
    create_db_and_tables()
    print('Database initialized successfully')
except Exception as e:
    print(f'Database initialization warning: {e}')
"

cd ..

# Setup Electron app
echo "🖥️  Setting up Electron app..."
cd apps/desktop-electron

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "⚠️  Node.js not found. Please install Node.js to use the desktop app."
    echo "   You can still use the server API directly."
else
    echo "✅ Node.js found: $(node --version)"
    
    # Install npm dependencies
    if [ -f "package.json" ]; then
        echo "📦 Installing npm dependencies..."
        npm install
        echo "✅ Electron app setup complete"
    else
        echo "⚠️  package.json not found in Electron app directory"
    fi
fi

cd ../..

# Test the system
echo "🧪 Testing system components..."
python test_system.py

# Test CCL functionality
echo "🔍 Testing CCL functionality..."
python test_ccl.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "💡 To get started:"
echo "   1. Activate virtual environment: source venv/bin/activate"
echo "   2. Start the server: cd server && python start.py"
echo "   3. Test the API: curl http://localhost:8000/v1/health"
echo "   4. Run demo: python demo.py"
echo ""
echo "🖥️  For desktop app (requires Node.js):"
echo "   cd apps/desktop-electron && npm start"
echo ""
echo "📚 API Documentation: http://localhost:8000/docs" 