# HRM Project Setup Guide

This guide will help you set up the HRM (Hierarchical Reasoning Model) project environment, resolve the `adam-atan2` compatibility issue, and configure everything for development.

## 🚀 Quick Setup

### Option 1: Automated Setup (Recommended)
```bash
# Navigate to the HRM directory
cd HRM

# Run the automated setup script
python setup_environment.py
```

### Option 2: Manual Setup
Follow the steps below if you prefer manual configuration.

## 📋 Prerequisites

- Python 3.8 or higher
- VS Code (recommended) or any Python IDE
- CUDA Toolkit (optional, for GPU training)

## 🔧 Step-by-Step Setup

### 1. Python Interpreter Selection

In VS Code:
1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Linux/Mac)

### 2. Environment Variables (.env file)

The `.env` file sets up:
- `PYTHONPATH`: Adds the project root to Python's module search path
- `CUDA_HOME`: Points to your CUDA installation
- `CUDA_PATH`: Alternative CUDA path variable

**Note**: If the `.env` file creation is blocked, you can manually create it or use the setup script.

### 3. VS Code Settings

The `.vscode/settings.json` file configures:
- Python interpreter path
- Environment file location
- Extra Python paths for imports
- Code formatting and linting
- Auto-import organization

### 4. Dependencies

#### Clean Requirements (requirements-clean.txt)
This file excludes the problematic `adam-atan2` package and includes:
- Core ML: PyTorch, torchvision, torchaudio
- Essential: einops, tqdm, pydantic, etc.
- Development: black, flake8, pytest, jupyter

#### Original Requirements (requirements.txt)
Contains the original dependencies including `adam-atan2`.

## 🔄 Adam-Atan2 Compatibility

### Problem
The `adam-atan2` package was causing compatibility issues and preventing the project from running.

### Solution
We've created `optimizer_adapter.py` which provides:

1. **AdamATan2**: A custom implementation that mimics the original package
2. **AdamWAdapter**: A simple wrapper around PyTorch's AdamW
3. **create_optimizer()**: Factory function for easy optimizer selection

### Usage
```python
# Instead of: from adam_atan2 import AdamATan2
from optimizer_adapter import AdamATan2, create_optimizer

# Use as before
optimizer = AdamATan2(model.parameters(), lr=1e-3)

# Or use the factory function
optimizer = create_optimizer("adam_atan2", params=model.parameters(), lr=1e-3)
```

## 🎯 Project Structure

```
HRM/
├── .vscode/                 # VS Code configuration
│   ├── settings.json       # Python and editor settings
│   └── launch.json         # Debug configurations
├── models/                  # Model implementations
│   └── hrm/               # HRM model classes
├── .env                    # Environment variables
├── requirements-clean.txt  # Clean dependencies
├── optimizer_adapter.py   # Adam-atan2 replacement
├── setup_environment.py   # Automated setup script
└── SETUP_GUIDE.md        # This file
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure PYTHONPATH is set correctly
   - Check that the virtual environment is activated
   - Verify VS Code is using the correct interpreter

2. **CUDA Issues**
   - Set `CUDA_HOME` to your CUDA installation path
   - Install PyTorch with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

3. **Package Installation**
   - Use `requirements-clean.txt` instead of `requirements.txt`
   - Ensure pip is up to date: `pip install --upgrade pip`

### Environment Variables

To manually set CUDA environment variables:

**Windows (PowerShell):**
```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
```

**Linux/Mac:**
```bash
export CUDA_HOME=/usr/local/cuda-12.0
export CUDA_PATH=/usr/local/cuda-12.0
```

## 🚀 Running the Project

After setup:

1. **Activate virtual environment:**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

2. **Test the setup:**
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

3. **Run training:**
   ```bash
   python pretrain.py
   ```

## 📚 Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit Documentation](https://developer.nvidia.com/cuda-toolkit)
- [VS Code Python Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)

## 🤝 Contributing

If you encounter issues or have improvements:
1. Check the troubleshooting section
2. Review the optimizer adapter implementation
3. Test with the clean requirements file
4. Report issues with detailed error messages

---

**Happy Coding! 🎉**
