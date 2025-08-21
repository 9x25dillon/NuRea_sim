#!/usr/bin/env python3
"""
Standalone Categorical Coherence Linter (CCL)
Entropy‑driven "ghost in the code" detector

This is a standalone version of the CCL tool that can be run independently
from the main CarryOn MVP project.

Usage:
    python ccl_standalone.py <path> [--samples 200] [--seed 13] [--report report.json]
    python ccl_standalone.py --serve  # Start FastAPI server
"""

import sys
import os

# Add the server app to the path so we can import CCL
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server', 'app'))

try:
    from tools.ccl import main
except ImportError as e:
    print(f"Error importing CCL: {e}")
    print("Make sure you're running this from the carryon-mvp directory")
    sys.exit(1)

if __name__ == "__main__":
    main() 