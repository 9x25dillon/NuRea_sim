#!/usr/bin/env python3
"""
CarryOn MVP Server Startup Script
"""

import uvicorn
from app.db import create_db_and_tables
from app.main import app

if __name__ == "__main__":
    print("🚀 Starting CarryOn MVP Server...")
    
    # Initialize database
    try:
        create_db_and_tables()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"⚠️  Database initialization warning: {e}")
    
    # Start server
    print("🌐 Starting FastAPI server on http://127.0.0.1:8000")
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    ) 