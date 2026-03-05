"""
Start the Explainable Medical AI Web Application
================================================

This script starts the FastAPI web server with the frontend interface.

Usage:
    python start_web_application.py
    
Then open your browser to:
    http://localhost:8000/app
"""

import uvicorn
import sys
from pathlib import Path

def main():
    """Start the web application."""
    print("=" * 70)
    print("🚀 Starting Explainable Medical AI Web Application")
    print("=" * 70)
    print()
    print("📦 Initializing FastAPI server...")
    print("🔧 Loading ML models...")
    print()
    print("Once started, access the application at:")
    print("   🌐 Web App:  http://localhost:8000/app")
    print("   📚 API Docs: http://localhost:8000/docs")
    print("   🔍 ReDoc:    http://localhost:8000/redoc")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    print()
    
    try:
        # Start uvicorn server
        uvicorn.run(
            "backend.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("\nMake sure you have:")
        print("  1. Installed dependencies: pip install -r backend/requirements.txt")
        print("  2. Trained models in the trained_models/ directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
