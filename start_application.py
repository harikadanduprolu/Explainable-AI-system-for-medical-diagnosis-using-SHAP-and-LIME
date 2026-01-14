#!/usr/bin/env python3
"""
Start the Clinical AI Application

This script verifies that trained models exist and launches the dashboard.
"""

import subprocess
import sys
from pathlib import Path

def check_models():
    """Verify models exist."""
    model_dir = Path("trained_models")
    if not model_dir.exists():
        print("❌ trained_models directory not found!")
        print("Run: python training_pipeline.py --quick-demo")
        return False
    
    models = list(model_dir.glob("*.pkl"))
    
    if not models:
        print("❌ No trained models found!")
        print("Run: python training_pipeline.py --quick-demo")
        return False
    
    print(f"✅ Found {len(models)} trained models:")
    for model in models:
        size_kb = model.stat().st_size / 1024
        print(f"   - {model.name} ({size_kb:.1f} KB)")
    
    return True

def main():
    """Start the application."""
    print("\n" + "="*60)
    print("🚀 Starting Clinical AI System...")
    print("="*60 + "\n")
    
    # Check models
    if not check_models():
        sys.exit(1)
    
    print("\n🌐 Launching dashboard...")
    print("   Access at: http://localhost:8050\n")
    
    # Start dashboard
    try:
        subprocess.run([sys.executable, "enhanced_dashboard_with_whatif.py"])
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
