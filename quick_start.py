#!/usr/bin/env python3
"""
Quick Start Script for Explainable Medical Diagnosis with Kaggle MIMIC-III

This script provides a simple way to get started with the explainable medical diagnosis
project using Kaggle's MIMIC-III dataset.

Usage:
    python quick_start.py

Requirements:
    - kagglehub
    - All packages from requirements.txt
"""

import sys
import subprocess
import importlib.util
from pathlib import Path

def check_and_install_package(package_name, install_name=None):
    """Check if package is installed, install if not."""
    if install_name is None:
        install_name = package_name
    
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"📦 Installing {package_name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', install_name])
        print(f"✅ {package_name} installed successfully")
    else:
        print(f"✅ {package_name} already installed")

def main():
    """Main setup and demo function."""
    print("🚀 Starting Explainable Medical Diagnosis Quick Setup")
    print("="*60)
    
    # Check and install required packages
    required_packages = [
        ('kagglehub', 'kagglehub'),
        ('shap', 'shap'),
        ('lime', 'lime'),
        ('xgboost', 'xgboost'),
        ('lightgbm', 'lightgbm'),
        ('plotly', 'plotly'),
        ('seaborn', 'seaborn')
    ]
    
    print("📋 Checking required packages...")
    for package, install_name in required_packages:
        try:
            check_and_install_package(package, install_name)
        except Exception as e:
            print(f"❌ Error installing {package}: {e}")
            print("🔧 Please install manually with: pip install requirements.txt")
            return
    
    # Download MIMIC-III dataset
    print("\n📥 Downloading MIMIC-III dataset from Kaggle...")
    try:
        import kagglehub
        
        # Download the dataset
        print("🔄 This may take several minutes...")
        mimic_path = kagglehub.dataset_download("asjad99/mimiciii")
        print(f"✅ Dataset downloaded to: {mimic_path}")
        
        # List files
        mimic_dir = Path(mimic_path)
        csv_files = list(mimic_dir.glob("*.csv"))
        print(f"📁 Found {len(csv_files)} CSV files")
        
        # Show file sizes
        total_size = 0
        for file in sorted(csv_files)[:10]:  # Show first 10 files
            size_mb = file.stat().st_size / (1024*1024)
            total_size += size_mb
            print(f"  📄 {file.name}: {size_mb:.1f} MB")
        
        if len(csv_files) > 10:
            print(f"  ... and {len(csv_files) - 10} more files")
        
        print(f"💾 Total dataset size: ~{total_size:.1f} MB")
        
        # Save path for later use
        with open('mimic_dataset_path.txt', 'w') as f:
            f.write(str(mimic_path))
        
        print("📝 Dataset path saved to 'mimic_dataset_path.txt'")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("🔧 Please ensure you have kagglehub properly configured")
        print("   Run: kaggle datasets download -d asjad99/mimiciii")
        return
    
    # Quick data exploration
    print("\n🔍 Quick data exploration...")
    try:
        import pandas as pd
        
        # Load a sample of key tables
        print("  📊 Loading ADMISSIONS table...")
        admissions = pd.read_csv(mimic_dir / "ADMISSIONS.csv")
        print(f"     Shape: {admissions.shape}")
        print(f"     Columns: {list(admissions.columns)}")
        
        # Show mortality statistics
        mortality_rate = admissions['HOSPITAL_EXPIRE_FLAG'].mean()
        print(f"  💀 In-hospital mortality rate: {mortality_rate:.1%}")
        
        print("  📊 Loading PATIENTS table...")
        patients = pd.read_csv(mimic_dir / "PATIENTS.csv")
        print(f"     Shape: {patients.shape}")
        
        # Gender distribution
        gender_dist = patients['GENDER'].value_counts()
        print(f"  👥 Gender distribution: {gender_dist.to_dict()}")
        
    except Exception as e:
        print(f"⚠️ Could not perform data exploration: {e}")
    
    # Next steps instructions
    print("\n🎯 Next Steps:")
    print("1. Open the Jupyter notebook: explainable_medical_diagnosis_demo.ipynb")
    print("2. Run all cells to see the complete analysis")
    print("3. Modify the notebook for your specific use case")
    print("4. Use the explainable_medical_diagnosis.py module for custom analysis")
    
    print("\n📚 Available Resources:")
    print("- explainable_medical_diagnosis_demo.ipynb: Complete tutorial notebook")
    print("- explainable_medical_diagnosis.py: Main analysis module")
    print("- explainable_dashboard.py: Interactive dashboard")
    print("- KAGGLE_MIMIC_GUIDE.md: Detailed setup guide")
    
    print("\n✅ Setup completed successfully!")
    print("🚀 You're ready to start explainable medical diagnosis with MIMIC-III!")

if __name__ == "__main__":
    main()
