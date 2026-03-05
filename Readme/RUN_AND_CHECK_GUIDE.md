# 🚀 Complete Run and Check Guide

**Last Updated:** January 14, 2026  
**Purpose:** Step-by-step guide to run and verify all system components

---

## 📋 Quick Status Check

### 1. Verify File Structure
```powershell
# Check all Python scripts exist
Get-ChildItem -Filter *.py | Select-Object Name, Length

# Check trained models
Get-ChildItem trained_models\ | Measure-Object

# Check documentation
Get-ChildItem -Filter *.md | Select-Object Name
```

---

## ✅ Step-by-Step Verification

### STEP 1: Verify Trained Models (23 models expected)

```powershell
# Count all model files
(Get-ChildItem trained_models\*.pkl).Count
# Expected: 23 files

# List all advanced models (best performance)
Get-ChildItem trained_models\*_advanced_*.pkl

# Check model sizes
Get-ChildItem trained_models\*_advanced_*.pkl | Select-Object Name, @{Name="Size(KB)";Expression={[math]::Round($_.Length/1KB,1)}}
```

**Expected Output:**
```
sepsis_advanced_v1.0.0.pkl          201 KB
kidney_failure_advanced_v1.0.0.pkl  212 KB
heart_disease_advanced_v1.0.0.pkl   226 KB
diabetes_advanced_v1.0.0.pkl        218 KB
anemia_advanced_v1.0.0.pkl          214 KB
thalassemia_advanced_v1.0.0.pkl     208 KB
thrombocytopenia_advanced_v1.0.0.pkl 206 KB
mortality_advanced_v1.0.0.pkl       205 KB
```

---

### STEP 2: Verify Model Loading (Quick Test)

```powershell
# Run model verification script
python verify_trained_models.py
```

**Expected Output:**
```
✅ All 8 models loaded successfully
✅ Predictions generated for test data
✅ Performance metrics displayed
```

**If errors occur:**
- Check if `trained_models/` directory exists
- Verify all 8 `*_advanced_v1.0.0.pkl` files are present
- Ensure scikit-learn and xgboost are installed

---

### STEP 3: Test Training Pipeline (Optional - takes time)

#### Quick Test (1,000 samples - 1 minute)
```powershell
# Test basic training pipeline
python training_pipeline.py
```

#### Advanced Test (10,000 samples - 5 minutes)
```powershell
# Test advanced training with smaller dataset
python train_advanced_models.py --n-samples 10000
```

#### Full Training (50,000 samples - 15-20 minutes)
```powershell
# Run full advanced training
python train_advanced_models.py --n-samples 50000
```

**Expected Output:**
```
Generating 50,000 samples...
Engineering 32 features...
Training 8 disease models...
✅ Sepsis: AUROC 0.862
✅ Kidney Failure: AUROC 0.907 (BEST)
✅ Heart Disease: AUROC 0.818
... (all 8 diseases)
Average AUROC: 0.833
```

---

### STEP 4: Test MIMIC Data Loading

```powershell
# Check if MIMIC-III demo dataset exists
python -c "import kagglehub; print(kagglehub.dataset_download('asjad99/mimiciii'))"

# Test MIMIC data extraction
python load_mimic_for_training.py
```

**Expected Output:**
```
MIMIC-III dataset found at: C:\Users\ADMIN\.cache\kagglehub\...
Extracted 120 patients, 1,761 diagnoses
Generated training data: mimic_training_data.csv
```

**If MIMIC data not found:**
```powershell
# Install kagglehub if needed
pip install kagglehub

# Download MIMIC-III demo
python -c "import kagglehub; kagglehub.dataset_download('asjad99/mimiciii')"
```

---

### STEP 5: Test Dashboard (CURRENTLY BROKEN - NEEDS FIX)

```powershell
# Try to run dashboard
python enhanced_dashboard_with_whatif.py
```

**Current Issue:** Exits with code 1 (model loading error)

**Temporary Fix:** Check what models the dashboard is trying to load
```powershell
# Check dashboard code for model references
Select-String -Path enhanced_dashboard_with_whatif.py -Pattern "trained_models" | Select-Object LineNumber, Line
```

**Expected Fix Needed:**
- Update disease list from 4 to 8 diseases
- Change model filenames from `*_xgboost_*` to `*_advanced_*`
- Add new disease imports

---

### STEP 6: Test Individual Components

#### Test XAI Engine (SHAP/LIME)
```powershell
# Test explainability engine
python -c "from xai_engine import XAIEngine; print('XAI Engine OK')"
```

#### Test What-If Engine
```powershell
# Test what-if analysis
python -c "from whatif_engine import WhatIfEngine; print('What-If Engine OK')"
```

#### Test Model Registry
```powershell
# Test model registry
python -c "from model_registry import ModelRegistry; print('Model Registry OK')"
```

---

## 🔍 Comprehensive System Check

### Run All Checks at Once

```powershell
# Create verification script
$checkScript = @"
Write-Host "=== SYSTEM VERIFICATION ===" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python version
Write-Host "1. Python Version:" -ForegroundColor Yellow
python --version
Write-Host ""

# 2. Check required packages
Write-Host "2. Required Packages:" -ForegroundColor Yellow
python -c "import numpy, pandas, sklearn, xgboost, shap, lime, plotly, dash; print('✅ All packages installed')"
Write-Host ""

# 3. Check trained models
Write-Host "3. Trained Models:" -ForegroundColor Yellow
Write-Host "Total models: $((Get-ChildItem trained_models\*.pkl).Count)"
Write-Host "Advanced models: $((Get-ChildItem trained_models\*_advanced_*.pkl).Count)"
Write-Host ""

# 4. Test model loading
Write-Host "4. Model Loading Test:" -ForegroundColor Yellow
python verify_trained_models.py
Write-Host ""

# 5. Check documentation
Write-Host "5. Documentation Files:" -ForegroundColor Yellow
Write-Host "Total .md files: $((Get-ChildItem *.md).Count)"
Write-Host ""

Write-Host "=== VERIFICATION COMPLETE ===" -ForegroundColor Green
"@

# Save and run
$checkScript | Out-File -FilePath check_system.ps1 -Encoding UTF8
.\check_system.ps1
```

---

## 📊 Expected System Status

### ✅ What Should Work

| Component | Command | Expected Result |
|-----------|---------|-----------------|
| Model Verification | `python verify_trained_models.py` | ✅ All 8 models load |
| Training Pipeline | `python training_pipeline.py` | ✅ Trains 4 original diseases |
| Advanced Training | `python train_advanced_models.py` | ✅ Trains 8 diseases |
| Ensemble Training | `python train_ensemble_models.py` | ✅ Trains 6 ensemble models |
| MIMIC Loading | `python load_mimic_for_training.py` | ✅ Extracts 120 patients |

### ⚠️ Known Issues

| Component | Command | Current Status | Fix Needed |
|-----------|---------|----------------|------------|
| Dashboard | `python enhanced_dashboard_with_whatif.py` | ❌ Exit code 1 | Update to load 8 models |
| Audit Logging | Import audit_logging | ⚠️ Bug at line 231 | Fix duplicate parameter |

---

## 🛠️ Quick Fixes

### Fix 1: Install Missing Packages
```powershell
# Install all required packages
pip install numpy pandas scikit-learn xgboost shap lime plotly dash kagglehub
```

### Fix 2: Verify Python Environment
```powershell
# Check Python version (3.8+ required)
python --version

# Check pip
pip --version

# List installed packages
pip list | Select-String "numpy|pandas|sklearn|xgboost|shap|lime"
```

### Fix 3: Re-download MIMIC Data
```powershell
# Clear cache and re-download
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\kagglehub\datasets\asjad99\mimiciii" -ErrorAction SilentlyContinue
python -c "import kagglehub; print(kagglehub.dataset_download('asjad99/mimiciii'))"
```

---

## 🎯 Testing Workflow (Recommended Order)

### Quick Check (5 minutes)
```powershell
# 1. Verify models exist
Get-ChildItem trained_models\*_advanced_*.pkl

# 2. Test model loading
python verify_trained_models.py

# 3. Check packages
python -c "import numpy, pandas, sklearn, xgboost, shap, lime; print('✅ OK')"
```

### Standard Check (15 minutes)
```powershell
# 1. Quick check (above)
# 2. Test MIMIC loading
python load_mimic_for_training.py

# 3. Test training (small dataset)
python train_advanced_models.py --n-samples 1000

# 4. Verify all components
python -c "from xai_engine import XAIEngine; from whatif_engine import WhatIfEngine; from model_registry import ModelRegistry; print('✅ All components OK')"
```

### Full Check (30+ minutes)
```powershell
# 1. Standard check (above)
# 2. Full training
python train_advanced_models.py --n-samples 50000

# 3. Ensemble training
python train_ensemble_models.py

# 4. Try dashboard (will fail but shows error)
python enhanced_dashboard_with_whatif.py
```

---

## 📈 Performance Benchmarks

### Expected Training Times

| Dataset Size | Training Time | Memory Usage | Expected AUROC |
|--------------|---------------|--------------|----------------|
| 1,000 samples | 1-2 min | ~500 MB | 0.65-0.75 |
| 10,000 samples | 5-8 min | ~1 GB | 0.75-0.82 |
| 50,000 samples | 15-20 min | ~2 GB | 0.83-0.85 |

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Disk: 5 GB free space

**Recommended:**
- CPU: 8+ cores
- RAM: 16 GB
- Disk: 10 GB free space

---

## 🐛 Troubleshooting

### Issue: Models not found
```powershell
# Check if trained_models directory exists
Test-Path trained_models
# If False, create it
New-Item -ItemType Directory -Path trained_models
# Re-run training
python train_advanced_models.py
```

### Issue: Import errors
```powershell
# Reinstall packages
pip install --upgrade numpy pandas scikit-learn xgboost shap lime plotly dash

# Or use requirements.txt
pip install -r requirements.txt
```

### Issue: Memory errors
```powershell
# Train with smaller dataset
python train_advanced_models.py --n-samples 10000

# Or increase virtual memory in Windows settings
```

### Issue: Dashboard crashes
```powershell
# Check error details
python enhanced_dashboard_with_whatif.py 2>&1 | Tee-Object -FilePath dashboard_error.log
cat dashboard_error.log
```

---

## ✅ Success Criteria

**System is working correctly if:**

1. ✅ 23 model files exist in `trained_models/`
2. ✅ `verify_trained_models.py` runs without errors
3. ✅ All 8 diseases show predictions
4. ✅ Average AUROC is ~0.83 for advanced models
5. ✅ MIMIC data loads successfully (120 patients)
6. ✅ Training pipeline completes without errors
7. ✅ All Python imports work (xai_engine, whatif_engine, etc.)

**Only issue:** Dashboard needs update to load 8 models (non-blocking)

---

## 📞 Quick Commands Summary

```powershell
# Check everything at once
python verify_trained_models.py  # Verify models
Get-ChildItem trained_models\    # List models
python -c "import sklearn, xgboost, shap, lime; print('✅')"  # Check packages

# Train models
python train_advanced_models.py --n-samples 10000  # Quick training
python train_advanced_models.py --n-samples 50000  # Full training

# Test MIMIC data
python load_mimic_for_training.py

# Test components
python -c "from xai_engine import XAIEngine; print('OK')"
python -c "from whatif_engine import WhatIfEngine; print('OK')"
```

---

**Ready to start? Run the Quick Check first!** ⚡

```powershell
python verify_trained_models.py
```
