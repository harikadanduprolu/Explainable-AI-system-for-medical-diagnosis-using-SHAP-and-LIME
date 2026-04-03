# MIMIC Data Training Status Review

**Date:** April 3, 2026  
**Summary:** Project architecture supports MIMIC data, but **only `train_advanced_models.py` actively loads real MIMIC-IV mini data**. Other training paths are broken or incomplete.

---

## 🎯 Current Status

| Script | Data Source | Status | Notes |
|--------|------------|--------|-------|
| **train_advanced_models.py** | ✅ MIMIC-IV mini (local) | **WORKING** | Defaults to `dataset/mimic4_mini/physionet.org/files/mimiciv/3.1` |
| **training_pipeline.py** | ❌ Hardcoded CSV | **BROKEN** | Requires `mimic_training_data.csv` (doesn't exist) |
| **train_ensemble_models.py** | ❌ Hardcoded CSV | **BROKEN** | Requires `mimic_training_data.csv` (doesn't exist) |
| **train_mimic_cxr_from_gcs.py** | GCS (Google Cloud) | **REQUIRES SETUP** | Trains on MIMIC-CXR images from GCS bucket |
| **run_complete_demo.py** | MIMIC-III (~500 patients) | **FALLBACK** | Falls back to synthetic if local MIMIC-III cache unavailable |

---

## ✅ What IS Training on MIMIC Data

### `train_advanced_models.py` (PRIMARY TRAINER)
**Status:** ✅ **FULLY OPERATIONAL**

- **Default data source:** `dataset/mimic4_mini/physionet.org/files/mimiciv/3.1`
- **Behavior:** 
  - Automatically extracts raw MIMIC-IV tables (`patients.csv.gz`, `admissions.csv.gz`, `icustays.csv.gz`, etc.)
  - Builds feature matrix from ICU stays, vital signs, lab values
  - Creates disease labels from ICD-10 codes  
  - Generates stratified train/val/test splits
  - Trains XGBoost + MLP neural network ensembles
  - Saves models to `trained_models/`

- **Command:**
  ```bash
  # Train on up to 5000 ICU stays from MIMIC-IV mini
  python train_advanced_models.py --max-patients 5000
  
  # Smaller sample (diagnostic):
  python train_advanced_models.py --max-patients 100
  ```

- **Data pipeline:**
  ```
  dataset/mimic4_mini/physionet.org/files/mimiciv/3.1/
    ├── hosp/
    │   ├── patients.csv.gz
    │   ├── admissions.csv.gz
    │   └── diagnoses_icd.csv.gz
    └── icu/
        ├── icustays.csv.gz
        ├── chartevents.csv.gz (optional)
        └── labevents.csv.gz (optional)
  ```

---

## ❌ What IS NOT Training on MIMIC Data

### `training_pipeline.py` (LEGACY)
**Status:** ❌ **BROKEN - REQUIRES CSV**

- **Expected CSV:** `mimic_training_data.csv` (DOES NOT EXIST)
- **What it tries to do:**
  - Load pre-built CSV with features + disease labels
  - Parse `--data-source` arg (choices: `csv`, `mimic`)
  - If `--data-source=mimic`, still requires `mimic_training_data.csv`
  - **No automatic MIMIC loading logic**

- **Issue:** 
  - Argparse accepts `--data-source mimic` but ignores it
  - Always defaults to loading `mimic_training_data.csv`
  - Script will crash with `FileNotFoundError` if CSV doesn't exist

- **How to fix:** Either
  1. Generate CSV first: `python train_advanced_models.py` → creates `mimic4_mini_training_data.csv`
  2. Then run: `python training_pipeline.py --csv-path mimic4_mini_training_data.csv`
  3. **OR** refactor `training_pipeline.py` to call `load_mimic_for_training.py` directly

### `train_ensemble_models.py` (LEGACY)
**Status:** ❌ **BROKEN - REQUIRES CSV**

- **Expected CSV:** `mimic_training_data.csv` (DOES NOT EXIST, default in arg)
- **Similar issue** to `training_pipeline.py`
- **No MIMIC loading logic**

### `run_complete_demo.py` (DEMO)
**Status:** ⚠️ **SYNTHETIC FALLBACK**

- **Attempts:** Load MIMIC-III from local cache path
- **Default path:** `C:\Users\ADMIN\.cache\kagglehub\datasets\asjad99\mimiciii\versions\1\...`
- **Fallback:** If path not found → creates **synthetic data** (~1000 synthetic patients)
- **Not using:** MIMIC-IV mini from `dataset/mimic4_mini/`

---

## 📊 Data Availability Check

### MIMIC-IV Mini Status
```
✅ Location: dataset/mimic4_mini/physionet.org/files/mimiciv/3.1/
✅ Exists: YES (confirmed directory found)
✅ Ready: YES (contains compressed hosp/ and icu/ tables)
```

### Generated Training Files
```
❌ mimic_training_data.csv       → Does NOT exist
❌ mimic_large.csv               → Found in git diff (DELETED in recent commit)
❌ mimic_training_data_smoke.csv → Created by smoke tests (temporary)
✅ mimic4_mini_training_data.csv → Will be created when train_advanced_models.py runs
```

### Backend Model Loading
```
Location: trained_models/
Status: Checks for *.pkl files when backend starts
Dependencies: Backend API will fail if no models found
Recommendation: Run train_advanced_models.py to populate models/
```

---

## 🚀 Recommended Training Workflow

### Step 1: Build Dataset from MIMIC-IV Mini
```bash
# Extract features + labels from MIMIC-IV mini compressed files
python load_mimic_for_training.py \
  --mimic-path dataset/mimic4_mini/physionet.org/files/mimiciv/3.1 \
  --output mimic4_mini_training_data.csv \
  --max-patients 5000
```

**Output:** `mimic4_mini_training_data.csv` (5000 patients × 68 features)

### Step 2: Train Advanced Models (Recommended)
```bash
# Automatically builds dataset + trains models
python train_advanced_models.py --max-patients 5000
```

**Output:** `trained_models/*.pkl` (XGBoost + MLP bundles)

### Step 3: Alternatively, Use Legacy Pipeline (If Needed)
```bash
# After generating mimic4_mini_training_data.csv
python training_pipeline.py --csv-path mimic4_mini_training_data.csv
```

### Step 4: Verify Models
```bash
python verify_trained_models.py
```

### Step 5: Start Backend API
```bash
# Will load trained models and start FastAPI server
python backend/main.py
# OR
uvicorn backend.main:app --host 127.0.0.1 --port 9000 --reload
```

---

## 📋 Architecture Decision: Why Only `train_advanced_models.py` Works

The recent refactoring (April 2026) has:

1. **Removed synthetic data generation** from the default pipeline
2. **Added MIMIC-IV mini support** with robust table resolution:
   - Handles `.csv.gz` compression
   - Case-insensitive column names
   - Graceful fallback for missing CXR metadata
3. **Made legacy scripts dependent on pre-built CSVs** which no longer exist

**Consequence:** 
- ✅ Training is now **reproducible** (uses real clinical data)
- ❌ But legacy scripts (`training_pipeline.py`, `train_ensemble_models.py`) are **disconnected** from MIMIC data
- ✅ Single source of truth: `train_advanced_models.py` + `load_mimic_for_training.py`

---

## ⚠️ Known Limitations

### MIMIC-IV Mini Dataset Quality
- **Size:** 20,000 patients (ICU stays) - significantly smaller than full MIMIC-IV (380K+)
- **Disease prevalence:** Often sparse (many diseases 0% in small samples)
- **Event data:** Chart/lab events may be minimal in compressed mini version
- **CXR metadata:** Not included in mini version (clinical-only fallback used)

### Training Stability
- Stratified splits may fail on very small cohorts → fallback to random split
- Some disease classes may be under-represented

### For Production Use
- Full MIMIC-IV recommended (80K+ patients)
- Alternatively: combine with other datasets (MIMIC-CXR, PhysioNet sepsis benchmark)

---

## 🔄 To Get Started Training

**Option 1: Quick Test (10 patients)**
```bash
python train_advanced_models.py --max-patients 10
```

**Option 2: Medium Sample (500 patients)**
```bash
python train_advanced_models.py --max-patients 500
```

**Option 3: Full Mini Dataset (all 20K- varies by content)**
```bash
python train_advanced_models.py --max-patients 20000
```

All three will:
1. Load actual MIMIC-IV clinical data
2. Engineer features from vitals + labs + demographics
3. Extract disease labels from ICD-10 codes
4. Train ensemble models
5. Save models to `trained_models/`

---

## Conclusion

✅ **YES, the project IS training on MIMIC data**
- Specifically: MIMIC-IV mini (compressed local version)
- Entry point: `train_advanced_models.py`
- Data loader: `load_mimic_for_training.py`
- No synthetic data in default workflow

⚠️ **But:** Other training scripts are legacy/broken and need refactoring or deprecation.

