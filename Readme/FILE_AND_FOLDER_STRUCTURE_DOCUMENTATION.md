# File and Folder Structure Documentation
## Explainable Medical AI System - Complete Directory Guide

**Generated:** March 3, 2026  
**Project:** Explainable AI System for Medical Diagnosis using SHAP and LIME

---

## Table of Contents

1. [Project Root Overview](#project-root-overview)
2. [Core Application Files](#core-application-files)
3. [Training and Model Files](#training-and-model-files)
4. [Dashboard and UI Files](#dashboard-and-ui-files)
5. [Novel Patentable Component Files](#novel-patentable-component-files)
6. [Governance and Compliance Files](#governance-and-compliance-files)
7. [Documentation Files](#documentation-files)
8. [Data Preprocessing](#data-preprocessing)
9. [Output Directories](#output-directories)
10. [Configuration Files](#configuration-files)

---

## 1. Project Root Overview

```
Explainable-AI-system-for-medical-diagnosis-using-SHAP-and-LIME/
│
├── 📁 mimic_preprocessing/          # MIMIC-III data preprocessing utilities
├── 📁 practice/                     # Practice/experimental code
├── 📁 trained_models/               # Saved model artifacts (.pkl files)
├── 📁 audit_logs/                   # Audit trail for compliance
├── 📁 regulatory_submission/        # Regulatory compliance documents
├── 📁 etc/                          # Configuration files
├── 📁 __pycache__/                  # Python bytecode cache
│
├── 🐍 Core Application Files        # Main executables
├── 📊 Training Pipeline Files        # Model training scripts
├── 🎨 Dashboard Files                # Web UI applications
├── ⚖️ Governance Files               # Audit, registry, compliance
├── 💡 Novel Components              # Patentable innovations
├── 📋 Documentation Files            # Guides and specifications
└── ⚙️ Configuration Files           # Setup and dependencies
```

---

## 2. Core Application Files

### 2.1 Main Entry Points

#### `main.py` ⭐
**Purpose:** Primary application integrating all three novel patented components.  
**Description:** Unified Dash application that combines:
- Physiological Coupling Engine
- Clinical Plausibility Scorer
- Hierarchical Intervention Recommender

**Key Features:**
- Patient state display with real-time risk calculation
- Disease risk prediction for 8 conditions
- Interactive coupling demonstration
- Plausibility scoring interface
- Hierarchical intervention recommendations

**Usage:**
```bash
python main.py
# Access at: http://127.0.0.1:8050
```

**Dependencies:**
- `physiological_coupling.py`
- `plausibility_scoring.py`
- `hierarchical_interventions.py`
- Trained models in `trained_models/`

---

#### `start_application.py`
**Purpose:** Launcher script with model verification.  
**Description:** Checks for trained models and launches the appropriate dashboard.

**Workflow:**
1. Checks if models exist in `trained_models/`
2. If missing, prompts to run training first
3. Launches `main.py` or fallback dashboard

**Usage:**
```bash
python start_application.py
```

---

#### `quick_start.py`
**Purpose:** Fast demo with dependency installation.  
**Description:** Automated setup script for new users.

**Steps:**
1. Checks for required packages
2. Auto-installs missing dependencies
3. Optionally trains models
4. Launches dashboard

**Usage:**
```bash
python quick_start.py
```

---

### 2.2 Core Engine Modules

#### `xai_engine.py` ⭐⭐⭐
**Purpose:** Comprehensive XAI engine supporting SHAP, LIME, counterfactuals.  
**Lines:** 1,265  
**Key Classes:**
- `XAIEngine`: Main explainer orchestrator
- `ClinicalTranslator`: AI → Clinical language
- `ExplanationCache`: Performance optimization

**Features:**
- SHAP (Tree, Kernel, Deep)
- LIME (Tabular, Image)
- Grad-CAM for CNNs
- Counterfactual generation
- Clinical context integration
- LRU caching for performance

**Usage Example:**
```python
from xai_engine import XAIEngine

engine = XAIEngine()
engine.register_model("sepsis", model, feature_names)
explanation = engine.explain("sepsis", patient_data, method="shap")
```

---

#### `whatif_engine.py` ⭐⭐⭐
**Purpose:** What-if scenario analysis with clinical constraints.  
**Lines:** 1,335  
**Key Classes:**
- `WhatIfEngine`: Scenario orchestrator
- `FeatureConstraint`: Clinical bounds
- `ScenarioResult`: Outcome analysis

**Features:**
- Baseline patient analysis
- Constrained perturbations
- Clinical plausibility validation
- Delta risk computation
- SHAP-guided recommendations
- Optimization solver for target risk

**Usage Example:**
```python
from whatif_engine import WhatIfEngine

engine = WhatIfEngine()
engine.register_model("sepsis", model, feature_names, constraints)
scenario = engine.run_scenario("sepsis", baseline, {"temperature": 37.0})
```

---

#### `disease_model_service.py` ⭐⭐
**Purpose:** Base framework for disease-specific models.  
**Lines:** 870  
**Key Classes:**
- `BaseModelService`: Abstract base class
- `SepsisModelService`: Sepsis prediction
- `KidneyFailureModelService`: AKI prediction

**Features:**
- Unified interface for all diseases
- Built-in SHAP/LIME integration
- Pydantic validation schemas
- Automatic feature scaling
- Training pipeline integration

**Supported Diseases:**
- Sepsis & Septic Shock
- Acute Kidney Injury
- Cardiovascular Disease
- Diabetes
- Anemia
- Thalassemia
- Thrombocytopenia
- In-Hospital Mortality

---

#### `multi_disease_explainable_system.py`
**Purpose:** Complete 4-module multi-disease system.  
**Lines:** 796+  
**Modules:**
1. **Disease Prediction Module**: ML models for 8 diseases
2. **Explainability Module**: SHAP & LIME integration
3. **Visualization Module**: Interactive dashboards
4. **Health Assistant Module**: Autonomous alerts

**Usage:**
```bash
python multi_disease_explainable_system.py
```

---

## 3. Training and Model Files

### 3.1 Training Scripts

#### `training_pipeline.py` ⭐⭐⭐
**Purpose:** Production training pipeline with full governance.  
**Lines:** 594  
**Features:**
- FDA/EU compliant training
- Synthetic data generation
- Multi-disease support
- ModelRegistry integration
- Audit logging
- Bootstrap evaluation (95% CI)

**Usage:**
```bash
# Train on synthetic data
python training_pipeline.py --data-source synthetic --n-samples 10000

# Train on CSV
python training_pipeline.py --data-source csv --csv-path mimic.csv

# Quick demo
python training_pipeline.py --quick-demo
```

**Outputs:**
- Model artifacts in `trained_models/`
- Metrics in JSON format
- Audit logs in `audit_logs/training_events.jsonl`

---

#### `train_advanced_models.py` ⭐⭐
**Purpose:** Advanced model training with feature engineering.  
**Features:**
- 50K+ sample generation
- 32 engineered features (from 14 base)
- Hyperparameter optimization
- Optimal threshold tuning (Youden's J)
- Clinical-grade performance metrics

**Feature Engineering:**
- MAP (Mean Arterial Pressure)
- Pulse Pressure
- Shock Index (HR/SBP)
- BUN/Cr Ratio
- SIRS Score
- qSOFA Score
- Age Risk Score
- Lab Abnormality Count

**Usage:**
```bash
python train_advanced_models.py --n-samples 50000
```

---

#### `train_ensemble_models.py`
**Purpose:** Ensemble model training (voting, stacking).  
**Features:**
- Random Forest + XGBoost + LogisticRegression
- Voting ensemble (soft/hard)
- Stacking with meta-learner
- Cross-validation averaging

**Usage:**
```bash
python train_ensemble_models.py
```

---

### 3.2 Model Verification

#### `verify_trained_models.py`
**Purpose:** Verify all trained models are loadable and functional.  
**Checks:**
- Model file exists
- Loads without errors
- Contains expected components (model, scaler, features)
- Can make predictions

**Usage:**
```bash
python verify_trained_models.py
```

**Output:**
```
✅ sepsis_advanced_50000.pkl - OK
✅ kidney_failure_advanced_50000.pkl - OK
✅ heart_disease_advanced_50000.pkl - OK
...
```

---

#### `check_model.py` & `check_model_type.py`
**Purpose:** Inspect model internals and architecture.  
**Usage:**
```bash
python check_model.py trained_models/sepsis_advanced_50000.pkl
```

---

### 3.3 Test and Demo Scripts

#### `test_inference.py`
**Purpose:** Test prediction pipeline end-to-end.  
**Tests:**
- Load model from disk
- Prepare input features
- Run prediction
- Verify output format

---

#### `test_shap_display.py`
**Purpose:** Test SHAP visualization generation.  
**Tests:**
- SHAP TreeExplainer initialization
- SHAP value computation
- Waterfall plot generation
- Force plot generation

---

#### `demo_explainable_diagnosis.py`
**Purpose:** Simple demo script showing XAI workflow.  
**Steps:**
1. Load trained model
2. Create sample patient
3. Make prediction
4. Generate SHAP explanation
5. Generate LIME explanation
6. Display results

---

## 4. Dashboard and UI Files

### 4.1 Dashboard Applications

#### `enhanced_dashboard_with_whatif.py` ⭐⭐⭐
**Purpose:** Complete interactive dashboard with what-if analysis.  
**Features:**
- Model performance visualization
- Risk distribution charts
- Patient risk cards
- **What-If Analysis Panel** (interactive sliders)
  - Age, Creatinine, Blood Pressure, Glucose
  - Real-time risk recalculation
  - Risk delta visualization
  - Clinical recommendations

**Layout Sections:**
1. **Header**: Title and description
2. **Model Performance**: AUROC for 8 diseases
3. **Risk Distribution**: Patient risk categories
4. **What-If Analysis**: Interactive scenario explorer
5. **Patient Details**: Individual patient cards

**Usage:**
```bash
python enhanced_dashboard_with_whatif.py
# Access at: http://127.0.0.1:8051
```

---

#### `explainable_dashboard.py`
**Purpose:** Basic dashboard without what-if features.  
**Features:**
- Model performance charts
- Risk distribution
- Disease prevalence
- Patient detail cards

**Usage:**
```bash
python explainable_dashboard.py
# Access at: http://127.0.0.1:8050
```

---

#### `dashboard_ui_specification.py`
**Purpose:** UI component specifications and design system.  
**Contains:**
- Color schemes
- Layout templates
- Component definitions
- Styling guidelines

---

### 4.2 Jupyter Notebook

#### `explainable_medical_diagnosis_demo.ipynb`
**Purpose:** Interactive tutorial notebook.  
**Sections:**
1. Introduction to XAI in healthcare
2. Data loading and preprocessing
3. Model training
4. SHAP explanations
5. LIME explanations
6. Dashboard preview
7. What-if scenarios

**Usage:** Open in Jupyter Lab/Notebook

---

## 5. Novel Patentable Component Files

### 5.1 Physiological Coupling

#### `physiological_coupling.py` ⭐ PATENT
**Purpose:** Evidence-based automatic parameter coupling.  
**Lines:** 499  
**Patent Claim:** Automatic coupled parameter adjustment using physiological relationships.

**Evidence-Based Couplings:**
1. Temperature ↔ Heart Rate (10 bpm/°C)
2. Blood Pressure → Heart Rate (baroreflex)
3. Heart Rate → Cardiac Output
4. Oxygen Saturation → Respiratory Rate
5. Age → Multiple parameter ranges

**Usage Example:**
```python
from physiological_coupling import PhysiologicalCouplingEngine

engine = PhysiologicalCouplingEngine()
patient = {"temperature": 37.0, "heart_rate": 80}

# Change temperature to 40°C
coupled = engine.apply_coupling("temperature", 40.0, patient)
# Result: {"temperature": 40.0, "heart_rate": 110}  # Auto-adjusted!
```

---

### 5.2 Clinical Plausibility

#### `plausibility_scoring.py` ⭐ PATENT
**Purpose:** Quantitative intervention feasibility scoring.  
**Patent Claim:** Multi-factor patient-specific plausibility assessment.

**Scoring Factors:**
- Physiological feasibility
- Time constraints (rate of change)
- Intervention type and intensity
- Patient modifiers (age, comorbidities)
- Urgency level

**Usage Example:**
```python
from plausibility_scoring import ClinicalPlausibilityScorer, UrgencyLevel

scorer = ClinicalPlausibilityScorer()
score = scorer.score_plausibility(
    parameter="creatinine",
    current_value=2.5,
    target_value=1.2,
    time_hours=6,
    intervention_type="medication",
    urgency=UrgencyLevel.URGENT
)
# Returns: 0.35 (low plausibility for medication alone)
```

---

### 5.3 Hierarchical Interventions

#### `hierarchical_interventions.py` ⭐ PATENT
**Purpose:** SHAP-guided multi-tier treatment planning.  
**Patent Claim:** Resource-aware intervention prioritization using SHAP values.

**Intervention Tiers:**
1. **Tier 1 - Monitoring**: Observation only
2. **Tier 2 - Non-invasive**: Oxygen, fluids, vitals
3. **Tier 3 - Medication**: Antibiotics, vasopressors
4. **Tier 4 - Invasive**: Dialysis, ventilation, surgery

**Usage Example:**
```python
from hierarchical_interventions import HierarchicalInterventionEngine, ResourceLevel

engine = HierarchicalInterventionEngine()
plan = engine.generate_intervention_plan(
    patient_data=patient,
    shap_values=shap_values,
    target_risk=0.3,
    resource_level=ResourceLevel.INTENSIVE_CARE
)
# Returns tiered plan with cost-benefit optimization
```

---

### 5.4 Governed What-If Engine

#### `governed_whatif_engine.py`
**Purpose:** What-if engine with integrated governance.  
**Features:**
- Combines all three novel components
- Automatic audit logging
- Constraint violation tracking
- Plausibility-weighted scenarios

---

## 6. Governance and Compliance Files

### 6.1 Audit and Registry

#### `audit_logging.py`
**Purpose:** Immutable audit trail for FDA compliance.  
**Features:**
- JSONL append-only logging
- Cryptographic hashing for integrity
- Event types: training, prediction, explanation, what-if
- Query interface with filters

**Audit Log Location:** `audit_logs/training_events.jsonl`

**Usage:**
```python
from audit_logging import AuditLogger, AuditEventType

logger = AuditLogger()
logger.log_event(
    event_type=AuditEventType.MODEL_TRAINING,
    event_data={"disease": "sepsis", "samples": 10000}
)
```

---

#### `model_registry.py`
**Purpose:** Central model tracking and versioning.  
**Features:**
- Model metadata storage
- Version control
- Performance tracking
- Data and model hashing
- Explainability metadata

**Usage:**
```python
from model_registry import ModelRegistry, ModelMetadata

registry = ModelRegistry()
metadata = ModelMetadata(
    model_name="sepsis",
    version="1.0.0",
    disease_target="sepsis",
    training_samples=50000,
    # ...
)
registry.register_model("sepsis", model, metadata)
```

---

### 6.2 Compliance

#### `compliance_matrix.py`
**Purpose:** Map system to regulatory requirements.  
**Frameworks:**
- FDA 510(k)
- FDA PMA
- EU MDR
- HIPAA
- GDPR

**Output Formats:**
- JSON (`regulatory_submission/compliance_matrix.json`)
- CSV (`regulatory_submission/compliance_matrix.csv`)
- Markdown (`regulatory_submission/compliance_matrix.md`)
- Plain text (`regulatory_submission/compliance_matrix.txt`)

**Usage:**
```bash
python compliance_matrix.py
```

---

#### `technical_effect_registry.py`
**Purpose:** Document technical effects for patent applications.  
**Features:**
- Technical problem definitions
- Novel solution descriptions
- Measurable effects with benchmarks
- Patent jurisdiction mapping

---

#### `alert_engine.py`
**Purpose:** Clinical alerting system with severity levels.  
**Alert Types:**
- Critical threshold violations
- Trend deterioration
- Pattern anomalies
- Risk escalation

---

## 7. Documentation Files

### 7.1 User Documentation

#### `README.md` ⭐
**Purpose:** Main project documentation.  
**Sections:**
- Project overview
- Key features
- Quick start guide
- Clinical interpretation guide
- Installation instructions
- Performance metrics

---

#### `USAGE_GUIDE.md`
**Purpose:** Detailed user guide for clinicians.  
**Sections:**
- How to interpret predictions
- Understanding SHAP/LIME
- Using what-if analysis
- Clinical workflow integration

---

#### `STEP_BY_STEP_GUIDE.md`
**Purpose:** Tutorial for new users.  
**Steps:**
1. Installation
2. Training models
3. Running dashboard
4. Interpreting results
5. Advanced features

---

#### `RUN_AND_CHECK_GUIDE.md`
**Purpose:** Quick reference for running and verifying the system.

---

### 7.2 Technical Documentation

#### `PROJECT_SUMMARY.md`
**Purpose:** High-level project overview.  
**Contents:**
- Architecture overview
- Module descriptions
- Performance results
- Clinical insights

---

#### `TECHNICAL_SPECIFICATION.md`
**Purpose:** Detailed technical specifications.  
**Sections:**
- System architecture
- API documentation
- Data schemas
- Algorithm descriptions

---

#### `MODULAR_ARCHITECTURE.md` ⭐⭐
**Purpose:** Complete modular refactoring plan.  
**Lines:** 1,258  
**Contents:**
- Proposed folder structure
- Module responsibilities
- Interface definitions
- Migration guide

---

#### `MODULAR_ARCHITECTURE_QUICK_REFERENCE.md`
**Purpose:** Quick reference for modular structure.

---

#### `MODULAR_ARCHITECTURE_STARTERS.md`
**Purpose:** Starter templates for each module.

---

### 7.3 Patent and Legal

#### `PATENT_DISCLOSURE.md`
**Purpose:** Full patent disclosure document.  
**Sections:**
- Invention overview
- Technical problems solved
- Novel solutions
- Claims (apparatus and method)
- Drawings and diagrams

---

#### `PATENTABLE_README.md`
**Purpose:** Summary of patentable innovations.

---

#### `LICENSE.md`
**Purpose:** Software license terms.

---

### 7.4 Development Documentation

#### `ARCHITECTURE_SYNC.md`
**Purpose:** Architecture synchronization notes.

---

#### `CHANGELOG.md`
**Purpose:** Version history and changes.

---

#### `COMMIT_MESSAGE.txt`
**Purpose:** Template for commit messages.

---

#### `REFACTORING_SUMMARY.md`
**Purpose:** Summary of code refactoring efforts.

---

#### `SYNC_SUMMARY.md`
**Purpose:** Synchronization status across modules.

---

#### `SYSTEM_STATUS.md`
**Purpose:** Current system status and known issues.

---

#### `TRAINING_SUMMARY.md`
**Purpose:** Summary of model training runs.

---

### 7.5 Deployment

#### `DEPLOYMENT_ROADMAP.md`
**Purpose:** Plan for production deployment.  
**Phases:**
1. Local testing
2. Clinical validation
3. Regulatory submission
4. Pilot deployment
5. Full rollout

---

#### `DELIVERY_SUMMARY.md`
**Purpose:** Summary of deliverables.

---

#### `OBJECTIVES_VERIFICATION.md`
**Purpose:** Verify all project objectives met.

---

### 7.6 Data Guides

#### `KAGGLE_MIMIC_GUIDE.md`
**Purpose:** Guide for accessing MIMIC-III on Kaggle.

---

#### `MIMIC_FULL_DATASET_GUIDE.md`
**Purpose:** Guide for full MIMIC-III dataset.

---

#### `PROJECT_FILE_GUIDE.md`
**Purpose:** Guide to all project files.

---

#### `VISUAL_ARCHITECTURE_DIAGRAMS.md`
**Purpose:** System architecture diagrams.

---

## 8. Data Preprocessing

### 8.1 MIMIC Preprocessing (`mimic_preprocessing/`)

#### `__init__.py`
**Purpose:** Package initialization.

---

#### `create_extended_mimic_dataset.py`
**Purpose:** Create extended MIMIC dataset with additional features.

---

#### `create_extended_listfile.py`
**Purpose:** Generate list files for MIMIC episodes.

---

#### `create_mimic_notes_bow.py`
**Purpose:** Create bag-of-words from clinical notes.

---

#### `extract_mimic_time_series_features.py`
**Purpose:** Extract time-series features from MIMIC.

---

#### `mp_filenames.py`
**Purpose:** Filename utilities for MIMIC preprocessing.

---

#### `explainable_medical_diagnosis.py`
**Purpose:** Explainable diagnosis on MIMIC data.

---

#### `explainable_dashboard.py`
**Purpose:** Dashboard for MIMIC results.

---

### 8.2 Data Loading

#### `load_mimic_for_training.py`
**Purpose:** Load MIMIC-III data for model training.  
**Features:**
- ICD-9 code mapping
- Feature extraction
- Label generation
- Train/test splitting

---

## 9. Output Directories

### 9.1 `trained_models/`
**Contents:** Saved model artifacts (.pkl files)

**File Naming Convention:**
```
{disease}_advanced_{n_samples}.pkl
```

**Examples:**
- `sepsis_advanced_50000.pkl`
- `kidney_failure_advanced_50000.pkl`
- `heart_disease_advanced_50000.pkl`
- `diabetes_advanced_50000.pkl`
- `anemia_advanced_50000.pkl`
- `thalassemia_advanced_50000.pkl`
- `thrombocytopenia_advanced_50000.pkl`
- `mortality_advanced_50000.pkl`

**Model Bundle Contents:**
```python
{
    'model': XGBoostClassifier,
    'scaler': StandardScaler,
    'feature_names': List[str],
    'metrics': {
        'auroc': float,
        'accuracy': float,
        'f1_score': float,
        'precision': float,
        'recall': float
    },
    'threshold': float,
    'training_date': str,
    'training_samples': int
}
```

---

### 9.2 `audit_logs/`
**Contents:** Audit trail logs (JSONL format)

**Files:**
- `training_events.jsonl`: Model training events
- `prediction_events.jsonl`: Prediction requests
- `explanation_events.jsonl`: Explanation generations
- `whatif_events.jsonl`: What-if scenario analyses

**Event Format:**
```json
{
  "event_id": "uuid-here",
  "timestamp": "2026-03-03T10:30:00Z",
  "event_type": "model_training",
  "user_id": "system",
  "event_data": {...},
  "data_hash": "sha256-hash"
}
```

---

### 9.3 `regulatory_submission/`
**Contents:** Regulatory compliance documents

**Files:**
- `compliance_matrix.json`: Machine-readable compliance
- `compliance_matrix.csv`: Spreadsheet format
- `compliance_matrix.md`: Human-readable Markdown
- `compliance_matrix.txt`: Plain text

---

## 10. Configuration Files

### 10.1 `etc/config.yaml`
**Purpose:** System configuration.  
**Sections:**
- Model parameters
- Training hyperparameters
- Dashboard settings
- Database connections

---

### 10.2 `requirements.txt`
**Purpose:** Python dependencies.  
**Key Packages:**
- numpy, pandas, scikit-learn
- xgboost
- shap, lime
- dash, plotly
- pydantic

**Usage:**
```bash
pip install -r requirements.txt
```

---

### 10.3 `setup.py`
**Purpose:** Package installation setup.  
**Usage:**
```bash
pip install -e .
```

---

### 10.4 Data Files

#### `mimic_large.csv`
**Purpose:** Large MIMIC dataset sample.

---

#### `mimic_training_data.csv`
**Purpose:** Preprocessed training data.

---

#### `mimic_dataset_path.txt`
**Purpose:** Path to MIMIC dataset.

---

#### `complete_multi_disease_results.json`
**Purpose:** Complete results from multi-disease system.  
**Contents:**
- Model performance metrics
- Patient risk assessments
- Clinical recommendations
- Population statistics

---

## Directory Size Summary

```
Total Files:       ~100+
Total Lines:       ~25,000+
Python Files:      ~60
Documentation:     ~30
Models:            8 diseases
Preprocessors:     10+ scripts
```

---

## File Type Breakdown

| Type | Count | Purpose |
|------|-------|---------|
| `.py` | 60+ | Python source code |
| `.md` | 30+ | Markdown documentation |
| `.pkl` | 8+ | Trained model artifacts |
| `.csv` | 3+ | Data files |
| `.json` | 2+ | Configuration and results |
| `.jsonl` | 1+ | Audit logs |
| `.yaml` | 1 | Configuration |
| `.ipynb` | 1 | Jupyter notebook |

---

## Critical Path Files

For a minimal working system, you need:

1. ✅ `xai_engine.py` - Explainability engine
2. ✅ `whatif_engine.py` - What-if scenarios
3. ✅ `training_pipeline.py` - Model training
4. ✅ `enhanced_dashboard_with_whatif.py` - UI
5. ✅ `physiological_coupling.py` - Novel component 1
6. ✅ `plausibility_scoring.py` - Novel component 2
7. ✅ `hierarchical_interventions.py` - Novel component 3
8. ✅ `model_registry.py` - Model tracking
9. ✅ `audit_logging.py` - Compliance
10. ✅ `requirements.txt` - Dependencies

---

*This documentation provides a complete guide to the file and folder structure. For implementation details, see individual files.*
