# System Process Flow Documentation
## How the Explainable Medical AI System Works

**Generated:** March 3, 2026  
**Project:** Explainable AI System for Medical Diagnosis using SHAP and LIME

---

## Table of Contents

1. [System Overview](#system-overview)
2. [End-to-End Process Flow](#end-to-end-process-flow)
3. [Module Interaction Diagrams](#module-interaction-diagrams)
4. [Detailed Process Flows](#detailed-process-flows)
5. [Data Flow Between Files](#data-flow-between-files)
6. [User Workflows](#user-workflows)
7. [Technical Workflows](#technical-workflows)

---

## 1. System Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                        │
│                                                                 │
│  ┌─────────────────────┐    ┌──────────────────────────────┐  │
│  │  Dash Dashboard     │◄───┤  Jupyter Notebook            │  │
│  │  (Web UI)           │    │  (Interactive Tutorial)      │  │
│  └──────────┬──────────┘    └──────────────────────────────┘  │
└─────────────┼──────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                            │
│                                                                 │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  XAI       │  │  What-If     │  │  Disease Model       │   │
│  │  Engine    │  │  Engine      │  │  Service             │   │
│  └─────┬──────┘  └──────┬───────┘  └──────┬───────────────┘   │
│        │                │                  │                   │
│        │    ┌───────────┴──────────┐       │                   │
│        └────┤  Novel Components    ├───────┘                   │
│             │  - Coupling          │                           │
│             │  - Plausibility      │                           │
│             │  - Interventions     │                           │
│             └──────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GOVERNANCE LAYER                              │
│                                                                 │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  Model     │  │  Audit       │  │  Compliance          │   │
│  │  Registry  │  │  Logger      │  │  Matrix              │   │
│  └────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                 │
│                                                                 │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  MIMIC-III │  │  Synthetic   │  │  Preprocessors       │   │
│  │  Loader    │  │  Generator   │  │  & Feature Eng.      │   │
│  └────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STORAGE LAYER                                │
│                                                                 │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  Trained   │  │  Audit       │  │  Configuration       │   │
│  │  Models    │  │  Logs        │  │  Files               │   │
│  │  (.pkl)    │  │  (.jsonl)    │  │  (.yaml, .csv)       │   │
│  └────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. End-to-End Process Flow

### 2.1 Complete Workflow from Setup to Deployment

```
┌─────────────────────────────────────────────────────────────────┐
│                      PHASE 1: SETUP                             │
└─────────────────────────────────────────────────────────────────┘

User runs: python quick_start.py
    │
    ├──► Checks requirements.txt
    │         │
    │         ├──► Installs missing packages
    │         └──► Verifies installation
    │
    └──► Checks for trained models
              │
              ├──► If missing: Runs training_pipeline.py
              └──► If exists: Proceeds to dashboard

┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 2: DATA PREPARATION                     │
└─────────────────────────────────────────────────────────────────┘

Option A: MIMIC-III Data
    │
    load_mimic_for_training.py
    │
    ├──► Reads CSV files
    ├──► Maps ICD-9 codes to diseases
    ├──► Extracts clinical features
    ├──► Creates train/test split
    └──► Returns: X_train, y_train, X_test, y_test

Option B: Synthetic Data
    │
    training_pipeline.py → SyntheticDataGenerator
    │
    ├──► Generates patient demographics (age, gender)
    ├──► Generates vital signs (HR, BP, temp, RR)
    ├──► Generates lab values (WBC, Hgb, Cr, glucose)
    ├──► Assigns disease labels (clinical criteria)
    └──► Returns: DataFrame with features and labels

┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 3: FEATURE ENGINEERING                  │
└─────────────────────────────────────────────────────────────────┘

train_advanced_models.py → AdvancedFeatureEngineer
    │
    ├──► Computes MAP = (SBP + 2*DBP) / 3
    ├──► Computes Pulse Pressure = SBP - DBP
    ├──► Computes Shock Index = HR / SBP
    ├──► Computes BUN/Cr Ratio
    ├──► Computes SIRS Score (0-4)
    ├──► Computes qSOFA Score (0-3)
    ├──► Computes Age Risk Score
    ├──► Computes Lab Abnormality Count
    └──► Returns: DataFrame with 32 features (from 14 base)

┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 4: MODEL TRAINING                     │
└─────────────────────────────────────────────────────────────────┘

training_pipeline.py → ClinicalModelTrainer
    │
    For each disease (Sepsis, AKI, Heart, Diabetes, etc.):
    │
    ├──► 1. Initialize XGBoost with optimal hyperparameters
    │         ├── max_depth=6
    │         ├── learning_rate=0.1
    │         ├── n_estimators=200
    │         └── scale_pos_weight=(negative/positive ratio)
    │
    ├──► 2. Fit model on training data
    │         └── model.fit(X_train, y_train)
    │
    ├──► 3. Evaluate on test data
    │         ├── AUROC (ROC curve)
    │         ├── Accuracy
    │         ├── Precision, Recall, F1
    │         ├── Brier Score
    │         └── Log Loss
    │
    ├──► 4. Find optimal threshold (Youden's J statistic)
    │         └── threshold = max(sensitivity + specificity - 1)
    │
    ├──► 5. Register with ModelRegistry
    │         └── model_registry.register_model(...)
    │
    ├──► 6. Log training event
    │         └── audit_logger.log_event(MODEL_TRAINING, {...})
    │
    └──► 7. Save model bundle
              └── joblib.dump({model, scaler, features, metrics},
                              f"trained_models/{disease}_advanced.pkl")

┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 5: MODEL VALIDATION                    │
└─────────────────────────────────────────────────────────────────┘

verify_trained_models.py
    │
    For each model in trained_models/:
    │
    ├──► Load model bundle
    ├──► Verify structure (model, scaler, features, metrics)
    ├──► Test prediction on sample data
    ├──► Check AUROC > 0.7 (minimum performance)
    └──► Print ✅ or ❌ status

┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 6: DASHBOARD LAUNCH                     │
└─────────────────────────────────────────────────────────────────┘

Enhanced Option: enhanced_dashboard_with_whatif.py
Patent Option: main.py

    │
    ├──► Load all trained models from trained_models/
    │         └── TRAINED_MODELS = {disease: bundle, ...}
    │
    ├──► Initialize Dash app
    │         └── app = dash.Dash(__name__)
    │
    ├──► Create layout
    │         ├── Header
    │         ├── Model Performance Charts
    │         ├── Risk Distribution Charts
    │         ├── What-If Analysis Panel
    │         └── Patient Details Cards
    │
    ├──► Register callbacks
    │         ├── update_whatif_analysis()
    │         ├── update_model_performance_chart()
    │         └── update_risk_distribution_chart()
    │
    └──► Run server
              └── app.run_server(debug=True, port=8051)

┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 7: USER INTERACTION                     │
└─────────────────────────────────────────────────────────────────┘

User accesses: http://127.0.0.1:8051
    │
    ├──► Views Model Performance
    │         └── Bar chart with AUROC for 8 diseases
    │
    ├──► Views Risk Distribution
    │         └── Pie chart: LOW/MODERATE/HIGH/CRITICAL
    │
    ├──► Adjusts What-If Sliders
    │         ├── Age: 30 → 70
    │         ├── Creatinine: 0.8 → 2.5
    │         ├── Blood Pressure: 90 → 140
    │         └── Glucose: 80 → 200
    │
    └──► Clicks "Run What-If Analysis"
              │
              └──► Triggers callback: update_whatif_analysis()

┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 8: WHAT-IF EXECUTION                     │
└─────────────────────────────────────────────────────────────────┘

Callback: update_whatif_analysis(n_clicks, age, cr, bp, glucose)
    │
    ├──► 1. Create baseline patient
    │         └── baseline = DEMO_PATIENT.copy()
    │
    ├──► 2. Apply user changes
    │         ├── baseline['age'] = age
    │         ├── baseline['creatinine'] = cr
    │         ├── baseline['systolic_bp'] = bp
    │         └── baseline['glucose'] = glucose
    │
    ├──► 3. Get baseline prediction
    │         └── baseline_risk = predict_with_real_model(
    │                                 DEMO_PATIENT, 'kidney_failure')
    │
    ├──► 4. Get new prediction
    │         └── new_risk = predict_with_real_model(
    │                            baseline, 'kidney_failure')
    │
    ├──► 5. Compute delta
    │         ├── risk_delta = new_risk - baseline_risk
    │         └── delta_pct = (risk_delta / baseline_risk) * 100
    │
    ├──► 6. Get SHAP explanation
    │         └── shap_values = get_shap_explanation(baseline)
    │
    ├──► 7. Apply physiological coupling (if enabled)
    │         └── coupling_engine.apply_coupling(...)
    │
    ├──► 8. Score plausibility
    │         └── plausibility_scorer.score_plausibility(...)
    │
    ├──► 9. Generate interventions
    │         └── intervention_engine.generate_intervention_plan(...)
    │
    └──► 10. Create result cards
              └── Returns: [Card1, Card2, Card3, ...]

┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 9: RESULT DISPLAY                       │
└─────────────────────────────────────────────────────────────────┘

Dashboard displays:
    │
    ├──► Baseline Risk Card
    │         ├── Risk: 45.2%
    │         ├── Risk Category: HIGH
    │         └── Color: Red
    │
    ├──► New Risk Card
    │         ├── Risk: 27.8%
    │         ├── Risk Category: MODERATE
    │         ├── Delta: -17.4% ↓
    │         └── Color: Orange
    │
    ├──► SHAP Explanation Card
    │         ├── Top 5 features
    │         ├── Contribution values
    │         └── Direction arrows
    │
    ├──► Coupled Parameters Card (if enabled)
    │         └── Shows auto-adjusted parameters
    │
    ├──► Plausibility Score Card
    │         └── Feasibility: 0.75 (Moderately Plausible)
    │
    └──► Intervention Recommendations Card
              ├── Tier 1: Monitoring
              ├── Tier 2: Fluids
              ├── Tier 3: Medications
              └── Tier 4: Dialysis (if needed)

┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 10: AUDIT & GOVERNANCE                   │
└─────────────────────────────────────────────────────────────────┘

Throughout all phases, the system logs:
    │
    audit_logging.py
    │
    ├──► Training events → audit_logs/training_events.jsonl
    ├──► Prediction events → audit_logs/prediction_events.jsonl
    ├──► Explanation events → audit_logs/explanation_events.jsonl
    └──► What-if events → audit_logs/whatif_events.jsonl

Each event includes:
    ├── Unique event ID (UUID)
    ├── Timestamp (ISO 8601)
    ├── Event type
    ├── User ID
    ├── Event data (parameters, results)
    └── Cryptographic hash (SHA-256)
```

---

## 3. Module Interaction Diagrams

### 3.1 Training Pipeline File Interactions

```
┌──────────────────────┐
│ User runs:           │
│ train_advanced_      │
│ models.py            │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  train_advanced_models.py                               │
│                                                          │
│  ┌────────────────────────────────────────────┐         │
│  │  1. AdvancedDataGenerator                 │         │
│  │     - generate_realistic_clinical_data()  │         │
│  │     - assign_disease_labels()             │         │
│  └─────────────┬──────────────────────────────┘         │
│                ▼                                         │
│  ┌────────────────────────────────────────────┐         │
│  │  2. AdvancedFeatureEngineer               │         │
│  │     - create_clinical_features()          │         │
│  │     - compute_severity_scores()           │         │
│  └─────────────┬──────────────────────────────┘         │
│                ▼                                         │
│  ┌────────────────────────────────────────────┐         │
│  │  3. AdvancedModelTrainer                  │         │
│  │     - train_disease_model()               │         │
│  │     - optimize_hyperparameters()          │         │
│  │     - find_optimal_threshold()            │         │
│  └─────────────┬──────────────────────────────┘         │
└────────────────┼──────────────────────────────────────────┘
                 │
                 ├──► Calls: model_registry.py
                 │           └── register_model()
                 │
                 ├──► Calls: audit_logging.py
                 │           └── log_event(MODEL_TRAINING)
                 │
                 └──► Saves: trained_models/{disease}.pkl
                             └── joblib.dump(model_bundle)
```

### 3.2 Dashboard Runtime File Interactions

```
┌──────────────────────┐
│ User runs:           │
│ enhanced_dashboard_  │
│ with_whatif.py       │
└──────────┬───────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  enhanced_dashboard_with_whatif.py                            │
│                                                                │
│  Startup:                                                     │
│  ├──► Loads: trained_models/*.pkl (joblib.load)              │
│  ├──► Initializes: Dash app                                  │
│  └──► Creates: UI layout                                     │
│                                                                │
│  User Interaction:                                            │
│  │                                                            │
│  └──► User adjusts sliders                                   │
│       │                                                       │
│       └──► Triggers: update_whatif_analysis()                │
│           │                                                   │
│           ├──► Calls: predict_with_real_model()              │
│           │           ├── Loads model from TRAINED_MODELS    │
│           │           ├── Prepares features                  │
│           │           ├── model.predict_proba(X)             │
│           │           └── Returns risk score                 │
│           │                                                   │
│           ├──► Calls: get_shap_explanation()                 │
│           │           ├── Loads model                        │
│           │           ├── shap.TreeExplainer(model)          │
│           │           ├── explainer.shap_values(X)           │
│           │           └── Returns feature importances        │
│           │                                                   │
│           └──► Returns: Updated UI components                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 3.3 Novel Components Integration

```
┌──────────────────────┐
│ User runs:           │
│ main.py              │
└──────────┬───────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  main.py                                                      │
│                                                                │
│  Initialization:                                              │
│  ├──► Imports: physiological_coupling                        │
│  │               └── PhysiologicalCouplingEngine()           │
│  │                                                            │
│  ├──► Imports: plausibility_scoring                          │
│  │               └── ClinicalPlausibilityScorer()            │
│  │                                                            │
│  ├──► Imports: hierarchical_interventions                    │
│  │               └── HierarchicalInterventionEngine()        │
│  │                                                            │
│  └──► Loads: trained_models/*.pkl                            │
│                                                                │
│  User Actions:                                                │
│  │                                                            │
│  ├──► Tab 1: Patient State                                   │
│  │     └── Display current vital signs and risk              │
│  │                                                            │
│  ├──► Tab 2: Apply Coupling                                  │
│  │     │                                                      │
│  │     └──► User changes: temperature = 40°C                 │
│  │         │                                                  │
│  │         └──► Callback: apply_coupling()                   │
│  │             │                                              │
│  │             ├──► coupling_engine.apply_coupling(          │
│  │             │        "temperature", 40.0, baseline)       │
│  │             │                                              │
│  │             ├──► Returns: {                                │
│  │             │         "temperature": 40.0,                │
│  │             │         "heart_rate": 110  # Auto-adjusted │
│  │             │     }                                        │
│  │             │                                              │
│  │             └──► Displays coupled parameters              │
│  │                                                            │
│  ├──► Tab 3: Score Plausibility                              │
│  │     │                                                      │
│  │     └──► User: Reduce creatinine 2.5 → 1.2 in 6 hours    │
│  │         │                                                  │
│  │         └──► Callback: score_plausibility()               │
│  │             │                                              │
│  │             ├──► plausibility_scorer.score_plausibility(  │
│  │             │        "creatinine", 2.5, 1.2, 6,          │
│  │             │        InterventionType.MEDICATION,         │
│  │             │        UrgencyLevel.URGENT)                 │
│  │             │                                              │
│  │             ├──► Returns: 0.35 (Low plausibility)        │
│  │             │                                              │
│  │             └──► Displays score with interpretation       │
│  │                                                            │
│  └──► Tab 4: Generate Interventions                          │
│      │                                                        │
│      └──► User: Target risk = 30%                            │
│          │                                                    │
│          └──► Callback: generate_interventions()             │
│              │                                                │
│              ├──► intervention_engine.                       │
│              │         generate_intervention_plan(           │
│              │            patient, shap_values, 0.30,        │
│              │            ResourceLevel.ICU)                 │
│              │                                                │
│              ├──► Returns: InterventionPlan with tiers       │
│              │                                                │
│              └──► Displays hierarchical recommendations      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Process Flows

### 4.1 Prediction Process Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION WORKFLOW                          │
└─────────────────────────────────────────────────────────────────┘

INPUT: Patient data (dict or DataFrame)
    │
    ├── age: 68
    ├── heart_rate: 115
    ├── systolic_bp: 95
    ├── temperature: 101.5
    ├── creatinine: 1.8
    └── ... (other features)

    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Load Model Bundle                                      │
│                                                                 │
│ File: disease_model_service.py or dashboard file               │
│                                                                 │
│ bundle = joblib.load('trained_models/sepsis_advanced.pkl')     │
│                                                                 │
│ Extracts:                                                       │
│ ├── model: XGBoostClassifier                                   │
│ ├── scaler: StandardScaler                                     │
│ ├── feature_names: List[str]                                   │
│ ├── threshold: float                                            │
│ └── metrics: Dict[str, float]                                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Feature Mapping and Validation                         │
│                                                                 │
│ Map user-friendly names to model features:                     │
│ ├── "age" → "age"          ✅                                  │
│ ├── "heart_rate" → "Heart Rate"  ✅                            │
│ ├── "systolic_bp" → "systolic_bp"  ✅                          │
│ └── ... (validate all required features present)              │
│                                                                 │
│ Check for missing values:                                      │
│ └── If missing: Impute with median or raise error             │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Feature Engineering (if needed)                        │
│                                                                 │
│ File: train_advanced_models.py → AdvancedFeatureEngineer      │
│                                                                 │
│ Computes derived features:                                     │
│ ├── MAP = (95 + 2*65) / 3 = 75 mmHg                           │
│ ├── Shock Index = 115 / 95 = 1.21                             │
│ ├── BUN/Cr Ratio = 25 / 1.8 = 13.9                            │
│ └── SIRS Score = 3 (based on temp, HR, WBC)                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Feature Standardization                                │
│                                                                 │
│ X_scaled = scaler.transform(X)                                 │
│                                                                 │
│ Example transformation:                                        │
│ ├── age: 68 → 0.45 (standardized)                             │
│ ├── heart_rate: 115 → 1.23 (standardized)                     │
│ └── temperature: 101.5 → 1.87 (standardized)                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Model Prediction                                       │
│                                                                 │
│ y_pred_proba = model.predict_proba(X_scaled)                  │
│                                                                 │
│ Returns probability array:                                     │
│ └── [[0.35, 0.65]]  # [P(negative), P(positive)]              │
│                                                                 │
│ Extract positive class probability:                            │
│ └── risk_score = 0.65 (65% risk)                              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Threshold Application                                  │
│                                                                 │
│ If risk_score >= threshold:                                    │
│     prediction = "POSITIVE" (Disease predicted)                │
│ Else:                                                           │
│     prediction = "NEGATIVE" (No disease)                       │
│                                                                 │
│ Example: 0.65 >= 0.45 → "POSITIVE"                            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: Risk Categorization                                    │
│                                                                 │
│ Based on risk_score:                                           │
│ ├── 0.00 - 0.30 → "LOW"                                        │
│ ├── 0.30 - 0.50 → "MODERATE"                                   │
│ ├── 0.50 - 0.70 → "HIGH"                                       │
│ └── 0.70 - 1.00 → "CRITICAL"                                   │
│                                                                 │
│ Example: 0.65 → "HIGH"                                         │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: Prediction Result                                      │
│                                                                 │
│ ModelPrediction(                                               │
│     patient_id="P123",                                         │
│     prediction=0.65,                                           │
│     prediction_class=1,                                        │
│     confidence=0.65,                                           │
│     risk_category="HIGH",                                      │
│     timestamp="2026-03-03T10:30:00Z"                          │
│ )                                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

### 4.2 SHAP Explanation Process Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                  SHAP EXPLANATION WORKFLOW                      │
└─────────────────────────────────────────────────────────────────┘

INPUT: Patient data + Trained model
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Initialize SHAP Explainer                              │
│                                                                 │
│ File: xai_engine.py or dashboard file                          │
│                                                                 │
│ explainer = shap.TreeExplainer(model)                          │
│                                                                 │
│ Note: TreeExplainer is optimal for XGBoost/Random Forest       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Compute SHAP Values                                    │
│                                                                 │
│ shap_values = explainer.shap_values(X_scaled)                  │
│                                                                 │
│ For binary classification, returns array:                      │
│ └── [contribution for each feature]                            │
│                                                                 │
│ Example:                                                        │
│ ┌────────────────┬─────────────┬──────────────┐               │
│ │ Feature        │ SHAP Value  │ Direction    │               │
│ ├────────────────┼─────────────┼──────────────┤               │
│ │ creatinine     │ +0.253      │ Increases    │               │
│ │ age            │ +0.188      │ Increases    │               │
│ │ systolic_bp    │ +0.161      │ Increases    │               │
│ │ temperature    │ +0.142      │ Increases    │               │
│ │ heart_rate     │ +0.089      │ Increases    │               │
│ │ glucose        │ -0.034      │ Decreases    │               │
│ │ hemoglobin     │ -0.067      │ Decreases    │               │
│ └────────────────┴─────────────┴──────────────┘               │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Get Baseline (Expected Value)                          │
│                                                                 │
│ baseline = explainer.expected_value                            │
│                                                                 │
│ This is the average prediction across all training data        │
│ Example: baseline = 0.35 (35% average risk)                    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Rank Feature Importances                               │
│                                                                 │
│ Sort features by absolute SHAP value:                          │
│                                                                 │
│ Top 5 Contributors:                                            │
│ 1. creatinine: +0.253 (↗️ Strongly increases risk)             │
│ 2. age: +0.188 (↗️ Moderately increases risk)                  │
│ 3. systolic_bp: +0.161 (↗️ Moderately increases risk)          │
│ 4. temperature: +0.142 (↗️ Increases risk)                     │
│ 5. heart_rate: +0.089 (↗️ Slightly increases risk)             │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Clinical Translation                                   │
│                                                                 │
│ File: xai_engine.py → ClinicalTranslator                       │
│                                                                 │
│ For each feature:                                              │
│ ├── Map technical name to clinical term                        │
│ ├── Add units and normal range                                 │
│ └── Generate clinical interpretation                           │
│                                                                 │
│ Example:                                                        │
│ "creatinine" → "Creatinine (kidney function marker)"          │
│                                                                 │
│ Clinical Interpretation:                                       │
│ "Elevated creatinine (1.8 mg/dL, normal <1.2) strongly        │
│  increases kidney failure risk. Consider nephrology consult."  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Verify SHAP Sum Property                               │
│                                                                 │
│ Verification:                                                   │
│ baseline + sum(shap_values) = prediction                       │
│                                                                 │
│ Example:                                                        │
│ 0.35 + (0.253 + 0.188 + ... + (-0.067)) = 0.65 ✅             │
│                                                                 │
│ This ensures SHAP values are accurate and trustworthy          │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: SHAP Explanation                                       │
│                                                                 │
│ SHAPExplanation(                                               │
│     prediction=0.65,                                           │
│     baseline_prediction=0.35,                                  │
│     feature_importances=[                                      │
│         FeatureImportance(                                     │
│             feature_name="creatinine",                         │
│             importance=0.253,                                  │
│             direction="increases",                             │
│             clinical_interpretation="Elevated creatinine..."   │
│         ),                                                      │
│         ...                                                     │
│     ],                                                          │
│     shap_values=array([0.253, 0.188, ...]),                   │
│     clinical_explanation=ClinicalExplanation(...)             │
│ )                                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

### 4.3 What-If Scenario Process Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                WHAT-IF SCENARIO WORKFLOW                        │
└─────────────────────────────────────────────────────────────────┘

INPUT: Baseline patient + Desired changes
    │
    Baseline: {creatinine: 2.5, age: 68, ...}
    Changes:  {creatinine: 1.2}  # Target value
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Baseline Analysis                                      │
│                                                                 │
│ File: whatif_engine.py → WhatIfEngine                          │
│                                                                 │
│ baseline_analysis = engine.analyze_baseline(                   │
│     model_name="kidney_failure",                               │
│     patient_data=baseline                                      │
│ )                                                               │
│                                                                 │
│ Returns:                                                        │
│ ├── current_risk: 0.65 (65%)                                   │
│ ├── modifiable_features: [creatinine, glucose, BP, ...]       │
│ ├── fixed_features: [age, gender]                             │
│ └── feature_risks: {creatinine: 0.253, age: 0.188, ...}       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Constraint Validation                                  │
│                                                                 │
│ Check clinical constraints:                                    │
│ ├── Is creatinine modifiable? ✅ (ACTIONABLE)                  │
│ ├── Is 1.2 in valid range? ✅ (0.5-10 mg/dL)                  │
│ ├── Is change physiologically possible? ⚠️ (check rate)        │
│ └── Are there dependencies? ✅ (check BUN coupling)            │
│                                                                 │
│ If intervention time specified (e.g., 6 hours):                │
│ └── Check max_change_per_hour constraint                       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Apply Physiological Coupling (NOVEL PATENT)            │
│                                                                 │
│ File: physiological_coupling.py                                │
│                                                                 │
│ When creatinine changes, check for coupled parameters:         │
│                                                                 │
│ coupling_engine.apply_coupling(                                │
│     "creatinine", 1.2, baseline                                │
│ )                                                               │
│                                                                 │
│ Automatic adjustments:                                         │
│ ├── BUN: 40 → 25 mg/dL (maintains BUN/Cr ratio ~20:1)        │
│ └── (Other couplings as defined)                              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Create Modified Patient                                │
│                                                                 │
│ new_patient = baseline.copy()                                  │
│ new_patient['creatinine'] = 1.2  # User change                 │
│ new_patient['BUN'] = 25          # Auto-coupled                │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Compute New Risk                                       │
│                                                                 │
│ new_risk = model.predict_proba(new_patient)                    │
│                                                                 │
│ Returns: 0.35 (35%)                                            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Compute Risk Delta                                     │
│                                                                 │
│ risk_delta = new_risk - baseline_risk                          │
│            = 0.35 - 0.65                                       │
│            = -0.30  (-30 percentage points)                    │
│                                                                 │
│ risk_delta_percent = (risk_delta / baseline_risk) * 100       │
│                    = (-0.30 / 0.65) * 100                      │
│                    = -46.2%  (46% reduction)                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: Score Clinical Plausibility (NOVEL PATENT)             │
│                                                                 │
│ File: plausibility_scoring.py                                  │
│                                                                 │
│ plausibility_score = scorer.score_plausibility(                │
│     parameter="creatinine",                                    │
│     current_value=2.5,                                         │
│     target_value=1.2,                                          │
│     time_hours=6,                                              │
│     intervention_type=InterventionType.MEDICATION,             │
│     urgency=UrgencyLevel.URGENT                                │
│ )                                                               │
│                                                                 │
│ Returns: 0.35                                                  │
│                                                                 │
│ Interpretation: "Low plausibility - Creatinine reduction of   │
│ 1.3 mg/dL in 6 hours via medication alone is aggressive.      │
│ Consider dialysis (plausibility: 0.85) instead."              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 8: Generate Alternative Interventions (NOVEL PATENT)      │
│                                                                 │
│ File: hierarchical_interventions.py                            │
│                                                                 │
│ intervention_plan = engine.generate_intervention_plan(         │
│     patient_data=baseline,                                     │
│     shap_values=shap_values,                                   │
│     target_risk=0.35,                                          │
│     resource_level=ResourceLevel.INTENSIVE_CARE                │
│ )                                                               │
│                                                                 │
│ Returns tiered plan:                                           │
│                                                                 │
│ Tier 1 - Monitoring:                                           │
│ ├── Monitor creatinine q6h                                     │
│ ├── Monitor urine output                                       │
│ └── Cost: $200, Risk Reduction: 5%                            │
│                                                                 │
│ Tier 2 - Non-invasive:                                        │
│ ├── IV fluid resuscitation (2L normal saline)                 │
│ ├── Hold nephrotoxic medications                              │
│ └── Cost: $500, Risk Reduction: 15%                           │
│                                                                 │
│ Tier 3 - Medication:                                           │
│ ├── Diuretics (furosemide 40mg IV)                            │
│ ├── Adjust vasopressors if hypotensive                        │
│ └── Cost: $1,200, Risk Reduction: 25%                         │
│                                                                 │
│ Tier 4 - Invasive:                                             │
│ ├── Continuous renal replacement therapy (CRRT)               │
│ ├── OR: Intermittent hemodialysis                             │
│ └── Cost: $8,000, Risk Reduction: 46%  ⭐ MATCHES TARGET      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: Scenario Result                                        │
│                                                                 │
│ ScenarioResult(                                                │
│     baseline_risk=0.65,                                        │
│     new_risk=0.35,                                             │
│     risk_delta=-0.30,                                          │
│     risk_delta_percent=-46.2,                                  │
│     modified_features={                                        │
│         "creatinine": (2.5, 1.2),                             │
│         "BUN": (40, 25)  # Auto-coupled                       │
│     },                                                          │
│     constraint_violations=[],                                  │
│     plausibility_score=0.35,                                   │
│     clinical_interpretation="Requires invasive intervention",  │
│     recommended_tier=4,                                        │
│     intervention_plan=InterventionPlan(...)                    │
│ )                                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Flow Between Files

### 5.1 Training Data Flow

```
mimic_large.csv (or synthetic generation)
    │
    └──► load_mimic_for_training.py
         OR
         training_pipeline.py → SyntheticDataGenerator
              │
              └──► Raw DataFrame (n_samples x features)
                   │
                   └──► train_advanced_models.py
                        │
                        ├──► AdvancedDataGenerator
                        │    └──► Enhanced DataFrame
                        │
                        ├──► AdvancedFeatureEngineer
                        │    └──► Engineered DataFrame (32 features)
                        │
                        └──► AdvancedModelTrainer
                             │
                             ├──► XGBoost model training
                             │    └──► Trained model object
                             │
                             ├──► model_registry.py
                             │    └──► Model metadata stored
                             │
                             ├──► audit_logging.py
                             │    └──► audit_logs/training_events.jsonl
                             │
                             └──► joblib.dump()
                                  └──► trained_models/{disease}.pkl
```

### 5.2 Prediction Data Flow

```
User Input (Dashboard or API)
    │
    └──► Patient Data Dict
         {age: 68, creatinine: 1.8, ...}
              │
              └──► enhanced_dashboard_with_whatif.py
                   OR main.py
                   OR disease_model_service.py
                        │
                        ├──► Load: trained_models/{disease}.pkl
                        │    └──► Model bundle
                        │
                        ├──► Feature Engineering
                        │    └──► 32 computed features
                        │
                        ├──► Scaling
                        │    └──► StandardScaler.transform()
                        │
                        ├──► Prediction
                        │    └──► model.predict_proba()
                        │         │
                        │         └──► Risk Score: 0.65
                        │
                        ├──► SHAP Explanation
                        │    └──► xai_engine.py
                        │         └──► Feature importances
                        │
                        ├──► What-If Analysis (if requested)
                        │    └──► whatif_engine.py
                        │         │
                        │         ├──► physiological_coupling.py
                        │         ├──► plausibility_scoring.py
                        │         └──► hierarchical_interventions.py
                        │
                        └──► Display Results
                             └──► Dashboard UI or JSON response
```

### 5.3 Audit Trail Data Flow

```
Any System Event
    │
    ├──► Training Event
    ├──► Prediction Event
    ├──► Explanation Event
    └──► What-If Event
         │
         └──► audit_logging.py → AuditLogger
              │
              ├──► Creates AuditEvent
              │    ├── event_id (UUID)
              │    ├── timestamp
              │    ├── event_type
              │    ├── event_data
              │    └── data_hash (SHA-256)
              │
              ├──► Serializes to JSON
              │    └──► {"event_id": "...", "timestamp": "...", ...}
              │
              └──► Appends to JSONL file
                   └──► audit_logs/{event_type}_events.jsonl
                        │
                        └──► Immutable append-only log
                             (No edits or deletes allowed)
```

---

## 6. User Workflows

### 6.1 Clinician Workflow: Assessing a New Patient

```
1. Clinician opens dashboard
   http://127.0.0.1:8051
   
2. Views current patient cohort
   ├──► Model Performance chart shows all 8 diseases
   └──► Risk Distribution shows patient breakdown

3. Selects individual patient
   └──► Click on patient card (e.g., Patient 001)
        │
        └──► Views:
             ├── Demographics: Age 68, Male
             ├── Vital Signs: HR 115, BP 95/65, Temp 38.5°C
             ├── Labs: Cr 1.8, WBC 12.5
             │
             ├── Risk Predictions:
             │   ├── Sepsis: 45% (MODERATE)
             │   ├── Kidney Failure: 65% (HIGH) ⚠️
             │   └── Mortality: 28% (MODERATE)
             │
             └── SHAP Explanations:
                 ├── Top contributor: Creatinine (+0.25)
                 ├── Secondary: Age (+0.19)
                 └── Clinical note: "Elevated creatinine..."

4. Explores "What-If" scenarios
   └──► Adjust sliders:
        ├── Creatinine: 1.8 → 1.2 mg/dL
        ├── Systolic BP: 95 → 120 mmHg
        └── Click "Run Analysis"
             │
             └──► Results:
                  ├── New Risk: 35% (was 65%)
                  ├── Risk Reduction: -30% ↓
                  └── Recommended: ICU admission, CRRT

5. Reviews intervention recommendations
   └──► Tiered plan:
        ├── Tier 1: Monitor vitals q1h
        ├── Tier 2: IV fluids 2L
        ├── Tier 3: Diuretics
        └── Tier 4: Dialysis (best outcome)

6. Makes clinical decision
   └──► Orders:
        ├── Transfer to ICU
        ├── Nephrology consult
        └── Prepare for potential dialysis
```

### 6.2 Researcher Workflow: Training New Model

```
1. Prepare data
   ├──► Option A: Download MIMIC-III
   │    └──► Follow KAGGLE_MIMIC_GUIDE.md
   │
   └──► Option B: Use synthetic data
        └──► Already built-in

2. Run training script
   $ python train_advanced_models.py --n-samples 50000
   
   └──► Training process:
        ├── Generate/load data: 50,000 patients
        ├── Engineer features: 32 clinical features
        ├── Train 8 disease models
        ├── Optimize thresholds
        ├── Evaluate performance
        └── Save to trained_models/

3. Verify models
   $ python verify_trained_models.py
   
   └──► Checks:
        ├── ✅ sepsis_advanced_50000.pkl
        ├── ✅ kidney_failure_advanced_50000.pkl
        └── ... (all 8 models)

4. Review results
   └──► Check TRAINING_SUMMARY.md
        ├── AUROC scores
        ├── Accuracy metrics
        └── Clinical grade assessments

5. Test inference
   $ python test_inference.py
   
   └──► Validates predictive pipeline

6. Launch dashboard
   $ python enhanced_dashboard_with_whatif.py
   
   └──► Explore model performance visually

7. Review audit logs
   └──► Check audit_logs/training_events.jsonl
        └──► Verify all events logged correctly
```

### 6.3 Regulatory Reviewer Workflow

```
1. Review compliance documentation
   └──► Open regulatory_submission/
        ├── compliance_matrix.md
        ├── compliance_matrix.csv
        └── compliance_matrix.json

2. Check audit trail
   └──► Examine audit_logs/
        ├── training_events.jsonl
        │   └── All model training events
        ├── prediction_events.jsonl
        │   └── All predictions made
        └── explanation_events.jsonl
            └──► All explanations generated

3. Verify model registry
   └──► Check model_registry.py outputs
        ├── Model versions
        ├── Training metadata
        ├── Performance metrics
        └── Data/model hashes (integrity)

4. Review technical documentation
   └──► Read:
        ├── PATENT_DISCLOSURE.md (novel claims)
        ├── TECHNICAL_SPECIFICATION.md
        ├── MODULAR_ARCHITECTURE.md
        └── README.md

5. Validate FDA requirements
   └──► Use compliance_matrix.py
        └──► Check FDA 510(k) mappings
             ├── Risk management ✅
             ├── Software validation ✅
             ├── Clinical evaluation ✅
             └── Labeling requirements ✅

6. Assess patentability
   └──► Review PATENT_DISCLOSURE.md
        ├── Physiological Coupling (Novel)
        ├── Plausibility Scoring (Novel)
        └──► Hierarchical Interventions (Novel)
```

---

## 7. Technical Workflows

### 7.1 Debugging Workflow

```
Problem: Model prediction seems incorrect

1. Check model file
   $ python check_model.py trained_models/sepsis_advanced.pkl
   
   └──► Verify:
        ├── Model loads successfully
        ├── Feature names match
        └── Metrics are reasonable

2. Test with known sample
   $ python test_inference.py
   
   └──► Compare output to expected

3. Check SHAP values
   $ python test_shap_display.py
   
   └──► Verify explainability works

4. Review audit logs
   └──► Check audit_logs/prediction_events.jsonl
        └──► Look for errors or anomalies

5. Run dashboard in debug mode
   └──► enhanced_dashboard_with_whatif.py
        ├── Set debug=True
        └── Check console for errors

6. Validate input features
   └──► Ensure feature engineering matches training
        └──► Check AdvancedFeatureEngineer.create_clinical_features()
```

### 7.2 Adding New Disease Workflow

```
1. Prepare disease-specific data
   └──► Define clinical criteria for disease label

2. Update SyntheticDataGenerator (if using)
   └──► training_pipeline.py
        └──► Add disease_label_generation logic

3. Create DiseaseModelService class
   └──► disease_model_service.py
        └──► Extend BaseModelService
             ├── Define disease_name
             ├── Define clinical_features
             └── Override train/predict/explain

4. Add to training pipeline
   └──► train_advanced_models.py
        └──► Add disease to DISEASES list

5. Train model
   $ python train_advanced_models.py --diseases new_disease

6. Verify
   $ python verify_trained_models.py

7. Update dashboard
   └──► enhanced_dashboard_with_whatif.py
        └──► Add disease to dropdown/charts

8. Update documentation
   └──► README.md, PROJECT_SUMMARY.md

9. Test end-to-end
   $ python test_inference.py new_disease
```

---

## Summary

This documentation covers:

✅ **End-to-end workflows** from setup to deployment  
✅ **Detailed process flows** for prediction, explanation, and what-if  
✅ **File interactions** showing how modules connect  
✅ **User workflows** for clinicians, researchers, and reviewers  
✅ **Technical workflows** for debugging and extending the system  

**Key Insight:** The system is designed as a pipeline where:
1. **Data flows** from sources → preprocessing → feature engineering
2. **Models** are trained → registered → deployed
3. **Explainability** is generated at prediction time
4. **Governance** captures everything in audit logs
5. **Novel components** enhance clinical utility

All components work together to create a complete, production-ready medical AI system.

---

*For implementation details, see individual source files and the comprehensive code documentation.*
