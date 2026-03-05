# MODULAR ARCHITECTURE - VISUAL DIAGRAMS

## Complete Visual Reference for Architecture

---

## 1. SYSTEM ARCHITECTURE OVERVIEW

```
┌──────────────────────────────────────────────────────────────────────┐
│         EXPLAINABLE MEDICAL AI - MODULAR ARCHITECTURE                │
└──────────────────────────────────────────────────────────────────────┘

                    ╔════════════════════════════════════╗
                    ║       CLINICAL DATA INPUT         ║
                    ║   (MIMIC-III or Synthetic)        ║
                    ╚════════════╤═══════════════════════╝
                                 │
                    ┌────────────▼────────────┐
                    │   1. DATA LAYER        │
                    │   ────────────────────  │
                    │  • Load MIMIC-III      │
                    │  • Preprocess data     │
                    │  • Feature engineer    │
                    │  • Split train/test    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  ProcessedDataset      │
                    │  ──────────────────    │
                    │  • X_train (n, 13)     │
                    │  • X_test (m, 13)      │
                    │  • y_train, y_test     │
                    │  • feature_names       │
                    │  • scaler              │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
        ┌────────▼────────┐ ┌────▼────────┐ ┌──▼───────────┐
        │  2. MODEL       │ │  3. XAI     │ │ 4. WHAT-IF   │
        │  SERVICES       │ │ ENGINE      │ │ ENGINE       │
        ├────────────────┤ ├────────────┤ ├──────────────┤
        │ • Sepsis       │ │ • SHAP     │ │ • Scenarios  │
        │ • Kidney       │ │ • LIME     │ │ • Impact     │
        │ • Cardio       │ │ • Clinical │ │ • Optimize   │
        │ • Mortality    │ │   Trans.   │ │              │
        └────────┬───────┘ └────┬───────┘ └──┬───────────┘
                 │               │           │
                 │     ┌─────────┴───────────┘
                 │     │
        ┌────────▼─────▼──────────┐
        │  5. DASHBOARD SERVER   │
        ├───────────────────────┤
        │ • Performance charts  │
        │ • Risk distribution   │
        │ • Patient analysis    │
        │ • What-If interative  │
        │ • Real-time updates   │
        └────────────┬──────────┘
                     │
                http://127.0.0.1:8051
                     │
        ┌────────────▼──────────┐
        │ 6. EVALUATION MODULE  │
        ├───────────────────────┤
        │ • Compute metrics     │
        │ • Validate clinical   │
        │ • Generate reports    │
        └───────────────────────┘
```

---

## 2. DATA LAYER DETAILED FLOW

```
┌───────────────────────────────────────────────────┐
│           INPUT: Raw MIMIC-III Data              │
├───────────────────────────────────────────────────┤
│ ADMISSIONS.csv   │  PATIENTS.csv   │ DIAGNOSES...│
│ (admissions)     │  (demographics) │ (ICD-9)    │
└─────────┬────────┴─────────┬───────┴─────────┬───┘
          │                  │                 │
          └──────────────────┼─────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │  MIMICDataSource.load_data()       │
          │  ───────────────────────────────   │
          │  • Read CSV files                  │
          │  • Validate tables                 │
          │  • Create dataframes               │
          └──────────────────┬──────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │  Preprocessor.preprocess()          │
          │  ──────────────────────────────────  │
          │  Step 1: Handle missing values      │
          │  Step 2: Encode categoricals       │
          │  Step 3: Engineer features         │
          │  Step 4: Normalize numericals      │
          │  Step 5: Create targets            │
          │  Step 6: Split train/test          │
          └──────────────────┬──────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │ ClinicalFeatureEngineer            │
          │ ────────────────────────────────── │
          │ From raw data create:              │
          │ • age (from DOB + admission)       │
          │ • vital signs features             │
          │ • lab features                     │
          │ • risk indicators                  │
          └──────────────────┬──────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │ OUTPUT: ProcessedDataset           │
          ├──────────────────────────────────┤
          │ X_train: (1000, 13)               │
          │ X_test: (428, 13)                 │
          │ y_train: (1000,)                  │
          │ y_test: (428,)                    │
          │ features: [age, hr, bp, ...]      │
          │ scaler: StandardScaler()          │
          └───────────────────────────────────┘
```

---

## 3. MODEL SERVICES ORCHESTRATION

```
                        ProcessedDataset
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
    ┌────────────┐         ┌────────────┐        ┌────────────┐
    │  Sepsis    │         │  Kidney    │        │Cardiovasc. │
    │  Service   │         │  Failure   │        │   Service  │
    │            │         │  Service   │        │            │
    │ RF Model   │         │ XGB Model  │        │ RF Model   │
    └──────┬─────┘         └──────┬─────┘        └──────┬─────┘
           │                      │                     │
           │     Train()          │     Train()         │
           │                      │                     │
           │      ┌───────────────┼──────────────┐      │
           │      │               │              │      │
           ▼      ▼               ▼              ▼      ▼
        ┌──────────────────────────────────────────────┐
        │      MODEL REGISTRY (Orchestrator)         │
        ├──────────────────────────────────────────────┤
        │ Services: {sepsis, kidney, cardio, mortality}│
        │                                              │
        │ Methods:                                     │
        │ • train_all(data)                            │
        │ • predict_all(X)                             │
        │ • evaluate_all(X_test, y_test)               │
        │ • save_all() / load_all()                     │
        └──────────────────┬───────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
    Predictions        Explanations      Evaluation
    (XAI Engine)       (Dashboard)       (Metrics)
```

---

## 4. XAI ENGINE FLOW

```
              Trained Model + Test Data
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
    ┌────────────┐          ┌────────────┐
    │   SHAP     │          │   LIME     │
    │ Explainer  │          │ Explainer  │
    └─────┬──────┘          └─────┬──────┘
          │                       │
    Global Feature         Local Patient
    Importance            Explanation
          │                       │
          │         ┌─────────────┤
          │         │             │
          ▼         ▼             ▼
    ┌────────────────────────────────┐
    │  Explanation Aggregator        │
    │  (Combine SHAP + LIME)         │
    └────────────┬───────────────────┘
                 │
    ┌────────────▼────────────┐
    │ Clinical Translator     │
    │ ──────────────────────  │
    │ Convert to medical lang │
    │ Example: "High creat.   │
    │ (+0.253 ↗️) increases   │
    │ risk"                   │
    └────────────┬────────────┘
                 │
    ┌────────────▼─────────────────┐
    │ LocalExplanation             │
    ├──────────────────────────────┤
    │ disease: "kidney_failure"    │
    │ feature_contributions: {...} │
    │ top_features: [...]          │
    │ clinical_summary: "str"      │
    │ confidence: 0.95             │
    └──────────────────────────────┘
```

---

## 5. WHAT-IF ENGINE ANALYSIS

```
          Patient Features (Original)
                    │
        ┌───────────┴────────────┐
        │                        │
        ▼                        ▼
    Parameter              WhatIfAnalyzer
    Ranges
    ├── age: 20-90         ├─ analyze_feature_variation()
    ├── creatinine: 0.5-4  │
    ├── BP: 90-180         ├─ optimize_parameters()
    ├── glucose: 70-300    │
    ├── HR: 60-120         └─ simulate_intervention()
    └── temp: 95-105
        │
        └─→ For each feature value:
            ├─ Modify patient data
            ├─ Call ModelRegistry.predict_all()
            └─ Record risk scores
                │
                ├─ Sepsis risk over range
                ├─ Kidney risk over range
                ├─ Cardio risk over range
                └─ Mortality risk over range
                    │
            ┌───────┴───────┐
            │               │
            ▼               ▼
        Optimal       Treatment Impact
        Parameters    Recommendations
            │               │
            └───────┬───────┘
                    ▼
        ┌──────────────────────┐
        │ WhatIfAnalysis       │
        ├──────────────────────┤
        │ parameter: "creat."  │
        │ original_value: 1.8  │
        │ test_values: array   │
        │ risks_per_disease:{} │
        │ optimal_value: 0.8   │
        │ recommendations:[]   │
        └──────────────────────┘
```

---

## 6. DASHBOARD REAL-TIME FLOW

```
User Interaction (Browser)
            │
            ▼
http://127.0.0.1:8051 (Dash)
            │
    ┌───────┴────────┐
    │                │
    ▼                ▼
Model         Patient
Performance   Selection
View            │
    │           │ Select Patient ID
    │           │
    │           ▼
    │    ┌──────────────────┐
    │    │ ModelRegistry    │
    │    │  .predict_all()  │
    │    └────────┬─────────┘
    │             │
    │             ▼
    │    ┌──────────────────┐
    │    │ XAIEngine        │
    │    │ .explain_local() │
    │    └────────┬─────────┘
    │             │
    │             ▼
    │    ┌──────────────────┐
    │    │ Patient Cards    │
    │    │ Risk Gauges      │
    │    │ SHAP Features    │
    │    └──────────────────┘
    │
    └────→ What-If Panel:
            User adjusts sliders
            (age, creatinine, BP, etc)
                    │
                    ▼
            WhatIfEngine.analyze()
                    │
                    ▼
            Real-time risk update
            (< 500ms per slider change)
                    │
                    ▼
            Updated charts + recommendations
```

---

## 7. MODULE DEPENDENCIES

```
        ┌──────────────────┐
        │  Core (Shared)   │  ← config, logging, types, constants
        │  ────────────────│
        └────────┬─────────┘
                 │
     ┌───────────┴───────────┐
     │                       │
     ▼                       ▼
┌──────────────┐     ┌──────────────────┐
│ Data Layer   │     │ Can operate      │
│ (isolated)   │     │ independently    │
└──────┬───────┘     │                  │
       │             └──────────────────┘
       │
       ▼
┌──────────────────┐
│ Model Services   │ ← Depends on Data Layer
├──────────────────┤
│ • Sepsis         │
│ • Kidney         │
│ • Cardio         │
│ • Mortality      │
└──────┬───────────┘
       │
       ├─────────────────┬──────────────────┐
       │                 │                  │
       ▼                 ▼                  ▼
┌──────────┐      ┌───────────┐      ┌──────────────┐
│ XAI      │      │ What-If   │      │ Evaluation   │
│ Engine   │      │ Engine    │      │ Module       │
└──────┬───┘      └─────┬─────┘      └──────┬───────┘
       │                │                   │
       └────────────────┼───────────────────┘
                        │
                        ▼
                ┌────────────────────┐
                │ Dashboard Server   │ ← Depends on all above
                │ (Orchestrates all) │
                └────────────────────┘
```

---

## 8. DATA TYPE FLOW

```
Data Layer Output
    │
    ▼ ProcessedDataset
    ├─ X_train: DataFrame (1000, 13)
    ├─ X_test: DataFrame (428, 13)
    ├─ y_train: Series (1000,)
    ├─ y_test: Series (428,)
    ├─ feature_names: List[13]
    └─ scaler: StandardScaler

Model Services Output
    │
    ▼ ModelPrediction
    ├─ disease: str
    ├─ risk_score: float (0-1)
    ├─ risk_category: str
    └─ confidence: float

XAI Engine Output
    │
    ▼ LocalExplanation
    ├─ disease: str
    ├─ feature_contributions: Dict
    ├─ top_features: List[5]
    └─ clinical_summary: str

What-If Engine Output
    │
    ▼ WhatIfAnalysis
    ├─ parameter: str
    ├─ test_values: ndarray
    ├─ risks_per_disease: Dict
    └─ recommendations: List[str]

Evaluation Module Output
    │
    ▼ EvaluationMetrics
    ├─ auc: float
    ├─ accuracy: float
    ├─ f1_score: float
    ├─ precision: float
    └─ recall: float

Dashboard Output
    │
    ▼ Interactive Web UI
    ├─ http://127.0.0.1:8051
    ├─ Charts & visualizations
    ├─ Real-time updates
    └─ Downloadable reports
```

---

## 9. REQUEST-RESPONSE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                    TYPICAL USER WORKFLOW                         │
└─────────────────────────────────────────────────────────────────┘

1. USER NAVIGATES TO DASHBOARD
   http://127.0.0.1:8051
   │
   ▼
   Dashboard.app loaded
   └─→ All models loaded from disk
   └─→ Test data loaded
   └─→ XAI engines initialized

2. USER SELECTS A PATIENT
   Patient ID: 42
   │
   ▼
   DashboardCallbacks.on_patient_selected(42)
   │
   ├─→ ModelRegistry.predict_all(patient_features)
   │   Returns: {sepsis: 0.2, kidney: 0.81, cardio: 0.15, mort: 0.35}
   │
   ├─→ XAIEngine.explain_patient(42)
   │   ├─→ SHAPExplainer.explain_global() → [creatinine, age, BP]
   │   └─→ LIMEExplainer.explain_local() → LocalExplanation
   │
   └─→ Display:
       ├─ Risk scores (gauges)
       ├─ Risk categories (colors)
       ├─ Top contributing features
       └─ Clinical recommendations

3. USER ADJUSTS WHAT-IF PARAMETER (Creatinine slider)
   Old: 1.8 mg/dL  →  New: 1.2 mg/dL
   │
   ▼
   DashboardCallbacks.on_creatinine_slider_change(1.2)
   │
   ├─→ Modify patient_features
   ├─→ ModelRegistry.predict_all(modified_features)
   │   Returns: {sepsis: 0.2, kidney: 0.45, cardio: 0.13, mort: 0.25}
   │
   ├─→ Calculate deltas:
   │   kidney: 0.81 → 0.45 (-0.36, -44% reduction) ✅
   │
   └─→ Update display (real-time):
       ├─ New risk gauges
       ├─ Risk reduction percentages
       ├─ Updated recommendations
       └─ Optimal value suggestion

4. USER EXPORTS REPORT
   Click "Download PDF"
   │
   ▼
   ReportGenerator.generate_report(patient_id=42)
   │
   ├─→ Collect all predictions
   ├─→ Collect all explanations
   ├─→ Create PDF with:
   │   ├─ Patient demographics
   │   ├─ Risk predictions
   │   ├─ Feature contributions
   │   ├─ What-If analysis
   │   └─ Clinical recommendations
   │
   └─→ Download PDF to browser
```

---

## 10. CLASS HIERARCHY

```
┌────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
├────────────────────────────────────────────────────────────┤

BaseDataSource (ABC)
├── MIMICDataSource
├── SyntheticDataSource
└── CSVDataSource

Preprocessor
└── Methods: preprocess(), _handle_missing(), _encode_cat(), etc.

ClinicalFeatureEngineer
└── Methods: calculate_age(), engineer_vitals(), etc.


┌────────────────────────────────────────────────────────────┐
│                   MODEL SERVICES LAYER                      │
├────────────────────────────────────────────────────────────┤

BaseModelService (ABC)
├── SepsisService
│   └── Uses: RandomForestClassifier
├── KidneyFailureService
│   └── Uses: XGBClassifier
├── CardiovascularService
│   └── Uses: RandomForestClassifier
└── MortalityService
    └── Uses: RandomForestClassifier

ModelRegistry
└── Contains: {sepsis, kidney, cardio, mortality} services


┌────────────────────────────────────────────────────────────┐
│                    XAI ENGINE LAYER                         │
├────────────────────────────────────────────────────────────┤

BaseExplainer (ABC)
├── SHAPExplainer
│   └── Uses: shap.TreeExplainer
├── LIMEExplainer
│   └── Uses: lime.LimeTabularExplainer
└── CounterfactualExplainer

ClinicalTranslator
└── Methods: translate_contribution(), create_summary()


┌────────────────────────────────────────────────────────────┐
│                  WHAT-IF ENGINE LAYER                       │
├────────────────────────────────────────────────────────────┤

WhatIfAnalyzer
├── Methods: analyze_feature_variation(), optimize_parameters()

ScenarioGenerator
└── Methods: generate_clinical_scenarios()

ParameterOptimizer
└── Methods: find_optimal_params()


┌────────────────────────────────────────────────────────────┐
│                DASHBOARD SERVER LAYER                       │
├────────────────────────────────────────────────────────────┤

DashboardApp
├── Layouts: performance, risk, patient, whatif
├── Components: charts, cards, tables
└── Callbacks: real-time updates


┌────────────────────────────────────────────────────────────┐
│               EVALUATION MODULE LAYER                       │
├────────────────────────────────────────────────────────────┤

ModelEvaluator
├── Methods: compute_metrics(), cross_validate()

XAIEvaluator
├── Methods: evaluate_shap_stability(), evaluate_lime_fidelity()

ClinicalValidator
└── Methods: validate_risk_stratification()

ReportGenerator
└── Methods: generate_model_report(), generate_xai_report()
```

---

## 11. DEPLOYMENT ARCHITECTURE

### Option 1: Monolithic
```
┌────────────────────────┐
│  Single Python Process │
├────────────────────────┤
│ • Data Layer           │
│ • Model Services       │
│ • XAI Engine           │
│ • What-If Engine       │
│ • Dashboard            │
│ • Evaluation           │
└────────────────────────┘
     │
     ▼
python main.py
```

### Option 2: Microservices
```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Data Service    │  │  Model Service   │  │  XAI Service     │
│  (port 5001)     │  │  (port 5002)     │  │  (port 5003)     │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Dashboard         │
                    │  (port 8051)       │
                    └───────────────────┘
```

### Option 3: Containerized
```
docker-compose.yml
├── data-layer service
│   └── Dockerfile
├── model-service
│   └── Dockerfile
├── xai-service
│   └── Dockerfile
└── dashboard-service
    └── Dockerfile
```

---

**END OF VISUAL DIAGRAMS**

Use these diagrams as reference during implementation and documentation.
