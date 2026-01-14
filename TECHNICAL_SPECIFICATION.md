# TECHNICAL SPECIFICATION: Multi-Disease Explainable AI Medical Diagnosis System

**Version:** 1.0  
**Status:** IMMUTABLE SPECIFICATION  
**Date:** January 13, 2026

---

## 1. SUPPORTED DISEASES

### 1.1 Disease List (4 Diseases)

| Disease ID | Disease Name | ICD-9 Codes | Description |
|------------|--------------|-------------|-------------|
| `sepsis` | Sepsis and Septic Shock | 038, 995.91, 995.92, 785.52 | Life-threatening infection response with organ dysfunction |
| `kidney_failure` | Acute and Chronic Kidney Injury | 584, 585, 586 | Acute kidney failure and chronic renal insufficiency |
| `cardiovascular` | Cardiovascular Events | 410, 411, 413, 414, 427.5 | Myocardial infarction, angina, heart failure, arrhythmias |
| `mortality` | In-Hospital Mortality | N/A (HOSPITAL_EXPIRE_FLAG) | Death during hospital admission |

### 1.2 Disease Prevalence Targets
- Sepsis: ~9.8%
- Kidney Failure: ~30.3%
- Cardiovascular: ~21.2%
- Mortality: ~15.9%

---

## 2. INPUT DATA MODALITIES

### 2.1 Data Source
**Primary:** MIMIC-III Clinical Database
- **Format:** CSV files from MIMIC-III dataset
- **Required Tables:**
  - `ADMISSIONS.csv` - Hospital admissions data
  - `PATIENTS.csv` - Patient demographics
  - `DIAGNOSES_ICD.csv` - ICD-9 diagnosis codes
  - `CHARTEVENTS.csv` - Vital signs and monitoring data
  - `LABEVENTS.csv` - Laboratory test results
  - `ICUSTAYS.csv` - ICU admission data (optional)

**Fallback:** Synthetic clinical data generator (for demonstration when MIMIC-III unavailable)

### 2.2 Feature Set (13 Clinical Features)

#### 2.2.1 Demographics
| Feature | Type | Range | Units | Description |
|---------|------|-------|-------|-------------|
| `age` | Continuous | 18-100 | years | Patient age at admission |
| `gender` | Categorical | M, F | - | Biological sex |
| `ethnicity` | Categorical | WHITE, BLACK, HISPANIC, ASIAN, OTHER | - | Self-reported ethnicity |
| `insurance` | Categorical | Medicare, Medicaid, Private | - | Insurance type |
| `admission_type` | Categorical | EMERGENCY, ELECTIVE, URGENT | - | Admission category |

#### 2.2.2 Vital Signs
| Feature | Type | Range | Units | Description |
|---------|------|-------|-------|-------------|
| `heart_rate` | Continuous | 40-150 | bpm | Heart rate |
| `systolic_bp` | Continuous | 80-200 | mmHg | Systolic blood pressure |
| `temperature` | Continuous | 95-105 | °F | Body temperature |

#### 2.2.3 Laboratory Values
| Feature | Type | Range | Units | Description |
|---------|------|-------|-------|-------------|
| `glucose` | Continuous | 70-400 | mg/dL | Blood glucose level |
| `creatinine` | Continuous | 0.5-8.0 | mg/dL | Serum creatinine (kidney function) |
| `hemoglobin` | Continuous | 6-18 | g/dL | Hemoglobin level |
| `white_blood_cells` | Continuous | 2-25 | K/µL | White blood cell count |
| `platelet_count` | Continuous | 50-600 | K/µL | Platelet count |

### 2.3 Data Preprocessing Pipeline
1. **Data Loading:** Load MIMIC-III CSV tables
2. **Feature Engineering:** 
   - Calculate age from DOB and admission time
   - Encode categorical variables using LabelEncoder
   - Create clinical risk indicators (boolean flags)
3. **Missing Value Handling:** Median imputation for continuous features
4. **Target Variable Creation:** Map ICD-9 codes to binary disease targets
5. **Train-Test Split:** 70% training, 30% testing with stratification

---

## 3. MACHINE LEARNING MODELS

### 3.1 Model Architecture Per Disease

**REQUIRED:** One model instance per disease (4 total models)

| Component | Specification | Justification |
|-----------|--------------|---------------|
| **Algorithm** | Random Forest Classifier | SHAP TreeExplainer compatibility |
| **Hyperparameters** | `n_estimators=100`, `max_depth=10`, `random_state=42` | Balanced performance vs. interpretability |
| **Class Weighting** | `class_weight='balanced'` | Handle imbalanced disease prevalence |
| **Feature Scaling** | StandardScaler (z-score normalization) | Required for numerical stability |

### 3.2 Alternative Algorithms (Optional)
- **XGBoost:** Gradient boosting (SHAP TreeExplainer compatible)
- **LightGBM:** Light gradient boosting (SHAP TreeExplainer compatible)
- **Logistic Regression:** Linear model (requires SHAP KernelExplainer)

**CONSTRAINT:** Model MUST be compatible with SHAP TreeExplainer (tree-based) OR KernelExplainer (model-agnostic)

### 3.3 Model Training Pipeline
```
For each disease:
  1. Extract features (X) and target (y) from dataset
  2. Verify target has at least 2 classes
  3. Split: train_test_split(test_size=0.3, stratify=y, random_state=42)
  4. Scale: StandardScaler().fit_transform(X_train)
  5. Train: RandomForestClassifier.fit(X_train_scaled, y_train)
  6. Store: model, scaler, metadata
```

---

## 4. EXPLAINABILITY TECHNIQUES

### 4.1 SHAP (SHapley Additive exPlanations)

#### 4.1.1 Specification
| Property | Value | Description |
|----------|-------|-------------|
| **Method** | SHAP TreeExplainer | Exact SHAP values for tree-based models |
| **Scope** | Global + Local | Feature importance across all patients AND per-patient |
| **Output** | Additive feature contributions | Quantifies each feature's contribution to prediction |

#### 4.1.2 Implementation Requirements
```python
# Per Disease Model
shap_explainer = shap.TreeExplainer(model)
shap_values = shap_explainer.shap_values(X_test_scaled)

# Global: Feature importance ranking
feature_importance = np.abs(shap_values).mean(axis=0)

# Local: Per-patient explanation
patient_shap = shap_values[patient_idx]
```

#### 4.1.3 SHAP Outputs
- **Summary Plot:** Global feature importance with distribution
- **Force Plot:** Individual prediction explanation with base value
- **Dependence Plot:** Feature interaction effects
- **Waterfall Plot:** Additive contribution breakdown

### 4.2 LIME (Local Interpretable Model-agnostic Explanations)

#### 4.2.1 Specification
| Property | Value | Description |
|----------|-------|-------------|
| **Method** | LIME Tabular Explainer | Model-agnostic local explanations |
| **Scope** | Local only | Individual patient predictions |
| **Technique** | Local linear approximation | Perturbs features to find important ones |

#### 4.2.2 Implementation Requirements
```python
# Per Disease Model
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train_sample,
    feature_names=feature_names,
    class_names=['No Disease', 'Disease'],
    mode='classification'
)

# Per Patient
lime_exp = lime_explainer.explain_instance(
    data_row=patient_features,
    predict_fn=model.predict_proba,
    num_features=len(feature_names)
)
```

#### 4.2.3 LIME Outputs
- **Feature Contributions:** List of (feature, weight) pairs
- **Local Fidelity Score:** R² of local linear model
- **Predicted Probabilities:** Model confidence for patient

### 4.3 Clinical Translation Requirements

**MANDATORY:** Convert explainability outputs to clinical language

| XAI Output | Clinical Translation | Format |
|------------|---------------------|--------|
| SHAP positive value | "Increases risk" | "creatinine (+0.253) ↗️ Increases risk" |
| SHAP negative value | "Decreases risk" | "hemoglobin (-0.15) ↘️ Decreases risk" |
| LIME positive weight | "Feature contributes to disease" | "creatinine > 1.5 (+0.253): High creatinine increases risk" |
| Feature importance ranking | "Top risk factors" | "1. creatinine, 2. age, 3. blood_pressure" |

---

## 5. REQUIRED DASHBOARD FEATURES

### 5.1 Technology Stack
- **Framework:** Dash (Plotly)
- **Port:** 8050 (basic), 8051 (enhanced with What-If)
- **Interface:** Web browser-based
- **Layout:** Multi-section responsive design

### 5.2 Dashboard Components (MANDATORY)

#### 5.2.1 Model Performance Section
**Required Visualizations:**
- **Bar Chart:** AUC scores per disease
  - X-axis: Disease names
  - Y-axis: AUC score (0.0-1.0)
  - Color coding: Green (>0.7), Yellow (0.5-0.7), Red (<0.5)
- **Metrics Table:** Disease, AUC, Accuracy, F1-Score, Prevalence

#### 5.2.2 Risk Distribution Section
**Required Visualizations:**
- **Pie Chart:** Patient risk categories
  - Categories: CRITICAL (red), HIGH (orange), MODERATE (yellow), LOW (green)
  - Percentages and counts
- **Bar Chart:** Risk category distribution with counts

#### 5.2.3 Patient Analysis Section
**Required Components:**
- **Patient Cards:** Individual patient summaries
  - Patient ID
  - Overall risk score (0-1)
  - Risk category with color coding
  - Per-disease predictions with risk scores
  - Clinical recommendations (bullet list)
- **Filtering:** Ability to filter by risk category

#### 5.2.4 Disease Prevalence Section
**Required Visualizations:**
- **Bar Chart:** Disease prevalence percentages
- **Statistics:** Total patients analyzed, high-risk count, critical alerts

### 5.3 What-If Analysis Module (MANDATORY)

#### 5.3.1 Interactive Controls
**Required Sliders:** (All with real-time update)

| Parameter | Range | Step | Default | Units |
|-----------|-------|------|---------|-------|
| Age | 20-90 | 1 | 68 | years |
| Creatinine | 0.5-4.0 | 0.1 | 1.8 | mg/dL |
| Systolic BP | 90-180 | 5 | 140 | mmHg |
| Glucose | 70-300 | 10 | 120 | mg/dL |
| Heart Rate | 60-120 | 5 | 85 | bpm |
| Temperature | 95-105 | 0.1 | 98.6 | °F |

#### 5.3.2 What-If Outputs
**Required Displays:**
- **Risk Gauge:** Visual risk meter per disease (0-100%)
- **Risk Change Indicator:** Delta from baseline (↑↓%)
- **Multi-Disease Panel:** All 4 diseases updated simultaneously
- **Optimal Values:** Show parameter values that minimize risk
- **Clinical Recommendations:** Auto-generate based on What-If results

#### 5.3.3 Risk Prediction Function
```python
def simulate_risk_prediction(patient_data, disease):
    """
    Calculates risk score based on clinical logic:
    - Kidney failure: creatinine + age + BP
    - Cardiovascular: age + BP + glucose
    - Sepsis: temperature + WBC + heart rate
    - Mortality: combined risk from all factors
    
    Returns: risk_score (0.0-1.0)
    """
```

---

## 6. EVALUATION METRICS

### 6.1 Required Metrics Per Disease Model

| Metric | Formula | Purpose | Acceptable Threshold |
|--------|---------|---------|---------------------|
| **AUC-ROC** | Area Under ROC Curve | Discrimination ability | > 0.70 (good), > 0.80 (excellent) |
| **Accuracy** | (TP + TN) / Total | Overall correctness | > 0.75 |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Balanced performance | > 0.40 (imbalanced data) |
| **Prevalence** | Positive cases / Total | Class distribution | Matches clinical reality |

### 6.2 Classification Metrics (Per Model)
- **True Positives (TP):** Correctly predicted disease
- **True Negatives (TN):** Correctly predicted no disease
- **False Positives (FP):** Incorrectly predicted disease
- **False Negatives (FN):** Missed disease diagnosis
- **Precision:** TP / (TP + FP)
- **Recall (Sensitivity):** TP / (TP + FN)
- **Specificity:** TN / (TN + FP)

### 6.3 Explainability Validation Metrics
- **SHAP Consistency:** Feature importance rankings align with clinical knowledge
- **LIME Fidelity:** R² > 0.5 for local linear approximation
- **Clinical Plausibility:** Top features match known disease risk factors

### 6.4 Risk Stratification Performance

| Risk Category | Threshold | Expected Outcome |
|---------------|-----------|------------------|
| **CRITICAL** | 70%+ in multiple diseases | Immediate ICU intervention |
| **HIGH** | 70%+ in single disease | Specialist consultation within 24h |
| **MODERATE** | 40-70% in any disease | Enhanced monitoring |
| **LOW** | <40% in all diseases | Standard care protocols |

---

## 7. ARCHITECTURAL COMPONENTS

### 7.1 System Architecture (4 Modules)

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-DISEASE EXPLAINABLE AI SYSTEM            │
└─────────────────────────────────────────────────────────────────┘

MODULE 1: DISEASE PREDICTION
┌──────────────────────────────────────────────────────────────────┐
│ Input: Patient Features (13 features)                             │
│ ├── Data Loading (MIMIC-III / Synthetic)                          │
│ ├── Preprocessing (encoding, scaling, imputation)                 │
│ ├── Feature Engineering (risk indicators)                         │
│ └── Target Creation (ICD-9 → Binary labels)                       │
│                                                                    │
│ Processing: ML Model Training                                     │
│ ├── Random Forest × 4 (one per disease)                           │
│ ├── Train-Test Split (70/30, stratified)                          │
│ ├── StandardScaler feature normalization                          │
│ └── Class-balanced training                                       │
│                                                                    │
│ Output: Trained Models + Performance Metrics                      │
│ └── {disease: {model, scaler, AUC, accuracy, F1}}                 │
└──────────────────────────────────────────────────────────────────┘

MODULE 2: EXPLAINABILITY (SHAP + LIME)
┌──────────────────────────────────────────────────────────────────┐
│ Input: Trained Models + Test Data                                │
│                                                                    │
│ SHAP Component:                                                   │
│ ├── TreeExplainer initialization per model                        │
│ ├── Compute SHAP values for all test patients                     │
│ ├── Global feature importance (mean |SHAP|)                       │
│ └── Local per-patient contributions                               │
│                                                                    │
│ LIME Component:                                                   │
│ ├── LimeTabularExplainer initialization per model                 │
│ ├── Sample training data (100 patients)                           │
│ ├── Per-patient local explanations                                │
│ └── Feature importance with fidelity scores                       │
│                                                                    │
│ Clinical Translation:                                             │
│ ├── Map SHAP values → Risk increase/decrease language             │
│ ├── Rank features by importance                                   │
│ └── Generate human-readable explanations                          │
│                                                                    │
│ Output: Explanations per Disease per Patient                      │
│ └── {disease: {shap_values, lime_exp, top_features}}              │
└──────────────────────────────────────────────────────────────────┘

MODULE 3: VISUALIZATION & INTERACTION (Dashboard)
┌──────────────────────────────────────────────────────────────────┐
│ Technology: Dash + Plotly                                         │
│                                                                    │
│ Components:                                                       │
│ ├── Section 1: Model Performance Charts                           │
│ │   ├── AUC bar chart (4 diseases)                                │
│ │   └── Metrics table (AUC, Acc, F1, Prev)                        │
│ │                                                                  │
│ ├── Section 2: Risk Distribution                                  │
│ │   ├── Pie chart (CRITICAL/HIGH/MODERATE/LOW)                    │
│ │   └── Bar chart with counts                                     │
│ │                                                                  │
│ ├── Section 3: Patient Analysis                                   │
│ │   ├── Patient cards (risk scores + recommendations)             │
│ │   └── Filter by risk category                                   │
│ │                                                                  │
│ └── Section 4: What-If Analysis                                   │
│     ├── Interactive sliders (6 parameters)                        │
│     ├── Real-time risk calculation                                │
│     ├── Multi-disease risk panel                                  │
│     └── Optimal parameter suggestions                             │
│                                                                    │
│ Interface: http://127.0.0.1:8051                                  │
│ Update: Real-time callback-driven                                 │
└──────────────────────────────────────────────────────────────────┘

MODULE 4: PERSONALIZED HEALTH ASSISTANT
┌──────────────────────────────────────────────────────────────────┐
│ Input: Patient Reports + Risk Scores                              │
│                                                                    │
│ Autonomous Decision Logic:                                        │
│ ├── Risk Stratification                                           │
│ │   ├── CRITICAL: ≥2 diseases at HIGH risk (70%+)                 │
│ │   ├── HIGH: ≥1 disease at HIGH risk                             │
│ │   ├── MODERATE: Any disease at 40-70%                           │
│ │   └── LOW: All diseases <40%                                    │
│ │                                                                  │
│ ├── Clinical Protocol Selection                                   │
│ │   ├── Sepsis: Antibiotics + blood cultures + ICU                │
│ │   ├── Kidney: Nephrology consult + electrolytes + fluids        │
│ │   ├── Cardio: Cardiology eval + ECG + cardiac markers           │
│ │   └── Mortality: Palliative care + family meeting               │
│ │                                                                  │
│ └── Autonomous Alerts                                             │
│     ├── Critical patient notifications                            │
│     ├── High-risk intervention triggers                           │
│     └── Population health insights                                │
│                                                                    │
│ Output: Clinical Recommendations + Alerts                         │
│ └── {patient_id: [recommendations], alert_level}                  │
└──────────────────────────────────────────────────────────────────┘
```

### 7.2 Data Flow Diagram

```
┌─────────────┐
│ MIMIC-III   │
│ Database    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│ Data Preprocessing       │
│ • Feature extraction     │
│ • Missing value handling │
│ • Target creation        │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐      ┌──────────────────┐
│ ML Training Pipeline     │      │ Test Dataset     │
│ • Random Forest × 4      │────▶│ • X_test         │
│ • StandardScaler         │      │ • y_test         │
│ • Class balancing        │      └────────┬─────────┘
└──────┬──────────────────┘               │
       │                                   │
       ▼                                   │
┌─────────────────────────┐               │
│ Trained Models           │               │
│ • Sepsis model          │◀──────────────┘
│ • Kidney model          │
│ • Cardiovascular model  │
│ • Mortality model       │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│ Explainability Engine    │
│ • SHAP TreeExplainer    │
│ • LIME TabularExplainer │
│ • Clinical translation   │
└──────┬──────────────────┘
       │
       ├─────────────┬───────────────┐
       ▼             ▼               ▼
┌──────────┐  ┌──────────┐  ┌──────────────────┐
│Dashboard │  │ Patient  │  │ Health Assistant │
│(Web UI)  │  │ Reports  │  │ (Decision Logic) │
└──────────┘  └──────────┘  └──────────────────┘
```

### 7.3 File Structure Specification

```
project_root/
├── final_demo.py                          # MAIN: Complete 4-module system
├── enhanced_dashboard_with_whatif.py      # MAIN: Web dashboard with What-If
├── complete_system_with_whatif.py         # Enhanced system with What-If
├── explainable_dashboard.py               # Basic dashboard
│
├── multi_disease_explainable_system.py    # Core system class
├── mimic_preprocessing/
│   └── explainable_medical_diagnosis.py   # Core AI class
│
├── requirements.txt                       # Python dependencies
├── complete_multi_disease_results.json    # OUTPUT: Analysis results
│
└── Documentation/
    ├── README.md                          # Complete documentation
    ├── USAGE_GUIDE.md                     # Usage instructions
    ├── PROJECT_SUMMARY.md                 # High-level overview
    ├── OBJECTIVES_VERIFICATION.md         # Verification document
    └── TECHNICAL_SPECIFICATION.md         # This document
```

### 7.4 Class Structure Specification

#### 7.4.1 Core Classes

**Class: `MultiDiseaseExplainableSystem`**
```python
class MultiDiseaseExplainableSystem:
    """Main system orchestrator"""
    
    # Required Attributes
    self.models: Dict[str, RandomForestClassifier]
    self.scalers: Dict[str, StandardScaler]
    self.explainers: Dict[str, Dict[str, Any]]  # {shap, lime}
    self.feature_names: List[str]
    self.diseases: Dict[str, Dict[str, Any]]
    
    # Required Methods
    def load_mimic_data() -> Dict[str, pd.DataFrame]
    def create_disease_targets() -> pd.DataFrame
    def engineer_features() -> pd.DataFrame
    def train_disease_models() -> Dict[str, Dict]
    def setup_explainability() -> Dict[str, Dict]
    def generate_patient_report() -> Dict
    def run_autonomous_assistant() -> List[str]
```

**Class: `WhatIfAnalyzer`**
```python
class WhatIfAnalyzer:
    """What-If scenario analysis engine"""
    
    # Required Attributes
    self.models: Dict[str, RandomForestClassifier]
    self.scalers: Dict[str, StandardScaler]
    self.feature_names: List[str]
    
    # Required Methods
    def analyze_patient_variations() -> Dict[str, Dict]
    def simulate_risk_prediction() -> float
    def calculate_optimal_parameters() -> Dict[str, float]
    def generate_what_if_recommendations() -> List[str]
```

---

## 8. IMPLEMENTATION REQUIREMENTS

### 8.1 Programming Language & Dependencies
- **Language:** Python 3.8+
- **Core Libraries:**
  - `scikit-learn >= 1.0.0` - ML models
  - `pandas >= 1.3.0` - Data manipulation
  - `numpy >= 1.21.0` - Numerical computing
  - `shap >= 0.42.0` - SHAP explanations
  - `lime >= 0.2.0.1` - LIME explanations
  - `dash >= 2.0.0` - Web dashboard
  - `plotly >= 5.0.0` - Visualizations
  - `matplotlib >= 3.5.0` - Static plots
  - `seaborn >= 0.11.0` - Statistical visualizations

### 8.2 Performance Requirements
- **Model Training:** < 5 minutes for 4 models on 1000 patients
- **SHAP Computation:** < 2 minutes for all models
- **Dashboard Load Time:** < 3 seconds
- **What-If Update:** Real-time (<500ms per parameter change)

### 8.3 Scalability Requirements
- **Minimum Dataset:** 100 patients
- **Target Dataset:** 1,000-10,000 patients
- **Maximum Features:** 20 clinical features
- **Concurrent Dashboard Users:** Support for 10+ simultaneous users

---

## 9. COMPLIANCE & VALIDATION

### 9.1 Clinical Validation Requirements
- All explanations must be medically plausible
- Risk thresholds aligned with clinical guidelines
- Disease prevalence matches epidemiological data
- Feature importance aligns with known risk factors

### 9.2 Regulatory Considerations
- System intended for research/educational use only
- NOT FDA-approved for clinical decision-making
- Requires clinical validation before deployment
- Human oversight required for all predictions

### 9.3 Ethical Requirements
- Transparent prediction explanations mandatory
- Bias detection through SHAP/LIME analysis
- No discrimination based on protected attributes
- Audit trail for all predictions and explanations

---

## 10. SUCCESS CRITERIA

### 10.1 Technical Success Criteria
- ✅ 4 disease models trained with AUC > 0.50
- ✅ SHAP explanations generated for all models
- ✅ LIME explanations functional for individual patients
- ✅ Dashboard accessible via web browser
- ✅ What-If analysis updates in real-time

### 10.2 Clinical Success Criteria
- ✅ Explanations align with clinical knowledge
- ✅ Risk stratification produces actionable categories
- ✅ Recommendations follow evidence-based protocols
- ✅ Clinicians can understand and trust predictions

### 10.3 Usability Success Criteria
- ✅ Non-technical users can operate dashboard
- ✅ What-If analysis produces clear guidance
- ✅ Patient reports readable by healthcare professionals
- ✅ System completes analysis in < 5 minutes

---

## APPENDIX A: RISK CALCULATION FORMULAS

### Sepsis Risk Formula
```
sepsis_risk = 
    0.3 × (temperature > 100.4°F) +
    0.3 × (WBC > 12 K/µL) +
    0.2 × (systolic_bp < 90 mmHg) +
    0.2 × (age > 70 years)
```

### Kidney Failure Risk Formula
```
kidney_risk = 
    0.4 × (creatinine > 1.5 mg/dL) +
    0.3 × (age > 70 years) +
    0.3 × (systolic_bp > 140 mmHg)
```

### Cardiovascular Risk Formula
```
cardiovascular_risk = 
    0.3 × (age > 65 years) +
    0.3 × (glucose > 180 mg/dL) +
    0.4 × (systolic_bp > 140 mmHg)
```

### Mortality Risk Formula
```
mortality_risk = 
    0.4 × sepsis_risk +
    0.3 × kidney_risk +
    0.3 × cardiovascular_risk +
    0.2 × (age > 75 years)
```

---

## APPENDIX B: CLINICAL DECISION PROTOCOLS

### Protocol: Sepsis Management
- **Trigger:** Risk > 70%
- **Actions:**
  1. Immediate blood cultures
  2. Broad-spectrum antibiotics within 1 hour
  3. Lactate measurement
  4. IV fluid resuscitation
  5. ICU consultation

### Protocol: Kidney Failure Management
- **Trigger:** Risk > 70%
- **Actions:**
  1. Nephrology consultation
  2. Monitor: Creatinine, BUN, electrolytes
  3. Assess fluid balance
  4. Review nephrotoxic medications
  5. Consider dialysis if AKI stage 3

### Protocol: Cardiovascular Management
- **Trigger:** Risk > 70%
- **Actions:**
  1. Cardiology evaluation
  2. 12-lead ECG
  3. Cardiac biomarkers (troponin, BNP)
  4. Continuous cardiac monitoring
  5. Optimize blood pressure and heart rate

### Protocol: Mortality Risk Management
- **Trigger:** Risk > 70%
- **Actions:**
  1. Goals of care discussion
  2. Palliative care consultation
  3. Family meeting
  4. Advance directive review
  5. Symptom management optimization

---

**END OF TECHNICAL SPECIFICATION**

*This specification is IMMUTABLE and serves as the authoritative reference for system implementation, validation, and compliance.*
