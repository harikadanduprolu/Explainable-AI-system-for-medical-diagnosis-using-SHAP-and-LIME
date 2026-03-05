# MODULAR ARCHITECTURE - QUICK REFERENCE GUIDE

## Module Overview (One-Page Summary)

```
┌────────────────────────────────────────────────────────────────────────────┐
│           EXPLAINABLE MEDICAL AI - MODULAR ARCHITECTURE OVERVIEW           │
└────────────────────────────────────────────────────────────────────────────┘

                          ┌─────────────────────┐
                          │   DATA LAYER (1)    │ Loads + Preprocesses Data
                          │                     │ Output: ProcessedDataset
                          └──────────┬──────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
         ┌──────────▼─────────────┐      ┌───────────▼──────────┐
         │  MODEL SERVICES (2)    │      │   XAI ENGINE (3)     │
         │  - Sepsis              │      │   - SHAP Explainer   │
         │  - Kidney Failure      │      │   - LIME Explainer   │
         │  - Cardiovascular      │      │   - Counterfactuals  │
         │  - Mortality           │      │   - Clinical Trans.  │
         │  Output: Predictions   │      │   Output: Explanations
         └──────────┬─────────────┘      └───────────┬──────────┘
                    │                                 │
                    └────────────────┬────────────────┘
                                     │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
    ┌────▼─────────┐         ┌──────▼───────┐         ┌─────────▼────┐
    │ WHAT-IF      │         │ DASHBOARD    │         │ EVALUATION   │
    │ ENGINE (4)   │         │ SERVER (5)   │         │ MODULE (6)   │
    │              │         │              │         │              │
    │ - Scenarios  │         │ - Charts     │         │ - Metrics    │
    │ - Impact     │         │ - Real-time  │         │ - Validation │
    │   Calc       │         │   What-If    │         │ - Reports    │
    │ - Optimize   │         │ - Patient    │         │              │
    │              │         │   Cards      │         │              │
    └──────────────┘         └──────────────┘         └──────────────┘
           │                        │                        │
           │                        ▼                        │
           │              http://127.0.0.1:8051              │
           │                                                 │
           └─────────────────────────┬──────────────────────┘
                                    ▼
                        ┌──────────────────────┐
                        │  CLINICAL INSIGHTS   │
                        │  & DECISIONS         │
                        └──────────────────────┘
```

---

## Module Details

### 1️⃣ DATA LAYER
**File Location:** `src/1_data_layer/`

| Component | Responsibility |
|-----------|-----------------|
| `base_data_source.py` | Abstract interface for data loading |
| `mimic_data_source.py` | Load MIMIC-III CSV files |
| `synthetic_data_source.py` | Generate synthetic clinical data |
| `preprocessor.py` | Handle missing values, encode, scale |
| `feature_engineering.py` | Extract 13 clinical features |
| `data_models.py` | Type definitions (Pydantic) |

**Input:** Raw patient data (CSV files)  
**Output:** `ProcessedDataset` {X_train, X_test, y_train, y_test, feature_names, scaler}  
**Key Method:** `preprocess(raw_data) → ProcessedDataset`

---

### 2️⃣ MODEL SERVICES
**File Location:** `src/2_model_services/`

| Component | Responsibility |
|-----------|-----------------|
| `base_service.py` | Abstract model service interface |
| `sepsis_service.py` | Sepsis prediction model |
| `kidney_failure_service.py` | Kidney failure prediction |
| `cardiovascular_service.py` | Cardiovascular events prediction |
| `mortality_service.py` | Mortality risk prediction |
| `model_registry.py` | Manage all 4 services |
| `training_pipeline.py` | Train all models orchestration |

**Input:** `ProcessedDataset`  
**Output:** `ModelPrediction` {disease, risk_score, risk_category, confidence}  
**Key Methods:**  
- `train(data) → Dict[metrics]`
- `predict(X) → np.ndarray`
- `evaluate(X, y) → Dict[metrics]`

---

### 3️⃣ XAI ENGINE
**File Location:** `src/3_xai_engine/`

| Component | Responsibility |
|-----------|-----------------|
| `base_explainer.py` | Abstract explainer interface |
| `shap_explainer.py` | Global feature importance (SHAP) |
| `lime_explainer.py` | Local patient explanations (LIME) |
| `counterfactual_explainer.py` | Generate counterfactuals |
| `explanation_aggregator.py` | Combine multiple explanations |
| `clinical_translator.py` | Convert to clinical language |
| `xai_models.py` | Explanation data types |

**Input:** Trained models + test instances  
**Output:** `LocalExplanation` {feature_contributions, clinical_summary}  
**Key Methods:**  
- `explain_global(X) → GlobalExplanation`
- `explain_local(instance) → LocalExplanation`
- `generate_counterfactuals(instance) → List[Dict]`
- `translate_to_clinical(explanation) → str`

---

### 4️⃣ WHAT-IF ENGINE
**File Location:** `src/4_whatif_engine/`

| Component | Responsibility |
|-----------|-----------------|
| `whatif_analyzer.py` | Core what-if analysis |
| `scenario_generator.py` | Generate clinical scenarios |
| `impact_calculator.py` | Quantify treatment impact |
| `optimization_solver.py` | Find optimal parameters |
| `whatif_models.py` | Scenario data types |

**Input:** Patient features + ModelRegistry  
**Output:** `WhatIfAnalysis` {parameter_ranges, risk_changes, recommendations}  
**Key Methods:**  
- `analyze_feature_variation(patient, feature, values)`
- `optimize_parameters(patient, target_risk)`
- `simulate_intervention(patient, intervention)`

---

### 5️⃣ DASHBOARD SERVER
**File Location:** `src/5_dashboard_server/`

| Component | Responsibility |
|-----------|-----------------|
| `app.py` | Main Dash application |
| `callbacks.py` | Interactive callbacks |
| `layouts/model_performance.py` | AUC/accuracy charts |
| `layouts/risk_distribution.py` | Risk stratification pie chart |
| `layouts/patient_analysis.py` | Individual patient cards |
| `layouts/whatif_panel.py` | Parameter sliders + analysis |
| `components/charts.py` | Reusable chart components |
| `components/cards.py` | Patient card components |

**Port:** 8051  
**URL:** `http://127.0.0.1:8051`  
**Key Features:**  
- 4 visualization sections
- 6 interactive What-If sliders
- Real-time updates
- Patient filtering

---

### 6️⃣ EVALUATION MODULE
**File Location:** `src/6_evaluation_module/`

| Component | Responsibility |
|-----------|-----------------|
| `model_evaluator.py` | Compute ML metrics |
| `xai_evaluator.py` | Evaluate explanation quality |
| `clinical_validator.py` | Validate clinical relevance |
| `metrics.py` | Metric implementations |
| `report_generator.py` | Generate evaluation reports |
| `evaluation_models.py` | Metric data types |

**Input:** Models + test data + explanations  
**Output:** `EvaluationReport` {metrics, validation, recommendations}  
**Key Methods:**  
- `compute_metrics(y_true, y_pred) → Dict`
- `evaluate_shap_stability(model, X) → float`
- `validate_risk_stratification(predictions, outcomes) → Dict`
- `generate_report() → str`

---

## Interface Summary

### Data Flow Types

```python
# 1. Raw Data → Processed Data
ProcessedDataset {
    X_train: pd.DataFrame (n_train, 13)
    X_test: pd.DataFrame (n_test, 13)
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    scaler: StandardScaler
}

# 2. Patient Data → Prediction
ModelPrediction {
    disease: str
    risk_score: float (0-1)
    risk_category: str (LOW/MODERATE/HIGH/CRITICAL)
    confidence: float
}

# 3. Prediction → Explanation
LocalExplanation {
    disease: str
    feature_contributions: Dict[str, float]
    top_features: List[str]
    clinical_summary: str
}

# 4. Parameters → What-If Results
WhatIfAnalysis {
    parameter: str
    original_value: float
    test_values: np.ndarray
    risks_per_disease: Dict[str, np.ndarray]
    optimal_value: float
    recommendations: List[str]
}

# 5. Metrics → Evaluation
EvaluationMetrics {
    auc: float
    accuracy: float
    f1_score: float
    precision: float
    recall: float
}
```

---

## File Mapping: Existing → New

| Existing File | Maps To | New Location |
|---|---|---|
| `final_demo.py` | Training script | `scripts/train_models.py` |
| `enhanced_dashboard_with_whatif.py` | Dashboard app | `src/5_dashboard_server/app.py` |
| `complete_system_with_whatif.py` | What-If logic | `src/4_whatif_engine/whatif_analyzer.py` |
| `multi_disease_explainable_system.py` | Model registry | `src/2_model_services/model_registry.py` |
| `mimic_preprocessing/explainable_medical_diagnosis.py` | Preprocessing | `src/1_data_layer/preprocessor.py` |
| `mimic_preprocessing/create_extended_mimic_dataset.py` | Feature eng. | `src/1_data_layer/feature_engineering.py` |

---

## Usage Examples

### Example 1: Train All Models
```python
from src.core.config import load_config
from src.data_layer import MIMICDataSource, Preprocessor
from src.model_services import ModelRegistry

# Load and preprocess data
data_source = MIMICDataSource(config['mimic_path'])
raw_data = data_source.load_data()
preprocessor = Preprocessor()
dataset = preprocessor.preprocess(raw_data)

# Train all models
registry = ModelRegistry()
registry.train_all(dataset)
registry.save_models('models/')
```

### Example 2: Get Patient Predictions + Explanations
```python
from src.model_services import ModelRegistry
from src.xai_engine import SHAPExplainer, LIMEExplainer

# Load trained models
registry = ModelRegistry.load('models/')

# Get predictions
predictions = registry.predict_all(patient_features)

# Get explanations
shap_exp = SHAPExplainer(registry.models['kidney_failure'], X_train)
lime_exp = LIMEExplainer(registry.models['kidney_failure'], X_train)

global_exp = shap_exp.explain_global(X_test)
local_exp = lime_exp.explain_local(patient_instance)
```

### Example 3: What-If Analysis
```python
from src.whatif_engine import WhatIfAnalyzer

analyzer = WhatIfAnalyzer(registry)

# Analyze what happens if we change creatinine
result = analyzer.analyze_feature_variation(
    patient=patient_features,
    feature='creatinine',
    values=np.linspace(0.5, 4.0, 20)
)

# Find optimal parameters
optimal = analyzer.optimize_parameters(
    patient=patient_features,
    target_risk=0.3
)
```

### Example 4: Launch Dashboard
```python
from src.dashboard_server import DashboardApp

app = DashboardApp(
    data_layer=preprocessor,
    models=registry,
    xai=xai_engine,
    whatif=analyzer
)

app.run(port=8051, debug=False)
# Opens: http://127.0.0.1:8051
```

### Example 5: Evaluate Models
```python
from src.evaluation_module import ModelEvaluator, XAIEvaluator

# Evaluate model performance
evaluator = ModelEvaluator()
metrics = evaluator.compute_metrics(y_test, y_pred)

# Evaluate explanation quality
xai_eval = XAIEvaluator()
shap_stability = xai_eval.evaluate_shap_stability(model, X_test)
lime_fidelity = xai_eval.evaluate_lime_fidelity(model, explanation)
```

---

## Dependency Graph

```
Layer 1: Data Layer (Independent)
    ↓
Layer 2: Model Services (Depends on Data Layer)
    ↓
Layer 3: XAI Engine (Depends on Model Services + Data Layer)
         What-If Engine (Depends on Model Services + Data Layer)
         Evaluation Module (Depends on Model Services + Data Layer)
    ↓
Layer 4: Dashboard Server (Depends on all above)
    ↓
Layer 5: Scripts/Orchestration (Depends on all above)
```

---

## Testing Strategy

### Unit Tests (Per Module)
- `test_data_layer.py` - Test preprocessing, feature engineering
- `test_model_services.py` - Test model training/prediction
- `test_xai_engine.py` - Test SHAP/LIME explanations
- `test_whatif_engine.py` - Test scenario analysis
- `test_evaluation.py` - Test metrics computation

### Integration Tests
- `test_end_to_end.py` - Full pipeline
- `test_dashboard.py` - Dashboard callbacks

### Sample Data Fixtures
- `fixtures/sample_data.py` - Synthetic test data
- `fixtures/mock_models.py` - Mock model objects

---

## Deployment Scenarios

### Scenario 1: Monolithic (Current)
- Single process runs everything
- Simple deployment
- Limited scalability

### Scenario 2: Microservices
- Data Layer (separate service)
- Model Services (REST API per disease)
- XAI Engine (separate service)
- Dashboard (separate service)
- Evaluation (separate service)

### Scenario 3: Containerized (Docker)
```dockerfile
# Dockerfile for Data Layer
FROM python:3.9
COPY src/1_data_layer /app
RUN pip install -r requirements.txt
CMD ["python", "data_service.py"]
```

---

## Configuration Files

### `configs/disease_configs.yaml`
```yaml
sepsis:
  icd9_codes: ['038', '995.91', '995.92']
  prevalence: 0.098
  model_type: 'random_forest'

kidney_failure:
  icd9_codes: ['584', '585', '586']
  prevalence: 0.303
  model_type: 'xgboost'

# ... more diseases
```

### `configs/model_configs.yaml`
```yaml
random_forest:
  n_estimators: 100
  max_depth: 10
  class_weight: 'balanced'

xgboost:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1

# ... more models
```

---

## Performance Considerations

| Operation | Estimated Time |
|-----------|-----------------|
| Data preprocessing (1000 patients) | 30-60 seconds |
| Train 4 models | 2-3 minutes |
| SHAP explanations (100 patients) | 1-2 minutes |
| Dashboard load | < 3 seconds |
| What-If update (parameter change) | < 500ms |

---

## Migration Checklist

- [ ] Create folder structure
- [ ] Extract Data Layer
- [ ] Extract Model Services
- [ ] Extract XAI Engine
- [ ] Extract What-If Engine
- [ ] Refactor Dashboard
- [ ] Implement Evaluation Module
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Create documentation
- [ ] Update requirements.txt
- [ ] Tag v2.0 release

---

**END OF QUICK REFERENCE GUIDE**
