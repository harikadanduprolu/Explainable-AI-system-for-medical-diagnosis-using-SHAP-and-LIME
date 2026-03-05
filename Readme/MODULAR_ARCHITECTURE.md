# MODULAR ARCHITECTURE DESIGN
# Explainable Medical AI System - Refactoring Plan

## Overview
This document provides a complete modular refactoring plan for the existing codebase, breaking it into 6 independent layers with clear interfaces and responsibilities.

---

## 1. PROPOSED FOLDER STRUCTURE

```
medical_ai_system/
│
├── src/
│   ├── __init__.py
│   │
│   ├── 1_data_layer/
│   │   ├── __init__.py
│   │   ├── base_data_source.py          # Abstract base class
│   │   ├── mimic_data_source.py          # MIMIC-III data loader
│   │   ├── synthetic_data_source.py      # Synthetic data generator
│   │   ├── preprocessor.py               # Feature engineering & preprocessing
│   │   ├── feature_engineering.py        # Clinical feature engineering
│   │   └── data_models.py                # Pydantic models for type safety
│   │
│   ├── 2_model_services/
│   │   ├── __init__.py
│   │   ├── base_service.py               # Abstract model service
│   │   ├── sepsis_service.py             # Sepsis prediction service
│   │   ├── kidney_failure_service.py     # Kidney failure service
│   │   ├── cardiovascular_service.py     # Cardiovascular service
│   │   ├── mortality_service.py          # Mortality prediction service
│   │   ├── model_registry.py             # Registry for all models
│   │   ├── model_config.py               # Model configurations
│   │   └── training_pipeline.py          # Training orchestration
│   │
│   ├── 3_xai_engine/
│   │   ├── __init__.py
│   │   ├── base_explainer.py             # Abstract explainer
│   │   ├── shap_explainer.py             # SHAP implementation
│   │   ├── lime_explainer.py             # LIME implementation
│   │   ├── counterfactual_explainer.py   # Counterfactual generation
│   │   ├── explanation_aggregator.py     # Combine explanations
│   │   ├── clinical_translator.py        # Translate to clinical language
│   │   └── xai_models.py                 # Data models for explanations
│   │
│   ├── 4_whatif_engine/
│   │   ├── __init__.py
│   │   ├── whatif_analyzer.py            # Core what-if engine
│   │   ├── scenario_generator.py         # Generate scenarios
│   │   ├── impact_calculator.py          # Calculate treatment impact
│   │   ├── optimization_solver.py        # Find optimal parameters
│   │   └── whatif_models.py              # Data models for scenarios
│   │
│   ├── 5_dashboard_server/
│   │   ├── __init__.py
│   │   ├── app.py                        # Main Dash application
│   │   ├── callbacks.py                  # Dash callbacks for interactivity
│   │   ├── layouts/
│   │   │   ├── __init__.py
│   │   │   ├── model_performance.py      # Model performance section
│   │   │   ├── risk_distribution.py      # Risk distribution section
│   │   │   ├── patient_analysis.py       # Patient analysis section
│   │   │   └── whatif_panel.py           # What-If analysis section
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── charts.py                 # Reusable chart components
│   │   │   ├── cards.py                  # Patient card components
│   │   │   └── tables.py                 # Data table components
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── colors.py                 # Color schemes
│   │   │   └── formatting.py             # Data formatting utilities
│   │   └── config.py                     # Dashboard configuration
│   │
│   ├── 6_evaluation_module/
│   │   ├── __init__.py
│   │   ├── model_evaluator.py            # Model performance metrics
│   │   ├── xai_evaluator.py              # XAI quality metrics
│   │   ├── clinical_validator.py         # Clinical validation
│   │   ├── metrics.py                    # Metric definitions
│   │   ├── report_generator.py           # Generate evaluation reports
│   │   └── evaluation_models.py          # Data models for metrics
│   │
│   └── core/
│       ├── __init__.py
│       ├── config.py                     # Global configuration
│       ├── logger.py                     # Logging configuration
│       ├── exceptions.py                 # Custom exceptions
│       ├── constants.py                  # System constants
│       └── types.py                      # Type definitions
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_data_layer.py
│   │   ├── test_model_services.py
│   │   ├── test_xai_engine.py
│   │   ├── test_whatif_engine.py
│   │   └── test_evaluation.py
│   ├── integration/
│   │   ├── test_end_to_end.py
│   │   └── test_dashboard.py
│   └── fixtures/
│       ├── sample_data.py
│       └── mock_models.py
│
├── configs/
│   ├── disease_configs.yaml              # Disease-specific settings
│   ├── model_configs.yaml                # ML model configurations
│   ├── xai_configs.yaml                  # XAI settings
│   └── dashboard_config.yaml             # Dashboard settings
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_explainability_analysis.ipynb
│   └── 04_whatif_scenarios.ipynb
│
├── scripts/
│   ├── train_models.py                   # Training orchestration script
│   ├── run_dashboard.py                  # Dashboard launcher
│   ├── run_evaluation.py                 # Evaluation script
│   └── generate_reports.py               # Report generation script
│
├── requirements.txt
├── setup.py
├── README.md
└── .env.example
```

---

## 2. MODULE RESPONSIBILITIES

### 2.1 DATA LAYER (`src/1_data_layer/`)

**Purpose:** Abstraction for all data operations - loading, preprocessing, feature engineering

**Key Components:**

```python
# base_data_source.py
class BaseDataSource(ABC):
    """Abstract base for data sources"""
    
    @abstractmethod
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load raw data tables"""
        pass
    
    @abstractmethod
    def get_patient_cohort(self, filters: Dict) -> pd.DataFrame:
        """Get filtered patient cohort"""
        pass

# mimic_data_source.py
class MIMICDataSource(BaseDataSource):
    """Load from MIMIC-III database"""
    - load_admissions()
    - load_patients()
    - load_diagnoses()
    - load_vitals()
    - load_labs()

# preprocessor.py
class Preprocessor:
    """Data preprocessing pipeline"""
    - handle_missing_values()
    - encode_categorical()
    - normalize_numerical()
    - create_targets()
    - split_train_test()
    - create_temporal_features()

# feature_engineering.py
class ClinicalFeatureEngineer:
    """Domain-specific feature engineering"""
    - calculate_age()
    - engineer_vital_signs_features()
    - engineer_lab_features()
    - create_risk_indicators()
    - create_interaction_features()
```

**Responsibilities:**
- ✅ Load data from multiple sources (MIMIC-III, synthetic, CSV)
- ✅ Data validation and quality checks
- ✅ Handle missing values and outliers
- ✅ Feature engineering and selection
- ✅ Data normalization/scaling
- ✅ Train-test splitting
- ✅ Cache preprocessed data

**Outputs to Consumers:**
```python
# Output Interface
class ProcessedDataset:
    X_train: pd.DataFrame       # Shape (n_train, 13)
    X_test: pd.DataFrame        # Shape (n_test, 13)
    y_train: pd.Series          # Binary labels
    y_test: pd.Series           # Binary labels
    feature_names: List[str]    # 13 feature names
    scaler: StandardScaler      # For scaling new data
```

---

### 2.2 MODEL SERVICES (`src/2_model_services/`)

**Purpose:** Independent service per disease with consistent interface

**Key Components:**

```python
# base_service.py
class BaseModelService(ABC):
    """Abstract model service template"""
    
    def __init__(self, disease_name: str):
        self.disease_name = disease_name
        self.model = None
        self.metadata = {}
    
    @abstractmethod
    def train(self, data: ProcessedDataset) -> Dict:
        """Train model and return metrics"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        pass
    
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate on test set"""
        pass
    
    def save_model(self, path: str) -> None:
        """Persist model"""
        pass
    
    def load_model(self, path: str) -> None:
        """Load model from disk"""
        pass

# sepsis_service.py
class SepsisService(BaseModelService):
    """Sepsis prediction service"""
    
    def __init__(self):
        super().__init__("sepsis")
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, 
            class_weight='balanced'
        )
    
    def train(self, data: ProcessedDataset) -> Dict:
        # Train sepsis-specific model
        # Returns {auc, accuracy, f1, precision, recall}
        pass

# kidney_failure_service.py
class KidneyFailureService(BaseModelService):
    """Kidney failure prediction service"""
    
    def __init__(self):
        super().__init__("kidney_failure")
        self.model = XGBClassifier(n_estimators=100, max_depth=6)

# cardiovascular_service.py
class CardiovascularService(BaseModelService):
    """Cardiovascular events prediction service"""

# mortality_service.py
class MortalityService(BaseModelService):
    """In-hospital mortality prediction service"""

# model_registry.py
class ModelRegistry:
    """Central registry for all disease models"""
    
    def __init__(self):
        self.services = {
            'sepsis': SepsisService(),
            'kidney_failure': KidneyFailureService(),
            'cardiovascular': CardiovascularService(),
            'mortality': MortalityService()
        }
    
    def get_service(self, disease: str) -> BaseModelService:
        """Retrieve service by disease name"""
        pass
    
    def train_all(self, data: ProcessedDataset) -> Dict:
        """Train all models"""
        pass
    
    def predict_all(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from all models"""
        pass
```

**Responsibilities:**
- ✅ Train models for individual disease
- ✅ Make predictions
- ✅ Evaluate model performance
- ✅ Manage model lifecycle (save/load)
- ✅ Store model metadata
- ✅ Handle hyperparameter tuning

**Input Interface:**
```python
ProcessedDataset {
    X_train, X_test, y_train, y_test,
    feature_names, scaler
}
```

**Output Interface:**
```python
class ModelPrediction:
    disease: str
    risk_score: float           # 0-1
    risk_category: str          # LOW, MODERATE, HIGH, CRITICAL
    confidence: float
    prediction_metadata: Dict
```

---

### 2.3 XAI ENGINE (`src/3_xai_engine/`)

**Purpose:** Generate explanations using SHAP, LIME, and Counterfactuals

**Key Components:**

```python
# base_explainer.py
class BaseExplainer(ABC):
    """Abstract explainer interface"""
    
    @abstractmethod
    def explain_global(self, X: pd.DataFrame) -> GlobalExplanation:
        """Generate global feature importance"""
        pass
    
    @abstractmethod
    def explain_local(self, instance: pd.Series) -> LocalExplanation:
        """Explain single prediction"""
        pass

# shap_explainer.py
class SHAPExplainer(BaseExplainer):
    """SHAP-based explanations"""
    
    def __init__(self, model: BaseEstimator, X_background: pd.DataFrame):
        self.explainer = shap.TreeExplainer(model)
        self.X_background = X_background
    
    def explain_global(self, X: pd.DataFrame) -> GlobalExplanation:
        # Compute SHAP values for dataset
        # Return feature importance rankings
        pass
    
    def explain_local(self, instance: pd.Series) -> LocalExplanation:
        # Explain single prediction
        # Return feature contributions (positive/negative)
        pass

# lime_explainer.py
class LIMEExplainer(BaseExplainer):
    """LIME-based local explanations"""
    
    def __init__(self, model: BaseEstimator, X_train: pd.DataFrame):
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns
        )
    
    def explain_local(self, instance: pd.Series) -> LocalExplanation:
        # Local linear approximation
        # Return contributing features
        pass

# counterfactual_explainer.py
class CounterfactualExplainer:
    """Generate counterfactual explanations"""
    
    def generate_counterfactuals(
        self, 
        instance: pd.Series,
        target_risk: float = 0.3
    ) -> List[Dict]:
        """
        Generate counterfactuals: minimal feature changes
        to reach target risk level
        
        Returns list of {feature, change, new_risk}
        """
        pass

# clinical_translator.py
class ClinicalTranslator:
    """Convert XAI outputs to clinical language"""
    
    def translate_shap_contribution(
        self, 
        feature: str, 
        value: float
    ) -> str:
        """
        Input: creatinine, 0.253
        Output: "High creatinine (+0.253 ↗️) strongly increases risk"
        """
        pass
    
    def create_clinical_summary(
        self,
        explanations: LocalExplanation
    ) -> str:
        """Create human-readable clinical report"""
        pass
```

**Responsibilities:**
- ✅ Compute global feature importance (SHAP)
- ✅ Generate local patient-specific explanations (LIME)
- ✅ Create counterfactual explanations
- ✅ Translate technical outputs to clinical language
- ✅ Validate explanation quality
- ✅ Cache explanations for performance

**Input Interface:**
```python
class ExplainerRequest:
    model: BaseEstimator
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    instance: pd.Series  # For local explanation
```

**Output Interface:**
```python
class GlobalExplanation:
    features: List[str]
    importances: List[float]
    positive_direction: List[bool]  # True=increases risk

class LocalExplanation:
    feature_contributions: Dict[str, float]
    top_features: List[str]
    clinical_summary: str
    confidence: float
```

---

### 2.4 WHAT-IF ENGINE (`src/4_whatif_engine/`)

**Purpose:** Analyze how parameter changes affect predictions

**Key Components:**

```python
# whatif_analyzer.py
class WhatIfAnalyzer:
    """What-If analysis engine"""
    
    def __init__(self, models: ModelRegistry):
        self.models = models
        self.feature_ranges = self._define_feature_ranges()
    
    def analyze_feature_variation(
        self,
        patient: pd.Series,
        feature: str,
        values: np.ndarray
    ) -> Dict:
        """
        Vary single feature across range
        Returns: {feature_values, risk_scores_per_disease}
        """
        pass
    
    def optimize_parameters(
        self,
        patient: pd.Series,
        target_risk: float = 0.3
    ) -> Dict:
        """
        Find optimal parameters to reach target risk
        Returns: {optimal_params, predicted_risks}
        """
        pass
    
    def simulate_intervention(
        self,
        patient: pd.Series,
        intervention: Dict  # {feature: new_value}
    ) -> Dict:
        """
        Show impact of clinical intervention
        Returns: {before_risks, after_risks, risk_reduction}
        """
        pass

# scenario_generator.py
class ScenarioGenerator:
    """Generate clinical scenarios for what-if"""
    
    def generate_clinical_scenarios(
        self,
        patient: pd.Series
    ) -> List[Dict]:
        """
        Generate realistic clinical scenarios:
        - Medication adjustment
        - Treatment interventions
        - Natural disease progression
        """
        pass

# impact_calculator.py
class ImpactCalculator:
    """Calculate treatment impact quantitatively"""
    
    def calculate_risk_reduction(
        self,
        before_risks: Dict,
        after_risks: Dict
    ) -> Dict:
        """
        Calculate absolute/relative risk reduction
        Returns: {disease: {abs_reduction, pct_reduction}}
        """
        pass

# optimization_solver.py
class ParameterOptimizer:
    """Find optimal parameter values"""
    
    def find_optimal_params(
        self,
        patient: pd.Series,
        objective_risks: Dict  # {disease: target_risk}
    ) -> Dict:
        """
        Solve: minimize disease risks subject to constraints
        Returns: optimal_params, predicted_risks, feasibility
        """
        pass
```

**Responsibilities:**
- ✅ Vary parameters and track risk changes
- ✅ Simulate clinical interventions
- ✅ Optimize parameters to minimize risk
- ✅ Generate clinical scenarios
- ✅ Calculate treatment impact
- ✅ Suggest evidence-based interventions

**Input Interface:**
```python
class WhatIfRequest:
    patient_data: pd.Series
    parameter: str  # Feature to vary
    min_value: float
    max_value: float
    step_count: int
```

**Output Interface:**
```python
class WhatIfAnalysis:
    parameter: str
    original_value: float
    test_values: np.ndarray
    risks_per_disease: Dict[str, np.ndarray]
    optimal_value: float
    optimal_risks: Dict[str, float]
    clinical_recommendations: List[str]
```

---

### 2.5 DASHBOARD SERVER (`src/5_dashboard_server/`)

**Purpose:** Interactive web interface for clinicians

**Key Components:**

```python
# app.py - Main Dash application
class DashboardApp:
    """Main Dash application"""
    
    def __init__(self, data_layer, models, xai, whatif):
        self.app = dash.Dash(__name__)
        self.data_layer = data_layer
        self.models = models
        self.xai = xai
        self.whatif = whatif
        
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Create multi-section layout"""
        # Section 1: Model Performance
        # Section 2: Risk Distribution
        # Section 3: Patient Analysis
        # Section 4: What-If Analysis
        pass
    
    def setup_callbacks(self):
        """Setup interactive callbacks"""
        pass
    
    def run(self, host='127.0.0.1', port=8051, debug=False):
        self.app.run_server(host=host, port=port, debug=debug)

# callbacks.py - Interactive callbacks
class DashboardCallbacks:
    """Manage interactive callbacks"""
    
    @callback(Output('risk-chart', 'figure'),
              Input('patient-selector', 'value'))
    def update_risk_chart(patient_id):
        """Update risk chart when patient selected"""
        pass
    
    @callback([Output('risk-gauge', 'figure'),
               Output('recommendations', 'children')],
              [Input('age-slider', 'value'),
               Input('creatinine-slider', 'value'),
               ...])
    def update_whatif_analysis(*values):
        """Real-time what-if analysis update"""
        pass

# layouts/model_performance.py
def create_model_performance_section(models):
    """
    Create model performance visualization section
    
    Displays:
    - AUC scores per disease (bar chart)
    - Accuracy, F1, prevalence (metrics table)
    - Model comparison
    """
    pass

# layouts/risk_distribution.py
def create_risk_distribution_section(patient_risks):
    """
    Create risk stratification visualization
    
    Displays:
    - Pie chart (CRITICAL/HIGH/MODERATE/LOW)
    - Risk distribution statistics
    - Patient counts per category
    """
    pass

# layouts/patient_analysis.py
def create_patient_analysis_section(patient_data):
    """
    Create individual patient analysis section
    
    Displays:
    - Patient cards with risk scores
    - Disease-specific predictions
    - SHAP/LIME explanations
    - Clinical recommendations
    """
    pass

# layouts/whatif_panel.py
def create_whatif_section():
    """
    Create interactive what-if analysis panel
    
    Features:
    - 6 parameter sliders (age, creatinine, BP, glucose, HR, temp)
    - Real-time risk recalculation
    - Risk reduction visualization
    - Optimal parameter suggestions
    - Clinical recommendations
    """
    pass

# components/charts.py
class ChartFactory:
    """Reusable chart components"""
    
    @staticmethod
    def create_auc_chart(metrics: Dict) -> go.Figure:
        pass
    
    @staticmethod
    def create_risk_gauge(risk_score: float) -> go.Figure:
        pass
    
    @staticmethod
    def create_feature_importance_chart(explanations) -> go.Figure:
        pass

# components/cards.py
class PatientCardComponent:
    """Patient information cards"""
    
    def __init__(self, patient_id, risks, explanations):
        self.patient_id = patient_id
        self.risks = risks
        self.explanations = explanations
    
    def render(self) -> dbc.Card:
        """Render patient card"""
        pass
```

**Responsibilities:**
- ✅ Serve interactive web interface
- ✅ Display model performance metrics
- ✅ Visualize risk distributions
- ✅ Show patient analysis with explanations
- ✅ Provide real-time what-if analysis
- ✅ Generate downloadable reports
- ✅ Handle user interactions

**Input Interface:**
- Consumes: ModelRegistry, XAIEngine, WhatIfEngine, DataLayer

**Output Interface:**
- HTTP web interface on port 8051
- JSON API endpoints for data retrieval

---

### 2.6 EVALUATION MODULE (`src/6_evaluation_module/`)

**Purpose:** Validate model and XAI quality

**Key Components:**

```python
# model_evaluator.py
class ModelEvaluator:
    """Evaluate ML model performance"""
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict:
        """
        Compute comprehensive metrics:
        - AUC, Accuracy, F1, Precision, Recall
        - Specificity, Sensitivity
        - ROC curve, Precision-Recall curve
        - Confusion matrix
        """
        pass
    
    def cross_validate(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict:
        """Perform k-fold cross-validation"""
        pass

# xai_evaluator.py
class XAIEvaluator:
    """Evaluate XAI quality"""
    
    def evaluate_shap_stability(
        self,
        model: BaseEstimator,
        X: pd.DataFrame
    ) -> float:
        """Check if SHAP values are stable (low variance)"""
        pass
    
    def evaluate_lime_fidelity(
        self,
        model: BaseEstimator,
        explanation
    ) -> float:
        """Check LIME local model R²"""
        pass
    
    def evaluate_explanation_plausibility(
        self,
        explanations: LocalExplanation,
        clinical_knowledge: Dict
    ) -> float:
        """Check if explanations align with medical knowledge"""
        pass

# clinical_validator.py
class ClinicalValidator:
    """Validate clinical relevance and safety"""
    
    def validate_risk_stratification(
        self,
        patient_predictions: Dict,
        clinical_outcomes: pd.Series
    ) -> Dict:
        """
        Check if model risk categories predict outcomes
        - Do HIGH RISK patients have worse outcomes?
        - Are risk categories well-calibrated?
        """
        pass
    
    def check_feature_clinical_relevance(
        self,
        important_features: List[str]
    ) -> Dict:
        """
        Validate that important features are clinically meaningful
        Returns: {feature: clinical_relevance_score}
        """
        pass

# metrics.py
class Metrics:
    """Metric definitions"""
    
    @staticmethod
    def auc_roc(y_true, y_pred_proba):
        pass
    
    @staticmethod
    def calculate_sensitivity(y_true, y_pred):
        pass
    
    @staticmethod
    def calculate_specificity(y_true, y_pred):
        pass

# report_generator.py
class EvaluationReportGenerator:
    """Generate comprehensive evaluation reports"""
    
    def generate_model_report(
        self,
        model_name: str,
        metrics: Dict,
        explanations: Dict
    ) -> str:
        """
        Generate markdown/PDF report with:
        - Model performance summary
        - Feature importance
        - Clinical validation results
        - Recommendations
        """
        pass
    
    def generate_xai_report(
        self,
        model_name: str,
        xai_metrics: Dict
    ) -> str:
        """Generate XAI quality report"""
        pass
```

**Responsibilities:**
- ✅ Compute performance metrics
- ✅ Validate model calibration
- ✅ Cross-validate models
- ✅ Evaluate XAI quality
- ✅ Clinical validation
- ✅ Generate evaluation reports

**Input Interface:**
```python
EvaluationRequest {
    models: ModelRegistry,
    test_data: ProcessedDataset,
    explanations: Dict[str, Explanation]
}
```

**Output Interface:**
```python
EvaluationReport {
    model_metrics: Dict,
    xai_metrics: Dict,
    clinical_validation: Dict,
    summary_string: str,
    report_path: str
}
```

---

## 3. MODULE INTERFACES & DATA FLOW

### 3.1 Interface Contracts (Type Definitions)

```python
# src/core/types.py
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# === Data Layer Output ===
class ProcessedDataset:
    X_train: pd.DataFrame           # (n_train, 13)
    X_test: pd.DataFrame            # (n_test, 13)
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    scaler: StandardScaler
    metadata: Dict[str, Any]

# === Model Services Output ===
class ModelPrediction:
    disease: str
    risk_score: float               # 0-1
    risk_category: str              # LOW, MODERATE, HIGH, CRITICAL
    confidence: float
    probabilities: Dict
    model_metadata: Dict

# === XAI Engine Output ===
class GlobalExplanation:
    disease: str
    features: List[str]
    importances: List[float]        # Sorted descending
    positive_direction: List[bool]
    visualization_data: Dict

class LocalExplanation:
    disease: str
    instance_id: int
    feature_contributions: Dict[str, float]
    top_features: List[str]
    top_contributions: List[float]
    clinical_summary: str
    confidence: float

# === What-If Engine Output ===
class WhatIfAnalysis:
    patient_id: int
    parameter: str
    original_value: float
    test_values: np.ndarray
    risks_per_disease: Dict[str, np.ndarray]
    optimal_value: float
    optimal_risks: Dict[str, float]
    recommendations: List[str]

# === Evaluation Module Output ===
class EvaluationMetrics:
    auc: float
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    specificity: float
    sensitivity: float
    confusion_matrix: np.ndarray
```

### 3.2 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER (1)                            │
│  Loads MIMIC-III → Preprocesses → Feature Engineering           │
│  Output: ProcessedDataset {X_train, X_test, features, scaler}   │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL SERVICES (2)                            │
│  Receives: ProcessedDataset                                     │
│  ├─ SepsisService.train() → ModelPrediction                     │
│  ├─ KidneyFailureService.train() → ModelPrediction              │
│  ├─ CardiovascularService.train() → ModelPrediction             │
│  └─ MortalityService.train() → ModelPrediction                  │
│  Output: ModelRegistry {4 trained models}                       │
└────────────────┬────────────────────────────────────────────────┘
                 │
        ┌────────┴────────┬─────────────┐
        │                 │             │
        ▼                 ▼             ▼
    ┌────────────┐  ┌──────────────┐ ┌──────────────────┐
    │ XAI Engine │  │ What-If      │ │ Dashboard        │
    │    (3)     │  │ Engine (4)   │ │ Server (5)       │
    └────────────┘  └──────────────┘ └──────────────────┘
        │ │                │              │
        │ │                │              │
    SHAP LIME         Scenarios        Web UI
    Global Local      Analysis         Charts
    Explns Explns     Optimization    Reports
        │ │                │              │
        └─┴────────────────┴──────────────┘
                 │
                 ▼
    ┌──────────────────────────────────┐
    │  EVALUATION MODULE (6)           │
    │  Metrics, Validation, Reports    │
    └──────────────────────────────────┘
```

### 3.3 Detailed Data Flow: End-to-End

```
1. USER RUNS: python scripts/train_models.py
   │
   ├─→ DataLayer.load_data()
   │   Output: ProcessedDataset
   │
   ├─→ ModelRegistry.train_all(ProcessedDataset)
   │   ├─→ SepsisService.train()
   │   ├─→ KidneyFailureService.train()
   │   ├─→ CardiovascularService.train()
   │   └─→ MortalityService.train()
   │   Output: Trained models + metrics
   │
   ├─→ Evaluator.evaluate_all()
   │   Output: Evaluation metrics
   │
   └─→ Save models to disk


2. USER RUNS: python scripts/run_dashboard.py
   │
   ├─→ Load trained models from disk
   │
   ├─→ Dashboard starts on http://127.0.0.1:8051
   │
   ├─→ USER SELECTS PATIENT
   │   │
   │   ├─→ ModelRegistry.predict_all(patient_features)
   │   │   Output: {sepsis_risk, kidney_risk, cardio_risk, mortality_risk}
   │   │
   │   ├─→ XAIEngine.explain_patient(patient_id)
   │   │   ├─→ SHAPExplainer.explain_local()
   │   │   │   Output: GlobalExplanation
   │   │   └─→ LIMEExplainer.explain_local()
   │   │       Output: LocalExplanation
   │   │
   │   └─→ Dashboard renders:
   │       - Risk gauge (per disease)
   │       - SHAP feature importance
   │       - LIME contributions
   │       - Clinical recommendations
   │
   ├─→ USER ADJUSTS WHAT-IF PARAMETERS (age slider, etc.)
   │   │
   │   ├─→ WhatIfEngine.simulate_intervention()
   │   │   ├─→ Update patient features
   │   │   ├─→ ModelRegistry.predict_all() with new values
   │   │   └─→ Calculate risk reduction
   │   │
   │   └─→ Dashboard updates in real-time:
   │       - New risk scores
   │       - Risk reduction % 
   │       - Optimal parameter suggestions
   │
   └─→ USER EXPORTS REPORT
       │
       └─→ ReportGenerator.generate_pdf()
           Output: Clinical report with all analyses
```

---

## 4. MAPPING EXISTING FILES TO NEW MODULES

### 4.1 Migration Matrix

| Existing File | New Location | Purpose |
|---|---|---|
| `final_demo.py` | `scripts/train_models.py` | Training orchestration |
| `enhanced_dashboard_with_whatif.py` | `src/5_dashboard_server/app.py` | Dashboard application |
| `complete_system_with_whatif.py` | `src/4_whatif_engine/whatif_analyzer.py` | What-If analysis logic |
| `explainable_dashboard.py` | Deprecated | Replaced by new dashboard |
| `multi_disease_explainable_system.py` | `src/2_model_services/model_registry.py` | Model orchestration |
| `mimic_preprocessing/explainable_medical_diagnosis.py` | `src/1_data_layer/preprocessor.py` | Data preprocessing |
| `mimic_preprocessing/create_extended_mimic_dataset.py` | `src/1_data_layer/feature_engineering.py` | Feature engineering |
| `mimic_preprocessing/create_mimic_notes_bow.py` | `src/1_data_layer/text_feature_extractor.py` | Text feature extraction |
| `demo_explainable_diagnosis.py` | `notebooks/01_data_exploration.ipynb` | Educational notebook |
| `quick_start.py` | `scripts/quickstart.py` | Quick verification script |

### 4.2 Code Extraction Guide

#### Extract Data Layer from multi_disease_explainable_system.py
```python
# Extract these methods → src/1_data_layer/preprocessor.py
- _create_synthetic_data()
- create_disease_targets()
- engineer_features()
- load_mimic_data()

# Extract these methods → src/1_data_layer/mimic_data_source.py
- All MIMIC CSV loading code
```

#### Extract Model Services from multi_disease_explainable_system.py
```python
# Extract these methods → src/2_model_services/base_service.py
- train_disease_models()
- _evaluate_model()

# Create one service per disease:
# - SepsisService (src/2_model_services/sepsis_service.py)
# - KidneyFailureService (src/2_model_services/kidney_failure_service.py)
# - CardiovascularService (src/2_model_services/cardiovascular_service.py)
# - MortalityService (src/2_model_services/mortality_service.py)
```

#### Extract XAI Engine from mimic_preprocessing/explainable_medical_diagnosis.py
```python
# Extract these methods → src/3_xai_engine/shap_explainer.py
- setup_shap_explainer()
- compute_shap_values()

# Extract these methods → src/3_xai_engine/lime_explainer.py
- setup_lime_explainer()
- compute_lime_explanations()

# Create → src/3_xai_engine/clinical_translator.py
- Translate SHAP/LIME to clinical language
```

#### Extract What-If Engine from complete_system_with_whatif.py
```python
# Extract entire WhatIfAnalyzer class
# → src/4_whatif_engine/whatif_analyzer.py

# Extract these methods → src/4_whatif_engine/impact_calculator.py
- analyze_patient_variations()
- calculate_risk_reduction()
```

#### Extract Dashboard from enhanced_dashboard_with_whatif.py
```python
# Refactor existing Dash code → src/5_dashboard_server/
# Split into:
- app.py (main Dash application)
- callbacks.py (interactive callbacks)
- layouts/ (sections)
- components/ (reusable components)
```

---

## 5. ADVANTAGES OF MODULAR ARCHITECTURE

### 5.1 Separation of Concerns
- Each module has single responsibility
- Easy to understand, test, and modify
- Data layer isolated from business logic

### 5.2 Scalability
- Add new diseases by creating new Service
- Replace model algorithm without affecting other modules
- Scale components independently

### 5.3 Testability
- Unit test each module independently
- Mock dependencies easily
- Create realistic test fixtures

### 5.4 Maintainability
- Clear interfaces reduce coupling
- Easy to locate bugs
- Simpler to onboard new developers

### 5.5 Reusability
- Dashboard can use any ModelRegistry
- What-If Engine works with any models
- XAI components model-agnostic

### 5.6 Deployment Flexibility
- Run models as microservices
- Deploy dashboard separately
- Scale data layer independently

---

## 6. IMPLEMENTATION ROADMAP

### Phase 1: Core Infrastructure (Week 1)
- [ ] Create folder structure
- [ ] Define type interfaces (data models)
- [ ] Create abstract base classes
- [ ] Setup logging and configuration

### Phase 2: Data Layer (Week 1-2)
- [ ] Implement BaseDataSource
- [ ] Implement MIMICDataSource
- [ ] Implement Preprocessor
- [ ] Write unit tests

### Phase 3: Model Services (Week 2-3)
- [ ] Implement BaseModelService
- [ ] Implement 4 disease services
- [ ] Implement ModelRegistry
- [ ] Write integration tests

### Phase 4: XAI Engine (Week 3-4)
- [ ] Implement SHAP explainer
- [ ] Implement LIME explainer
- [ ] Implement ClinicalTranslator
- [ ] Write explanation tests

### Phase 5: What-If Engine (Week 4)
- [ ] Implement WhatIfAnalyzer
- [ ] Implement ScenarioGenerator
- [ ] Implement ParameterOptimizer
- [ ] Write scenario tests

### Phase 6: Dashboard (Week 5)
- [ ] Refactor dashboard components
- [ ] Implement callbacks
- [ ] Add real-time features
- [ ] Performance optimization

### Phase 7: Evaluation (Week 5-6)
- [ ] Implement ModelEvaluator
- [ ] Implement XAIEvaluator
- [ ] Implement ClinicalValidator
- [ ] Generate reports

---

## 7. BACKWARD COMPATIBILITY

### 7.1 Migration Strategy
- Keep existing scripts working during transition
- Create adapter layer to existing code
- Gradually replace components
- Full refactor completion: 6 weeks

### 7.2 Version Management
- Tag current codebase as `v1.0-monolithic`
- New branch: `v2.0-modular`
- Maintain v1.0 for documentation
- Increment to v2.0 on completion

---

**END OF MODULAR ARCHITECTURE DESIGN**
