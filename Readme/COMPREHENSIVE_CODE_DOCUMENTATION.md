# Comprehensive Code Documentation
## Explainable Medical AI System - Classes, Methods, and Objects

**Generated:** March 3, 2026  
**Project:** Explainable AI System for Medical Diagnosis using SHAP and LIME

---

## Table of Contents

1. [Core Engine Classes](#core-engine-classes)
2. [Model Service Classes](#model-service-classes)
3. [Novel Patentable Components](#novel-patentable-components)
4. [Data Processing Classes](#data-processing-classes)
5. [Governance and Compliance Classes](#governance-and-compliance-classes)
6. [Dashboard and UI Classes](#dashboard-and-ui-classes)
7. [Utility Classes and Enums](#utility-classes-and-enums)

---

## 1. Core Engine Classes

### 1.1 `XAIEngine` (xai_engine.py)

**Purpose:** Comprehensive explainable AI engine supporting multiple explanation methods.

**Key Attributes:**
- `registered_models`: Dict[str, RegisteredModel] - Registry of models with metadata
- `explainers`: Dict[str, Any] - Cache of SHAP/LIME explainers
- `cache`: ExplanationCache - Performance caching system
- `clinical_translator`: ClinicalTranslator - Converts AI outputs to clinical language

**Key Methods:**

```python
def register_model(
    model_name: str,
    model: Any,
    feature_names: List[str],
    data_type: DataType = DataType.TABULAR,
    background_data: Optional[np.ndarray] = None
) -> bool:
    """
    Register a trained model for explanation.
    
    Args:
        model_name: Unique identifier for the model
        model: Trained ML model (sklearn, XGBoost, etc.)
        feature_names: List of feature names in order
        data_type: Type of data (tabular, image, time_series)
        background_data: Background/training data for SHAP
        
    Returns:
        bool: True if registration successful
    """
```

```python
def explain(
    model_name: str,
    X: Union[np.ndarray, pd.DataFrame],
    method: str = "shap",
    clinical_context: Optional[Dict] = None,
    **kwargs
) -> Union[SHAPExplanation, LIMEExplanation]:
    """
    Generate explanation for a prediction.
    
    Args:
        model_name: Name of registered model
        X: Input features (single sample or batch)
        method: Explanation method (shap_tree, lime_tabular, etc.)
        clinical_context: Patient metadata for clinical translation
        
    Returns:
        Explanation object with feature importances and clinical translation
    """
```

```python
def generate_counterfactuals(
    model_name: str,
    X: np.ndarray,
    desired_outcome: int,
    num_samples: int = 5,
    diversity_weight: float = 1.0
) -> CounterfactualExplanation:
    """
    Generate counterfactual examples.
    
    Args:
        model_name: Name of registered model
        X: Original input features
        desired_outcome: Target class for counterfactuals
        num_samples: Number of counterfactuals to generate
        diversity_weight: How diverse counterfactuals should be
        
    Returns:
        CounterfactualExplanation with alternative scenarios
    """
```

**Supporting Classes:**

```python
class ExplanationMethod(str, Enum):
    """Supported explanation methods"""
    SHAP_TREE = "shap_tree"
    SHAP_KERNEL = "shap_kernel"
    SHAP_DEEP = "shap_deep"
    LIME_TABULAR = "lime_tabular"
    LIME_IMAGE = "lime_image"
    GRADCAM = "gradcam"
    COUNTERFACTUAL = "counterfactual"

class FeatureImportance(BaseModel):
    """Single feature importance value"""
    feature_name: str
    importance: float
    direction: str  # "increases" or "decreases"
    clinical_interpretation: Optional[str] = None

class SHAPExplanation(BaseModel):
    """SHAP explanation output"""
    prediction: float
    baseline_prediction: float
    feature_importances: List[FeatureImportance]
    shap_values: np.ndarray
    clinical_explanation: Optional[ClinicalExplanation] = None
```

---

### 1.2 `WhatIfEngine` (whatif_engine.py)

**Purpose:** Clinical decision support for exploring "what-if" scenarios.

**Key Attributes:**
- `registered_models`: Dict[str, WhatIfModel] - Models with clinical constraints
- `constraint_validator`: ConstraintValidator - Validates clinical plausibility
- `optimization_solver`: OptimizationSolver - Finds optimal interventions

**Key Methods:**

```python
def analyze_baseline(
    model_name: str,
    patient_data: Union[np.ndarray, pd.DataFrame],
    shap_explainer: Optional[Any] = None
) -> BaselineAnalysis:
    """
    Analyze current patient state.
    
    Args:
        model_name: Name of registered model
        patient_data: Current patient features
        shap_explainer: Optional SHAP explainer for feature importance
        
    Returns:
        BaselineAnalysis with current risk, modifiable features, and risks
    """
```

```python
def run_scenario(
    model_name: str,
    baseline_data: Union[np.ndarray, pd.DataFrame],
    interventions: Dict[str, float],
    validate_constraints: bool = True
) -> ScenarioResult:
    """
    Run a what-if scenario with specified interventions.
    
    Args:
        model_name: Name of registered model
        baseline_data: Current patient state
        interventions: Dict mapping feature names to new values
        validate_constraints: Whether to check clinical constraints
        
    Returns:
        ScenarioResult with new risk, delta, and constraint violations
    """
```

```python
def suggest_interventions(
    model_name: str,
    patient_data: Union[np.ndarray, pd.DataFrame],
    target_risk: float,
    max_features: int = 3,
    shap_explainer: Optional[Any] = None
) -> InterventionPlan:
    """
    Generate intervention suggestions to reach target risk.
    
    Args:
        model_name: Name of registered model
        patient_data: Current patient features
        target_risk: Desired risk level (0.0 to 1.0)
        max_features: Maximum number of features to modify
        shap_explainer: SHAP explainer for prioritization
        
    Returns:
        InterventionPlan with ranked suggestions and feasibility scores
    """
```

**Supporting Classes:**

```python
class FeatureType(str, Enum):
    """Feature modifiability classification"""
    FIXED = "fixed"           # Cannot change (age, gender)
    ACTIONABLE = "actionable" # Can change with treatment (BP, glucose)
    SLOW = "slow"             # Changes slowly (weight, BMI)
    DERIVED = "derived"       # Computed from other features

class FeatureConstraint:
    """Clinical constraint for a feature"""
    feature_name: str
    feature_type: FeatureType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    normal_range: Optional[Tuple[float, float]] = None
    max_change_per_hour: Optional[float] = None
    
class BaselineAnalysis(BaseModel):
    """Baseline patient analysis"""
    current_risk: float
    current_features: Dict[str, float]
    modifiable_features: List[str]
    fixed_features: List[str]
    feature_risks: Dict[str, float]  # SHAP-based risk contribution
    
class ScenarioResult(BaseModel):
    """What-if scenario outcome"""
    baseline_risk: float
    new_risk: float
    risk_delta: float
    risk_delta_percent: float
    modified_features: Dict[str, Tuple[float, float]]  # name -> (old, new)
    constraint_violations: List[str]
    plausibility_score: float
    clinical_interpretation: str
```

---

## 2. Model Service Classes

### 2.1 `BaseModelService` (disease_model_service.py)

**Purpose:** Abstract base class for disease-specific prediction services.

**Key Methods:**

```python
@abstractmethod
def train(
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: np.ndarray,
    X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    y_val: Optional[np.ndarray] = None,
    **kwargs
) -> TrainingMetrics:
    """Train the disease prediction model"""
    
@abstractmethod
def predict(
    X: Union[np.ndarray, pd.DataFrame],
    return_proba: bool = True
) -> ModelPrediction:
    """Make predictions on new data"""
    
@abstractmethod
def explain(
    X: Union[np.ndarray, pd.DataFrame],
    method: ExplainerType = ExplainerType.SHAP,
    **kwargs
) -> ExplanationOutput:
    """Generate explanations for predictions"""
```

**Concrete Implementations:**

### 2.2 `SepsisModelService` (disease_model_service.py)

**Purpose:** Sepsis and septic shock prediction.

**Clinical Features:**
- Temperature, Heart Rate, Respiratory Rate
- White Blood Cell Count, Platelet Count
- Systolic/Diastolic Blood Pressure
- Lactate, Creatinine

**Key Clinical Thresholds:**
- SIRS Criteria: Temp >38°C or <36°C, HR >90, RR >20, WBC >12K or <4K
- qSOFA Score: SBP ≤100, RR ≥22, Altered Mental Status
- Sepsis-3: SOFA score ≥2 with infection

### 2.3 `KidneyFailureModelService` (disease_model_service.py)

**Purpose:** Acute Kidney Injury (AKI) prediction.

**Clinical Features:**
- Creatinine, BUN, GFR
- Urine Output, Fluid Balance
- Blood Pressure, Age
- Diabetes and Hypertension comorbidities

**Key Clinical Criteria:**
- KDIGO Stage 1: Creatinine 1.5-1.9x baseline
- KDIGO Stage 2: Creatinine 2.0-2.9x baseline
- KDIGO Stage 3: Creatinine ≥3.0x baseline

---

## 3. Novel Patentable Components

### 3.1 `PhysiologicalCouplingEngine` (physiological_coupling.py)

**Patent Claim:** Automatic computation of physiologically coupled parameter adjustments using evidence-based coupling coefficients.

**Purpose:** Ensures what-if scenarios remain medically plausible by automatically adjusting related parameters.

**Key Classes:**

```python
class CouplingType(Enum):
    """Types of physiological coupling"""
    LINEAR = "linear"              # Y = a*X + b
    NONLINEAR = "nonlinear"        # Y = f(X)
    THRESHOLD = "threshold"        # Y changes when X crosses threshold
    BIDIRECTIONAL = "bidirectional" # X ↔ Y mutual influence
    PROPORTIONAL = "proportional"  # Y/X ratio maintained

class CouplingRelationship:
    """Evidence-based coupling definition"""
    primary_parameter: str
    coupled_parameter: str
    coupling_type: CouplingType
    coefficient: float              # Coupling strength
    baseline_offset: float = 0.0
    evidence_source: str            # Medical citation
    confidence_level: float = 0.95
    applicable_range: Tuple[float, float]
```

**Key Methods:**

```python
def apply_coupling(
    primary_param: str,
    primary_value: float,
    baseline_values: Dict[str, float]
) -> Dict[str, float]:
    """
    Apply physiological coupling when a parameter changes.
    
    Example: Increasing temperature from 37°C to 40°C automatically
    increases heart rate by ~30 bpm based on coupling coefficient.
    
    Args:
        primary_param: Parameter being changed
        primary_value: New value for primary parameter
        baseline_values: Current patient state
        
    Returns:
        Dict of automatically adjusted coupled parameters
    """
```

**Evidence-Based Couplings:**

1. **Temperature ↔ Heart Rate**
   - Type: LINEAR
   - Coefficient: 10 bpm per °C
   - Source: "Mackowiak, PA (1992). A critical appraisal of 98.6°F"
   - Range: 35-42°C

2. **Blood Pressure → Baroreflex → Heart Rate**
   - Type: THRESHOLD (bidirectional)
   - Coefficient: -0.5 to -0.8 bpm/mmHg
   - Source: Baroreceptor sensitivity studies

3. **Heart Rate → Cardiac Output**
   - Type: PROPORTIONAL
   - Ratio: Maintains stroke volume relationship
   - Source: Frank-Starling mechanism

---

### 3.2 `ClinicalPlausibilityScorer` (plausibility_scoring.py)

**Patent Claim:** Quantitative intervention feasibility scoring using patient-specific modifiers.

**Purpose:** Scores how realistic a clinical intervention is based on:
- Physiological feasibility
- Time constraints
- Patient-specific factors
- Intervention type and urgency

**Key Classes:**

```python
class InterventionType(Enum):
    """Types of clinical interventions"""
    MEDICATION = "medication"
    PROCEDURE = "procedure"
    LIFESTYLE = "lifestyle"
    MONITORING = "monitoring"

class UrgencyLevel(Enum):
    """Clinical urgency levels"""
    ELECTIVE = "elective"      # Planned, weeks-months
    URGENT = "urgent"          # Hours-days
    EMERGENT = "emergent"      # Minutes-hours
    IMMEDIATE = "immediate"    # Seconds-minutes

class PatientModifier(Enum):
    """Patient factors affecting feasibility"""
    AGE_PEDIATRIC = "age_pediatric"
    AGE_GERIATRIC = "age_geriatric"
    PREGNANCY = "pregnancy"
    RENAL_IMPAIRMENT = "renal_impairment"
    HEPATIC_IMPAIRMENT = "hepatic_impairment"
    IMMUNOCOMPROMISED = "immunocompromised"
```

**Key Methods:**

```python
def score_plausibility(
    parameter: str,
    current_value: float,
    target_value: float,
    time_hours: float,
    intervention_type: InterventionType,
    urgency: UrgencyLevel,
    patient_modifiers: List[PatientModifier] = []
) -> float:
    """
    Compute clinical plausibility score (0.0 to 1.0).
    
    Example: Reducing creatinine from 2.5 to 1.2 mg/dL in 6 hours
    might score 0.3 (low plausibility) for medication, but 0.8 for
    dialysis in an ICU setting.
    
    Returns:
        float: Plausibility score where:
            1.0 = Highly plausible
            0.75 = Moderately plausible
            0.5 = Marginally plausible
            0.25 = Implausible but theoretically possible
            0.0 = Physiologically impossible
    """
```

---

### 3.3 `HierarchicalInterventionEngine` (hierarchical_interventions.py)

**Patent Claim:** SHAP-guided multi-tier treatment planning with cost-benefit optimization.

**Purpose:** Generates tiered intervention plans based on resource level and SHAP importance.

**Key Classes:**

```python
class InterventionTier(Enum):
    """Intervention tiers by intensity"""
    TIER_1_MONITORING = "tier_1_monitoring"
    TIER_2_NONINVASIVE = "tier_2_noninvasive"
    TIER_3_MEDICATION = "tier_3_medication"
    TIER_4_INVASIVE = "tier_4_invasive"

class ResourceLevel(Enum):
    """Available clinical resources"""
    OUTPATIENT = "outpatient"
    EMERGENCY_DEPT = "emergency_dept"
    GENERAL_WARD = "general_ward"
    INTENSIVE_CARE = "intensive_care"
    
class ClinicalIntervention:
    """Single intervention recommendation"""
    tier: InterventionTier
    parameter: str
    current_value: float
    target_value: float
    intervention_name: str
    estimated_time_hours: float
    estimated_cost: float
    expected_risk_reduction: float
    contraindications: List[str]
    monitoring_required: List[str]
```

**Key Methods:**

```python
def generate_intervention_plan(
    patient_data: Dict[str, float],
    shap_values: Dict[str, float],
    target_risk: float,
    resource_level: ResourceLevel
) -> InterventionPlan:
    """
    Generate hierarchical intervention plan.
    
    Uses SHAP values to prioritize interventions by impact.
    Selects interventions appropriate for resource level.
    Orders by cost-benefit ratio.
    
    Returns:
        InterventionPlan with tiered recommendations
    """
```

---

## 4. Data Processing Classes

### 4.1 `SyntheticDataGenerator` (training_pipeline.py)

**Purpose:** Generate realistic synthetic clinical data for training.

**Key Methods:**

```python
@staticmethod
def generate_patient_data(
    n_samples: int = 10000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic patient features with realistic distributions.
    
    Features:
        - Demographics: age, gender
        - Vital signs: HR, BP, temp, RR
        - Labs: WBC, Hgb, Plt, Cr, BUN, glucose, lactate
        
    Returns:
        DataFrame with n_samples rows and all features
    """

@staticmethod
def assign_disease_labels(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Assign disease labels based on clinical criteria.
    
    Diseases:
        - Sepsis: Based on SIRS + suspected infection
        - AKI: Based on creatinine and BUN
        - Heart Disease: Based on age, BP, glucose
        - Diabetes: Based on glucose levels
        - Anemia: Based on hemoglobin
        - Mortality: Based on severity scores
    """
```

---

### 4.2 `AdvancedFeatureEngineer` (train_advanced_models.py)

**Purpose:** Create clinical features from raw data.

**Key Methods:**

```python
def create_clinical_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Engineer advanced clinical features.
    
    Features Created:
        - MAP (Mean Arterial Pressure)
        - Pulse Pressure
        - Shock Index (HR/SBP)
        - BUN/Cr Ratio
        - SIRS Score
        - qSOFA Score
        - Age Risk Score
        - Lab Abnormality Count
        - Vital Sign Severity
        
    Returns:
        DataFrame with original + engineered features
    """
```

---

### 4.3 `ClinicalModelTrainer` (training_pipeline.py)

**Purpose:** Train disease prediction models with governance.

**Key Attributes:**
- `model_registry`: ModelRegistry - Tracks all trained models
- `audit_logger`: AuditLogger - Logs all training events

**Key Methods:**

```python
def train_model(
    disease_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = "xgboost"
) -> Dict[str, Any]:
    """
    Train a disease prediction model.
    
    Steps:
        1. Initialize model with optimal hyperparameters
        2. Fit on training data
        3. Evaluate on validation set
        4. Register with ModelRegistry
        5. Log training event with AuditLogger
        6. Save model artifact
        
    Returns:
        Dict with model, scaler, metrics, and metadata
    """
```

---

## 5. Governance and Compliance Classes

### 5.1 `ModelRegistry` (model_registry.py)

**Purpose:** Track and manage trained models for FDA compliance.

**Key Classes:**

```python
class ModelMetadata:
    """Metadata for a trained model"""
    model_name: str
    version: str
    disease_target: str
    model_type: str
    training_date: datetime
    training_samples: int
    feature_names: List[str]
    performance_metrics: Dict[str, float]
    data_hash: str
    model_hash: str
    
class ExplanationMetadata:
    """Metadata for explanations"""
    explanation_method: str
    shap_available: bool
    lime_available: bool
    feature_importance_available: bool
```

**Key Methods:**

```python
def register_model(
    model_name: str,
    model: Any,
    metadata: ModelMetadata
) -> str:
    """
    Register a trained model.
    
    Creates model artifact bundle with:
        - Trained model object
        - Feature names and preprocessing
        - Performance metrics
        - Training metadata
        - Cryptographic hash for integrity
        
    Returns:
        model_id: Unique identifier for registered model
    """

def get_model(
    model_name: str,
    version: Optional[str] = None
) -> Tuple[Any, ModelMetadata]:
    """
    Retrieve a registered model.
    
    Args:
        model_name: Name of model
        version: Specific version (None = latest)
        
    Returns:
        (model, metadata) tuple
    """
```

---

### 5.2 `AuditLogger` (audit_logging.py)

**Purpose:** Immutable audit trail for all system events.

**Key Classes:**

```python
class AuditEventType(str, Enum):
    """Types of auditable events"""
    MODEL_TRAINING = "model_training"
    MODEL_PREDICTION = "model_prediction"
    EXPLANATION_GENERATED = "explanation_generated"
    WHATIF_SCENARIO = "whatif_scenario"
    DATA_ACCESS = "data_access"
    CONFIG_CHANGE = "config_change"
    
class AuditEvent:
    """Single audit event"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: str
    session_id: str
    event_data: Dict[str, Any]
    data_hash: str
```

**Key Methods:**

```python
def log_event(
    event_type: AuditEventType,
    event_data: Dict[str, Any],
    user_id: Optional[str] = None
) -> str:
    """
    Log an immutable audit event.
    
    Creates JSONL entry with:
        - Unique event ID
        - Timestamp (ISO 8601)
        - Event type and data
        - Cryptographic hash for integrity
        
    Returns:
        event_id: Unique identifier for this event
    """

def query_events(
    event_type: Optional[AuditEventType] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[AuditEvent]:
    """Query audit log with filters"""
```

---

### 5.3 `ComplianceMatrix` (compliance_matrix.py)

**Purpose:** Map system features to regulatory requirements.

**Key Classes:**

```python
class RegulatoryFramework(str, Enum):
    """Regulatory frameworks"""
    FDA_510K = "FDA_510K"
    FDA_PMA = "FDA_PMA"
    EU_MDR = "EU_MDR"
    HIPAA = "HIPAA"
    GDPR = "GDPR"
    
class ComplianceStatus(str, Enum):
    """Compliance status"""
    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"
    
class RegulatoryRequirement:
    """Single regulatory requirement"""
    requirement_id: str
    framework: RegulatoryFramework
    section: str
    description: str
    risk_class: str
```

**Key Methods:**

```python
def check_compliance(
    framework: RegulatoryFramework
) -> Dict[str, ComplianceStatus]:
    """
    Check compliance against regulatory framework.
    
    Returns:
        Dict mapping requirement_id to ComplianceStatus
    """
```

---

## 6. Dashboard and UI Classes

### 6.1 Dashboard Functions (enhanced_dashboard_with_whatif.py)

**Purpose:** Interactive Dash web application for clinical users.

**Key Functions:**

```python
def load_results() -> Dict[str, Any]:
    """Load model results and patient data"""

def predict_with_real_model(
    patient_data: Dict[str, float],
    disease: str = 'kidney_failure'
) -> float:
    """Get prediction from trained model"""

def get_shap_explanation(
    patient_data: Dict[str, float],
    disease: str = 'kidney_failure'
) -> Dict[str, float]:
    """Get SHAP feature importances"""

@app.callback(
    Output('whatif-results', 'children'),
    Input('run-whatif-btn', 'n_clicks'),
    State('age-slider', 'value'),
    State('creatinine-slider', 'value'),
    State('bp-slider', 'value'),
    State('glucose-slider', 'value')
)
def update_whatif_analysis(
    n_clicks, age, creatinine, bp, glucose
) -> List[dbc.Card]:
    """
    Run what-if scenario and display results.
    
    Compares baseline risk to new risk after interventions.
    Shows risk delta and clinical recommendations.
    """
```

---

## 7. Utility Classes and Enums

### 7.1 Clinical Translation

```python
class ClinicalTranslator:
    """Translate AI outputs to clinical language"""
    
    def translate_feature_importance(
        feature_name: str,
        importance: float,
        direction: str
    ) -> str:
        """
        Convert feature importance to clinical statement.
        
        Example:
            Input: ("creatinine", 0.25, "increases")
            Output: "Elevated creatinine (kidney function) 
                    strongly increases risk"
        """
```

### 7.2 Data Models (Pydantic)

All classes use Pydantic BaseModel for:
- **Type validation**: Ensures correct data types
- **Serialization**: Easy JSON export
- **Documentation**: Self-documenting schemas

Example:
```python
class ModelPrediction(BaseModel):
    """Prediction output schema"""
    patient_id: str
    prediction: float
    prediction_class: int
    confidence: float
    risk_category: str = Field(..., regex="^(LOW|MODERATE|HIGH|CRITICAL)$")
    timestamp: datetime = Field(default_factory=datetime.now)
```

---

## Method Naming Conventions

The codebase follows these naming conventions:

- **Public methods**: `snake_case` (e.g., `register_model()`)
- **Private methods**: `_leading_underscore` (e.g., `_validate_input()`)
- **Class names**: `PascalCase` (e.g., `XAIEngine`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_THRESHOLD`)
- **Enums**: `PascalCase` class, `UPPER_SNAKE_CASE` members

---

## Type Annotations

All methods use Python type hints:

```python
def predict(
    X: Union[np.ndarray, pd.DataFrame],
    return_proba: bool = True
) -> ModelPrediction:
    """Examples of type annotations"""
```

This enables:
- IDE autocomplete
- Static type checking with mypy
- Better documentation
- Runtime validation with Pydantic

---

## Error Handling

Custom exceptions are defined in respective modules:

```python
class ModelNotFoundError(ValueError):
    """Raised when model not in registry"""
    
class InvalidFeatureError(ValueError):
    """Raised when feature validation fails"""
    
class ConstraintViolationError(RuntimeError):
    """Raised when clinical constraints violated"""
```

---

## Testing Patterns

The codebase uses these testing patterns:

1. **Unit Tests**: Test individual methods in isolation
2. **Integration Tests**: Test module interactions
3. **Fixtures**: Reusable test data in `tests/fixtures/`
4. **Mocking**: Mock external dependencies (models, data sources)

Example:
```python
def test_model_prediction():
    """Test prediction output format"""
    service = SepsisModelService()
    service.train(X_train, y_train)
    
    prediction = service.predict(X_test[0])
    
    assert isinstance(prediction, ModelPrediction)
    assert 0 <= prediction.confidence <= 1
    assert prediction.risk_category in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
```

---

## Summary Statistics

**Total Classes:** ~70+  
**Total Methods:** ~300+  
**Total Enums:** ~20+  
**Lines of Code:** ~15,000+  
**Documentation Coverage:** 90%+  

**Core Technologies:**
- Python 3.8+
- NumPy, Pandas
- Scikit-learn, XGBoost
- SHAP, LIME
- Pydantic for data validation
- Dash/Plotly for UI

---

*This documentation covers the main classes and methods. For implementation details, see individual source files.*
