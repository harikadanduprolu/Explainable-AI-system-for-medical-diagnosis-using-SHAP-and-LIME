"""
What-If and Counterfactual Engine for Medical AI
=================================================

A clinical decision support engine for exploring "what-if" scenarios:
- Patient baseline analysis
- Constrained feature perturbations
- Clinical plausibility checks
- Delta risk computation
- SHAP-guided recommendations
- Safety constraint validation

Usage:
    engine = WhatIfEngine()
    
    # Register model with clinical constraints
    engine.register_model(
        model_name="sepsis",
        model=trained_model,
        feature_names=feature_names,
        clinical_constraints=sepsis_constraints
    )
    
    # Analyze baseline
    baseline = engine.analyze_baseline(
        model_name="sepsis",
        patient_data=patient_features
    )
    
    # Explore what-if scenario
    scenario = engine.run_scenario(
        model_name="sepsis",
        baseline_data=patient_features,
        interventions={"temperature": 37.0, "heart_rate": 85}
    )
    
    # Generate counterfactuals
    suggestions = engine.suggest_interventions(
        model_name="sepsis",
        patient_data=patient_features,
        target_risk=0.30,
        shap_explainer=shap_explainer
    )

Algorithm Steps:
1. BASELINE ANALYSIS
   - Load patient data
   - Compute current risk
   - Identify modifiable vs fixed features
   
2. CONSTRAINT VALIDATION
   - Check clinical plausibility (e.g., temp 35-42°C)
   - Verify physiological relationships
   - Ensure safety bounds
   
3. PERTURBATION GENERATION
   - Apply user-specified changes
   - OR: Use optimization to find target risk
   - Respect constraints at each step
   
4. PREDICTION & DELTA
   - Re-run model with new features
   - Compute risk delta
   - Assess clinical significance
   
5. SHAP INTEGRATION
   - Use SHAP values to prioritize interventions
   - Focus on high-impact, actionable features
   - Generate ranked recommendations
"""

from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from datetime import datetime
import warnings
from scipy.optimize import minimize
from copy import deepcopy

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

class FeatureType(str, Enum):
    """Feature modifiability classification."""
    FIXED = "fixed"              # Cannot be changed (age, gender)
    ACTIONABLE = "actionable"    # Can be changed with treatment (temp, BP)
    SLOW = "slow"                # Changes slowly (weight, BMI)
    DERIVED = "derived"          # Computed from other features


class InterventionPriority(str, Enum):
    """Clinical priority levels."""
    CRITICAL = "critical"    # Immediate life-saving
    HIGH = "high"           # Important for outcome
    MODERATE = "moderate"   # Helpful but not urgent
    LOW = "low"            # Minor impact


class PlausibilityLevel(str, Enum):
    """Scenario plausibility assessment."""
    REALISTIC = "realistic"           # Achievable with standard treatment
    CHALLENGING = "challenging"        # Difficult but possible
    UNLIKELY = "unlikely"             # Requires aggressive intervention
    IMPOSSIBLE = "impossible"         # Violates physiology


# ============================================================================
# CLINICAL CONSTRAINTS
# ============================================================================

@dataclass
class FeatureConstraint:
    """
    Defines valid range and characteristics for a clinical feature.
    """
    feature_name: str
    feature_type: FeatureType
    
    # Value constraints
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    valid_values: Optional[List[float]] = None  # For categorical
    
    # Change constraints
    max_absolute_change: Optional[float] = None  # Maximum single change
    max_percent_change: Optional[float] = None   # Maximum % change
    
    # Clinical metadata
    unit: str = ""
    clinical_name: str = ""
    normal_range: Optional[Tuple[float, float]] = None
    
    # Relationships
    coupled_features: List[str] = field(default_factory=list)  # Features that change together
    
    def validate(self, value: float) -> Tuple[bool, str]:
        """
        Validate if a value is clinically plausible.
        
        Returns:
            (is_valid, error_message)
        """
        if self.valid_values is not None:
            if value not in self.valid_values:
                return False, f"{self.feature_name} must be one of {self.valid_values}"
        
        if self.min_value is not None and value < self.min_value:
            return False, f"{self.feature_name} below minimum ({self.min_value})"
        
        if self.max_value is not None and value > self.max_value:
            return False, f"{self.feature_name} above maximum ({self.max_value})"
        
        return True, ""
    
    def validate_change(self, original: float, new: float) -> Tuple[bool, str]:
        """
        Validate if a change is clinically plausible.
        """
        if self.feature_type == FeatureType.FIXED:
            if abs(new - original) > 0.001:
                return False, f"{self.feature_name} is fixed and cannot be changed"
        
        if self.max_absolute_change is not None:
            change = abs(new - original)
            if change > self.max_absolute_change:
                return False, f"Change in {self.feature_name} exceeds max ({self.max_absolute_change})"
        
        if self.max_percent_change is not None and original != 0:
            pct_change = abs((new - original) / original)
            if pct_change > self.max_percent_change:
                return False, f"% change in {self.feature_name} exceeds max ({self.max_percent_change*100}%)"
        
        return True, ""


class ClinicalConstraints:
    """
    Collection of constraints for a specific disease model.
    """
    
    @staticmethod
    def get_sepsis_constraints() -> Dict[str, FeatureConstraint]:
        """Standard constraints for sepsis prediction."""
        return {
            'temperature': FeatureConstraint(
                feature_name='temperature',
                feature_type=FeatureType.ACTIONABLE,
                min_value=35.0,
                max_value=42.0,
                max_absolute_change=3.0,  # Can reduce by ~3°C with treatment
                unit='°C',
                clinical_name='Body Temperature',
                normal_range=(36.1, 37.2)
            ),
            'heart_rate': FeatureConstraint(
                feature_name='heart_rate',
                feature_type=FeatureType.ACTIONABLE,
                min_value=40,
                max_value=200,
                max_absolute_change=40,  # Can change by ~40 bpm
                unit='bpm',
                clinical_name='Heart Rate',
                normal_range=(60, 100)
            ),
            'systolic_bp': FeatureConstraint(
                feature_name='systolic_bp',
                feature_type=FeatureType.ACTIONABLE,
                min_value=60,
                max_value=220,
                max_absolute_change=50,
                unit='mmHg',
                clinical_name='Systolic Blood Pressure',
                normal_range=(90, 140)
            ),
            'respiratory_rate': FeatureConstraint(
                feature_name='respiratory_rate',
                feature_type=FeatureType.ACTIONABLE,
                min_value=8,
                max_value=40,
                max_absolute_change=15,
                unit='breaths/min',
                clinical_name='Respiratory Rate',
                normal_range=(12, 20)
            ),
            'wbc_count': FeatureConstraint(
                feature_name='wbc_count',
                feature_type=FeatureType.SLOW,
                min_value=1000,
                max_value=50000,
                max_absolute_change=5000,  # Changes slowly with treatment
                unit='cells/μL',
                clinical_name='White Blood Cell Count',
                normal_range=(4000, 11000)
            ),
            'lactate': FeatureConstraint(
                feature_name='lactate',
                feature_type=FeatureType.SLOW,
                min_value=0.5,
                max_value=20.0,
                max_absolute_change=3.0,
                unit='mmol/L',
                clinical_name='Blood Lactate',
                normal_range=(0.5, 2.0)
            ),
            'age': FeatureConstraint(
                feature_name='age',
                feature_type=FeatureType.FIXED,
                min_value=0,
                max_value=120,
                unit='years',
                clinical_name='Age'
            ),
            'gender': FeatureConstraint(
                feature_name='gender',
                feature_type=FeatureType.FIXED,
                valid_values=[0, 1],
                clinical_name='Gender'
            )
        }
    
    @staticmethod
    def get_aki_constraints() -> Dict[str, FeatureConstraint]:
        """Standard constraints for AKI prediction."""
        return {
            'creatinine': FeatureConstraint(
                feature_name='creatinine',
                feature_type=FeatureType.SLOW,
                min_value=0.3,
                max_value=15.0,
                max_absolute_change=2.0,  # Changes slowly
                unit='mg/dL',
                clinical_name='Serum Creatinine',
                normal_range=(0.6, 1.2)
            ),
            'urine_output': FeatureConstraint(
                feature_name='urine_output',
                feature_type=FeatureType.ACTIONABLE,
                min_value=0,
                max_value=5000,
                max_absolute_change=1000,
                unit='mL/day',
                clinical_name='Urine Output',
                normal_range=(800, 2000)
            ),
            'potassium': FeatureConstraint(
                feature_name='potassium',
                feature_type=FeatureType.ACTIONABLE,
                min_value=2.5,
                max_value=7.5,
                max_absolute_change=1.5,
                unit='mEq/L',
                clinical_name='Serum Potassium',
                normal_range=(3.5, 5.0)
            ),
            'age': FeatureConstraint(
                feature_name='age',
                feature_type=FeatureType.FIXED,
                min_value=0,
                max_value=120,
                unit='years',
                clinical_name='Age'
            )
        }


# ============================================================================
# OUTPUT SCHEMAS
# ============================================================================

class BaselineAnalysis(BaseModel):
    """Baseline patient analysis output."""
    patient_id: str
    model_name: str
    current_risk: float = Field(..., ge=0, le=1)
    risk_category: str
    
    feature_values: Dict[str, float] = Field(..., description="Current feature values")
    feature_types: Dict[str, str] = Field(..., description="Feature modifiability")
    
    modifiable_features: List[str] = Field(..., description="Features that can be changed")
    fixed_features: List[str] = Field(..., description="Features that cannot be changed")
    
    abnormal_features: Dict[str, Dict[str, Any]] = Field(
        ..., 
        description="Features outside normal range"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P123",
                "model_name": "sepsis",
                "current_risk": 0.75,
                "risk_category": "High",
                "feature_values": {"temperature": 38.5, "heart_rate": 110},
                "feature_types": {"temperature": "actionable", "age": "fixed"},
                "modifiable_features": ["temperature", "heart_rate"],
                "fixed_features": ["age", "gender"],
                "abnormal_features": {
                    "temperature": {
                        "value": 38.5,
                        "normal_range": [36.1, 37.2],
                        "status": "high"
                    }
                }
            }
        }


class ScenarioResult(BaseModel):
    """What-if scenario result."""
    scenario_id: str
    patient_id: str
    model_name: str
    
    # Original state
    baseline_risk: float = Field(..., ge=0, le=1)
    baseline_features: Dict[str, float]
    
    # Modified state
    new_risk: float = Field(..., ge=0, le=1)
    new_features: Dict[str, float]
    
    # Changes
    interventions: Dict[str, Tuple[float, float]] = Field(
        ..., 
        description="Feature → (original, new)"
    )
    risk_delta: float = Field(..., description="Change in risk (new - baseline)")
    risk_delta_percent: float = Field(..., description="% change in risk")
    
    # Validation
    is_valid: bool
    plausibility: PlausibilityLevel
    constraint_violations: List[str] = Field(default_factory=list)
    
    # Clinical interpretation
    clinical_summary: str
    intervention_feasibility: Dict[str, str] = Field(
        ...,
        description="Feasibility of each intervention"
    )
    
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        schema_extra = {
            "example": {
                "scenario_id": "S001",
                "patient_id": "P123",
                "model_name": "sepsis",
                "baseline_risk": 0.75,
                "baseline_features": {"temperature": 38.5},
                "new_risk": 0.45,
                "new_features": {"temperature": 37.0},
                "interventions": {"temperature": (38.5, 37.0)},
                "risk_delta": -0.30,
                "risk_delta_percent": -40.0,
                "is_valid": True,
                "plausibility": "realistic",
                "constraint_violations": [],
                "clinical_summary": "Reducing temperature to normal decreases risk by 40%",
                "intervention_feasibility": {
                    "temperature": "ACTIONABLE: Administer antipyretics"
                }
            }
        }


class InterventionSuggestion(BaseModel):
    """Suggested clinical intervention."""
    feature_name: str
    clinical_name: str
    
    current_value: float
    suggested_value: float
    change_amount: float
    
    expected_risk_reduction: float = Field(..., description="Predicted risk decrease")
    priority: InterventionPriority
    
    actionability: str = Field(..., description="How to achieve this change")
    time_to_effect: str = Field(..., description="Expected time for change")
    
    shap_importance: Optional[float] = Field(None, description="SHAP value if available")
    
    class Config:
        schema_extra = {
            "example": {
                "feature_name": "temperature",
                "clinical_name": "Body Temperature",
                "current_value": 38.5,
                "suggested_value": 37.0,
                "change_amount": -1.5,
                "expected_risk_reduction": 0.20,
                "priority": "high",
                "actionability": "Administer acetaminophen 1000mg PO/IV",
                "time_to_effect": "1-2 hours",
                "shap_importance": 0.15
            }
        }


class InterventionPlan(BaseModel):
    """Complete intervention plan."""
    patient_id: str
    model_name: str
    
    current_risk: float
    target_risk: float
    
    suggestions: List[InterventionSuggestion] = Field(..., description="Ranked interventions")
    
    expected_final_risk: float = Field(..., description="Risk if all interventions applied")
    total_risk_reduction: float
    
    plausibility: PlausibilityLevel
    estimated_time: str = Field(..., description="Time to achieve target")
    
    clinical_summary: str
    safety_warnings: List[str] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P123",
                "model_name": "sepsis",
                "current_risk": 0.75,
                "target_risk": 0.30,
                "suggestions": [],
                "expected_final_risk": 0.32,
                "total_risk_reduction": 0.43,
                "plausibility": "realistic",
                "estimated_time": "4-6 hours",
                "clinical_summary": "Achievable with standard sepsis management",
                "safety_warnings": []
            }
        }


# ============================================================================
# WHAT-IF ENGINE (Main Class)
# ============================================================================

class WhatIfEngine:
    """
    What-If and Counterfactual Engine for medical AI.
    
    Explores alternative clinical scenarios with safety constraints.
    """
    
    def __init__(self):
        """Initialize What-If Engine."""
        self.models: Dict[str, Dict[str, Any]] = {}
        self.constraints: Dict[str, Dict[str, FeatureConstraint]] = {}
        self.scenario_history: List[ScenarioResult] = []
        
        print("[What-If Engine] Initialized")
    
    # ------------------------------------------------------------------------
    # STEP 1: MODEL REGISTRATION
    # ------------------------------------------------------------------------
    
    def register_model(
        self,
        model_name: str,
        model: Any,
        feature_names: List[str],
        clinical_constraints: Optional[Dict[str, FeatureConstraint]] = None,
        shap_explainer: Optional[Any] = None
    ):
        """
        Register a model for what-if analysis.
        
        Args:
            model_name: Unique model identifier
            model: Trained model
            feature_names: List of feature names
            clinical_constraints: Feature constraints (if None, use defaults)
            shap_explainer: Pre-initialized SHAP explainer (optional)
        """
        # Use default constraints if not provided
        if clinical_constraints is None:
            if 'sepsis' in model_name.lower():
                clinical_constraints = ClinicalConstraints.get_sepsis_constraints()
            elif 'kidney' in model_name.lower() or 'aki' in model_name.lower():
                clinical_constraints = ClinicalConstraints.get_aki_constraints()
            else:
                # Generic constraints
                clinical_constraints = {
                    fname: FeatureConstraint(
                        feature_name=fname,
                        feature_type=FeatureType.ACTIONABLE,
                        clinical_name=fname.replace('_', ' ').title()
                    ) for fname in feature_names
                }
        
        self.models[model_name] = {
            'model': model,
            'feature_names': feature_names,
            'shap_explainer': shap_explainer
        }
        
        self.constraints[model_name] = clinical_constraints
        
        print(f"[What-If Engine] Registered model: {model_name}")
        print(f"  Features: {len(feature_names)}")
        print(f"  Constraints: {len(clinical_constraints)}")
    
    # ------------------------------------------------------------------------
    # STEP 2: BASELINE ANALYSIS
    # ------------------------------------------------------------------------
    
    def analyze_baseline(
        self,
        model_name: str,
        patient_data: Union[np.ndarray, pd.DataFrame, Dict[str, float]],
        patient_id: str = "Unknown"
    ) -> BaselineAnalysis:
        """
        Analyze patient's baseline state.
        
        Args:
            model_name: Registered model name
            patient_data: Patient features
            patient_id: Patient identifier
        
        Returns:
            BaselineAnalysis with current risk and feature categorization
        """
        # Validate model exists
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        model_info = self.models[model_name]
        model = model_info['model']
        feature_names = model_info['feature_names']
        constraints = self.constraints[model_name]
        
        # Convert to feature dict
        if isinstance(patient_data, dict):
            feature_values = patient_data
        elif isinstance(patient_data, pd.DataFrame):
            feature_values = patient_data.iloc[0].to_dict()
        else:
            feature_values = dict(zip(feature_names, patient_data.flatten()))
        
        # Get current risk
        X = np.array([feature_values[f] for f in feature_names]).reshape(1, -1)
        if hasattr(model, 'predict_proba'):
            current_risk = float(model.predict_proba(X)[0, 1])
        else:
            current_risk = float(model.predict(X)[0])
        
        # Categorize features
        feature_types = {}
        modifiable_features = []
        fixed_features = []
        
        for fname in feature_names:
            constraint = constraints.get(fname)
            if constraint:
                ftype = constraint.feature_type
                feature_types[fname] = ftype.value
                
                if ftype == FeatureType.FIXED:
                    fixed_features.append(fname)
                else:
                    modifiable_features.append(fname)
            else:
                feature_types[fname] = "unknown"
                modifiable_features.append(fname)  # Assume modifiable if unknown
        
        # Identify abnormal features
        abnormal_features = {}
        for fname, value in feature_values.items():
            constraint = constraints.get(fname)
            if constraint and constraint.normal_range:
                normal_min, normal_max = constraint.normal_range
                if value < normal_min:
                    abnormal_features[fname] = {
                        'value': value,
                        'normal_range': list(constraint.normal_range),
                        'status': 'low',
                        'deviation': normal_min - value
                    }
                elif value > normal_max:
                    abnormal_features[fname] = {
                        'value': value,
                        'normal_range': list(constraint.normal_range),
                        'status': 'high',
                        'deviation': value - normal_max
                    }
        
        # Risk category
        risk_category = self._categorize_risk(current_risk)
        
        return BaselineAnalysis(
            patient_id=patient_id,
            model_name=model_name,
            current_risk=current_risk,
            risk_category=risk_category,
            feature_values=feature_values,
            feature_types=feature_types,
            modifiable_features=modifiable_features,
            fixed_features=fixed_features,
            abnormal_features=abnormal_features
        )
    
    # ------------------------------------------------------------------------
    # STEP 3: RUN WHAT-IF SCENARIO
    # ------------------------------------------------------------------------
    
    def run_scenario(
        self,
        model_name: str,
        baseline_data: Union[np.ndarray, pd.DataFrame, Dict[str, float]],
        interventions: Dict[str, float],
        patient_id: str = "Unknown",
        validate: bool = True
    ) -> ScenarioResult:
        """
        Run a what-if scenario with specified interventions.
        
        Algorithm:
        1. Load baseline data
        2. Apply interventions
        3. Validate constraints
        4. Compute new risk
        5. Calculate delta
        6. Assess plausibility
        
        Args:
            model_name: Registered model name
            baseline_data: Original patient data
            interventions: Dict of {feature: new_value}
            patient_id: Patient identifier
            validate: Whether to validate constraints
        
        Returns:
            ScenarioResult with risk delta and validation
        """
        model_info = self.models[model_name]
        model = model_info['model']
        feature_names = model_info['feature_names']
        constraints = self.constraints[model_name]
        
        # Convert baseline to dict
        if isinstance(baseline_data, dict):
            baseline_features = baseline_data.copy()
        elif isinstance(baseline_data, pd.DataFrame):
            baseline_features = baseline_data.iloc[0].to_dict()
        else:
            baseline_features = dict(zip(feature_names, baseline_data.flatten()))
        
        # Get baseline risk
        X_baseline = np.array([baseline_features[f] for f in feature_names]).reshape(1, -1)
        if hasattr(model, 'predict_proba'):
            baseline_risk = float(model.predict_proba(X_baseline)[0, 1])
        else:
            baseline_risk = float(model.predict(X_baseline)[0])
        
        # Apply interventions
        new_features = baseline_features.copy()
        intervention_details = {}
        constraint_violations = []
        
        for feature, new_value in interventions.items():
            if feature not in feature_names:
                constraint_violations.append(f"Feature '{feature}' not in model")
                continue
            
            original_value = baseline_features[feature]
            new_features[feature] = new_value
            intervention_details[feature] = (original_value, new_value)
            
            # Validate if requested
            if validate:
                constraint = constraints.get(feature)
                if constraint:
                    # Check value validity
                    is_valid, msg = constraint.validate(new_value)
                    if not is_valid:
                        constraint_violations.append(msg)
                    
                    # Check change validity
                    is_valid, msg = constraint.validate_change(original_value, new_value)
                    if not is_valid:
                        constraint_violations.append(msg)
        
        # Compute new risk
        X_new = np.array([new_features[f] for f in feature_names]).reshape(1, -1)
        if hasattr(model, 'predict_proba'):
            new_risk = float(model.predict_proba(X_new)[0, 1])
        else:
            new_risk = float(model.predict(X_new)[0])
        
        # Calculate delta
        risk_delta = new_risk - baseline_risk
        risk_delta_percent = (risk_delta / baseline_risk * 100) if baseline_risk > 0 else 0
        
        # Assess plausibility
        is_valid = len(constraint_violations) == 0
        plausibility = self._assess_plausibility(
            intervention_details,
            constraints,
            risk_delta
        )
        
        # Clinical summary
        clinical_summary = self._generate_scenario_summary(
            intervention_details,
            risk_delta,
            plausibility
        )
        
        # Intervention feasibility
        intervention_feasibility = {}
        for feature in intervention_details.keys():
            constraint = constraints.get(feature)
            if constraint:
                if constraint.feature_type == FeatureType.FIXED:
                    intervention_feasibility[feature] = "NON-ACTIONABLE: Fixed patient characteristic"
                elif constraint.feature_type == FeatureType.ACTIONABLE:
                    intervention_feasibility[feature] = "ACTIONABLE: Achievable with treatment"
                elif constraint.feature_type == FeatureType.SLOW:
                    intervention_feasibility[feature] = "SLOW: Requires time for effect"
                else:
                    intervention_feasibility[feature] = "DERIVED: Cannot directly modify"
        
        # Create result
        result = ScenarioResult(
            scenario_id=f"S{len(self.scenario_history)+1:03d}",
            patient_id=patient_id,
            model_name=model_name,
            baseline_risk=baseline_risk,
            baseline_features=baseline_features,
            new_risk=new_risk,
            new_features=new_features,
            interventions=intervention_details,
            risk_delta=risk_delta,
            risk_delta_percent=risk_delta_percent,
            is_valid=is_valid,
            plausibility=plausibility,
            constraint_violations=constraint_violations,
            clinical_summary=clinical_summary,
            intervention_feasibility=intervention_feasibility
        )
        
        # Save to history
        self.scenario_history.append(result)
        
        return result
    
    # ------------------------------------------------------------------------
    # STEP 4: SHAP-GUIDED INTERVENTION SUGGESTIONS
    # ------------------------------------------------------------------------
    
    def suggest_interventions(
        self,
        model_name: str,
        patient_data: Union[np.ndarray, pd.DataFrame, Dict[str, float]],
        target_risk: float = 0.30,
        max_suggestions: int = 5,
        patient_id: str = "Unknown"
    ) -> InterventionPlan:
        """
        Generate SHAP-guided intervention suggestions.
        
        Algorithm:
        1. Compute SHAP values for baseline
        2. Identify high-impact, actionable features
        3. Suggest realistic changes toward normal
        4. Rank by expected risk reduction
        5. Validate clinical plausibility
        
        Args:
            model_name: Registered model name
            patient_data: Patient features
            target_risk: Desired risk level
            max_suggestions: Maximum interventions to suggest
            patient_id: Patient identifier
        
        Returns:
            InterventionPlan with ranked suggestions
        """
        model_info = self.models[model_name]
        model = model_info['model']
        feature_names = model_info['feature_names']
        shap_explainer = model_info.get('shap_explainer')
        constraints = self.constraints[model_name]
        
        # Convert to array
        if isinstance(patient_data, dict):
            X = np.array([patient_data[f] for f in feature_names]).reshape(1, -1)
            feature_values = patient_data
        elif isinstance(patient_data, pd.DataFrame):
            X = patient_data.values
            feature_values = patient_data.iloc[0].to_dict()
        else:
            X = np.array(patient_data).reshape(1, -1)
            feature_values = dict(zip(feature_names, X.flatten()))
        
        # Get current risk
        if hasattr(model, 'predict_proba'):
            current_risk = float(model.predict_proba(X)[0, 1])
        else:
            current_risk = float(model.predict(X)[0])
        
        # Get SHAP values if available
        shap_values = None
        if shap_explainer is not None and SHAP_AVAILABLE:
            try:
                shap_values = shap_explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Positive class
                if shap_values.ndim > 1:
                    shap_values = shap_values[0]
            except Exception as e:
                warnings.warn(f"Could not compute SHAP values: {e}")
        
        # Generate suggestions
        suggestions = []
        
        for i, fname in enumerate(feature_names):
            constraint = constraints.get(fname)
            if not constraint:
                continue
            
            # Skip fixed features
            if constraint.feature_type == FeatureType.FIXED:
                continue
            
            current_value = feature_values[fname]
            
            # Suggest moving toward normal range
            if constraint.normal_range:
                normal_min, normal_max = constraint.normal_range
                normal_mid = (normal_min + normal_max) / 2
                
                # If outside normal, suggest normal midpoint
                if current_value < normal_min or current_value > normal_max:
                    suggested_value = normal_mid
                else:
                    # Already in normal range, skip
                    continue
            else:
                # No normal range defined, skip
                continue
            
            # Validate suggested change
            is_valid, _ = constraint.validate_change(current_value, suggested_value)
            if not is_valid:
                # Adjust to max allowed change
                if constraint.max_absolute_change:
                    if suggested_value > current_value:
                        suggested_value = current_value + constraint.max_absolute_change
                    else:
                        suggested_value = current_value - constraint.max_absolute_change
            
            # Estimate risk reduction (using SHAP if available)
            if shap_values is not None:
                shap_value = float(shap_values[i])
                # Approximate: risk_reduction ≈ shap_value * feature_change
                feature_change = suggested_value - current_value
                expected_reduction = abs(shap_value * feature_change / current_value) if current_value != 0 else 0
            else:
                # Without SHAP, use heuristic
                expected_reduction = 0.05  # Default small reduction
            
            # Determine priority
            if expected_reduction > 0.15:
                priority = InterventionPriority.CRITICAL
            elif expected_reduction > 0.08:
                priority = InterventionPriority.HIGH
            elif expected_reduction > 0.03:
                priority = InterventionPriority.MODERATE
            else:
                priority = InterventionPriority.LOW
            
            # Actionability and timing
            if constraint.feature_type == FeatureType.ACTIONABLE:
                actionability = self._get_actionability_text(fname, current_value, suggested_value)
                time_to_effect = "1-4 hours"
            elif constraint.feature_type == FeatureType.SLOW:
                actionability = f"Requires sustained treatment over days"
                time_to_effect = "12-48 hours"
            else:
                actionability = "Unknown intervention"
                time_to_effect = "Unknown"
            
            suggestions.append(InterventionSuggestion(
                feature_name=fname,
                clinical_name=constraint.clinical_name,
                current_value=current_value,
                suggested_value=suggested_value,
                change_amount=suggested_value - current_value,
                expected_risk_reduction=max(0, expected_reduction),
                priority=priority,
                actionability=actionability,
                time_to_effect=time_to_effect,
                shap_importance=float(shap_values[i]) if shap_values is not None else None
            ))
        
        # Sort by expected risk reduction (highest first)
        suggestions.sort(key=lambda x: x.expected_risk_reduction, reverse=True)
        suggestions = suggestions[:max_suggestions]
        
        # Estimate final risk if all applied
        test_interventions = {s.feature_name: s.suggested_value for s in suggestions}
        scenario = self.run_scenario(
            model_name=model_name,
            baseline_data=feature_values,
            interventions=test_interventions,
            patient_id=patient_id,
            validate=False  # Don't validate here
        )
        
        expected_final_risk = scenario.new_risk
        total_risk_reduction = current_risk - expected_final_risk
        
        # Assess plausibility
        plausibility = self._assess_plan_plausibility(suggestions, total_risk_reduction)
        
        # Estimated time (max of all interventions)
        time_estimates = []
        for s in suggestions:
            if 'hours' in s.time_to_effect:
                hours = int(s.time_to_effect.split('-')[1].split()[0])
                time_estimates.append(hours)
        estimated_hours = max(time_estimates) if time_estimates else 6
        estimated_time = f"{estimated_hours//2}-{estimated_hours} hours"
        
        # Clinical summary
        clinical_summary = self._generate_plan_summary(
            current_risk,
            expected_final_risk,
            suggestions,
            plausibility
        )
        
        # Safety warnings
        safety_warnings = []
        if len(suggestions) > 3:
            safety_warnings.append("⚠️ Multiple simultaneous interventions may interact")
        if plausibility == PlausibilityLevel.UNLIKELY:
            safety_warnings.append("⚠️ Target risk may be difficult to achieve")
        
        return InterventionPlan(
            patient_id=patient_id,
            model_name=model_name,
            current_risk=current_risk,
            target_risk=target_risk,
            suggestions=suggestions,
            expected_final_risk=expected_final_risk,
            total_risk_reduction=total_risk_reduction,
            plausibility=plausibility,
            estimated_time=estimated_time,
            clinical_summary=clinical_summary,
            safety_warnings=safety_warnings
        )
    
    # ------------------------------------------------------------------------
    # UTILITY METHODS
    # ------------------------------------------------------------------------
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize risk probability."""
        if probability < 0.25:
            return "Low"
        elif probability < 0.50:
            return "Moderate"
        elif probability < 0.75:
            return "High"
        else:
            return "Critical"
    
    def _assess_plausibility(
        self,
        interventions: Dict[str, Tuple[float, float]],
        constraints: Dict[str, FeatureConstraint],
        risk_delta: float
    ) -> PlausibilityLevel:
        """Assess clinical plausibility of scenario."""
        violation_count = 0
        
        for feature, (orig, new) in interventions.items():
            constraint = constraints.get(feature)
            if constraint:
                is_valid, _ = constraint.validate_change(orig, new)
                if not is_valid:
                    violation_count += 1
        
        # Assess based on violations and risk change
        if violation_count == 0:
            if abs(risk_delta) < 0.50:
                return PlausibilityLevel.REALISTIC
            else:
                return PlausibilityLevel.CHALLENGING
        elif violation_count <= 1:
            return PlausibilityLevel.CHALLENGING
        elif violation_count <= 2:
            return PlausibilityLevel.UNLIKELY
        else:
            return PlausibilityLevel.IMPOSSIBLE
    
    def _assess_plan_plausibility(
        self,
        suggestions: List[InterventionSuggestion],
        total_reduction: float
    ) -> PlausibilityLevel:
        """Assess plausibility of intervention plan."""
        if total_reduction < 0.20:
            return PlausibilityLevel.REALISTIC
        elif total_reduction < 0.40:
            return PlausibilityLevel.CHALLENGING
        elif total_reduction < 0.60:
            return PlausibilityLevel.UNLIKELY
        else:
            return PlausibilityLevel.IMPOSSIBLE
    
    def _generate_scenario_summary(
        self,
        interventions: Dict[str, Tuple[float, float]],
        risk_delta: float,
        plausibility: PlausibilityLevel
    ) -> str:
        """Generate clinical summary for scenario."""
        if risk_delta < -0.10:
            direction = "significantly decreases"
        elif risk_delta < 0:
            direction = "slightly decreases"
        elif risk_delta < 0.10:
            direction = "slightly increases"
        else:
            direction = "significantly increases"
        
        feature_list = ", ".join(list(interventions.keys())[:3])
        
        summary = f"Modifying {feature_list} {direction} risk by {abs(risk_delta)*100:.1f}%. "
        summary += f"Scenario is {plausibility.value}."
        
        return summary
    
    def _generate_plan_summary(
        self,
        current_risk: float,
        expected_risk: float,
        suggestions: List[InterventionSuggestion],
        plausibility: PlausibilityLevel
    ) -> str:
        """Generate clinical summary for intervention plan."""
        risk_reduction = (current_risk - expected_risk) * 100
        
        summary = f"Implementing {len(suggestions)} interventions could reduce risk "
        summary += f"from {current_risk*100:.1f}% to {expected_risk*100:.1f}% "
        summary += f"({risk_reduction:.1f}% reduction). "
        
        if plausibility == PlausibilityLevel.REALISTIC:
            summary += "Plan is clinically realistic with standard care."
        elif plausibility == PlausibilityLevel.CHALLENGING:
            summary += "Plan is achievable but requires intensive monitoring."
        elif plausibility == PlausibilityLevel.UNLIKELY:
            summary += "Plan may be difficult to achieve in practice."
        else:
            summary += "Plan exceeds clinical feasibility."
        
        return summary
    
    def _get_actionability_text(
        self,
        feature: str,
        current: float,
        target: float
    ) -> str:
        """Get specific actionability text for common features."""
        change = target - current
        
        if 'temperature' in feature.lower():
            if change < 0:
                return "Administer antipyretics (acetaminophen/ibuprofen)"
            else:
                return "Warm patient, investigate hypothermia"
        elif 'heart_rate' in feature.lower():
            if change < 0:
                return "Beta-blockers or calcium channel blockers"
            else:
                return "Investigate and treat underlying cause"
        elif 'bp' in feature.lower() or 'blood_pressure' in feature.lower():
            if change < 0:
                return "Antihypertensives (ACE-I, ARB, diuretics)"
            else:
                return "IV fluids, vasopressors if needed"
        elif 'lactate' in feature.lower():
            if change < 0:
                return "Fluid resuscitation, improve tissue perfusion"
            else:
                return "N/A - should reduce lactate"
        else:
            return f"Clinical intervention to modify {feature}"
    
    def get_history(self, patient_id: Optional[str] = None) -> List[ScenarioResult]:
        """Get scenario history, optionally filtered by patient."""
        if patient_id:
            return [s for s in self.scenario_history if s.patient_id == patient_id]
        return self.scenario_history


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """Comprehensive usage examples."""
    
    print("=" * 80)
    print("WHAT-IF ENGINE - USAGE EXAMPLES")
    print("=" * 80)
    
    # Setup
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=8, n_informative=5, random_state=42)
    feature_names = ['temperature', 'heart_rate', 'wbc_count', 'lactate', 'age', 
                     'respiratory_rate', 'systolic_bp', 'creatinine']
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Initialize engine
    engine = WhatIfEngine()
    
    # Register model with constraints
    engine.register_model(
        model_name="sepsis_predictor",
        model=model,
        feature_names=feature_names,
        clinical_constraints=ClinicalConstraints.get_sepsis_constraints()
    )
    
    # Patient data
    patient_features = {
        'temperature': 38.5,  # High
        'heart_rate': 110.0,  # High
        'wbc_count': 15000.0,  # High
        'lactate': 2.5,       # Slightly high
        'age': 65.0,
        'respiratory_rate': 22.0,  # Slightly high
        'systolic_bp': 95.0,  # Low
        'creatinine': 1.0
    }
    
    # Example 1: Baseline Analysis
    print("\n[Example 1] Baseline Analysis")
    print("-" * 80)
    
    baseline = engine.analyze_baseline(
        model_name="sepsis_predictor",
        patient_data=patient_features,
        patient_id="P12345"
    )
    
    print(f"\n🏥 Baseline Assessment:")
    print(f"  Patient: {baseline.patient_id}")
    print(f"  Current Risk: {baseline.current_risk:.1%} ({baseline.risk_category})")
    print(f"  Modifiable Features: {', '.join(baseline.modifiable_features)}")
    print(f"  Fixed Features: {', '.join(baseline.fixed_features)}")
    
    print(f"\n⚠️  Abnormal Features:")
    for feature, info in baseline.abnormal_features.items():
        print(f"  • {feature}: {info['value']:.1f} ({info['status']}) - "
              f"Normal: {info['normal_range']}")
    
    # Example 2: What-If Scenario
    print("\n[Example 2] What-If Scenario - Reduce Temperature")
    print("-" * 80)
    
    scenario = engine.run_scenario(
        model_name="sepsis_predictor",
        baseline_data=patient_features,
        interventions={
            'temperature': 37.0,  # Normalize temperature
            'heart_rate': 85.0    # Normalize heart rate
        },
        patient_id="P12345"
    )
    
    print(f"\n📊 Scenario Results:")
    print(f"  Baseline Risk: {scenario.baseline_risk:.1%}")
    print(f"  New Risk: {scenario.new_risk:.1%}")
    print(f"  Risk Delta: {scenario.risk_delta:+.1%} ({scenario.risk_delta_percent:+.1f}%)")
    print(f"  Valid: {scenario.is_valid}")
    print(f"  Plausibility: {scenario.plausibility.value}")
    
    print(f"\n💉 Interventions:")
    for feature, (orig, new) in scenario.interventions.items():
        print(f"  • {feature}: {orig:.1f} → {new:.1f}")
        print(f"    {scenario.intervention_feasibility[feature]}")
    
    print(f"\n💬 Clinical Summary:")
    print(f"  {scenario.clinical_summary}")
    
    # Example 3: Invalid Scenario
    print("\n[Example 3] Invalid Scenario - Constraint Violation")
    print("-" * 80)
    
    invalid_scenario = engine.run_scenario(
        model_name="sepsis_predictor",
        baseline_data=patient_features,
        interventions={
            'temperature': 32.0,  # Too low!
            'age': 70.0           # Cannot change age!
        },
        patient_id="P12345"
    )
    
    print(f"\n⚠️  Validation:")
    print(f"  Valid: {invalid_scenario.is_valid}")
    print(f"  Violations: {len(invalid_scenario.constraint_violations)}")
    for violation in invalid_scenario.constraint_violations:
        print(f"    • {violation}")
    
    # Example 4: SHAP-Guided Suggestions
    print("\n[Example 4] SHAP-Guided Intervention Suggestions")
    print("-" * 80)
    
    # Initialize SHAP explainer
    if SHAP_AVAILABLE:
        shap_explainer = shap.TreeExplainer(model)
        engine.models['sepsis_predictor']['shap_explainer'] = shap_explainer
    
    plan = engine.suggest_interventions(
        model_name="sepsis_predictor",
        patient_data=patient_features,
        target_risk=0.30,
        max_suggestions=5,
        patient_id="P12345"
    )
    
    print(f"\n🎯 Intervention Plan:")
    print(f"  Current Risk: {plan.current_risk:.1%}")
    print(f"  Target Risk: {plan.target_risk:.1%}")
    print(f"  Expected Final Risk: {plan.expected_final_risk:.1%}")
    print(f"  Total Reduction: {plan.total_risk_reduction:.1%}")
    print(f"  Plausibility: {plan.plausibility.value}")
    print(f"  Estimated Time: {plan.estimated_time}")
    
    print(f"\n📋 Suggested Interventions:")
    for i, suggestion in enumerate(plan.suggestions, 1):
        print(f"\n  {i}. {suggestion.clinical_name} ({suggestion.priority.value.upper()})")
        print(f"     Current: {suggestion.current_value:.2f}")
        print(f"     Target: {suggestion.suggested_value:.2f}")
        print(f"     Expected Risk ↓: {suggestion.expected_risk_reduction:.1%}")
        print(f"     How: {suggestion.actionability}")
        print(f"     Time: {suggestion.time_to_effect}")
        if suggestion.shap_importance:
            print(f"     SHAP: {suggestion.shap_importance:+.3f}")
    
    print(f"\n💬 Clinical Summary:")
    print(f"  {plan.clinical_summary}")
    
    if plan.safety_warnings:
        print(f"\n⚠️  Safety Warnings:")
        for warning in plan.safety_warnings:
            print(f"  {warning}")
    
    # Example 5: Scenario History
    print("\n[Example 5] Scenario History")
    print("-" * 80)
    
    history = engine.get_history(patient_id="P12345")
    print(f"\n📚 Scenarios for P12345: {len(history)}")
    for scenario in history:
        print(f"  • {scenario.scenario_id}: Risk {scenario.baseline_risk:.1%} → "
              f"{scenario.new_risk:.1%} ({scenario.plausibility.value})")
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
