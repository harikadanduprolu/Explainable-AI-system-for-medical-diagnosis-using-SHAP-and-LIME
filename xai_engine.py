"""
XAI Engine Module for Clinical Decision Support
================================================

A comprehensive explainable AI engine supporting multiple explanation methods:
- SHAP (TreeExplainer, KernelExplainer, DeepExplainer)
- LIME (Tabular, Image)
- Grad-CAM (CNN visualization)
- Counterfactual explanations (DiCE-style)
- Clinical translation
- Performance caching

Usage:
    engine = XAIEngine()
    
    # Register a trained model
    engine.register_model("sepsis", model, feature_names, data_type="tabular")
    
    # Get SHAP explanation
    explanation = engine.explain(
        model_name="sepsis",
        X=patient_features,
        method="shap",
        clinical_context={"patient_id": "P123", "age": 65}
    )
    
    # Generate counterfactuals
    counterfactuals = engine.generate_counterfactuals(
        model_name="sepsis",
        X=patient_features,
        desired_outcome=0,  # Change prediction to negative
        num_samples=5
    )
"""

from typing import Dict, List, Optional, Union, Any, Literal, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
import hashlib
import json
from datetime import datetime
from functools import lru_cache
import pickle
from pathlib import Path

# Optional imports with graceful fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

try:
    from lime.lime_tabular import LimeTabularExplainer
    from lime.lime_image import LimeImageExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not available. Install with: pip install lime")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# ENUMS AND TYPE DEFINITIONS
# ============================================================================

class ExplanationMethod(str, Enum):
    """Supported explanation methods."""
    SHAP_TREE = "shap_tree"
    SHAP_KERNEL = "shap_kernel"
    SHAP_DEEP = "shap_deep"
    LIME_TABULAR = "lime_tabular"
    LIME_IMAGE = "lime_image"
    GRADCAM = "gradcam"
    COUNTERFACTUAL = "counterfactual"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    CONTRASTIVE_ANCHOR = "contrastive_anchor"
    CONSTRAINED_RESPONSE_SURFACE = "constrained_response_surface"


class DataType(str, Enum):
    """Data types for explanation."""
    TABULAR = "tabular"
    IMAGE = "image"
    TIME_SERIES = "time_series"
    TEXT = "text"


class ExplanationLevel(str, Enum):
    """Granularity of explanation."""
    GLOBAL = "global"  # Overall model behavior
    LOCAL = "local"    # Single prediction
    COHORT = "cohort"  # Group of similar patients


# ============================================================================
# OUTPUT SCHEMAS (Pydantic Models)
# ============================================================================

class FeatureImportance(BaseModel):
    """Individual feature importance with clinical context."""
    feature_name: str = Field(..., description="Technical feature name")
    importance_score: float = Field(..., description="Importance value (SHAP/LIME)")
    feature_value: Optional[float] = Field(None, description="Actual feature value")
    baseline_value: Optional[float] = Field(None, description="Population baseline")
    percentile: Optional[float] = Field(None, ge=0, le=100, description="Patient percentile")
    clinical_label: str = Field(..., description="Clinician-readable name")
    clinical_interpretation: str = Field(..., description="What this means clinically")
    direction: Literal["increases", "decreases", "neutral"] = Field(..., description="Effect on risk")
    
    class Config:
        schema_extra = {
            "example": {
                "feature_name": "temperature",
                "importance_score": 0.15,
                "feature_value": 38.5,
                "baseline_value": 36.8,
                "percentile": 85.0,
                "clinical_label": "Body Temperature",
                "clinical_interpretation": "Elevated temperature (fever) strongly indicates infection",
                "direction": "increases"
            }
        }


class SHAPExplanation(BaseModel):
    """SHAP-specific explanation output."""
    method: Literal["shap_tree", "shap_kernel", "shap_deep"] = "shap_tree"
    base_value: float = Field(..., description="Expected model output (population baseline)")
    prediction_value: float = Field(..., description="Actual prediction for this patient")
    feature_contributions: List[FeatureImportance] = Field(..., description="SHAP values per feature")
    interaction_effects: Optional[Dict[str, Dict[str, float]]] = Field(
        None, 
        description="Feature interactions (if computed)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "method": "shap_tree",
                "base_value": 0.098,
                "prediction_value": 0.75,
                "feature_contributions": [
                    {
                        "feature_name": "temperature",
                        "importance_score": 0.15,
                        "clinical_label": "Body Temperature",
                        "clinical_interpretation": "Fever indicates infection",
                        "direction": "increases"
                    }
                ]
            }
        }


class LIMEExplanation(BaseModel):
    """LIME-specific explanation output."""
    method: Literal["lime_tabular", "lime_image"] = "lime_tabular"
    prediction_confidence: float = Field(..., ge=0, le=1, description="Model confidence in prediction")
    feature_contributions: List[FeatureImportance] = Field(..., description="LIME weights per feature")
    local_model_r2: float = Field(..., description="R² of linear approximation")
    intercept: float = Field(..., description="Local model intercept")
    
    class Config:
        schema_extra = {
            "example": {
                "method": "lime_tabular",
                "prediction_confidence": 0.89,
                "feature_contributions": [],
                "local_model_r2": 0.92,
                "intercept": 0.15
            }
        }


class GradCAMExplanation(BaseModel):
    """Grad-CAM explanation for CNN models."""
    method: Literal["gradcam"] = "gradcam"
    heatmap: List[List[float]] = Field(..., description="2D attention heatmap")
    layer_name: str = Field(..., description="CNN layer used for Grad-CAM")
    prediction_class: int = Field(..., description="Predicted class index")
    prediction_confidence: float = Field(..., ge=0, le=1)
    high_attention_regions: List[Dict[str, int]] = Field(
        ..., 
        description="Bounding boxes of important regions"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "method": "gradcam",
                "heatmap": [[0.1, 0.5, 0.9], [0.2, 0.8, 0.7]],
                "layer_name": "conv5_block3_out",
                "prediction_class": 1,
                "prediction_confidence": 0.87,
                "high_attention_regions": [
                    {"x": 50, "y": 100, "width": 80, "height": 60}
                ]
            }
        }


class CounterfactualExample(BaseModel):
    """A single counterfactual example."""
    original_prediction: float = Field(..., description="Original risk probability")
    counterfactual_prediction: float = Field(..., description="New risk probability")
    feature_changes: Dict[str, Tuple[float, float]] = Field(
        ..., 
        description="Feature → (original, modified) mapping"
    )
    clinical_actionability: Dict[str, str] = Field(
        ..., 
        description="Which changes are clinically actionable"
    )
    distance: float = Field(..., description="Distance from original (L2 norm)")
    
    class Config:
        schema_extra = {
            "example": {
                "original_prediction": 0.75,
                "counterfactual_prediction": 0.25,
                "feature_changes": {
                    "temperature": (38.5, 37.0),
                    "wbc_count": (15000, 9000)
                },
                "clinical_actionability": {
                    "temperature": "ACTIONABLE: Administer antipyretics",
                    "wbc_count": "NON-ACTIONABLE: Requires time/treatment"
                },
                "distance": 1.8
            }
        }


class CounterfactualExplanation(BaseModel):
    """Counterfactual explanation output."""
    method: Literal["counterfactual"] = "counterfactual"
    desired_outcome: Union[int, float] = Field(..., description="Target outcome value")
    original_prediction: float
    counterfactuals: List[CounterfactualExample] = Field(..., description="Alternative scenarios")
    feasibility_scores: List[float] = Field(..., description="Clinical feasibility [0-1]")
    
    class Config:
        schema_extra = {
            "example": {
                "method": "counterfactual",
                "desired_outcome": 0,
                "original_prediction": 0.75,
                "counterfactuals": [],
                "feasibility_scores": [0.8, 0.6, 0.3]
            }
        }


class ContrastiveAnchorFeature(FeatureImportance):
    """Feature-level output for contrastive anchor explanations."""
    anchor_value: float = Field(..., description="Reference value used as the local anchor")
    anchor_prediction: float = Field(..., description="Prediction at the anchor value")
    raw_delta: float = Field(..., description="Signed change from current prediction to anchor prediction")
    stability_score: float = Field(..., ge=0, le=1, description="Agreement of perturbation signs")
    sensitivity_score: float = Field(..., ge=0, description="Average local sensitivity magnitude")
    anchor_distance: float = Field(..., ge=0, description="Normalized distance from current value to anchor")
    context_support: float = Field(..., ge=0, le=1, description="Fraction of context samples supporting the same direction")
    absolute_low: float = Field(..., description="Clinical absolute lower bound")
    absolute_high: float = Field(..., description="Clinical absolute upper bound")
    preferred_low: float = Field(..., description="Preferred clinical lower bound")
    preferred_high: float = Field(..., description="Preferred clinical upper bound")
    plausibility_score: float = Field(..., ge=0, le=1, description="Clinical plausibility of the anchor")


class ContrastiveAnchorExplanation(BaseModel):
    """Model-agnostic contrastive anchor explanation output."""
    method: Literal["contrastive_anchor"] = "contrastive_anchor"
    prediction_value: float = Field(..., description="Prediction for the original patient")
    baseline_prediction: float = Field(..., description="Mean prediction across the local background context")
    feature_contributions: List[ContrastiveAnchorFeature] = Field(..., description="Feature-level contrastive anchor scores")
    anchor_strategy: str = Field(..., description="How anchor values were selected")
    local_context_size: int = Field(..., description="Number of perturbed context samples used")
    confidence: float = Field(..., ge=0, le=1, description="Reliability of the local explanation")

    class Config:
        schema_extra = {
            "example": {
                "method": "contrastive_anchor",
                "prediction_value": 0.74,
                "baseline_prediction": 0.31,
                "feature_contributions": [],
                "anchor_strategy": "median-path local perturbation",
                "local_context_size": 32,
                "confidence": 0.84
            }
        }


class ResponseSurfacePoint(BaseModel):
    """One sampled point on a feature response curve."""
    feature_value: float
    predicted_risk: float


class ResponseSurfaceFeature(FeatureImportance):
    """Feature-level output for constrained response-surface explanations."""
    response_area: float = Field(..., ge=0, description="Average absolute risk displacement across sampled curve")
    peak_risk: float = Field(..., ge=0, le=1)
    trough_risk: float = Field(..., ge=0, le=1)
    elasticity_up: float = Field(..., description="Average local slope moving feature upward")
    elasticity_down: float = Field(..., description="Average local slope moving feature downward")
    monotonicity_score: float = Field(..., ge=0, le=1, description="Sign consistency of local slopes")
    nonlinearity_score: float = Field(..., ge=0, le=1, description="Curvature intensity of local response")
    absolute_low: float = Field(...)
    absolute_high: float = Field(...)
    preferred_low: float = Field(...)
    preferred_high: float = Field(...)
    sampled_points: List[ResponseSurfacePoint] = Field(default_factory=list)


class ResponseSurfaceExplanation(BaseModel):
    """Model-agnostic constrained response-surface explanation output."""
    method: Literal["constrained_response_surface"] = "constrained_response_surface"
    prediction_value: float = Field(..., ge=0, le=1)
    baseline_prediction: float = Field(..., ge=0, le=1)
    feature_contributions: List[ResponseSurfaceFeature] = Field(default_factory=list)
    sampling_steps: int = Field(..., ge=5, le=25)
    local_context_size: int = Field(..., ge=4)
    strategy: str
    confidence: float = Field(..., ge=0, le=1)

    class Config:
        schema_extra = {
            "example": {
                "method": "constrained_response_surface",
                "prediction_value": 0.71,
                "baseline_prediction": 0.29,
                "feature_contributions": [],
                "sampling_steps": 11,
                "local_context_size": 32,
                "strategy": "clinically bounded univariate response-curve sampling",
                "confidence": 0.82
            }
        }


class ClinicalExplanation(BaseModel):
    """Unified clinician-readable explanation."""
    patient_id: str
    model_name: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Prediction details
    predicted_risk: float = Field(..., ge=0, le=1, description="Risk probability")
    risk_category: Literal["Low", "Moderate", "High", "Critical"]
    confidence: float = Field(..., ge=0, le=1)
    
    # Explanation
    explanation_method: ExplanationMethod
    top_risk_factors: List[FeatureImportance] = Field(..., description="Top contributing factors")
    protective_factors: List[FeatureImportance] = Field(..., description="Factors reducing risk")
    
    # Clinical summary
    clinical_summary: str = Field(..., description="Plain English explanation")
    clinical_recommendations: List[str] = Field(..., description="Suggested actions")
    
    # Detailed explanations (method-specific)
    shap_details: Optional[SHAPExplanation] = None
    lime_details: Optional[LIMEExplanation] = None
    gradcam_details: Optional[GradCAMExplanation] = None
    counterfactual_details: Optional[CounterfactualExplanation] = None
    contrastive_anchor_details: Optional[ContrastiveAnchorExplanation] = None
    response_surface_details: Optional[ResponseSurfaceExplanation] = None
    
    # Metadata
    computation_time_ms: float
    cached: bool = Field(default=False, description="Was this retrieved from cache?")
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P12345",
                "model_name": "sepsis_predictor",
                "predicted_risk": 0.75,
                "risk_category": "High",
                "confidence": 0.89,
                "explanation_method": "shap_tree",
                "top_risk_factors": [],
                "protective_factors": [],
                "clinical_summary": "Patient shows high sepsis risk due to fever and elevated WBC",
                "clinical_recommendations": ["Monitor vitals every 2 hours", "Consider broad-spectrum antibiotics"],
                "computation_time_ms": 45.2,
                "cached": False
            }
        }


# ============================================================================
# CLINICAL TRANSLATOR
# ============================================================================

class ClinicalTranslator:
    """
    Translates technical ML features into clinician-readable language.
    """
    
    # Default clinical mappings (can be extended per-disease)
    CLINICAL_MAPPINGS = {
        # Vital signs
        'temperature': {
            'label': 'Body Temperature',
            'unit': '°C',
            'normal_range': (36.1, 37.2),
            'interpretation': {
                'high': 'Fever indicates possible infection or inflammatory response',
                'low': 'Hypothermia may indicate shock or severe infection',
                'normal': 'Normal temperature'
            }
        },
        'heart_rate': {
            'label': 'Heart Rate',
            'unit': 'bpm',
            'normal_range': (60, 100),
            'interpretation': {
                'high': 'Tachycardia suggests stress, pain, or cardiovascular compromise',
                'low': 'Bradycardia may indicate heart block or medication effect',
                'normal': 'Normal heart rate'
            }
        },
        'respiratory_rate': {
            'label': 'Respiratory Rate',
            'unit': 'breaths/min',
            'normal_range': (12, 20),
            'interpretation': {
                'high': 'Tachypnea indicates respiratory distress or metabolic acidosis',
                'low': 'Hypoventilation may indicate CNS depression',
                'normal': 'Normal breathing rate'
            }
        },
        'systolic_bp': {
            'label': 'Systolic Blood Pressure',
            'unit': 'mmHg',
            'normal_range': (90, 140),
            'interpretation': {
                'high': 'Hypertension increases cardiovascular risk',
                'low': 'Hypotension indicates shock or poor perfusion',
                'normal': 'Normal blood pressure'
            }
        },
        
        # Lab values
        'wbc_count': {
            'label': 'White Blood Cell Count',
            'unit': 'cells/μL',
            'normal_range': (4000, 11000),
            'interpretation': {
                'high': 'Leukocytosis suggests infection or inflammation',
                'low': 'Leukopenia may indicate immunosuppression',
                'normal': 'Normal white cell count'
            }
        },
        'creatinine': {
            'label': 'Serum Creatinine',
            'unit': 'mg/dL',
            'normal_range': (0.6, 1.2),
            'interpretation': {
                'high': 'Elevated creatinine indicates kidney dysfunction',
                'low': 'Low creatinine may indicate muscle loss',
                'normal': 'Normal kidney function'
            }
        },
        'lactate': {
            'label': 'Blood Lactate',
            'unit': 'mmol/L',
            'normal_range': (0.5, 2.0),
            'interpretation': {
                'high': 'Elevated lactate suggests tissue hypoperfusion or shock',
                'low': 'Normal lactate',
                'normal': 'Normal tissue perfusion'
            }
        },
        'glucose': {
            'label': 'Blood Glucose',
            'unit': 'mg/dL',
            'normal_range': (70, 140),
            'interpretation': {
                'high': 'Hyperglycemia increases infection risk and complications',
                'low': 'Hypoglycemia is immediately life-threatening',
                'normal': 'Normal glucose level'
            }
        },
        
        # Demographics
        'age': {
            'label': 'Patient Age',
            'unit': 'years',
            'normal_range': None,
            'interpretation': {
                'high': 'Advanced age increases risk for most acute conditions',
                'low': 'Younger age generally more resilient',
                'normal': 'Age is a significant risk factor'
            }
        }
    }
    
    @classmethod
    def translate_feature(
        cls, 
        feature_name: str, 
        value: float,
        importance: float
    ) -> Dict[str, Any]:
        """
        Translate a technical feature to clinical language.
        
        Args:
            feature_name: Technical name (e.g., 'temperature')
            value: Feature value
            importance: SHAP/LIME importance score
        
        Returns:
            Dict with clinical_label, interpretation, direction
        """
        mapping = cls.CLINICAL_MAPPINGS.get(feature_name, {
            'label': feature_name.replace('_', ' ').title(),
            'unit': '',
            'normal_range': None,
            'interpretation': {'normal': f'Clinical factor: {feature_name}'}
        })
        
        # Determine if value is high/low/normal
        normal_range = mapping.get('normal_range')
        if normal_range:
            if value < normal_range[0]:
                status = 'low'
            elif value > normal_range[1]:
                status = 'high'
            else:
                status = 'normal'
        else:
            status = 'normal'
        
        # Direction based on SHAP value
        if importance > 0.01:
            direction = "increases"
        elif importance < -0.01:
            direction = "decreases"
        else:
            direction = "neutral"
        
        return {
            'clinical_label': mapping['label'],
            'unit': mapping.get('unit', ''),
            'status': status,
            'interpretation': mapping['interpretation'].get(status, 'Unknown clinical significance'),
            'direction': direction
        }
    
    @classmethod
    def generate_summary(
        cls,
        top_factors: List[FeatureImportance],
        risk_level: str
    ) -> str:
        """Generate plain English clinical summary."""
        if not top_factors:
            return f"Patient classified as {risk_level} risk based on overall clinical profile."
        
        # Build summary
        main_factors = [f.clinical_label for f in top_factors[:3]]
        
        if risk_level in ["High", "Critical"]:
            summary = f"Patient shows {risk_level.lower()} risk primarily due to: {', '.join(main_factors)}. "
            summary += f"Key concern: {top_factors[0].clinical_interpretation}"
        else:
            summary = f"{risk_level} risk profile. Notable factors include: {', '.join(main_factors)}."
        
        return summary
    
    @classmethod
    def generate_recommendations(
        cls,
        top_factors: List[FeatureImportance],
        risk_level: str,
        disease: str = "condition"
    ) -> List[str]:
        """Generate clinical action recommendations."""
        recommendations = []
        
        if risk_level == "Critical":
            recommendations.append(f"🚨 URGENT: Immediate medical attention required")
            recommendations.append("Consider ICU admission and specialist consultation")
        elif risk_level == "High":
            recommendations.append("Close monitoring recommended (vitals every 2-4 hours)")
            recommendations.append(f"Consider early intervention to prevent {disease} progression")
        elif risk_level == "Moderate":
            recommendations.append("Regular monitoring and reassessment in 4-8 hours")
        else:
            recommendations.append("Continue standard care protocol")
        
        # Feature-specific recommendations
        for factor in top_factors[:2]:
            if 'fever' in factor.clinical_interpretation.lower():
                recommendations.append("→ Administer antipyretics; investigate infection source")
            elif 'kidney' in factor.clinical_interpretation.lower():
                recommendations.append("→ Monitor renal function; ensure adequate hydration")
            elif 'shock' in factor.clinical_interpretation.lower():
                recommendations.append("→ Fluid resuscitation; consider vasopressor support")
        
        return recommendations


CONTRASTIVE_ANCHOR_LIMITS: Dict[str, Tuple[float, float]] = {
    "age": (18.0, 100.0),
    "gender": (0.0, 1.0),
    "temperature": (95.0, 106.0),
    "heart_rate": (40.0, 200.0),
    "systolic_bp": (70.0, 250.0),
    "diastolic_bp": (40.0, 150.0),
    "respiratory_rate": (8.0, 50.0),
    "wbc_count": (1.0, 50.0),
    "hemoglobin": (5.0, 20.0),
    "platelet_count": (20.0, 700.0),
    "creatinine": (0.3, 15.0),
    "bun": (5.0, 200.0),
    "glucose": (50.0, 700.0),
    "lactate": (0.5, 25.0),
}


# ============================================================================
# EXPLANATION CACHE
# ============================================================================

class ExplanationCache:
    """
    LRU cache for explanations to avoid redundant computation.
    """
    
    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 3600):
        """
        Args:
            maxsize: Maximum number of cached explanations
            ttl_seconds: Time-to-live for cache entries (default 1 hour)
        """
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}
    
    def _generate_key(
        self, 
        model_name: str,
        X: np.ndarray,
        method: str,
        **kwargs
    ) -> str:
        """Generate unique cache key from inputs."""
        # Hash the input data
        X_hash = hashlib.md5(X.tobytes()).hexdigest()
        
        # Include kwargs in key
        kwargs_str = json.dumps(kwargs, sort_keys=True)
        
        full_key = f"{model_name}:{method}:{X_hash}:{kwargs_str}"
        return hashlib.md5(full_key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve from cache if exists and not expired."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if (datetime.now().timestamp() - timestamp) < self.ttl_seconds:
                return value
            else:
                # Expired, remove
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Store in cache with timestamp."""
        # Evict oldest if at capacity
        if len(self.cache) >= self.maxsize:
            oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
            del self.cache[oldest_key]
        
        self.cache[key] = (value, datetime.now().timestamp())
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()


# ============================================================================
# XAI ENGINE (Main Class)
# ============================================================================

class XAIEngine:
    """
    Comprehensive XAI Engine for clinical decision support.
    
    Supports:
    - SHAP (Tree, Kernel, Deep)
    - LIME (Tabular, Image)
    - Grad-CAM (CNN visualization)
    - Counterfactual generation
    - Clinical translation
    - Performance caching
    """
    
    def __init__(
        self,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        enable_cache: bool = True
    ):
        """
        Initialize XAI Engine.
        
        Args:
            cache_size: Maximum cached explanations
            cache_ttl: Cache time-to-live in seconds
            enable_cache: Whether to use caching
        """
        self.models: Dict[str, Dict[str, Any]] = {}
        self.explainers: Dict[str, Dict[str, Any]] = {}
        self.enable_cache = enable_cache
        self.cache = ExplanationCache(maxsize=cache_size, ttl_seconds=cache_ttl)
        self.translator = ClinicalTranslator()
        
        print(f"[XAI Engine] Initialized with cache_size={cache_size}, ttl={cache_ttl}s")
    
    # ------------------------------------------------------------------------
    # MODEL REGISTRATION
    # ------------------------------------------------------------------------
    
    def register_model(
        self,
        model_name: str,
        model: Any,
        feature_names: List[str],
        data_type: DataType = DataType.TABULAR,
        background_data: Optional[np.ndarray] = None,
        disease_name: Optional[str] = None
    ):
        """
        Register a trained model for explanation.
        
        Args:
            model_name: Unique identifier for this model
            model: Trained ML model (sklearn, xgboost, keras, etc.)
            feature_names: List of feature names
            data_type: Type of input data
            background_data: Background dataset for SHAP KernelExplainer
            disease_name: Clinical disease name (for translation)
        """
        self.models[model_name] = {
            'model': model,
            'feature_names': feature_names,
            'data_type': data_type,
            'disease_name': disease_name or model_name,
            'background_data': background_data
        }
        
        # Pre-initialize explainers
        self._initialize_explainers(model_name)
        
        print(f"[XAI Engine] Registered model: {model_name} ({data_type.value})")
    
    def _initialize_explainers(self, model_name: str):
        """Pre-initialize SHAP and LIME explainers for a model."""
        model_info = self.models[model_name]
        model = model_info['model']
        data_type = model_info['data_type']
        
        self.explainers[model_name] = {}
        
        # SHAP Tree Explainer (for tree-based models)
        if SHAP_AVAILABLE:
            try:
                if hasattr(model, 'tree_') or hasattr(model, 'estimators_') or \
                   model.__class__.__name__ in ['XGBClassifier', 'XGBRegressor', 'LGBMClassifier']:
                    self.explainers[model_name]['shap_tree'] = shap.TreeExplainer(model)
                    print(f"  ✓ SHAP TreeExplainer initialized")
            except Exception as e:
                print(f"  ✗ Could not init TreeExplainer: {e}")
            
            # SHAP Kernel Explainer (model-agnostic, slower)
            try:
                background = model_info.get('background_data')
                if background is not None:
                    predict_fn = lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x)
                    self.explainers[model_name]['shap_kernel'] = shap.KernelExplainer(
                        predict_fn,
                        background[:100]  # Use sample for efficiency
                    )
                    print(f"  ✓ SHAP KernelExplainer initialized")
            except Exception as e:
                print(f"  ✗ Could not init KernelExplainer: {e}")
        
        # LIME Explainer
        if LIME_AVAILABLE and data_type == DataType.TABULAR:
            try:
                background = model_info.get('background_data')
                if background is not None:
                    self.explainers[model_name]['lime_tabular'] = LimeTabularExplainer(
                        background,
                        feature_names=model_info['feature_names'],
                        mode='classification'
                    )
                    print(f"  ✓ LIME TabularExplainer initialized")
            except Exception as e:
                print(f"  ✗ Could not init LIME: {e}")
    
    # ------------------------------------------------------------------------
    # MAIN EXPLANATION METHOD
    # ------------------------------------------------------------------------
    
    def explain(
        self,
        model_name: str,
        X: Union[np.ndarray, pd.DataFrame],
        method: ExplanationMethod = ExplanationMethod.SHAP_TREE,
        clinical_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        use_cache: bool = True
    ) -> ClinicalExplanation:
        """
        Generate explanation for a prediction.
        
        Args:
            model_name: Name of registered model
            X: Input features (single sample)
            method: Explanation method to use
            clinical_context: Additional context (patient_id, age, etc.)
            top_k: Number of top features to include
            use_cache: Whether to use cached results
        
        Returns:
            ClinicalExplanation with clinician-readable output
        """
        import time
        start_time = time.time()
        
        # Validate model exists
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered. Use register_model() first.")
        
        model_info = self.models[model_name]
        model = model_info['model']
        feature_names = model_info['feature_names']
        
        # Convert input to numpy
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
        
        # Check cache
        cached_result = None
        if use_cache and self.enable_cache:
            cache_key = self.cache._generate_key(model_name, X_array, method.value)
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                cached_result.cached = True
                cached_result.computation_time_ms = (time.time() - start_time) * 1000
                print(f"[XAI Engine] Cache hit for {model_name}")
                return cached_result
        
        # Get prediction
        if hasattr(model, 'predict_proba'):
            prediction = float(model.predict_proba(X_array)[0, 1])
        else:
            prediction = float(model.predict(X_array)[0])
        
        # Generate method-specific explanation
        if method in [ExplanationMethod.SHAP_TREE, ExplanationMethod.SHAP_KERNEL]:
            explanation_details = self._explain_shap(model_name, X_array, method)
        elif method == ExplanationMethod.LIME_TABULAR:
            explanation_details = self._explain_lime(model_name, X_array)
        elif method == ExplanationMethod.GRADCAM:
            explanation_details = self._explain_gradcam(model_name, X_array, clinical_context or {})
        elif method == ExplanationMethod.CONTRASTIVE_ANCHOR:
            explanation_details = self._explain_contrastive_anchor(model_name, X_array)
        elif method == ExplanationMethod.CONSTRAINED_RESPONSE_SURFACE:
            explanation_details = self._explain_constrained_response_surface(model_name, X_array)
        else:
            raise NotImplementedError(f"Method {method} not yet implemented")
        
        # Extract feature importances
        if isinstance(explanation_details, SHAPExplanation):
            feature_importances = explanation_details.feature_contributions
            shap_details = explanation_details
            lime_details = None
            gradcam_details = None
        elif isinstance(explanation_details, LIMEExplanation):
            feature_importances = explanation_details.feature_contributions
            shap_details = None
            lime_details = explanation_details
            gradcam_details = None
        elif isinstance(explanation_details, GradCAMExplanation):
            feature_importances = []
            shap_details = None
            lime_details = None
            gradcam_details = explanation_details
        elif isinstance(explanation_details, ContrastiveAnchorExplanation):
            feature_importances = list(explanation_details.feature_contributions)
            shap_details = None
            lime_details = None
            gradcam_details = None
        elif isinstance(explanation_details, ResponseSurfaceExplanation):
            feature_importances = list(explanation_details.feature_contributions)
            shap_details = None
            lime_details = None
            gradcam_details = None
        else:
            feature_importances = []
            shap_details = None
            lime_details = None
            gradcam_details = None
        
        # Sort by importance
        feature_importances.sort(key=lambda x: abs(x.importance_score), reverse=True)
        
        # Separate risk vs protective factors
        top_risk_factors = [f for f in feature_importances if f.direction == "increases"][:top_k]
        protective_factors = [f for f in feature_importances if f.direction == "decreases"][:top_k]
        
        # Determine risk category
        risk_category = self._categorize_risk(prediction)
        
        # Generate clinical summary
        clinical_summary = self.translator.generate_summary(top_risk_factors, risk_category)
        clinical_recommendations = self.translator.generate_recommendations(
            top_risk_factors, 
            risk_category,
            model_info['disease_name']
        )
        
        # Compute confidence (distance from decision boundary)
        confidence = 2 * abs(prediction - 0.5)
        
        # Build final explanation
        explanation = ClinicalExplanation(
            patient_id=clinical_context.get('patient_id', 'Unknown') if clinical_context else 'Unknown',
            model_name=model_name,
            predicted_risk=prediction,
            risk_category=risk_category,
            confidence=confidence,
            explanation_method=method,
            top_risk_factors=top_risk_factors,
            protective_factors=protective_factors,
            clinical_summary=clinical_summary,
            clinical_recommendations=clinical_recommendations,
            shap_details=shap_details,
            lime_details=lime_details,
            gradcam_details=gradcam_details,
            contrastive_anchor_details=explanation_details if isinstance(explanation_details, ContrastiveAnchorExplanation) else None,
            response_surface_details=explanation_details if isinstance(explanation_details, ResponseSurfaceExplanation) else None,
            computation_time_ms=(time.time() - start_time) * 1000,
            cached=False
        )
        
        # Cache result
        if use_cache and self.enable_cache:
            self.cache.set(cache_key, explanation)
        
        return explanation
    
    # ------------------------------------------------------------------------
    # METHOD-SPECIFIC EXPLAINERS
    # ------------------------------------------------------------------------
    
    def _explain_shap(
        self, 
        model_name: str, 
        X: np.ndarray,
        method: ExplanationMethod
    ) -> SHAPExplanation:
        """Generate SHAP explanation."""
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP not installed. Install with: pip install shap")
        
        explainer_key = 'shap_tree' if method == ExplanationMethod.SHAP_TREE else 'shap_kernel'
        explainer = self.explainers[model_name].get(explainer_key)
        
        if explainer is None:
            raise RuntimeError(f"SHAP explainer not initialized for {model_name}")
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X)
        
        # Handle multi-output (binary classification)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        if shap_values.ndim > 1:
            shap_values = shap_values[0]
        
        # Get base value
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]
        
        # Get prediction
        prediction_value = base_value + shap_values.sum()
        
        # Build feature contributions
        model_info = self.models[model_name]
        feature_names = model_info['feature_names']
        
        feature_contributions = []
        for i, (fname, shap_val) in enumerate(zip(feature_names, shap_values)):
            feature_value = float(X[0, i])
            
            # Translate to clinical
            translation = self.translator.translate_feature(fname, feature_value, shap_val)
            
            feature_contributions.append(FeatureImportance(
                feature_name=fname,
                importance_score=float(shap_val),
                feature_value=feature_value,
                baseline_value=None,  # Could compute from background
                percentile=None,
                clinical_label=translation['clinical_label'],
                clinical_interpretation=translation['interpretation'],
                direction=translation['direction']
            ))
        
        return SHAPExplanation(
            method=explainer_key,
            base_value=float(base_value),
            prediction_value=float(prediction_value),
            feature_contributions=feature_contributions,
            interaction_effects=None
        )
    
    def _explain_lime(self, model_name: str, X: np.ndarray) -> LIMEExplanation:
        """Generate LIME explanation."""
        if not LIME_AVAILABLE:
            raise RuntimeError("LIME not installed. Install with: pip install lime")
        
        explainer = self.explainers[model_name].get('lime_tabular')
        if explainer is None:
            raise RuntimeError(f"LIME explainer not initialized for {model_name}")
        
        model = self.models[model_name]['model']
        feature_names = self.models[model_name]['feature_names']
        
        # Generate explanation
        exp = explainer.explain_instance(
            X[0],
            model.predict_proba,
            num_features=len(feature_names)
        )
        
        # Extract feature weights
        feature_contributions = []
        for fname, weight in exp.as_list():
            # Parse feature name (LIME returns "feature <= value" strings)
            clean_fname = fname.split()[0] if ' ' in fname else fname
            
            # Find feature value
            if clean_fname in feature_names:
                idx = feature_names.index(clean_fname)
                feature_value = float(X[0, idx])
            else:
                feature_value = None
            
            # Translate
            translation = self.translator.translate_feature(clean_fname, feature_value or 0, weight)
            
            feature_contributions.append(FeatureImportance(
                feature_name=clean_fname,
                importance_score=float(weight),
                feature_value=feature_value,
                baseline_value=None,
                percentile=None,
                clinical_label=translation['clinical_label'],
                clinical_interpretation=translation['interpretation'],
                direction=translation['direction']
            ))
        
        return LIMEExplanation(
            method="lime_tabular",
            prediction_confidence=float(exp.predict_proba[1]),
            feature_contributions=feature_contributions,
            local_model_r2=float(exp.score),
            intercept=float(exp.intercept[1])
        )
    
    def _explain_gradcam(
        self, 
        model_name: str, 
        X: np.ndarray,
        context: Dict
    ) -> GradCAMExplanation:
        """Generate Grad-CAM explanation (placeholder for CNN models)."""
        # This is a simplified placeholder - full implementation requires CNN model
        return GradCAMExplanation(
            method="gradcam",
            heatmap=[[0.5]],
            layer_name="conv_layer",
            prediction_class=1,
            prediction_confidence=0.85,
            high_attention_regions=[]
        )

    def _predict_scores(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Return model scores as a 1D numpy array, preferring positive-class probability."""
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(X), dtype=float)
            if proba.ndim == 1:
                return proba.reshape(-1)
            if proba.shape[1] == 1:
                return proba[:, 0].reshape(-1)
            return proba[:, 1].reshape(-1)

        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(X), dtype=float).reshape(-1)
            clipped = np.clip(scores, -50.0, 50.0)
            return 1.0 / (1.0 + np.exp(-clipped))

        return np.asarray(model.predict(X), dtype=float).reshape(-1)

    def _explain_contrastive_anchor(self, model_name: str, X: np.ndarray) -> ContrastiveAnchorExplanation:
        """Generate a model-agnostic contrastive anchor explanation."""
        model_info = self.models[model_name]
        model = model_info['model']
        feature_names = model_info['feature_names']

        background = model_info.get('background_data')
        if isinstance(background, pd.DataFrame):
            background_array = background.values
        elif background is not None:
            background_array = np.asarray(background)
        else:
            background_array = None

        rng = np.random.default_rng()
        context_size = 32

        if background_array is None or background_array.size == 0:
            feature_scale = np.maximum(np.abs(X[0]), 1.0)
            background_array = np.repeat(X, repeats=context_size, axis=0).astype(float)
            background_array = background_array + rng.normal(
                loc=0.0,
                scale=0.05 * feature_scale,
                size=background_array.shape
            )

        if background_array.ndim == 1:
            background_array = background_array.reshape(1, -1)

        local_size = min(context_size, max(8, background_array.shape[0]))
        if background_array.shape[0] >= local_size:
            context_indices = rng.choice(background_array.shape[0], size=local_size, replace=False)
        else:
            context_indices = rng.choice(background_array.shape[0], size=local_size, replace=True)
        local_context = np.asarray(background_array[context_indices], dtype=float)

        prediction_value = float(self._predict_scores(model, X)[0])
        baseline_prediction = float(np.mean(self._predict_scores(model, local_context)))

        feature_contributions: List[ContrastiveAnchorFeature] = []
        stability_values: List[float] = []
        prediction_gaps: List[float] = []

        for idx, feature_name in enumerate(feature_names):
            current_value = float(X[0, idx])
            reference_values = np.asarray(background_array[:, idx], dtype=float)
            if reference_values.size == 0:
                reference_values = np.array([current_value], dtype=float)

            absolute_low, absolute_high = CONTRASTIVE_ANCHOR_LIMITS.get(feature_name, (float(np.min(reference_values)), float(np.max(reference_values))))
            preferred = ClinicalTranslator.CLINICAL_MAPPINGS.get(feature_name, {}).get('normal_range')
            if preferred:
                preferred_low, preferred_high = float(preferred[0]), float(preferred[1])
            else:
                preferred_low, preferred_high = absolute_low, absolute_high

            anchor_value = float(np.median(reference_values))
            anchor_value = float(np.clip(anchor_value, absolute_low, absolute_high))
            if anchor_value < preferred_low:
                anchor_value = preferred_low
            elif anchor_value > preferred_high:
                anchor_value = preferred_high

            lower_quantile = float(np.percentile(reference_values, 25))
            upper_quantile = float(np.percentile(reference_values, 75))
            spread = float(np.std(reference_values))
            iqr = float(upper_quantile - lower_quantile)
            scale = max(spread, iqr, abs(current_value - anchor_value), 1e-6)

            path_points = np.linspace(current_value, anchor_value, num=9)
            path_predictions: List[float] = []

            for path_value in path_points:
                perturbed_context = local_context.copy()
                perturbed_context[:, idx] = np.clip(path_value, absolute_low, absolute_high)
                path_predictions.append(float(np.mean(self._predict_scores(model, perturbed_context))))

            path_predictions_array = np.asarray(path_predictions, dtype=float)
            anchor_index = int(np.argmin(np.abs(path_predictions_array - baseline_prediction)))
            anchor_prediction = float(path_predictions_array[anchor_index])
            selected_anchor_value = float(path_points[anchor_index])

            raw_delta = float(prediction_value - anchor_prediction)
            local_deltas = path_predictions_array - prediction_value
            sign_mask = np.sign(local_deltas[np.abs(local_deltas) > 1e-9])
            if sign_mask.size == 0:
                stability_score = 1.0
                context_support = 1.0
            else:
                reference_sign = np.sign(raw_delta) if abs(raw_delta) > 1e-9 else np.sign(np.mean(local_deltas))
                stability_score = float(np.mean(sign_mask == reference_sign))
                context_support = stability_score

            sensitivity_score = float(np.mean(np.abs(local_deltas)) / scale)
            anchor_distance = float(abs(current_value - selected_anchor_value) / scale)
            weighted_importance = float(raw_delta * (0.5 + 0.5 * stability_score))
            if preferred_low <= selected_anchor_value <= preferred_high:
                plausibility_score = 1.0
            else:
                distance_from_preferred = min(abs(selected_anchor_value - preferred_low), abs(selected_anchor_value - preferred_high))
                plausibility_score = float(np.clip(1.0 - distance_from_preferred / max(absolute_high - absolute_low, 1e-6), 0.0, 1.0))

            translation = self.translator.translate_feature(feature_name, current_value, weighted_importance)
            percentile = float(np.mean(reference_values <= current_value) * 100.0)

            feature_contributions.append(ContrastiveAnchorFeature(
                feature_name=feature_name,
                importance_score=weighted_importance,
                feature_value=current_value,
                baseline_value=float(np.median(reference_values)),
                percentile=percentile,
                clinical_label=translation['clinical_label'],
                clinical_interpretation=translation['interpretation'],
                direction=translation['direction'],
                anchor_value=selected_anchor_value,
                anchor_prediction=anchor_prediction,
                raw_delta=raw_delta,
                stability_score=max(0.0, min(1.0, stability_score)),
                sensitivity_score=max(0.0, sensitivity_score),
                anchor_distance=max(0.0, anchor_distance),
                context_support=max(0.0, min(1.0, context_support))
                ,absolute_low=float(absolute_low),
                absolute_high=float(absolute_high),
                preferred_low=float(preferred_low),
                preferred_high=float(preferred_high),
                plausibility_score=max(0.0, min(1.0, plausibility_score))
            ))

            stability_values.append(max(0.0, min(1.0, stability_score)))
            prediction_gaps.append(abs(prediction_value - baseline_prediction))

        feature_contributions.sort(key=lambda item: abs(item.importance_score), reverse=True)

        confidence = float(np.clip(
            0.45 * np.mean(stability_values) +
            0.25 * (1.0 - min(1.0, float(np.std(prediction_gaps)))) +
            0.20 * (1.0 - min(1.0, abs(prediction_value - baseline_prediction))) +
            0.10 * float(np.mean([item.plausibility_score for item in feature_contributions])),
            0.0,
            1.0
        ))

        return ContrastiveAnchorExplanation(
            prediction_value=prediction_value,
            baseline_prediction=baseline_prediction,
            feature_contributions=feature_contributions,
            anchor_strategy="median-path local perturbation against background context",
            local_context_size=local_size,
            confidence=confidence
        )

    def _explain_constrained_response_surface(self, model_name: str, X: np.ndarray) -> ResponseSurfaceExplanation:
        """Generate a clinically bounded, model-agnostic local response-surface explanation."""
        model_info = self.models[model_name]
        model = model_info['model']
        feature_names = model_info['feature_names']

        background = model_info.get('background_data')
        if isinstance(background, pd.DataFrame):
            background_array = background.values
        elif background is not None:
            background_array = np.asarray(background)
        else:
            background_array = None

        rng = np.random.default_rng()
        context_size = 32
        sampling_steps = 11

        if background_array is None or background_array.size == 0:
            feature_scale = np.maximum(np.abs(X[0]), 1.0)
            background_array = np.repeat(X, repeats=context_size, axis=0).astype(float)
            background_array = background_array + rng.normal(
                loc=0.0,
                scale=0.04 * feature_scale,
                size=background_array.shape
            )

        if background_array.ndim == 1:
            background_array = background_array.reshape(1, -1)

        local_size = min(context_size, max(8, background_array.shape[0]))
        if background_array.shape[0] >= local_size:
            context_indices = rng.choice(background_array.shape[0], size=local_size, replace=False)
        else:
            context_indices = rng.choice(background_array.shape[0], size=local_size, replace=True)
        local_context = np.asarray(background_array[context_indices], dtype=float)

        prediction_value = float(self._predict_scores(model, X)[0])
        baseline_prediction = float(np.mean(self._predict_scores(model, local_context)))

        contributions: List[ResponseSurfaceFeature] = []
        monotonicity_scores: List[float] = []
        nonlinearity_scores: List[float] = []

        for idx, feature_name in enumerate(feature_names):
            current_value = float(X[0, idx])
            ref_values = np.asarray(background_array[:, idx], dtype=float)
            if ref_values.size == 0:
                ref_values = np.array([current_value], dtype=float)

            absolute_low, absolute_high = CONTRASTIVE_ANCHOR_LIMITS.get(
                feature_name,
                (float(np.min(ref_values)), float(np.max(ref_values)))
            )

            preferred = ClinicalTranslator.CLINICAL_MAPPINGS.get(feature_name, {}).get('normal_range')
            if preferred is not None:
                preferred_low, preferred_high = float(preferred[0]), float(preferred[1])
            else:
                preferred_low, preferred_high = absolute_low, absolute_high

            preferred_low = max(absolute_low, preferred_low)
            preferred_high = min(absolute_high, preferred_high)

            low_bound = max(absolute_low, min(current_value, preferred_low))
            high_bound = min(absolute_high, max(current_value, preferred_high))
            if high_bound - low_bound < 1e-6:
                span = max((absolute_high - absolute_low) * 0.15, 1e-3)
                low_bound = max(absolute_low, current_value - span)
                high_bound = min(absolute_high, current_value + span)

            grid = np.linspace(low_bound, high_bound, num=sampling_steps)
            sampled_risks: List[float] = []

            for value in grid:
                perturbed_context = local_context.copy()
                perturbed_context[:, idx] = value
                sampled_risks.append(float(np.mean(self._predict_scores(model, perturbed_context))))

            sampled_risks_arr = np.asarray(sampled_risks, dtype=float)
            deltas = sampled_risks_arr - prediction_value

            denom = max(high_bound - low_bound, 1e-6)
            response_area = float(np.trapz(np.abs(deltas), x=grid) / denom)

            first_diff = np.diff(sampled_risks_arr)
            if first_diff.size == 0:
                monotonicity = 1.0
                nonlinearity = 0.0
            else:
                dominant_sign = np.sign(np.mean(first_diff))
                if dominant_sign == 0:
                    dominant_sign = 1.0
                monotonicity = float(np.mean(np.sign(first_diff) == dominant_sign))
                second_diff = np.diff(first_diff)
                slope_scale = max(np.mean(np.abs(first_diff)), 1e-6)
                nonlinearity = float(np.clip(np.mean(np.abs(second_diff)) / slope_scale, 0.0, 1.0)) if second_diff.size > 0 else 0.0

            target_value = float(np.clip((preferred_low + preferred_high) / 2.0, absolute_low, absolute_high))
            target_context = local_context.copy()
            target_context[:, idx] = target_value
            target_prediction = float(np.mean(self._predict_scores(model, target_context)))

            signed_effect = float(prediction_value - target_prediction)
            weighted_importance = float(signed_effect * (0.50 + 0.35 * monotonicity + 0.15 * (1.0 - nonlinearity)))

            current_idx = int(np.argmin(np.abs(grid - current_value)))
            up_slopes = np.diff(sampled_risks_arr[current_idx:]) / np.maximum(np.diff(grid[current_idx:]), 1e-6)
            down_slopes = np.diff(sampled_risks_arr[:current_idx + 1]) / np.maximum(np.diff(grid[:current_idx + 1]), 1e-6)

            elasticity_up = float(np.mean(up_slopes)) if up_slopes.size > 0 else 0.0
            elasticity_down = float(np.mean(down_slopes)) if down_slopes.size > 0 else 0.0

            translation = self.translator.translate_feature(feature_name, current_value, weighted_importance)
            percentile = float(np.mean(ref_values <= current_value) * 100.0)

            sampled_points = [
                ResponseSurfacePoint(feature_value=float(v), predicted_risk=float(r))
                for v, r in zip(grid, sampled_risks_arr)
            ]

            contributions.append(ResponseSurfaceFeature(
                feature_name=feature_name,
                importance_score=weighted_importance,
                feature_value=current_value,
                baseline_value=float(np.median(ref_values)),
                percentile=percentile,
                clinical_label=translation['clinical_label'],
                clinical_interpretation=translation['interpretation'],
                direction=translation['direction'],
                response_area=max(0.0, response_area),
                peak_risk=float(np.max(sampled_risks_arr)),
                trough_risk=float(np.min(sampled_risks_arr)),
                elasticity_up=elasticity_up,
                elasticity_down=elasticity_down,
                monotonicity_score=max(0.0, min(1.0, monotonicity)),
                nonlinearity_score=max(0.0, min(1.0, nonlinearity)),
                absolute_low=float(absolute_low),
                absolute_high=float(absolute_high),
                preferred_low=float(preferred_low),
                preferred_high=float(preferred_high),
                sampled_points=sampled_points,
            ))

            monotonicity_scores.append(max(0.0, min(1.0, monotonicity)))
            nonlinearity_scores.append(max(0.0, min(1.0, nonlinearity)))

        contributions.sort(key=lambda item: abs(item.importance_score), reverse=True)

        confidence = float(np.clip(
            0.45 * (float(np.mean(monotonicity_scores)) if monotonicity_scores else 0.0) +
            0.30 * (1.0 - (float(np.mean(nonlinearity_scores)) if nonlinearity_scores else 0.0)) +
            0.25 * (1.0 - min(1.0, abs(prediction_value - baseline_prediction))),
            0.0,
            1.0,
        ))

        return ResponseSurfaceExplanation(
            prediction_value=prediction_value,
            baseline_prediction=baseline_prediction,
            feature_contributions=contributions,
            sampling_steps=sampling_steps,
            local_context_size=local_size,
            strategy="clinically bounded univariate response-curve sampling",
            confidence=confidence,
        )
    
    # ------------------------------------------------------------------------
    # COUNTERFACTUAL GENERATION
    # ------------------------------------------------------------------------
    
    def generate_counterfactuals(
        self,
        model_name: str,
        X: Union[np.ndarray, pd.DataFrame],
        desired_outcome: Union[int, float],
        num_samples: int = 5,
        max_iterations: int = 100,
        step_size: float = 0.1
    ) -> CounterfactualExplanation:
        """
        Generate counterfactual explanations using gradient-based search.
        
        Args:
            model_name: Name of registered model
            X: Original input features
            desired_outcome: Target prediction value
            num_samples: Number of counterfactuals to generate
            max_iterations: Maximum optimization iterations
            step_size: Learning rate for gradient descent
        
        Returns:
            CounterfactualExplanation with alternative scenarios
        """
        model_info = self.models[model_name]
        model = model_info['model']
        feature_names = model_info['feature_names']
        
        # Convert input
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
        
        # Get original prediction
        if hasattr(model, 'predict_proba'):
            original_pred = float(model.predict_proba(X_array)[0, 1])
        else:
            original_pred = float(model.predict(X_array)[0])
        
        counterfactuals = []
        
        # Simple random search for counterfactuals (can be improved with optimization)
        for _ in range(num_samples):
            X_modified = X_array.copy()
            
            # Randomly perturb features
            perturbation = np.random.normal(0, 0.5, X_array.shape)
            X_modified += perturbation
            
            # Get new prediction
            if hasattr(model, 'predict_proba'):
                new_pred = float(model.predict_proba(X_modified)[0, 1])
            else:
                new_pred = float(model.predict(X_modified)[0])
            
            # Calculate feature changes
            feature_changes = {}
            clinical_actionability = {}
            
            for i, fname in enumerate(feature_names):
                orig_val = float(X_array[0, i])
                new_val = float(X_modified[0, i])
                
                if abs(new_val - orig_val) > 0.01:
                    feature_changes[fname] = (orig_val, new_val)
                    
                    # Assess actionability
                    if fname in ['age', 'gender']:
                        clinical_actionability[fname] = "NON-ACTIONABLE: Fixed patient characteristic"
                    elif fname in ['temperature', 'heart_rate', 'blood_pressure']:
                        clinical_actionability[fname] = "ACTIONABLE: Can be modified with treatment"
                    else:
                        clinical_actionability[fname] = "PARTIALLY ACTIONABLE: Requires time/treatment"
            
            distance = float(np.linalg.norm(X_modified - X_array))
            
            counterfactuals.append(CounterfactualExample(
                original_prediction=original_pred,
                counterfactual_prediction=new_pred,
                feature_changes=feature_changes,
                clinical_actionability=clinical_actionability,
                distance=distance
            ))
        
        # Sort by distance (most feasible first)
        counterfactuals.sort(key=lambda x: x.distance)
        
        # Compute feasibility scores
        feasibility_scores = [1.0 / (1.0 + cf.distance) for cf in counterfactuals]
        
        return CounterfactualExplanation(
            method="counterfactual",
            desired_outcome=desired_outcome,
            original_prediction=original_pred,
            counterfactuals=counterfactuals,
            feasibility_scores=feasibility_scores
        )
    
    # ------------------------------------------------------------------------
    # UTILITY METHODS
    # ------------------------------------------------------------------------
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize risk probability into clinical levels."""
        if probability < 0.25:
            return "Low"
        elif probability < 0.50:
            return "Moderate"
        elif probability < 0.75:
            return "High"
        else:
            return "Critical"
    
    def clear_cache(self):
        """Clear all cached explanations."""
        self.cache.clear()
        print("[XAI Engine] Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'size': len(self.cache.cache),
            'maxsize': self.cache.maxsize,
            'ttl_seconds': self.cache.ttl_seconds
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """Comprehensive usage examples."""
    
    print("=" * 80)
    print("XAI ENGINE - USAGE EXAMPLES")
    print("=" * 80)
    
    # Setup
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate synthetic medical data
    X, y = make_classification(n_samples=1000, n_features=8, n_informative=5, random_state=42)
    feature_names = ['temperature', 'heart_rate', 'wbc_count', 'lactate', 'age', 
                     'respiratory_rate', 'systolic_bp', 'creatinine']
    
    # Train a model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Initialize XAI Engine
    engine = XAIEngine(cache_size=100, enable_cache=True)
    
    # Register model
    engine.register_model(
        model_name="sepsis_predictor",
        model=model,
        feature_names=feature_names,
        data_type=DataType.TABULAR,
        background_data=X[:100],
        disease_name="Sepsis"
    )
    
    # Example 1: SHAP Explanation
    print("\n[Example 1] SHAP Tree Explanation")
    print("-" * 80)
    
    patient_data = X[0:1]
    explanation = engine.explain(
        model_name="sepsis_predictor",
        X=patient_data,
        method=ExplanationMethod.SHAP_TREE,
        clinical_context={'patient_id': 'P12345', 'age': 65},
        top_k=3
    )
    
    print(f"\n🏥 Clinical Summary:")
    print(f"  Patient: {explanation.patient_id}")
    print(f"  Risk: {explanation.predicted_risk:.1%} ({explanation.risk_category})")
    print(f"  Confidence: {explanation.confidence:.1%}")
    print(f"  Method: {explanation.explanation_method.value}")
    print(f"  Computation Time: {explanation.computation_time_ms:.1f}ms")
    
    print(f"\n📊 Top Risk Factors:")
    for i, factor in enumerate(explanation.top_risk_factors, 1):
        print(f"  {i}. {factor.clinical_label} = {factor.feature_value:.2f}")
        print(f"     Impact: {factor.importance_score:+.3f} ({factor.direction} risk)")
        print(f"     → {factor.clinical_interpretation}")
    
    print(f"\n💬 Clinical Summary:")
    print(f"  {explanation.clinical_summary}")
    
    print(f"\n📋 Recommendations:")
    for rec in explanation.clinical_recommendations:
        print(f"  {rec}")
    
    # Example 2: LIME Explanation (with caching)
    print("\n[Example 2] LIME Explanation (Testing Cache)")
    print("-" * 80)
    
    explanation_lime = engine.explain(
        model_name="sepsis_predictor",
        X=patient_data,
        method=ExplanationMethod.LIME_TABULAR,
        clinical_context={'patient_id': 'P12345'},
        top_k=3
    )
    
    print(f"  Risk: {explanation_lime.predicted_risk:.1%}")
    print(f"  Cached: {explanation_lime.cached}")
    print(f"  Computation Time: {explanation_lime.computation_time_ms:.1f}ms")
    
    # Call again to test cache
    explanation_lime2 = engine.explain(
        model_name="sepsis_predictor",
        X=patient_data,
        method=ExplanationMethod.LIME_TABULAR,
        clinical_context={'patient_id': 'P12345'},
        top_k=3
    )
    print(f"  Second call - Cached: {explanation_lime2.cached}")
    print(f"  Second call - Time: {explanation_lime2.computation_time_ms:.1f}ms")

    # Example 2.5: Contrastive Anchor Explanation
    print("\n[Example 2.5] Contrastive Anchor Explanation")
    print("-" * 80)

    explanation_anchor = engine.explain(
        model_name="sepsis_predictor",
        X=patient_data,
        method=ExplanationMethod.CONTRASTIVE_ANCHOR,
        clinical_context={'patient_id': 'P12345'},
        top_k=3
    )

    print(f"  Risk: {explanation_anchor.predicted_risk:.1%}")
    print(f"  Confidence: {explanation_anchor.confidence:.1%}")
    print(f"  Method: {explanation_anchor.explanation_method.value}")
    print(f"  Contrastive Detail Confidence: {explanation_anchor.contrastive_anchor_details.confidence:.1%}")
    for factor in explanation_anchor.top_risk_factors[:3]:
        print(f"    • {factor.clinical_label}: {factor.importance_score:+.3f} -> anchor {factor.feature_value:.2f} to {getattr(factor, 'anchor_value', factor.feature_value):.2f}")

    # Example 2.6: Constrained Response Surface Explanation
    print("\n[Example 2.6] Constrained Response Surface Explanation")
    print("-" * 80)

    explanation_surface = engine.explain(
        model_name="sepsis_predictor",
        X=patient_data,
        method=ExplanationMethod.CONSTRAINED_RESPONSE_SURFACE,
        clinical_context={'patient_id': 'P12345'},
        top_k=3
    )

    print(f"  Risk: {explanation_surface.predicted_risk:.1%}")
    print(f"  Method: {explanation_surface.explanation_method.value}")
    print(f"  Surface Detail Confidence: {explanation_surface.response_surface_details.confidence:.1%}")
    for factor in explanation_surface.top_risk_factors[:3]:
        print(
            f"    • {factor.clinical_label}: {factor.importance_score:+.3f} | "
            f"response area {getattr(factor, 'response_area', 0.0):.4f} | "
            f"nonlinearity {getattr(factor, 'nonlinearity_score', 0.0):.2f}"
        )
    
    # Example 3: Counterfactual Generation
    print("\n[Example 3] Counterfactual Explanations")
    print("-" * 80)
    
    counterfactuals = engine.generate_counterfactuals(
        model_name="sepsis_predictor",
        X=patient_data,
        desired_outcome=0,  # Want negative prediction
        num_samples=3
    )
    
    print(f"\n  Original Prediction: {counterfactuals.original_prediction:.1%}")
    print(f"  Desired Outcome: {counterfactuals.desired_outcome}")
    print(f"\n  Alternative Scenarios:")
    
    for i, cf in enumerate(counterfactuals.counterfactuals, 1):
        print(f"\n  Scenario {i}:")
        print(f"    New Prediction: {cf.counterfactual_prediction:.1%}")
        print(f"    Feasibility: {counterfactuals.feasibility_scores[i-1]:.1%}")
        print(f"    Changes Required:")
        for feat, (orig, new) in list(cf.feature_changes.items())[:3]:
            actionable = cf.clinical_actionability.get(feat, "Unknown")
            print(f"      • {feat}: {orig:.2f} → {new:.2f}")
            print(f"        {actionable}")
    
    # Example 4: Cache Statistics
    print("\n[Example 4] Cache Statistics")
    print("-" * 80)
    
    stats = engine.get_cache_stats()
    print(f"  Cache Size: {stats['size']} / {stats['maxsize']}")
    print(f"  TTL: {stats['ttl_seconds']}s")
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
