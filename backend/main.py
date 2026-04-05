"""
FastAPI Backend for Explainable Medical AI System
==================================================

Complete REST API for disease prediction with SHAP/LIME explanations.

Usage:
    uvicorn backend.main:app --reload --port 8000
    
Access:
    API: http://localhost:8000
    Docs: http://localhost:8000/docs
    Frontend: http://localhost:8000/app
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import os
from dotenv import load_dotenv

# Compatibility shim: some serialized bundles reference numpy._core modules
# (NumPy 2.x internal path) while this runtime may expose numpy.core.
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
if "numpy._core.multiarray" not in sys.modules and hasattr(np.core, "multiarray"):
    sys.modules["numpy._core.multiarray"] = np.core.multiarray

from feature_engineering import BASE_FEATURES, add_derived_features
from multimodal_fusion import MultimodalFusionEngine
from backend.clinical_recommendation_engine import get_recommendation_engine
from audit_logging import AuditLogger, AuditEventType
from model_registry import ModelRegistry
from alert_engine import (
    AlertEngine,
    DEFAULT_ALERT_POLICIES,
    RegulatoryViolationError as AlertRegulatoryViolationError,
)
from governed_whatif_engine import (
    GovernedWhatIfEngine,
    FeatureConstraint,
    FeatureType,
    PlausibilityLevel,
    RegulatoryViolationError as GovernedRegulatoryViolationError,
)

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None

try:
    from lime import lime_tabular
except ImportError:  # pragma: no cover - optional dependency
    lime_tabular = None

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration from environment
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_API_DOCS = os.getenv("ENABLE_API_DOCS", "true").lower() == "true"
CXR_LABELS_PATH = os.getenv(
    "CXR_LABELS_PATH",
    str(Path("dataset") / "mimiccxr" / "mimic-cxr-2.0.0-chexpert.csv.gz"),
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info(f"Starting application in {APP_ENV} mode")
logger.info(f"CORS origins: {CORS_ORIGINS}")

# Import authentication router (after logger is initialized)
AUTH_ENABLED = False
try:
    from backend.auth import router as auth_router
    from backend.database import connect_to_mongodb, close_mongodb_connection
    AUTH_ENABLED = True
    logger.info("✅ Authentication module loaded")
except ImportError as e:
    logger.warning(f"⚠️ Authentication disabled: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Explainable Medical AI API",
    description="Multi-disease prediction with SHAP & LIME explanations",
    version="2.0.0",
    docs_url="/docs" if ENABLE_API_DOCS else None,
    redoc_url="/redoc" if ENABLE_API_DOCS else None,
    debug=DEBUG
)

# Security: Trusted Host Middleware (prevent host header attacks)
if APP_ENV == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure with your domain
    )

# CORS middleware with environment-based configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600,
)

# ============================================================================
# APPLICATION LIFECYCLE EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("🚀 Application startup...")
    
    # Connect to MongoDB
    if AUTH_ENABLED:
        try:
            await connect_to_mongodb()
            logger.info("✅ MongoDB connected")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            logger.warning("⚠️ Running without authentication")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Application shutdown...")
    
    # Close MongoDB connection
    if AUTH_ENABLED:
        try:
            await close_mongodb_connection()
        except Exception as e:
            logger.error(f"Error closing MongoDB: {e}")

# ============================================================================
# INCLUDE ROUTERS
# ============================================================================

# Authentication routes
if AUTH_ENABLED:
    app.include_router(auth_router)
    logger.info("✅ Authentication routes enabled")

# Static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@dataclass
class _SimpleDecisionTrace:
    """Minimal trace payload needed by governance engines."""
    trace_id: str
    patient_id: str
    disease: str
    model_name: str
    model_version: str
    input_summary: Dict[str, Any]
    prediction_event: Any = None
    explanation_event: Any = None
    alert_event: Any = None


class _BackendGovernanceContext:
    """Adapter exposing the context methods expected by governance engines."""

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    def log_alert(
        self,
        prediction_event: Any,
        explanation_event: Any,
        threshold: float,
        payload: Dict[str, Any],
        human_message: str,
    ):
        patient_id = payload.get("patient_id") or getattr(prediction_event, "patient_id", None)
        disease = payload.get("disease") or getattr(prediction_event, "disease", None)
        return self.audit_logger.log_event(
            event_type=AuditEventType.ALERT,
            patient_id=patient_id,
            disease=disease,
            prediction_id=getattr(prediction_event, "event_id", None),
            explanation_id=getattr(explanation_event, "event_id", None),
            threshold=threshold,
            payload=payload,
            human_message=human_message,
        )


def _build_explanation_summary(prediction: "DiseasePrediction") -> str:
    top = prediction.shap_top_features or prediction.top_features
    if not top:
        return f"{prediction.disease} risk score is {prediction.risk_score:.2f}"
    pieces = [
        f"{feat.feature_name.replace('_', ' ')}={feat.value:.2f} (impact {feat.importance:+.3f})"
        for feat in top[:3]
    ]
    return "; ".join(pieces)


def _trace_id(patient_id: Optional[str], disease: str) -> str:
    safe_pid = patient_id or "anonymous"
    return f"trace-{safe_pid}-{disease}-{int(datetime.now().timestamp() * 1000)}"


ALERT_POLICY_BY_DISEASE: Dict[str, str] = {
    "sepsis": "Sepsis",
    "kidney_failure": "Acute Kidney Injury",
    "mortality": "Mortality Risk",
}

# ============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================================

class PatientFeatures(BaseModel):
    """Patient clinical features."""
    age: float = Field(..., ge=18, le=100, description="Age in years")
    gender: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    heart_rate: float = Field(..., ge=40, le=200, description="Heart rate (bpm)")
    systolic_bp: float = Field(..., ge=70, le=250, description="Systolic BP (mmHg)")
    diastolic_bp: float = Field(..., ge=40, le=150, description="Diastolic BP (mmHg)")
    temperature: float = Field(..., ge=95, le=106, description="Temperature (°F)")
    respiratory_rate: float = Field(..., ge=8, le=50, description="Respiratory rate")
    wbc_count: float = Field(..., ge=1, le=50, description="WBC count (K/µL)")
    hemoglobin: float = Field(..., ge=5, le=20, description="Hemoglobin (g/dL)")
    platelet_count: float = Field(..., ge=20, le=700, description="Platelet count (K/µL)")
    creatinine: float = Field(..., ge=0.3, le=15, description="Creatinine (mg/dL)")
    bun: float = Field(..., ge=5, le=200, description="BUN (mg/dL)")
    glucose: float = Field(..., ge=50, le=700, description="Glucose (mg/dL)")
    lactate: float = Field(..., ge=0.5, le=25, description="Lactate (mmol/L)")


class ImagingEvidence(BaseModel):
    """Optional imaging findings derived from MIMIC-CXR or CheXpert pipeline."""
    dicom_id: Optional[str] = Field(None, description="MIMIC-CXR DICOM identifier")
    source: Optional[str] = Field(
        default="manual-entry",
        description="Where the imaging probabilities originated (manual-entry, auto-inference, repository)",
    )
    findings: Optional[Dict[str, float]] = Field(
        default=None,
        description="Normalized CheXpert-style findings (0-1) keyed by label name",
    )


class PredictionRequest(BaseModel):
    """Request for disease prediction."""
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    features: PatientFeatures
    diseases: Optional[List[str]] = Field(None, description="Specific diseases to predict")
    imaging: Optional[ImagingEvidence] = Field(
        default=None,
        description="Optional imaging evidence (CXR) linked to the patient",
    )


class FeatureImportance(BaseModel):
    """Feature importance from model."""
    feature_name: str
    importance: float
    value: float


class DiseasePrediction(BaseModel):
    """Single disease prediction result."""
    disease: str
    risk_score: float
    risk_category: str
    prediction: int
    model_type: str
    threshold: float
    top_features: List[FeatureImportance] = Field(default_factory=list)
    shap_top_features: List[FeatureImportance] = Field(default_factory=list)
    lime_top_features: List[FeatureImportance] = Field(default_factory=list)
    explanation_methods: List[str] = Field(default_factory=list)
    explanation_warnings: List[str] = Field(default_factory=list)
    clinical_decision_support_report: Optional[str] = None


class FusedPrediction(BaseModel):
    """Fused output that blends tabular, imaging, and severity evidence."""
    disease: str
    tabular_risk: float
    fused_score: float
    agreement_index: float
    severity_modifier: Optional[float] = None
    imaging_signal: Optional[float] = None
    governance_flag: Optional[str] = None


class MultimodalSummary(BaseModel):
    """Summary of fusion process for governance/patent claims."""
    consistency_index: float
    fused_predictions: List[FusedPrediction]
    alerts: List[str]
    imaging_channels_used: List[str]
    severity_channels: Dict[str, float]
    data_sources: Dict[str, str]


class PredictionResponse(BaseModel):
    """Complete prediction response."""
    patient_id: Optional[str]
    timestamp:str
    predictions: List[DiseasePrediction]
    overall_risk_category: str
    multimodal_summary: Optional[MultimodalSummary] = None
    clinical_recommendations: Optional[Dict[str, Any]] = None
    alerts: List[Dict[str, Any]] = Field(default_factory=list)


class WhatIfRequest(BaseModel):
    """What-if scenario analysis request."""
    baseline_features: Dict[str, Any]
    modified_features: Dict[str, float]
    disease: str


class WhatIfResponse(BaseModel):
    """What-if analysis response."""
    disease: str
    baseline_risk: float
    new_risk: float
    risk_delta: float
    risk_delta_percent: float
    modified_features: Dict[str, Dict[str, float]]
    recommendation: str
    recommended_changes: List[Dict[str, Any]] = Field(default_factory=list)
    clinical_summary: Optional[str] = None
    counterfactual_explanations: List[str] = Field(default_factory=list)


class ContrastiveAnchorRequest(BaseModel):
    """Request for contrastive anchor explanation."""
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    baseline_features: PatientFeatures
    disease: str = Field(..., description="Disease to explain")
    focus_features: Optional[List[str]] = Field(
        default=None,
        description="Optional ordered list of features to prioritize in the explanation",
    )
    top_k: int = Field(default=5, ge=1, le=8, description="Number of feature explanations to return")
    context_samples: int = Field(
        default=24,
        ge=8,
        le=64,
        description="Number of local plausibility samples used around each anchor",
    )


class ContrastiveAnchorFeatureResult(BaseModel):
    """Single feature result for contrastive anchor explanation."""
    feature_name: str
    current_value: float
    anchor_value: float
    baseline_risk: float
    anchor_risk: float
    risk_delta: float
    importance: float
    direction: str
    absolute_low: float
    absolute_high: float
    preferred_low: Optional[float] = None
    preferred_high: Optional[float] = None
    stability_score: float
    plausibility_score: float
    constraint_status: str
    recommendation: str


class ContrastiveAnchorResponse(BaseModel):
    """Response for contrastive anchor explanation."""
    patient_id: Optional[str]
    disease: str
    baseline_risk: float
    representative_anchor_risk: float
    risk_delta: float
    risk_delta_percent: float
    risk_category: str
    confidence: float
    anchor_strategy: str
    local_context_size: int
    focus_features: List[str]
    feature_contributions: List[ContrastiveAnchorFeatureResult]
    clinical_boundaries: Dict[str, Dict[str, float]]
    clinical_summary: str
    clinical_recommendations: List[str] = Field(default_factory=list)


class ResponseSurfacePointResult(BaseModel):
    """Single sampled point on a feature response surface."""
    feature_value: float
    risk_score: float


class ResponseSurfaceFeatureResult(BaseModel):
    """Per-feature constrained response-surface metrics and sampled curve."""
    feature_name: str
    current_value: float
    best_value: float
    baseline_risk: float
    best_risk: float
    risk_delta: float
    importance: float
    confidence: float
    response_area: float
    monotonicity: float
    nonlinearity: float
    elasticity_mean: float
    elasticity_max: float
    suggested_direction: str
    absolute_low: float
    absolute_high: float
    preferred_low: Optional[float] = None
    preferred_high: Optional[float] = None
    sample_points: List[ResponseSurfacePointResult]


class ConstrainedResponseSurfaceRequest(BaseModel):
    """Request for constrained response-surface explanation."""
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    baseline_features: PatientFeatures
    disease: str = Field(..., description="Disease to explain")
    focus_features: Optional[List[str]] = Field(
        default=None,
        description="Optional ordered list of features to prioritize in the explanation",
    )
    top_k: int = Field(default=5, ge=1, le=8, description="Number of feature response curves to return")
    num_points: int = Field(default=11, ge=7, le=31, description="Sample points per feature curve")


class ConstrainedResponseSurfaceResponse(BaseModel):
    """Response for constrained response-surface explanation."""
    patient_id: Optional[str]
    disease: str
    baseline_risk: float
    representative_best_risk: float
    risk_delta: float
    risk_delta_percent: float
    risk_category: str
    confidence: float
    analysis_strategy: str
    local_context_size: int
    focus_features: List[str]
    feature_surfaces: List[ResponseSurfaceFeatureResult]
    clinical_boundaries: Dict[str, Dict[str, float]]
    clinical_summary: str
    clinical_recommendations: List[str] = Field(default_factory=list)


WHATIF_TARGETS: Dict[str, Dict[str, Any]] = {
    "temperature": {
        "preferred_low": 97.0,
        "preferred_high": 99.0,
        "action": "Normalize fever control"
    },
    "heart_rate": {
        "preferred_low": 60.0,
        "preferred_high": 100.0,
        "action": "Stabilize heart rate"
    },
    "systolic_bp": {
        "preferred_low": 100.0,
        "preferred_high": 140.0,
        "action": "Optimize blood pressure and perfusion"
    },
    "respiratory_rate": {
        "preferred_low": 12.0,
        "preferred_high": 20.0,
        "action": "Improve respiratory stability"
    },
    "wbc_count": {
        "preferred_low": 4.0,
        "preferred_high": 11.0,
        "action": "Reduce inflammatory burden"
    },
    "hemoglobin": {
        "preferred_low": 12.0,
        "preferred_high": 16.0,
        "action": "Correct anemia / improve oxygen carrying capacity"
    },
    "platelet_count": {
        "preferred_low": 150.0,
        "preferred_high": 400.0,
        "action": "Restore platelet reserve"
    },
    "creatinine": {
        "preferred_low": 0.6,
        "preferred_high": 1.2,
        "action": "Protect kidney function"
    },
    "bun": {
        "preferred_low": 7.0,
        "preferred_high": 20.0,
        "action": "Reduce renal stress"
    },
    "glucose": {
        "preferred_low": 70.0,
        "preferred_high": 140.0,
        "action": "Improve glycemic control"
    },
    "lactate": {
        "preferred_low": 0.8,
        "preferred_high": 2.0,
        "action": "Improve perfusion / lactate clearance"
    },
}


DISEASE_FEATURE_PRIORITIES: Dict[str, List[str]] = {
    "sepsis": ["lactate", "temperature", "wbc_count", "systolic_bp", "heart_rate", "respiratory_rate"],
    "kidney_failure": ["creatinine", "bun", "systolic_bp", "lactate"],
    "diabetes": ["glucose", "systolic_bp", "creatinine"],
    "anemia": ["hemoglobin", "platelet_count", "creatinine"],
    "thrombocytopenia": ["platelet_count", "hemoglobin"],
    "hypertension": ["systolic_bp", "diastolic_bp", "heart_rate", "creatinine"],
    "mortality": ["lactate", "systolic_bp", "creatinine", "heart_rate", "respiratory_rate"],
}


DISEASE_ACTION_OVERRIDES: Dict[str, Dict[str, str]] = {
    "anemia": {
        "hemoglobin": "Increase hemoglobin and evaluate iron stores/supplementation plan",
        "platelet_count": "Support marrow/hemostatic stability and reassess bleeding risk",
    },
    "sepsis": {
        "lactate": "Improve perfusion and monitor lactate clearance protocol",
        "temperature": "Control infectious burden and fever trajectory",
        "wbc_count": "Reassess inflammatory/infectious response and antimicrobial strategy",
    },
    "kidney_failure": {
        "creatinine": "Reduce kidney stress and avoid nephrotoxic exposure",
        "bun": "Optimize renal support and monitor uremic burden",
    },
    "hypertension": {
        "systolic_bp": "Lower systolic blood pressure toward guideline-consistent range",
        "diastolic_bp": "Normalize diastolic blood pressure and reassess perfusion",
        "heart_rate": "Reduce sympathetic stress and stabilize hemodynamics",
    },
}


DISEASE_DIRECT_FACTORS: Dict[str, set[str]] = {
    "sepsis": {"temperature", "wbc_count", "lactate", "respiratory_rate", "systolic_bp", "heart_rate"},
    "kidney_failure": {"creatinine", "bun", "systolic_bp", "lactate"},
    "diabetes": {"glucose", "systolic_bp"},
    "anemia": {"hemoglobin", "platelet_count"},
    "thrombocytopenia": {"platelet_count", "hemoglobin"},
    "hypertension": {"systolic_bp", "diastolic_bp", "heart_rate", "creatinine"},
    "mortality": {"lactate", "systolic_bp", "respiratory_rate", "heart_rate", "creatinine"},
}


DISEASE_MONITORING: Dict[str, List[str]] = {
    "sepsis": [
        "Trend lactate and complete blood count at short intervals",
        "Monitor temperature, respiratory rate, and blood pressure trajectory",
        "Repeat organ-function panel if instability persists",
    ],
    "kidney_failure": [
        "Repeat creatinine and BUN trend with urine output review",
        "Monitor blood pressure and fluid-balance markers",
        "Recheck electrolytes and renal function panel",
    ],
    "diabetes": [
        "Trend glucose and repeat metabolic panel",
        "Monitor blood pressure and renal markers",
        "Reassess glycemic profile at follow-up",
    ],
    "anemia": [
        "Repeat hemoglobin and complete blood count",
        "Trend platelet count and bleeding-risk indicators",
        "Reassess oxygen-delivery related parameters if symptoms evolve",
    ],
    "thrombocytopenia": [
        "Repeat platelet trend and complete blood count",
        "Monitor bleeding-risk signs and hemodynamic stability",
        "Reassess related hematologic markers at follow-up",
    ],
    "hypertension": [
        "Trend systolic and diastolic blood pressure over serial measurements",
        "Monitor heart-rate variability and signs of end-organ stress",
        "Reassess renal and metabolic profile at follow-up",
    ],
    "mortality": [
        "Trend lactate, blood pressure, and respiratory status",
        "Repeat renal and metabolic profile in short intervals",
        "Monitor multi-system instability markers over time",
    ],
}


NORMAL_RANGES: Dict[str, tuple[float, float]] = {
    "heart_rate": (60.0, 100.0),
    "systolic_bp": (90.0, 130.0),
    "diastolic_bp": (60.0, 85.0),
    "temperature": (97.0, 99.5),
    "respiratory_rate": (12.0, 20.0),
    "wbc_count": (4.0, 11.0),
    "hemoglobin": (12.0, 16.5),
    "platelet_count": (150.0, 400.0),
    "creatinine": (0.6, 1.2),
    "bun": (7.0, 20.0),
    "glucose": (70.0, 140.0),
    "lactate": (0.5, 2.0),
}


CLINICAL_ABSOLUTE_BOUNDS: Dict[str, tuple[float, float]] = {
    "age": (18.0, 100.0),
    "gender": (0.0, 1.0),
    "heart_rate": (40.0, 200.0),
    "systolic_bp": (70.0, 250.0),
    "diastolic_bp": (40.0, 150.0),
    "temperature": (95.0, 106.0),
    "respiratory_rate": (8.0, 50.0),
    "wbc_count": (1.0, 50.0),
    "hemoglobin": (5.0, 20.0),
    "platelet_count": (20.0, 700.0),
    "creatinine": (0.3, 15.0),
    "bun": (5.0, 200.0),
    "glucose": (50.0, 700.0),
    "lactate": (0.5, 25.0),
}


def _in_range(value: float, low: float, high: float) -> bool:
    return low <= float(value) <= high


def _is_globally_stable_profile(features: PatientFeatures) -> bool:
    """True when all core vitals/labs are within normal adult ranges."""
    return all(
        _in_range(getattr(features, field), bounds[0], bounds[1])
        for field, bounds in NORMAL_RANGES.items()
    )


def _apply_rule_based_risk_overrides(
    disease: str,
    raw_risk_score: float,
    features: PatientFeatures,
) -> tuple[float, List[str]]:
    """Apply lightweight clinical calibration + category override flags."""
    adjusted = float(raw_risk_score)
    applied: List[str] = []

    # Anemia calibration around physiologic hemoglobin ranges.
    if disease == "anemia":
        if features.hemoglobin >= 12.0:
            adjusted = max(0.0, adjusted - min((features.hemoglobin - 12.0) / 8.0, 0.20))
            applied.append("anemia_normal_hemoglobin_override")
        else:
            adjusted = min(1.0, adjusted + min((12.0 - features.hemoglobin) / 6.0, 0.25))

    # Hypertension calibration to keep BP-risk direction clinically monotonic.
    if disease == "hypertension":
        sys = float(features.systolic_bp)
        dia = float(features.diastolic_bp)

        # Lower BP in normal range should reduce hypertension risk.
        if sys < 130.0 and dia < 85.0:
            lower_gain = max((130.0 - sys) / 50.0, (85.0 - dia) / 25.0)
            adjusted = max(0.0, adjusted - min(0.20, 0.05 + 0.20 * lower_gain))
            applied.append("hypertension_normal_bp_override")

        # Elevated BP should increase hypertension risk.
        if sys >= 140.0 or dia >= 90.0:
            high_gain = max(max(0.0, (sys - 140.0) / 40.0), max(0.0, (dia - 90.0) / 20.0))
            adjusted = min(1.0, adjusted + min(0.30, 0.10 + 0.25 * high_gain))

    # Global normal-range awareness: stable physiology should not be flagged as moderate/high.
    if _is_globally_stable_profile(features):
        applied.append("global_normal_profile_override")

    return adjusted, applied


def _format_feature_name(name: str) -> str:
    return name.replace("_", " ")


def _impact_band(value: float, max_abs: float) -> str:
    if max_abs <= 1e-9:
        return "Low Impact"
    ratio = abs(value) / max_abs
    if ratio >= 0.67:
        return "High Impact"
    if ratio >= 0.33:
        return "Moderate Impact"
    return "Low Impact"


def _target_adjustment(feature_name: str, current: float, importance: float) -> Optional[float]:
    config = WHATIF_TARGETS.get(feature_name)
    if not config:
        return None

    preferred_low = float(config["preferred_low"])
    preferred_high = float(config["preferred_high"])
    preferred_mid = (preferred_low + preferred_high) / 2.0

    # Positive contribution means the current direction increases risk.
    if importance > 0:
        if current > preferred_high:
            return preferred_high
        if current > preferred_mid:
            return preferred_mid
        return None

    # Negative contribution means higher values may be protective in this local context.
    if current < preferred_low:
        return preferred_low
    if current < preferred_mid:
        return preferred_mid
    return None


def _build_clinical_decision_support_report(
    disease: str,
    risk_score: float,
    features_row: pd.Series,
    shap_summary: List[FeatureImportance],
) -> str:
    disease_label = _format_feature_name(disease)
    contributors = shap_summary[:5] if shap_summary else []

    if not contributors:
        return (
            "### 🩺 Prediction Summary\n"
            f"* Risk Score: {risk_score:.2f}\n"
            f"* Risk Level: {'High' if risk_score >= 0.50 else 'Moderate' if risk_score >= 0.25 else 'Low'}\n\n"
            "### 🔍 Key Drivers\n"
            "* No reliable feature-attribution summary is available for this prediction.\n\n"
            "### 🧬 Clinical Insight\n"
            "* This output should be interpreted with clinical context because local explanatory attributions are unavailable.\n\n"
            "### 🎯 Recommended Changes\n"
            "* No targeted change recommendation can be generated without valid feature-attribution data.\n\n"
            "### 🎯 Minimal Change Plan\n"
            "* Stabilize key disease-relevant vital signs and laboratory values, then reassess.\n\n"
            "### 📉 Expected Outcome\n"
            "* Risk trend cannot be estimated reliably until explainability outputs are available.\n\n"
            "### ⚠️ Low Impact Factors\n"
            "* Not assessable due to missing local feature-attribution data.\n\n"
            "### 🩺 Suggested Monitoring\n"
            f"* Continue disease-focused trend monitoring for {disease_label}."
        )

    max_abs = max(abs(item.importance) for item in contributors) if contributors else 0.0
    risk_level = "High" if risk_score >= 0.50 else "Moderate" if risk_score >= 0.25 else "Low"

    key_driver_lines: List[str] = []
    high_or_moderate: List[FeatureImportance] = []
    low_impact: List[FeatureImportance] = []

    for item in contributors:
        impact_level = _impact_band(item.importance, max_abs)
        direction = "Increases risk" if item.importance > 0 else "Decreases risk"
        key_driver_lines.append(
            f"* {_format_feature_name(item.feature_name)} -> {impact_level} -> {direction}"
        )
        if impact_level == "Low Impact":
            low_impact.append(item)
        else:
            high_or_moderate.append(item)

    direct_factors = DISEASE_DIRECT_FACTORS.get(disease, set())
    clinical_reasoning: List[str] = []
    for item in high_or_moderate[:4]:
        feature = item.feature_name
        feature_label = _format_feature_name(feature)
        if feature in direct_factors:
            clinical_reasoning.append(
                f"* {feature_label} is a direct causal factor for {disease_label} risk in this model context because abnormal values are tightly linked to disease pathophysiology."
            )
        else:
            clinical_reasoning.append(
                f"* {feature_label} is an indirect or associated indicator that reflects systemic stress relevant to {disease_label} risk rather than acting as a standalone causal driver."
            )

    recommended_changes: List[str] = []
    actionable: List[tuple[str, float, float]] = []
    for item in high_or_moderate:
        if item.importance <= 0:
            continue
        target = _target_adjustment(item.feature_name, float(item.value), item.importance)
        if target is None:
            continue
        actionable.append((item.feature_name, float(item.value), float(target)))
        if len(actionable) >= 3:
            break

    if actionable:
        for feature_name, current, target in actionable:
            recommended_changes.append(
                f"* {_format_feature_name(feature_name)}: {current:.2f} -> {target:.2f}"
            )
    else:
        recommended_changes.append(
            "* No strong single-feature out-of-range driver was identified; prioritize close monitoring and repeat assessment."
        )

    minimal_plan: List[str] = []
    for feature_name, _, target in actionable[:2]:
        minimal_plan.append(f"* {_format_feature_name(feature_name)} correction to {target:.2f}")
    if not minimal_plan:
        minimal_plan.append("* Maintain current status and re-evaluate with updated labs/vitals.")

    if actionable:
        change_strength = min(0.35, 0.08 + 0.09 * len(actionable[:2]))
        projected_low = max(0.0, risk_score - change_strength)
        projected_high = max(projected_low, risk_score - change_strength * 0.65)
        expected_outcome = (
            f"* Risk may reduce from {risk_score:.2f} to approximately {projected_low:.2f}-{projected_high:.2f} if high-impact features move toward suggested targets."
        )
    else:
        projected_low = max(0.0, risk_score - 0.05)
        expected_outcome = (
            f"* Risk may reduce modestly from {risk_score:.2f} to approximately {projected_low:.2f}-{risk_score:.2f} with monitoring-guided optimization."
        )

    if low_impact:
        low_impact_lines = [
            f"* {_format_feature_name(item.feature_name)} has negligible independent effect on this risk estimate."
            for item in low_impact[:2]
        ]
    else:
        low_impact_lines = [
            "* Remaining lower-ranked features have negligible independent impact relative to the key drivers."
        ]

    monitoring = DISEASE_MONITORING.get(
        disease,
        [
            "Trend high-impact vitals/laboratory features over short intervals",
            "Repeat disease-relevant panel if clinical status changes",
        ],
    )
    monitoring_lines = [f"* {item}" for item in monitoring[:3]]

    report_sections = [
        "### 🩺 Prediction Summary",
        f"* Risk Score: {risk_score:.2f}",
        f"* Risk Level: {risk_level}",
        "",
        "### 🔍 Key Drivers",
        *key_driver_lines,
        "",
        "### 🧬 Clinical Insight",
        *clinical_reasoning,
        "",
        "### 🎯 Recommended Changes",
        *recommended_changes,
        "",
        "### 🎯 Minimal Change Plan",
        *minimal_plan,
        "",
        "### 📉 Expected Outcome",
        expected_outcome,
        "",
        "### ⚠️ Low Impact Factors",
        *low_impact_lines,
        "",
        "### 🩺 Suggested Monitoring",
        *monitoring_lines,
    ]
    return "\n".join(report_sections)


def _suggest_target_value(feature_name: str, current_value: float) -> Optional[float]:
    config = WHATIF_TARGETS.get(feature_name)
    if not config:
        return None

    preferred_low = config["preferred_low"]
    preferred_high = config["preferred_high"]

    if current_value < preferred_low:
        return preferred_low
    if current_value > preferred_high:
        return preferred_high

    return None


def _build_whatif_recommendations(baseline_features: PatientFeatures, disease: str) -> List[Dict[str, Any]]:
    default_candidate_fields = [
        "temperature",
        "heart_rate",
        "systolic_bp",
        "respiratory_rate",
        "wbc_count",
        "hemoglobin",
        "platelet_count",
        "creatinine",
        "bun",
        "glucose",
        "lactate",
    ]
    candidate_fields = DISEASE_FEATURE_PRIORITIES.get(disease, default_candidate_fields)

    baseline_risk = model_manager.predict(baseline_features, disease).risk_score
    recommendations: List[Dict[str, Any]] = []

    for feature_name in candidate_fields:
        current_value = getattr(baseline_features, feature_name)
        target_value = _suggest_target_value(feature_name, current_value)
        if target_value is None or abs(target_value - current_value) < 1e-9:
            continue

        modified_dict = baseline_features.dict()
        modified_dict[feature_name] = target_value
        candidate_features = PatientFeatures(**modified_dict)
        candidate_risk = model_manager.predict(candidate_features, disease).risk_score
        risk_delta = candidate_risk - baseline_risk

        action = WHATIF_TARGETS.get(feature_name, {}).get("action", "Improve this feature")
        action = DISEASE_ACTION_OVERRIDES.get(disease, {}).get(feature_name, action)
        if target_value > current_value:
            direction = "Increase"
        else:
            direction = "Decrease"

        recommendations.append(
            {
                "feature": feature_name,
                "direction": direction,
                "current_value": current_value,
                "target_value": target_value,
                "expected_risk": candidate_risk,
                "risk_delta": risk_delta,
                "action": action,
            }
        )

    recommendations.sort(key=lambda item: item["risk_delta"])

    if recommendations:
        strongest = abs(recommendations[0]["risk_delta"])
        for item in recommendations:
            if strongest == 0:
                impact_label = "minimal"
            else:
                ratio = abs(item["risk_delta"]) / strongest
                if ratio >= 0.85:
                    impact_label = "strong"
                elif ratio >= 0.45:
                    impact_label = "moderate"
                else:
                    impact_label = "slight"
            item["impact_label"] = impact_label

    return recommendations[:3]


def _feature_bounds(feature_name: str) -> tuple[float, float, float, float]:
    absolute_low, absolute_high = CLINICAL_ABSOLUTE_BOUNDS.get(feature_name, (0.0, 1.0))
    preferred = WHATIF_TARGETS.get(feature_name)
    if preferred:
        preferred_low = float(preferred["preferred_low"])
        preferred_high = float(preferred["preferred_high"])
    else:
        preferred_low, preferred_high = NORMAL_RANGES.get(feature_name, (absolute_low, absolute_high))
    return absolute_low, absolute_high, preferred_low, preferred_high


def _clamp_value(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _build_governed_constraints() -> Dict[str, Dict[str, FeatureConstraint]]:
    """Create bounded constraints in the same units used by API patient features."""
    disease_constraints: Dict[str, Dict[str, FeatureConstraint]] = {}
    for disease in [
        "sepsis",
        "kidney_failure",
        "diabetes",
        "anemia",
        "thrombocytopenia",
        "hypertension",
        "mortality",
    ]:
        per_feature: Dict[str, FeatureConstraint] = {}
        for feature_name, (low, high) in CLINICAL_ABSOLUTE_BOUNDS.items():
            feature_type = FeatureType.FIXED if feature_name in {"age", "gender"} else FeatureType.ACTIONABLE
            span = max(float(high - low), 1e-6)
            per_feature[feature_name] = FeatureConstraint(
                feature_name=feature_name,
                feature_type=feature_type,
                min_value=float(low),
                max_value=float(high),
                max_increase=span,
                max_decrease=span,
                normal_range=NORMAL_RANGES.get(feature_name),
            )
        disease_constraints[disease] = per_feature
    return disease_constraints


def _select_anchor_value(feature_name: str, current_value: float) -> float:
    absolute_low, absolute_high, preferred_low, preferred_high = _feature_bounds(feature_name)
    preferred_low = max(absolute_low, preferred_low)
    preferred_high = min(absolute_high, preferred_high)

    if current_value < preferred_low:
        return preferred_low
    if current_value > preferred_high:
        return preferred_high

    anchor_value = (preferred_low + preferred_high) / 2.0
    return _clamp_value(anchor_value, absolute_low, absolute_high)


def _build_plausible_anchor_samples(
    base_features: PatientFeatures,
    feature_name: str,
    anchor_value: float,
    context_samples: int,
) -> List[PatientFeatures]:
    absolute_low, absolute_high, preferred_low, preferred_high = _feature_bounds(feature_name)
    feature_span = max(absolute_high - absolute_low, 1e-6)
    jitter_scale = max(feature_span * 0.04, abs(anchor_value) * 0.03, 0.05)
    if feature_name in {"gender"}:
        jitter_scale = 0.15

    rng = np.random.default_rng()
    base_dict = base_features.dict()
    samples: List[PatientFeatures] = []

    for _ in range(context_samples):
        sample_dict = dict(base_dict)
        noisy_value = rng.normal(loc=anchor_value, scale=jitter_scale)
        if feature_name == "gender":
            noisy_value = float(int(round(_clamp_value(noisy_value, absolute_low, absolute_high))))
        else:
            noisy_value = _clamp_value(noisy_value, absolute_low, absolute_high)
            if preferred_low <= preferred_high:
                if noisy_value < preferred_low:
                    noisy_value = preferred_low
                elif noisy_value > preferred_high:
                    noisy_value = preferred_high
        sample_dict[feature_name] = noisy_value
        samples.append(PatientFeatures(**sample_dict))

    return samples


def _plausibility_score(
    current_value: float,
    anchor_value: float,
    absolute_low: float,
    absolute_high: float,
    preferred_low: float,
    preferred_high: float,
) -> float:
    span = max(absolute_high - absolute_low, 1e-6)
    distance_score = 1.0 - min(1.0, abs(anchor_value - current_value) / span)
    if preferred_low <= anchor_value <= preferred_high:
        range_score = 1.0
    elif absolute_low <= anchor_value <= absolute_high:
        range_score = 0.7
    else:
        range_score = 0.0
    return max(0.0, min(1.0, 0.55 * range_score + 0.45 * distance_score))


def _contrastive_anchor_recommendation(
    disease: str,
    feature_name: str,
    risk_delta: float,
    anchor_value: float,
    preferred_low: float,
    preferred_high: float,
) -> str:
    label = _format_feature_name(feature_name)
    if risk_delta > 0:
        return f"Shift {label} toward {anchor_value:.2f} within the clinically plausible target range."
    if preferred_low <= anchor_value <= preferred_high:
        return f"Keep {label} near {anchor_value:.2f}; it is already in a clinically plausible range."
    return f"Keep {label} within bounds and avoid moving it farther from the preferred range."


def _build_contrastive_anchor_explanation(
    baseline_features: PatientFeatures,
    disease: str,
    focus_features: Optional[List[str]],
    top_k: int,
    context_samples: int,
    patient_id: Optional[str] = None,
) -> ContrastiveAnchorResponse:
    if disease not in model_manager.models:
        raise HTTPException(status_code=400, detail=f"Unknown disease: {disease}")

    baseline_prediction = model_manager.predict(baseline_features, disease)
    baseline_risk = baseline_prediction.risk_score

    default_candidate_fields = [
        "temperature",
        "heart_rate",
        "systolic_bp",
        "respiratory_rate",
        "wbc_count",
        "hemoglobin",
        "platelet_count",
        "creatinine",
        "bun",
        "glucose",
        "lactate",
    ]

    candidate_fields = focus_features or DISEASE_FEATURE_PRIORITIES.get(disease, default_candidate_fields)
    candidate_fields = [field for field in candidate_fields if field in CLINICAL_ABSOLUTE_BOUNDS]
    if not candidate_fields:
        candidate_fields = default_candidate_fields

    feature_results: List[ContrastiveAnchorFeatureResult] = []
    stability_scores: List[float] = []
    plausibility_scores: List[float] = []
    representative_anchor_risk = baseline_risk
    clinical_boundaries: Dict[str, Dict[str, float]] = {}

    for feature_name in candidate_fields[:max(1, top_k)]:
        current_value = float(getattr(baseline_features, feature_name))
        absolute_low, absolute_high, preferred_low, preferred_high = _feature_bounds(feature_name)
        anchor_value = _select_anchor_value(feature_name, current_value)
        anchor_value = _clamp_value(anchor_value, absolute_low, absolute_high)

        anchor_samples = _build_plausible_anchor_samples(
            base_features=baseline_features,
            feature_name=feature_name,
            anchor_value=anchor_value,
            context_samples=context_samples,
        )
        anchor_scores = [model_manager.predict(sample, disease).risk_score for sample in anchor_samples]
        anchor_risk = float(np.mean(anchor_scores))
        representative_anchor_risk = anchor_risk if feature_name == candidate_fields[0] else representative_anchor_risk

        risk_delta = baseline_risk - anchor_risk
        stability_score = float(np.clip(1.0 - min(1.0, np.std(anchor_scores) / 0.15), 0.0, 1.0))
        plausibility_score = _plausibility_score(
            current_value=current_value,
            anchor_value=anchor_value,
            absolute_low=absolute_low,
            absolute_high=absolute_high,
            preferred_low=preferred_low,
            preferred_high=preferred_high,
        )

        importance = float(risk_delta * (0.45 + 0.30 * stability_score + 0.25 * plausibility_score))
        direction = "increases" if importance > 0.0 else "decreases" if importance < 0.0 else "neutral"
        constraint_status = "within_preferred_range" if preferred_low <= anchor_value <= preferred_high else "within_absolute_bounds"

        clinical_boundaries[feature_name] = {
            "absolute_low": absolute_low,
            "absolute_high": absolute_high,
            "preferred_low": preferred_low,
            "preferred_high": preferred_high,
        }

        feature_results.append(
            ContrastiveAnchorFeatureResult(
                feature_name=feature_name,
                current_value=current_value,
                anchor_value=anchor_value,
                baseline_risk=baseline_risk,
                anchor_risk=anchor_risk,
                risk_delta=risk_delta,
                importance=importance,
                direction=direction,
                absolute_low=absolute_low,
                absolute_high=absolute_high,
                preferred_low=preferred_low,
                preferred_high=preferred_high,
                stability_score=stability_score,
                plausibility_score=plausibility_score,
                constraint_status=constraint_status,
                recommendation=_contrastive_anchor_recommendation(
                    disease=disease,
                    feature_name=feature_name,
                    risk_delta=risk_delta,
                    anchor_value=anchor_value,
                    preferred_low=preferred_low,
                    preferred_high=preferred_high,
                ),
            )
        )
        stability_scores.append(stability_score)
        plausibility_scores.append(plausibility_score)

    feature_results.sort(key=lambda item: abs(item.importance), reverse=True)
    if feature_results:
        representative_anchor_risk = feature_results[0].anchor_risk

    confidence = float(np.clip(
        0.45 * (float(np.mean(stability_scores)) if stability_scores else 0.0) +
        0.35 * (float(np.mean(plausibility_scores)) if plausibility_scores else 0.0) +
        0.20 * (1.0 - min(1.0, abs(baseline_risk - representative_anchor_risk))),
        0.0,
        1.0,
    ))

    top_features = feature_results[:top_k]
    top_lines = [
        f"{idx + 1}. {_format_feature_name(item.feature_name)}: {item.current_value:.2f} -> {item.anchor_value:.2f}"
        for idx, item in enumerate(top_features)
    ]
    if top_lines:
        clinical_summary = (
            f"Contrastive anchor analysis suggests the strongest bounded adjustments are: {', '.join(top_lines)}. "
            f"The representative anchor state moves risk from {baseline_risk:.2f} to {representative_anchor_risk:.2f}."
        )
    else:
        clinical_summary = (
            "No bounded contrastive adjustment could be derived from the available clinical features."
        )

    clinical_recommendations = [
        f"Maintain every adjusted feature inside its absolute clinical bounds before reassessing {disease.replace('_', ' ')} risk.",
        *DISEASE_MONITORING.get(disease, [])[:2],
    ]

    return ContrastiveAnchorResponse(
        patient_id=patient_id,
        disease=disease,
        baseline_risk=baseline_risk,
        representative_anchor_risk=representative_anchor_risk,
        risk_delta=baseline_risk - representative_anchor_risk,
        risk_delta_percent=((baseline_risk - representative_anchor_risk) / baseline_risk * 100.0) if baseline_risk > 0 else 0.0,
        risk_category=baseline_prediction.risk_category,
        confidence=confidence,
        anchor_strategy="bounded local anchor toward clinically plausible target ranges",
        local_context_size=context_samples,
        focus_features=top_features and [item.feature_name for item in top_features] or candidate_fields[:top_k],
        feature_contributions=top_features,
        clinical_boundaries=clinical_boundaries,
        clinical_summary=clinical_summary,
        clinical_recommendations=clinical_recommendations,
    )


def _build_constrained_response_surface_explanation(
    baseline_features: PatientFeatures,
    disease: str,
    focus_features: Optional[List[str]],
    top_k: int,
    num_points: int,
    patient_id: Optional[str] = None,
) -> ConstrainedResponseSurfaceResponse:
    if disease not in model_manager.models:
        raise HTTPException(status_code=400, detail=f"Unknown disease: {disease}")

    baseline_prediction = model_manager.predict(baseline_features, disease)
    baseline_risk = float(baseline_prediction.risk_score)

    default_candidate_fields = [
        "temperature",
        "heart_rate",
        "systolic_bp",
        "respiratory_rate",
        "wbc_count",
        "hemoglobin",
        "platelet_count",
        "creatinine",
        "bun",
        "glucose",
        "lactate",
    ]

    candidate_fields = focus_features or DISEASE_FEATURE_PRIORITIES.get(disease, default_candidate_fields)
    candidate_fields = [field for field in candidate_fields if field in CLINICAL_ABSOLUTE_BOUNDS]
    if not candidate_fields:
        candidate_fields = default_candidate_fields

    feature_surfaces: List[ResponseSurfaceFeatureResult] = []
    clinical_boundaries: Dict[str, Dict[str, float]] = {}
    feature_confidences: List[float] = []

    for feature_name in candidate_fields[: max(1, top_k)]:
        current_value = float(getattr(baseline_features, feature_name))
        absolute_low, absolute_high, preferred_low, preferred_high = _feature_bounds(feature_name)

        sweep_low = max(absolute_low, preferred_low)
        sweep_high = min(absolute_high, preferred_high)
        if sweep_high - sweep_low < 1e-6:
            sweep_low = absolute_low
            sweep_high = absolute_high

        values = np.linspace(sweep_low, sweep_high, int(num_points), dtype=float)
        scores: List[float] = []
        for value in values:
            probe_dict = baseline_features.dict()
            probe_dict[feature_name] = float(value)
            probe_features = PatientFeatures(**probe_dict)
            scores.append(float(model_manager.predict(probe_features, disease).risk_score))

        curve_scores = np.array(scores, dtype=float)
        deltas = curve_scores - baseline_risk
        span = max(float(sweep_high - sweep_low), 1e-6)

        response_area = float(np.trapz(np.abs(deltas), values) / span)
        slopes = np.diff(curve_scores) / np.maximum(np.diff(values), 1e-6)
        monotonicity = float(abs(np.mean(np.sign(slopes)))) if slopes.size else 1.0
        curvature = np.diff(slopes)
        nonlinearity = float(np.mean(np.abs(curvature))) if curvature.size else 0.0

        numerator = np.abs(np.diff(curve_scores) / np.maximum(np.abs(curve_scores[:-1]), 1e-4))
        denominator = np.abs(np.diff(values) / np.maximum(np.abs(values[:-1]), 1e-4))
        elasticity = numerator / np.maximum(denominator, 1e-6)
        elasticity_mean = float(np.mean(elasticity)) if elasticity.size else 0.0
        elasticity_max = float(np.max(elasticity)) if elasticity.size else 0.0

        best_idx = int(np.argmin(curve_scores))
        best_value = float(values[best_idx])
        best_risk = float(curve_scores[best_idx])
        risk_delta = float(baseline_risk - best_risk)

        importance = float(response_area * (1.0 + 0.25 * nonlinearity + 0.15 * elasticity_mean))
        confidence = float(np.clip(
            0.45 * monotonicity +
            0.30 * (1.0 - min(1.0, nonlinearity / 0.5)) +
            0.25 * (1.0 - min(1.0, np.std(curve_scores) / 0.2)),
            0.0,
            1.0,
        ))

        if abs(best_value - current_value) < 1e-9:
            suggested_direction = "hold"
        elif best_value > current_value:
            suggested_direction = "increase"
        else:
            suggested_direction = "decrease"

        sample_points = [
            ResponseSurfacePointResult(feature_value=float(v), risk_score=float(r))
            for v, r in zip(values, curve_scores)
        ]

        clinical_boundaries[feature_name] = {
            "absolute_low": absolute_low,
            "absolute_high": absolute_high,
            "preferred_low": preferred_low,
            "preferred_high": preferred_high,
        }

        feature_surfaces.append(
            ResponseSurfaceFeatureResult(
                feature_name=feature_name,
                current_value=current_value,
                best_value=best_value,
                baseline_risk=baseline_risk,
                best_risk=best_risk,
                risk_delta=risk_delta,
                importance=importance,
                confidence=confidence,
                response_area=response_area,
                monotonicity=monotonicity,
                nonlinearity=nonlinearity,
                elasticity_mean=elasticity_mean,
                elasticity_max=elasticity_max,
                suggested_direction=suggested_direction,
                absolute_low=absolute_low,
                absolute_high=absolute_high,
                preferred_low=preferred_low,
                preferred_high=preferred_high,
                sample_points=sample_points,
            )
        )
        feature_confidences.append(confidence)

    feature_surfaces.sort(key=lambda item: abs(item.importance), reverse=True)
    top_feature_surfaces = feature_surfaces[:top_k]

    representative_best_risk = min((item.best_risk for item in top_feature_surfaces), default=baseline_risk)
    confidence = float(np.mean(feature_confidences)) if feature_confidences else 0.0

    top_lines = [
        f"{idx + 1}. {_format_feature_name(item.feature_name)}: {item.current_value:.2f} -> {item.best_value:.2f}"
        for idx, item in enumerate(top_feature_surfaces[:3])
    ]
    if top_lines:
        clinical_summary = (
            f"Constrained response-surface analysis identifies the strongest bounded levers as {', '.join(top_lines)}. "
            f"Best observed bounded local risk moves from {baseline_risk:.2f} to {representative_best_risk:.2f}."
        )
    else:
        clinical_summary = (
            "No bounded response-surface profile could be generated from the available clinical features."
        )

    clinical_recommendations = [
        f"Adjust only inside absolute and preferred bounds before reassessing {disease.replace('_', ' ')} risk.",
        *DISEASE_MONITORING.get(disease, [])[:2],
    ]

    return ConstrainedResponseSurfaceResponse(
        patient_id=patient_id,
        disease=disease,
        baseline_risk=baseline_risk,
        representative_best_risk=representative_best_risk,
        risk_delta=baseline_risk - representative_best_risk,
        risk_delta_percent=((baseline_risk - representative_best_risk) / baseline_risk * 100.0) if baseline_risk > 0 else 0.0,
        risk_category=baseline_prediction.risk_category,
        confidence=confidence,
        analysis_strategy="clinically bounded local response curves with shape-aware importance",
        local_context_size=int(num_points),
        focus_features=top_feature_surfaces and [item.feature_name for item in top_feature_surfaces] or candidate_fields[:top_k],
        feature_surfaces=top_feature_surfaces,
        clinical_boundaries=clinical_boundaries,
        clinical_summary=clinical_summary,
        clinical_recommendations=clinical_recommendations,
    )


# ============================================================================
# MODEL MANAGER
# ============================================================================

class ModelManager:
    """Manages all trained ML models."""
    
    DISEASES = [
        'sepsis', 'kidney_failure', 'diabetes',
        'anemia', 'thrombocytopenia', 'hypertension', 'mortality'
    ]
    
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            # Default to trained_models relative to backend script location
            models_dir = Path(__file__).parent.parent / "trained_models"
        self.models_dir = Path(models_dir)
        self.models: Dict[str, Any] = {}
        self.model_variants: Dict[str, Dict[str, Any]] = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all trained models from disk."""
        logger.info(f"📦 Loading models from: {self.models_dir}")

        self.models = {}
        self.model_variants = {}
        
        for disease in self.DISEASES:
            self.model_variants[disease] = {}

            # Load explicit model variants when present.
            for variant_name in ["xgb", "nn"]:
                variant_path = self.models_dir / f"{disease}_advanced_{variant_name}_v1.0.0.pkl"
                if variant_path.exists():
                    try:
                        variant_bundle = joblib.load(variant_path)
                        self.model_variants[disease][variant_name] = variant_bundle
                        logger.info(f"  ✅ {disease:20s} - variant {variant_name}")
                    except Exception as e:
                        logger.error(f"  ❌ {disease:20s} - variant {variant_name} error: {e}")

            # Try advanced model first, then fallback to xgboost
            for suffix in ["_advanced_v1.0.0.pkl", "_xgboost_v1.0.0.pkl"]:
                model_path = self.models_dir / f"{disease}{suffix}"
                
                if model_path.exists():
                    try:
                        bundle = joblib.load(model_path)
                        self.models[disease] = bundle
                        logger.info(f"  ✅ {disease:20s} - {bundle.get('model_type', 'unknown')}")
                        break
                    except Exception as e:
                        logger.error(f"  ❌ {disease:20s} - Error: {e}")
            else:
                if self.model_variants[disease]:
                    # If only variants are present, pick XGB first then NN as serving default.
                    default_key = "xgb" if "xgb" in self.model_variants[disease] else "nn"
                    self.models[disease] = self.model_variants[disease][default_key]
                    logger.info(f"  ✅ {disease:20s} - defaulted to variant {default_key}")
                else:
                    logger.warning(f"  ⚠️  {disease:20s} - Not found")
        
        total_variants = sum(len(v) for v in self.model_variants.values())
        logger.info(f"📊 Loaded {len(self.models)}/{len(self.DISEASES)} primary models")
        logger.info(f"📊 Loaded {total_variants} model variants (xgb/nn)")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features consistent with training pipeline."""
        base_df = df.copy()
        missing = [col for col in BASE_FEATURES if col not in base_df.columns]
        if missing:
            raise ValueError(f"Missing base features for inference: {missing}")

        # Keep legacy derived features for backward compatibility with older bundles.
        engineered = add_derived_features(base_df[BASE_FEATURES]).copy()

        # Add advanced features used by train_advanced_models.py bundles.
        X = base_df[BASE_FEATURES].copy()
        X['hr_bp_ratio'] = X['heart_rate'] / (X['systolic_bp'] + 1)
        X['shock_index'] = X['heart_rate'] / (X['systolic_bp'] + 1)
        X['map'] = (X['systolic_bp'] + 2 * X['diastolic_bp']) / 3
        X['pulse_pressure'] = X['systolic_bp'] - X['diastolic_bp']

        X['creat_bun_ratio'] = X['creatinine'] / (X['bun'] + 1)
        X['kidney_damage'] = X['creatinine'] * X['bun'] / 100

        X['age_glucose'] = X['age'] * X['glucose'] / 100
        X['bmi_proxy'] = X['age'] / 2

        X['hemoglobin_age'] = X['hemoglobin'] * (100 - X['age']) / 100
        X['platelet_wbc_ratio'] = X['platelet_count'] / (X['wbc_count'] + 1)

        X['sepsis_score'] = (
            (X['temperature'] > 100.4).astype(int) +
            (X['temperature'] < 96.8).astype(int) +
            (X['heart_rate'] > 90).astype(int) +
            (X['respiratory_rate'] > 20).astype(int) +
            (X['wbc_count'] > 12).astype(int) +
            (X['wbc_count'] < 4).astype(int)
        )

        X['kidney_score'] = (
            (X['creatinine'] > 1.5).astype(int) +
            (X['bun'] > 25).astype(int) +
            (X['creatinine'] > 2.5).astype(int)
        )

        X['cardiac_score'] = (
            (X['heart_rate'] > 100).astype(int) +
            (X['systolic_bp'] > 140).astype(int) +
            (X['systolic_bp'] < 90).astype(int)
        )

        X['age_squared'] = X['age'] ** 2
        X['glucose_squared'] = (X['glucose'] / 100) ** 2
        X['lactate_squared'] = X['lactate'] ** 2

        X['hr_map_interaction'] = X['heart_rate'] * X['map'] / 1000
        X['temp_wbc_interaction'] = X['temperature'] * X['wbc_count'] / 100

        for col in X.columns:
            if col not in engineered.columns:
                engineered[col] = X[col]

        return engineered

    def _compute_shap_top_features(
        self,
        model,
        X_scaled: np.ndarray,
        X_row: pd.Series,
        feature_names: List[str],
        top_k: int = 5,
    ) -> List[FeatureImportance]:
        """Compute top SHAP contributors for one sample (safe fallback to empty)."""
        if shap is None:
            return []

        try:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_scaled)
            except Exception:
                explainer = shap.Explainer(model)
                shap_output = explainer(X_scaled)
                shap_values = shap_output.values

            if isinstance(shap_values, list):
                # Binary classifiers may return [class0, class1].
                class_one = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                shap_row = np.array(class_one)[0]
            else:
                shap_row = np.array(shap_values)[0]

            indices = np.argsort(np.abs(shap_row))[-top_k:][::-1]
            return [
                FeatureImportance(
                    feature_name=feature_names[idx],
                    importance=float(shap_row[idx]),
                    value=float(X_row[feature_names[idx]]),
                )
                for idx in indices
            ]
        except Exception:
            return []

    def _compute_lime_top_features(
        self,
        model,
        X_scaled: np.ndarray,
        X_row: pd.Series,
        feature_names: List[str],
        top_k: int = 5,
    ) -> List[FeatureImportance]:
        """Compute top LIME contributors for one sample (safe fallback to empty)."""
        if lime_tabular is None:
            return []

        try:
            # Use standardized synthetic background in scaled feature space.
            background = np.random.normal(0.0, 1.0, size=(256, len(feature_names)))
            explainer = lime_tabular.LimeTabularExplainer(
                training_data=background,
                feature_names=feature_names,
                class_names=["negative", "positive"],
                mode="classification",
                discretize_continuous=True,
            )
            explanation = explainer.explain_instance(
                data_row=X_scaled[0],
                predict_fn=model.predict_proba,
                num_features=min(top_k, len(feature_names)),
            )

            lime_features: List[FeatureImportance] = []
            for descriptor, weight in explanation.as_list(label=1):
                matched_idx = next(
                    (i for i, name in enumerate(feature_names) if name in descriptor),
                    None,
                )
                if matched_idx is None:
                    continue
                lime_features.append(
                    FeatureImportance(
                        feature_name=feature_names[matched_idx],
                        importance=float(weight),
                        value=float(X_row[feature_names[matched_idx]]),
                    )
                )

            return lime_features[:top_k]
        except Exception:
            return []
    
    def predict(self, features: PatientFeatures, disease: str) -> DiseasePrediction:
        """Make prediction for a single disease."""
        if disease not in self.models:
            raise ValueError(f"Model not found: {disease}")
        
        bundle = self.models[disease]
        model = bundle['model']
        scaler = bundle['scaler']
        threshold = bundle.get('optimal_threshold', 0.5)
        
        explanation_warnings: List[str] = []

        # Prepare features
        df = pd.DataFrame([features.dict()])
        X = self.engineer_features(df)
        
        # Get feature names from scaler
        if hasattr(scaler, 'feature_names_in_'):
            feature_names = list(scaler.feature_names_in_)
            missing_features = [name for name in feature_names if name not in X.columns]
            for name in missing_features:
                X[name] = 0.0
            X = X[feature_names]
            if missing_features:
                logger.warning(
                    f"{disease}: missing engineered features filled with 0.0: {missing_features}"
                )
        else:
            feature_names = list(X.columns)
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        raw_risk_score = float(model.predict_proba(X_scaled)[0, 1])
        risk_score, applied_overrides = _apply_rule_based_risk_overrides(
            disease=disease,
            raw_risk_score=raw_risk_score,
            features=features,
        )
        prediction = 1 if risk_score >= threshold else 0
        
        # Risk category
        if risk_score < 0.25:
            risk_category = "LOW"
        elif risk_score < 0.50:
            risk_category = "MODERATE"
        elif risk_score < 0.70:
            risk_category = "HIGH"
        else:
            risk_category = "CRITICAL"

        # Enforce category-level guardrails while preserving score granularity.
        if any(
            key in applied_overrides
            for key in [
                "anemia_normal_hemoglobin_override",
                "hypertension_normal_bp_override",
                "global_normal_profile_override",
            ]
        ):
            risk_category = "LOW"
            prediction = 0
        
        # Baseline top features from model importances.
        top_features = []
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-5:][::-1]
            
            for idx in indices:
                fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                top_features.append(FeatureImportance(
                    feature_name=fname,
                    importance=float(importances[idx]),
                    value=float(X.iloc[0, idx]) if idx < X.shape[1] else 0.0
                ))

        shap_top_features = self._compute_shap_top_features(
            model=model,
            X_scaled=X_scaled,
            X_row=X.iloc[0],
            feature_names=feature_names,
        )
        lime_top_features = self._compute_lime_top_features(
            model=model,
            X_scaled=X_scaled,
            X_row=X.iloc[0],
            feature_names=feature_names,
        )

        explanation_methods: List[str] = []
        if top_features:
            explanation_methods.append("feature_importance")
        if shap_top_features:
            explanation_methods.append("shap")
        elif shap is None:
            explanation_warnings.append("SHAP package not installed")
        else:
            explanation_warnings.append("SHAP explanation unavailable for this model; using feature-importance proxy")
            shap_top_features = [
                FeatureImportance(
                    feature_name=feat.feature_name,
                    importance=feat.importance,
                    value=feat.value,
                )
                for feat in top_features
            ]
            if shap_top_features:
                explanation_methods.append("shap_proxy")
        if lime_top_features:
            explanation_methods.append("lime")
        elif lime_tabular is None:
            explanation_warnings.append("LIME package not installed")
        else:
            explanation_warnings.append("LIME explanation unavailable for this model")

        if applied_overrides:
            explanation_warnings.append(
                "rule_based_overrides_applied: " + ",".join(applied_overrides)
            )

        report_sources = shap_top_features if shap_top_features else top_features
        clinical_decision_support_report = _build_clinical_decision_support_report(
            disease=disease,
            risk_score=risk_score,
            features_row=X.iloc[0],
            shap_summary=report_sources,
        )
        
        return DiseasePrediction(
            disease=disease,
            risk_score=risk_score,
            risk_category=risk_category,
            prediction=prediction,
            model_type=bundle.get('model_type', 'unknown'),
            threshold=threshold,
            top_features=top_features,
            shap_top_features=shap_top_features,
            lime_top_features=lime_top_features,
            explanation_methods=explanation_methods,
            explanation_warnings=explanation_warnings,
            clinical_decision_support_report=clinical_decision_support_report,
        )
    
    def predict_all(self, features: PatientFeatures) -> List[DiseasePrediction]:
        """Predict all diseases."""
        predictions = []
        for disease in self.models.keys():
            try:
                pred = self.predict(features, disease)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error predicting {disease}: {e}")
        
        # Sort by risk score
        predictions.sort(key=lambda x: x.risk_score, reverse=True)
        return predictions


# Initialize model + fusion managers
model_manager = ModelManager()
fusion_engine = MultimodalFusionEngine(
    labels_path=(
        CXR_LABELS_PATH
        if (str(CXR_LABELS_PATH).startswith("gs://") or Path(CXR_LABELS_PATH).exists())
        else None
    )
)
recommendation_engine = get_recommendation_engine()
audit_logger = AuditLogger(log_path="audit_logs/training_events.jsonl", system_id="ED-AI-API")
governance_ctx = _BackendGovernanceContext(audit_logger)
alert_engine = AlertEngine(ctx=governance_ctx, alert_policies=DEFAULT_ALERT_POLICIES)
governed_whatif_engine = GovernedWhatIfEngine(
    ctx=governance_ctx,
    model_registry=ModelRegistry(),
    constraints=_build_governed_constraints(),
)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api")
async def api_info():
    """General API metadata endpoint."""
    return {
        "name": "Explainable Medical AI API",
        "version": "2.0.0",
        "status": "operational",
        "models_loaded": len(model_manager.models),
        "diseases": list(model_manager.models.keys()),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/api/predict",
            "whatif": "/api/what-if",
            "contrastive_anchor": "/api/contrastive-anchor",
            "constrained_response_surface": "/api/constrained-response-surface",
            "models": "/api/models",
            "samples": "/api/sample-patients",
            "frontend": "/app"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.
    Returns detailed system status in production.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": APP_ENV,
        "version": "2.0.0",
        "models_loaded": len(model_manager.models),
        "diseases_available": list(model_manager.models.keys()) if DEBUG else len(model_manager.models)
    }
    
    # Add detailed checks in development/debug mode
    if DEBUG:
        health_status.update({
            "debug_mode": True,
            "api_docs_enabled": ENABLE_API_DOCS,
            "cors_origins": CORS_ORIGINS
        })
    
    return health_status


@app.get("/api/models")
async def get_models():
    """Get information about loaded models."""
    models_info = []
    
    for disease, bundle in model_manager.models.items():
        metrics = bundle.get('metrics', {})
        variants = sorted(list(model_manager.model_variants.get(disease, {}).keys()))
        models_info.append({
            "disease": disease,
            "model_type": bundle.get('model_type', 'unknown'),
            "available_variants": variants,
            "optimal_threshold": bundle.get('optimal_threshold', 0.5),
            "metrics": {
                "auroc": metrics.get('auroc', 0),
                "accuracy": metrics.get('accuracy', 0),
                "f1_score": metrics.get('f1_score', 0)
            }
        })
    
    return {
        "total_models": len(models_info),
        "models": models_info
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_diseases(request: PredictionRequest):
    """
    Predict disease risks for a patient.
    
    Returns predictions for all diseases with feature importances.
    """
    try:
        # Predict specified or all diseases
        if request.diseases:
            predictions = []
            for disease in request.diseases:
                if disease in model_manager.models:
                    pred = model_manager.predict(request.features, disease)
                    predictions.append(pred)
        else:
            predictions = model_manager.predict_all(request.features)
        
        # Determine overall risk
        max_risk = max([p.risk_score for p in predictions]) if predictions else 0
        if _is_globally_stable_profile(request.features):
            overall_risk = "LOW"
        elif max_risk < 0.30:
            overall_risk = "LOW"
        elif max_risk < 0.50:
            overall_risk = "MODERATE"
        elif max_risk < 0.70:
            overall_risk = "HIGH"
        else:
            overall_risk = "CRITICAL"

        alert_outcomes: List[Dict[str, Any]] = []

        for pred in predictions:
            policy_name = ALERT_POLICY_BY_DISEASE.get(pred.disease)
            if not policy_name:
                alert_outcomes.append(
                    {
                        "disease": pred.disease,
                        "policy_mapped": False,
                        "status": "not_configured",
                        "triggered": False,
                        "reason": "No alert policy mapping for this disease",
                    }
                )
                continue
            policy = DEFAULT_ALERT_POLICIES.get(policy_name)
            if not policy:
                alert_outcomes.append(
                    {
                        "disease": pred.disease,
                        "policy_mapped": True,
                        "status": "policy_missing",
                        "triggered": False,
                        "reason": f"Alert policy missing for {policy_name}",
                    }
                )
                continue

            model_bundle = model_manager.models.get(pred.disease, {})
            model_name = model_bundle.get("model_type", "unknown")
            model_version = model_bundle.get("version", "current")
            patient_id = request.patient_id or "anonymous"

            prediction_event = audit_logger.log_event(
                event_type=AuditEventType.PREDICTION,
                patient_id=patient_id,
                disease=policy_name,
                model_name=model_name,
                model_version=model_version,
                threshold=policy.moderate_threshold,
                payload={
                    "probability": float(pred.risk_score),
                    "risk_category": pred.risk_category,
                },
                human_message=f"{pred.disease} risk predicted at {pred.risk_score:.2f}",
            )

            explanation_summary = _build_explanation_summary(pred)
            explanation_event = audit_logger.log_event(
                event_type=AuditEventType.EXPLANATION,
                patient_id=patient_id,
                disease=policy_name,
                model_name=model_name,
                model_version=model_version,
                prediction_id=prediction_event.event_id,
                payload={
                    "methods": pred.explanation_methods,
                    "summary": explanation_summary,
                },
                human_message=f"Explanation generated for {pred.disease}: {explanation_summary}",
            )

            trace = _SimpleDecisionTrace(
                trace_id=_trace_id(request.patient_id, pred.disease),
                patient_id=patient_id,
                disease=policy_name,
                model_name=model_name,
                model_version=model_version,
                input_summary=request.features.dict(),
                prediction_event=prediction_event,
                explanation_event=explanation_event,
            )

            try:
                alert = alert_engine.trigger_alert(
                    trace=trace,
                    prediction_probability=float(pred.risk_score),
                    explanation_summary=explanation_summary,
                    threshold_used=policy.moderate_threshold,
                    explanation_details={
                        "top_features": [feat.dict() for feat in (pred.shap_top_features or pred.top_features)[:5]],
                    },
                )

                if alert:
                    alert_outcomes.append(
                        {
                            "disease": pred.disease,
                            "policy_mapped": True,
                            "status": "triggered",
                            "triggered": True,
                            "alert_id": alert.alert_id,
                            "severity": alert.severity.value,
                            "probability": float(pred.risk_score),
                            "threshold": float(policy.moderate_threshold),
                            "explanation_summary": explanation_summary,
                            "recommended_actions": list(alert.recommended_actions),
                        }
                    )
                else:
                    alert_outcomes.append(
                        {
                            "disease": pred.disease,
                            "policy_mapped": True,
                            "status": "suppressed_or_below_threshold",
                            "triggered": False,
                            "probability": float(pred.risk_score),
                            "threshold": float(policy.moderate_threshold),
                        }
                    )
            except AlertRegulatoryViolationError as alert_error:
                logger.warning(f"Alert suppressed due to governance rule: {alert_error}")
                alert_outcomes.append(
                    {
                        "disease": pred.disease,
                        "policy_mapped": True,
                        "status": "suppressed_regulatory",
                        "triggered": False,
                        "reason": str(alert_error),
                    }
                )
            except Exception as alert_error:
                logger.warning(f"Alert engine fallback: {alert_error}")
                alert_outcomes.append(
                    {
                        "disease": pred.disease,
                        "policy_mapped": True,
                        "status": "error",
                        "triggered": False,
                        "reason": str(alert_error),
                    }
                )

        fusion_summary = None
        try:
            fusion_payload = fusion_engine.fuse(
                base_features=request.features.dict(),
                predictions=predictions,
                imaging_signal=request.imaging.findings if request.imaging else None,
                dicom_id=request.imaging.dicom_id if request.imaging else None,
                imaging_source=request.imaging.source if request.imaging else None,
            )
            if fusion_payload:
                fusion_summary = MultimodalSummary(**fusion_payload)
        except Exception as fusion_error:
            logger.warning(f"Fusion engine fallback: {fusion_error}")
        
        return PredictionResponse(
            patient_id=request.patient_id,
            timestamp=datetime.now().isoformat(),
            predictions=predictions,
            overall_risk_category=overall_risk,
            multimodal_summary=fusion_summary,
            clinical_recommendations=recommendation_engine.generate_recommendations(
                patient_age=int(request.features.age),
                predictions={
                    pred.disease: {
                        "risk_score": pred.risk_score,
                        "risk_category": pred.risk_category,
                    }
                    for pred in predictions
                },
            ),
            alerts=alert_outcomes,
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/what-if", response_model=WhatIfResponse)
@app.post("/api/whatif", response_model=WhatIfResponse)
async def whatif_analysis(request: WhatIfRequest):
    """
    Perform what-if scenario analysis.
    
    Compares baseline risk vs modified features risk.
    """
    try:
        if request.disease not in model_manager.models:
            raise HTTPException(status_code=400, detail=f"Unknown disease: {request.disease}")

        baseline_payload = request.baseline_features
        if isinstance(baseline_payload, dict) and "features" in baseline_payload:
            baseline_payload = baseline_payload["features"]

        baseline_features = PatientFeatures(**baseline_payload)

        # Baseline prediction
        baseline_pred = model_manager.predict(baseline_features, request.disease)
        baseline_risk = baseline_pred.risk_score
        
        # Normalize modified payload (frontend may send nested {"features": {...}})
        modified_payload = request.modified_features
        if isinstance(modified_payload, dict) and "features" in modified_payload and isinstance(modified_payload["features"], dict):
            modified_payload = modified_payload["features"]

        if hasattr(PatientFeatures, "model_fields"):
            # Pydantic v2
            allowed_feature_fields = set(PatientFeatures.model_fields.keys())
        else:
            # Pydantic v1
            allowed_feature_fields = set(PatientFeatures.__fields__.keys())
        invalid_fields = [name for name in modified_payload.keys() if name not in allowed_feature_fields]
        if invalid_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid modified feature fields: {invalid_fields}. Allowed fields: {sorted(allowed_feature_fields)}",
            )

        policy_name = ALERT_POLICY_BY_DISEASE.get(request.disease, request.disease)
        model_bundle = model_manager.models.get(request.disease, {})
        model_name = model_bundle.get("model_type", "unknown")
        model_version = model_bundle.get("version", "current")
        patient_id = request.baseline_features.get("patientId") or "anonymous"

        baseline_prediction_event = audit_logger.log_event(
            event_type=AuditEventType.PREDICTION,
            patient_id=patient_id,
            disease=policy_name,
            model_name=model_name,
            model_version=model_version,
            payload={
                "probability": float(baseline_risk),
                "request": "whatif_baseline",
            },
            human_message=f"Baseline what-if prediction for {request.disease}: {baseline_risk:.2f}",
        )
        baseline_explanation = _build_explanation_summary(baseline_pred)
        baseline_explanation_event = audit_logger.log_event(
            event_type=AuditEventType.EXPLANATION,
            patient_id=patient_id,
            disease=policy_name,
            model_name=model_name,
            model_version=model_version,
            prediction_id=baseline_prediction_event.event_id,
            payload={"summary": baseline_explanation, "methods": baseline_pred.explanation_methods},
            human_message=f"Baseline explanation for {request.disease}: {baseline_explanation}",
        )

        trace = _SimpleDecisionTrace(
            trace_id=_trace_id(patient_id, request.disease),
            patient_id=patient_id,
            disease=request.disease,
            model_name=model_name,
            model_version=model_version,
            input_summary=baseline_features.dict(),
            prediction_event=baseline_prediction_event,
            explanation_event=baseline_explanation_event,
        )

        try:
            simulation_result = governed_whatif_engine.run_simulation(
                trace=trace,
                proposed_changes={k: float(v) for k, v in modified_payload.items()},
                clinician_id=patient_id,
                model=model_manager.models.get(request.disease, {}).get("model"),
            )
            if simulation_result.plausibility == PlausibilityLevel.IMPOSSIBLE:
                raise HTTPException(
                    status_code=400,
                    detail="Simulation rejected: physiologically impossible scenario.",
                )
        except GovernedRegulatoryViolationError as gov_error:
            raise HTTPException(status_code=400, detail=str(gov_error)) from gov_error

        # Modified prediction
        modified_dict = baseline_features.dict()
        modified_dict.update(modified_payload)
        modified_features = PatientFeatures(**modified_dict)
        
        modified_pred = model_manager.predict(modified_features, request.disease)
        new_risk = modified_pred.risk_score
        
        # Calculate delta
        risk_delta = new_risk - baseline_risk
        risk_delta_percent = (risk_delta / baseline_risk * 100) if baseline_risk > 0 else 0
        
        # Track changes
        feature_changes = {}
        for feature, new_value in modified_payload.items():
            old_value = getattr(baseline_features, feature)
            feature_changes[feature] = {"old": old_value, "new": new_value}
        
        # Generate recommendation
        if risk_delta < -0.1:
            recommendation = f"✅ Positive intervention: Risk reduced by {abs(risk_delta_percent):.1f}%"
        elif risk_delta > 0.1:
            recommendation = f"⚠️ Warning: Risk increased by {risk_delta_percent:.1f}%"
        else:
            recommendation = "ℹ️ Minimal impact on risk"

        recommended_changes = _build_whatif_recommendations(baseline_features, request.disease)
        if recommended_changes:
            top_change = recommended_changes[0]
            clinical_summary = (
                f"To reduce {request.disease.replace('_', ' ')} risk, prioritize changing "
                f"{top_change['feature'].replace('_', ' ')} {top_change['direction'].lower()} "
                f"from {top_change['current_value']:.2f} to {top_change['target_value']:.2f}."
            )
            if len(recommended_changes) > 1:
                second_change = recommended_changes[1]
                clinical_summary += (
                    f" Secondary lever: {second_change['feature'].replace('_', ' ')} "
                    f"{second_change['direction'].lower()} to {second_change['target_value']:.2f}."
                )
            counterfactual_explanations = []
            for item in recommended_changes:
                direction_word = item['direction'].lower()
                effect_word = 'decrease' if item['risk_delta'] < 0 else 'increase'
                counterfactual_explanations.append(
                    f"If {item['feature'].replace('_', ' ')} is changed to {item['target_value']:.2f}, "
                    f"risk is expected to {effect_word} by {abs(item['risk_delta'] * 100):.1f} percentage points. "
                    f"This is a {item['impact_label']} effect relative to the other suggested changes."
                )
        else:
            clinical_summary = "Current baseline is already close to preferred feature ranges; focus on monitoring and reassessment."
            counterfactual_explanations = [
                "The current baseline is already near preferred ranges, so large risk shifts are not expected from single-feature changes."
            ]
        
        return WhatIfResponse(
            disease=request.disease,
            baseline_risk=baseline_risk,
            new_risk=new_risk,
            risk_delta=risk_delta,
            risk_delta_percent=risk_delta_percent,
            modified_features=feature_changes,
            recommendation=recommendation,
            recommended_changes=recommended_changes,
            clinical_summary=clinical_summary,
            counterfactual_explanations=counterfactual_explanations,
        )
    
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"What-if error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/contrastive-anchor", response_model=ContrastiveAnchorResponse)
async def contrastive_anchor_analysis(request: ContrastiveAnchorRequest):
    """Generate a bounded, model-agnostic contrastive anchor explanation."""
    try:
        return _build_contrastive_anchor_explanation(
            baseline_features=request.baseline_features,
            disease=request.disease,
            focus_features=request.focus_features,
            top_k=request.top_k,
            context_samples=request.context_samples,
            patient_id=request.patient_id,
        )
    except HTTPException:
        raise
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as e:
        logger.error(f"Contrastive anchor error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/constrained-response-surface", response_model=ConstrainedResponseSurfaceResponse)
async def constrained_response_surface_analysis(request: ConstrainedResponseSurfaceRequest):
    """Generate a constrained, model-agnostic local response-surface explanation."""
    try:
        return _build_constrained_response_surface_explanation(
            baseline_features=request.baseline_features,
            disease=request.disease,
            focus_features=request.focus_features,
            top_k=request.top_k,
            num_points=request.num_points,
            patient_id=request.patient_id,
        )
    except HTTPException:
        raise
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as e:
        logger.error(f"Constrained response-surface error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sample-patients")
async def get_sample_patients():
    """Get sample patient data for testing."""
    return {
        "samples": [
            {
                "name": "Healthy Patient",
                "features": {
                    "age": 45, "gender": 0, "heart_rate": 72, "systolic_bp": 120,
                    "diastolic_bp": 80, "temperature": 98.6, "respiratory_rate": 16,
                    "wbc_count": 7.5, "hemoglobin": 14.0, "platelet_count": 250,
                    "creatinine": 1.0, "bun": 15, "glucose": 95, "lactate": 1.2
                }
            },
            {
                "name": "High Sepsis Risk",
                "features": {
                    "age": 68, "gender": 1, "heart_rate": 115, "systolic_bp": 95,
                    "diastolic_bp": 65, "temperature": 101.5, "respiratory_rate": 24,
                    "wbc_count": 16.5, "hemoglobin": 10.5, "platelet_count": 150,
                    "creatinine": 2.1, "bun": 40, "glucose": 180, "lactate": 3.2
                }
            },
            {
                "name": "Kidney Failure Risk",
                "features": {
                    "age": 72, "gender": 1, "heart_rate": 88, "systolic_bp": 135,
                    "diastolic_bp": 85, "temperature": 98.4, "respiratory_rate": 18,
                    "wbc_count": 8.2, "hemoglobin": 9.5, "platelet_count": 220,
                    "creatinine": 4.5, "bun": 85, "glucose": 110, "lactate": 1.8
                }
            },
            {
                "name": "Diabetes Risk",
                "features": {
                    "age": 55, "gender": 0, "heart_rate": 82, "systolic_bp": 140,
                    "diastolic_bp": 90, "temperature": 98.7, "respiratory_rate": 17,
                    "wbc_count": 7.8, "hemoglobin": 12.5, "platelet_count": 280,
                    "creatinine": 1.2, "bun": 18, "glucose": 325, "lactate": 1.5
                }
            }
        ]
    }


# ============================================================================
# FRONTEND ROUTES
# ============================================================================

FRONTEND_ENTRY = STATIC_DIR / "index.html"
SPA_RESERVED_PATHS = ("api", "docs", "redoc", "openapi.json", "health")


def serve_frontend_entry():
    """Return the built React application entry file."""
    if FRONTEND_ENTRY.exists():
        return FileResponse(FRONTEND_ENTRY)
    return JSONResponse(
        content={
            "error": "Frontend build not found",
            "detail": "Run `npm install` and `npm run build` inside the frontend folder."
        },
        status_code=503
    )


@app.get("/", include_in_schema=False)
async def serve_frontend_root():
    """Serve the React SPA root."""
    return serve_frontend_entry()


@app.get("/app", include_in_schema=False)
async def serve_frontend_app():
    """Support legacy /app route for the SPA."""
    return serve_frontend_entry()


@app.get("/login", include_in_schema=False)
async def serve_frontend_login():
    """Serve the SPA for the login route."""
    return serve_frontend_entry()


@app.get("/signup", include_in_schema=False)
async def serve_frontend_signup():
    """Serve the SPA for the signup route."""
    return serve_frontend_entry()


@app.get("/clinical", include_in_schema=False)
async def serve_frontend_clinical():
    """Serve the SPA for the protected clinical dashboard route."""
    return serve_frontend_entry()


@app.get("/predict", include_in_schema=False)
async def serve_frontend_predict():
    """Serve the SPA for the prediction workspace."""
    return serve_frontend_entry()


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend_catchall(full_path: str):
    """
    Return the SPA for any non-API route so BrowserRouter works on refresh/deep links.
    """
    normalized = full_path.strip("/")
    if not normalized:
        return serve_frontend_entry()

    # Skip API/docs/openapi paths so FastAPI can continue handling them.
    if any(normalized.startswith(prefix) for prefix in SPA_RESERVED_PATHS):
        raise HTTPException(status_code=404, detail="Not Found")

    candidate_file = STATIC_DIR / normalized
    if candidate_file.exists() and candidate_file.is_file():
        return FileResponse(candidate_file)

    return serve_frontend_entry()


# Serve medical icon
@app.get("/medical-icon.svg")
async def serve_icon():
    """Serve medical icon."""
    icon_file = STATIC_DIR / "medical-icon.svg"
    if icon_file.exists():
        return FileResponse(icon_file, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Icon not found")


# Mount static files for assets
try:
    ASSETS_DIR = STATIC_DIR / "assets"
    if ASSETS_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
except Exception as e:
    logger.warning(f"Could not mount assets: {e}")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on startup."""
    logger.info("🚀 Explainable Medical AI API Starting...")
    logger.info(f"📦 Models loaded: {len(model_manager.models)}")
    logger.info(f"🌐 Diseases: {', '.join(model_manager.models.keys())}")
    logger.info("✅ API ready at http://localhost:8000")
    logger.info("📚 API Docs at http://localhost:8000/docs")
    logger.info("🎨 Frontend at http://localhost:8000/app")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on shutdown."""
    logger.info("👋 Shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
