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
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import sys
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import os
from dotenv import load_dotenv

from feature_engineering import BASE_FEATURES, add_derived_features
from multimodal_fusion import MultimodalFusionEngine

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
    str(Path("dataset") / "mimiccxr" / "mimic-cxr-jpg-2.1.0-chexpert.csv.gz"),
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
    "heart_disease": ["systolic_bp", "heart_rate", "glucose", "creatinine"],
    "cardiovascular": ["systolic_bp", "heart_rate", "glucose", "lactate"],
    "diabetes": ["glucose", "systolic_bp", "creatinine"],
    "anemia": ["hemoglobin", "platelet_count", "creatinine"],
    "thalassemia": ["hemoglobin", "platelet_count"],
    "thrombocytopenia": ["platelet_count", "hemoglobin"],
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
}


DISEASE_DIRECT_FACTORS: Dict[str, set[str]] = {
    "sepsis": {"temperature", "wbc_count", "lactate", "respiratory_rate", "systolic_bp", "heart_rate"},
    "kidney_failure": {"creatinine", "bun", "systolic_bp", "lactate"},
    "heart_disease": {"systolic_bp", "heart_rate", "glucose", "creatinine"},
    "cardiovascular": {"systolic_bp", "heart_rate", "lactate", "glucose"},
    "diabetes": {"glucose", "systolic_bp"},
    "anemia": {"hemoglobin", "platelet_count"},
    "thalassemia": {"hemoglobin", "platelet_count"},
    "thrombocytopenia": {"platelet_count", "hemoglobin"},
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
    "heart_disease": [
        "Trend blood pressure and heart-rate variability",
        "Repeat metabolic profile including glucose and renal markers",
        "Consider guideline-based cardiac monitoring if clinically indicated",
    ],
    "cardiovascular": [
        "Trend blood pressure, heart rate, and lactate trajectory",
        "Reassess metabolic profile over time",
        "Consider cardiovascular monitoring as clinically indicated",
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
    "thalassemia": [
        "Trend hemoglobin and complete blood count",
        "Monitor platelet profile and blood indices",
        "Reassess hematology panel at follow-up",
    ],
    "thrombocytopenia": [
        "Repeat platelet trend and complete blood count",
        "Monitor bleeding-risk signs and hemodynamic stability",
        "Reassess related hematologic markers at follow-up",
    ],
    "mortality": [
        "Trend lactate, blood pressure, and respiratory status",
        "Repeat renal and metabolic profile in short intervals",
        "Monitor multi-system instability markers over time",
    ],
}


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


# ============================================================================
# MODEL MANAGER
# ============================================================================

class ModelManager:
    """Manages all trained ML models."""
    
    DISEASES = [
        'sepsis', 'kidney_failure', 'heart_disease', 'diabetes',
        'anemia', 'thalassemia', 'thrombocytopenia', 'cardiovascular', 'mortality'
    ]
    
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            # Default to trained_models relative to backend script location
            models_dir = Path(__file__).parent.parent / "trained_models"
        self.models_dir = Path(models_dir)
        self.models: Dict[str, Any] = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all trained models from disk."""
        logger.info(f"📦 Loading models from: {self.models_dir}")
        
        for disease in self.DISEASES:
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
                logger.warning(f"  ⚠️  {disease:20s} - Not found")
        
        logger.info(f"📊 Loaded {len(self.models)}/{len(self.DISEASES)} models")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features consistent with training pipeline."""
        base_df = df.copy()
        missing = [col for col in BASE_FEATURES if col not in base_df.columns]
        if missing:
            raise ValueError(f"Missing base features for inference: {missing}")
        engineered = add_derived_features(base_df[BASE_FEATURES])
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
        risk_score = float(model.predict_proba(X_scaled)[0, 1])
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
    labels_path=CXR_LABELS_PATH if Path(CXR_LABELS_PATH).exists() else None
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
        models_info.append({
            "disease": disease,
            "model_type": bundle.get('model_type', 'unknown'),
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
        if max_risk < 0.30:
            overall_risk = "LOW"
        elif max_risk < 0.50:
            overall_risk = "MODERATE"
        elif max_risk < 0.70:
            overall_risk = "HIGH"
        else:
            overall_risk = "CRITICAL"

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
            multimodal_summary=fusion_summary
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
        
        # Modified prediction
        modified_dict = baseline_features.dict()
        modified_dict.update(request.modified_features)
        modified_features = PatientFeatures(**modified_dict)
        
        modified_pred = model_manager.predict(modified_features, request.disease)
        new_risk = modified_pred.risk_score
        
        # Calculate delta
        risk_delta = new_risk - baseline_risk
        risk_delta_percent = (risk_delta / baseline_risk * 100) if baseline_risk > 0 else 0
        
        # Track changes
        feature_changes = {}
        for feature, new_value in request.modified_features.items():
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
    
    except Exception as e:
        logger.error(f"What-if error: {e}")
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
