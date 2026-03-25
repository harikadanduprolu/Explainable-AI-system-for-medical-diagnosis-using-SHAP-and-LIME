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


class PredictionRequest(BaseModel):
    """Request for disease prediction."""
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    features: PatientFeatures
    diseases: Optional[List[str]] = Field(None, description="Specific diseases to predict")


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
    top_features: List[FeatureImportance] = []


class PredictionResponse(BaseModel):
    """Complete prediction response."""
    patient_id: Optional[str]
    timestamp:str
    predictions: List[DiseasePrediction]
    overall_risk_category: str


class WhatIfRequest(BaseModel):
    """What-if scenario analysis request."""
    baseline_features: PatientFeatures
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
        """Engineer advanced features."""
        X = df.copy()
        eps = 1e-6
        
        # Physiological ratios
        X['hr_bp_ratio'] = X['heart_rate'] / (X['systolic_bp'] + eps)
        X['shock_index'] = X['heart_rate'] / (X['systolic_bp'] + eps)
        X['map'] = (X['systolic_bp'] + 2 * X['diastolic_bp']) / 3
        X['pulse_pressure'] = X['systolic_bp'] - X['diastolic_bp']
        
        # Kidney function
        X['creat_bun_ratio'] = X['creatinine'] / (X['bun'] + eps)
        X['kidney_damage'] = X['creatinine'] * X['bun'] / 100
        
        # Metabolic
        X['age_glucose'] = X['age'] * X['glucose'] / 100
        X['bmi_proxy'] = X['age'] / 2
        
        # Hematologic
        X['hemoglobin_age'] = X['hemoglobin'] * (100 - X['age']) / 100
        X['platelet_wbc_ratio'] = X['platelet_count'] / (X['wbc_count'] + eps)
        
        # Clinical scores
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
        
        # Interaction terms
        X['age_squared'] = X['age'] ** 2
        X['glucose_squared'] = (X['glucose'] / 100) ** 2
        X['lactate_squared'] = X['lactate'] ** 2
        X['hr_map_interaction'] = X['heart_rate'] * X['map'] / 1000
        X['temp_wbc_interaction'] = X['temperature'] * X['wbc_count'] / 100
        
        return X
    
    def predict(self, features: PatientFeatures, disease: str) -> DiseasePrediction:
        """Make prediction for a single disease."""
        if disease not in self.models:
            raise ValueError(f"Model not found: {disease}")
        
        bundle = self.models[disease]
        model = bundle['model']
        scaler = bundle['scaler']
        threshold = bundle.get('optimal_threshold', 0.5)
        
        # Prepare features
        df = pd.DataFrame([features.dict()])
        X = self.engineer_features(df)
        
        # Get feature names from scaler
        if hasattr(scaler, 'feature_names_in_'):
            feature_names = list(scaler.feature_names_in_)
            X = X[feature_names]
        
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
        
        # Get top features (simplified - using model feature importances)
        top_features = []
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-5:][::-1]
            
            for idx in indices:
                fname = feature_names[idx] if hasattr(scaler, 'feature_names_in_') else f"feature_{idx}"
                top_features.append(FeatureImportance(
                    feature_name=fname,
                    importance=float(importances[idx]),
                    value=float(X.iloc[0, idx])
                ))
        
        return DiseasePrediction(
            disease=disease,
            risk_score=risk_score,
            risk_category=risk_category,
            prediction=prediction,
            model_type=bundle.get('model_type', 'unknown'),
            threshold=threshold,
            top_features=top_features
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


# Initialize model manager
model_manager = ModelManager()


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
            "whatif": "/api/whatif",
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
        
        return PredictionResponse(
            patient_id=request.patient_id,
            timestamp=datetime.now().isoformat(),
            predictions=predictions,
            overall_risk_category=overall_risk
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/whatif", response_model=WhatIfResponse)
async def whatif_analysis(request: WhatIfRequest):
    """
    Perform what-if scenario analysis.
    
    Compares baseline risk vs modified features risk.
    """
    try:
        # Baseline prediction
        baseline_pred = model_manager.predict(request.baseline_features, request.disease)
        baseline_risk = baseline_pred.risk_score
        
        # Modified prediction
        modified_dict = request.baseline_features.dict()
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
            old_value = getattr(request.baseline_features, feature)
            feature_changes[feature] = {"old": old_value, "new": new_value}
        
        # Generate recommendation
        if risk_delta < -0.1:
            recommendation = f"✅ Positive intervention: Risk reduced by {abs(risk_delta_percent):.1f}%"
        elif risk_delta > 0.1:
            recommendation = f"⚠️ Warning: Risk increased by {risk_delta_percent:.1f}%"
        else:
            recommendation = "ℹ️ Minimal impact on risk"
        
        return WhatIfResponse(
            disease=request.disease,
            baseline_risk=baseline_risk,
            new_risk=new_risk,
            risk_delta=risk_delta,
            risk_delta_percent=risk_delta_percent,
            modified_features=feature_changes,
            recommendation=recommendation
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
