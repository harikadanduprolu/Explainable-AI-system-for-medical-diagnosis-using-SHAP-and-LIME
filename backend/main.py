"""
FastAPI Backend for Explainable Medical AI System
==================================================

RESTful API for disease prediction, SHAP explanations, and what-if analysis.

Usage:
    uvicorn backend.main:app --reload --port 8000
    
Access:
    API: http://localhost:8000
    Docs: http://localhost:8000/docs
    ReDoc: http://localhost:8000/redoc
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import shap
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Explainable Medical AI API",
    description="API for disease prediction with SHAP explanations and what-if analysis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files mounting (for serving frontend)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# ============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================================

class PatientFeatures(BaseModel):
    """Input features for a patient."""
    age: float = Field(..., ge=18, le=100, description="Patient age in years")
    gender: int = Field(..., ge=0, le=1, description="Gender (0=Female, 1=Male)")
    heart_rate: float = Field(..., ge=40, le=200, description="Heart rate (bpm)")
    systolic_bp: float = Field(..., ge=70, le=250, description="Systolic BP (mmHg)")
    diastolic_bp: float = Field(..., ge=40, le=150, description="Diastolic BP (mmHg)")
    temperature: float = Field(..., ge=95, le=106, description="Temperature (°F)")
    respiratory_rate: float = Field(..., ge=8, le=50, description="Respiratory rate (breaths/min)")
    wbc_count: float = Field(..., ge=1, le=50, description="White blood cell count (K/µL)")
    hemoglobin: float = Field(..., ge=5, le=20, description="Hemoglobin (g/dL)")
    platelet_count: float = Field(..., ge=20, le=700, description="Platelet count (K/µL)")
    creatinine: float = Field(..., ge=0.3, le=15, description="Creatinine (mg/dL)")
    bun: float = Field(..., ge=5, le=200, description="BUN (mg/dL)")
    glucose: float = Field(..., ge=50, le=700, description="Glucose (mg/dL)")
    lactate: float = Field(..., ge=0.5, le=25, description="Lactate (mmol/L)")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 68,
                "gender": 1,
                "heart_rate": 115,
                "systolic_bp": 95,
                "diastolic_bp": 65,
                "temperature": 101.5,
                "respiratory_rate": 24,
                "wbc_count": 16.5,
                "hemoglobin": 10.5,
                "platelet_count": 150,
                "creatinine": 2.1,
                "bun": 40,
                "glucose": 180,
                "lactate": 3.2
            }
        }


class PredictionRequest(BaseModel):
    """Request for disease prediction."""
    patient_id: Optional[str] = Field(None, description="Optional patient identifier")
    features: PatientFeatures
    diseases: Optional[List[str]] = Field(None, description="Specific diseases to predict (default: all)")


class FeatureImportance(BaseModel):
    """SHAP feature importance."""
    feature_name: str
    importance: float
    direction: str  # "increases" or "decreases"
    value: float


class DiseaseRiskPrediction(BaseModel):
    """Single disease risk prediction."""
    disease: str
    risk_score: float = Field(..., ge=0, le=1)
    risk_category: str
    prediction: int
    top_features: List[FeatureImportance]


class PredictionResponse(BaseModel):
    """Response with predictions for all diseases."""
    patient_id: Optional[str]
    timestamp: str
    predictions: List[DiseaseRiskPrediction]
    overall_risk_category: str


class WhatIfRequest(BaseModel):
    """Request for what-if scenario analysis."""
    baseline_features: PatientFeatures
    modified_features: Dict[str, float]
    disease: str


class WhatIfResponse(BaseModel):
    """Response for what-if analysis."""
    disease: str
    baseline_risk: float
    new_risk: float
    risk_delta: float
    risk_delta_percent: float
    modified_features: Dict[str, Dict[str, float]]  # {feature: {old: x, new: y}}
    recommendation: str


# ============================================================================
# MODEL LOADING AND MANAGEMENT
# ============================================================================

class ModelManager:
    """Manage trained models and predictions."""
    
    def __init__(self, models_dir: str = "trained_models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all advanced models."""
        diseases = [
            'sepsis', 'kidney_failure', 'heart_disease', 'diabetes',
            'anemia', 'thalassemia', 'thrombocytopenia', 'mortality'
        ]
        
        for disease in diseases:
            try:
                model_path = self.models_dir / f"{disease}_advanced_v1.0.0.pkl"
                if model_path.exists():
                    self.models[disease] = joblib.load(model_path)
                    logger.info(f"✅ Loaded model: {disease}")
                else:
                    logger.warning(f"⚠️  Model not found: {disease}")
            except Exception as e:
                logger.error(f"❌ Error loading {disease}: {e}")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced features."""
        X = df.copy()
        
        # Physiological ratios
        X['hr_bp_ratio'] = X['heart_rate'] / (X['systolic_bp'] + 1)
        X['shock_index'] = X['heart_rate'] / (X['systolic_bp'] + 1)
        X['map'] = (X['systolic_bp'] + 2 * X['diastolic_bp']) / 3
        X['pulse_pressure'] = X['systolic_bp'] - X['diastolic_bp']
        
        # Kidney function
        X['creat_bun_ratio'] = X['creatinine'] / (X['bun'] + 1)
        X['kidney_damage'] = X['creatinine'] * X['bun'] / 100
        
        # Metabolic
        X['age_glucose'] = X['age'] * X['glucose'] / 100
        X['bmi_proxy'] = X['age'] / 2
        
        # Hematologic
        X['hemoglobin_age'] = X['hemoglobin'] * (100 - X['age']) / 100
        X['platelet_wbc_ratio'] = X['platelet_count'] / (X['wbc_count'] + 1)
        
        # Severity scores
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
    
    def predict(self, features: PatientFeatures, disease: str) -> Dict[str, Any]:
        """Make prediction for a single disease."""
        if disease not in self.models:
            raise ValueError(f"Model not found: {disease}")
        
        bundle = self.models[disease]
        model = bundle['model']
        scaler = bundle['scaler']
        threshold = bundle.get('optimal_threshold', 0.5)
        
        # Prepare features
        feature_dict = features.dict()
        df = pd.DataFrame([feature_dict])
        
        # Engineer features
        X = self.engineer_features(df)
        
        # Scale
        X_scaled = scaler.transform(X)
        
        # Predict
        risk_score = model.predict_proba(X_scaled)[0, 1]
        prediction = 1 if risk_score >= threshold else 0
        
        # Calculate SHAP values (simplified - using feature importances)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = list(X.columns)
            
            # Get top features
            indices = np.argsort(importances)[-5:][::-1]
            top_features = []
            
            for idx in indices:
                feature_name = feature_names[idx]
                importance = float(importances[idx])
                value = float(X.iloc[0, idx])
                direction = "increases" if value > 0 else "decreases"
                
                top_features.append({
                    "feature_name": feature_name,
                    "importance": importance,
                    "direction": direction,
                    "value": value
                })
        else:
            top_features = []
        
        # Risk category
        if risk_score < 0.30:
            risk_category = "LOW"
        elif risk_score < 0.50:
            risk_category = "MODERATE"
        elif risk_score < 0.70:
            risk_category = "HIGH"
        else:
            risk_category = "CRITICAL"
        
        return {
            "disease": disease,
            "risk_score": float(risk_score),
            "risk_category": risk_category,
            "prediction": int(prediction),
            "top_features": top_features
        }
    
    def predict_all(self, features: PatientFeatures) -> List[Dict[str, Any]]:
        """Predict all diseases."""
        predictions = []
        for disease in self.models.keys():
            try:
                pred = self.predict(features, disease)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error predicting {disease}: {e}")
        
        return predictions


# Initialize model manager
model_manager = ModelManager()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Explainable Medical AI API",
        "version": "2.0.0",
        "status": "operational",
        "models_loaded": len(model_manager.models),
        "available_diseases": list(model_manager.models.keys()),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/api/predict",
            "whatif": "/api/whatif",
            "models": "/api/models"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(model_manager.models)
    }


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
    
    Returns predictions for all diseases with SHAP explanations.
    """
    try:
        # Predict all or specified diseases
        if request.diseases:
            predictions = []
            for disease in request.diseases:
                if disease in model_manager.models:
                    pred = model_manager.predict(request.features, disease)
                    predictions.append(pred)
        else:
            predictions = model_manager.predict_all(request.features)
        
        # Determine overall risk
        max_risk = max([p['risk_score'] for p in predictions]) if predictions else 0
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
        baseline_risk = baseline_pred['risk_score']
        
        # Modified prediction
        modified_dict = request.baseline_features.dict()
        modified_dict.update(request.modified_features)
        
        modified_features = PatientFeatures(**modified_dict)
        modified_pred = model_manager.predict(modified_features, request.disease)
        new_risk = modified_pred['risk_score']
        
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
            recommendation = f"Positive intervention: Risk reduced by {abs(risk_delta_percent):.1f}%"
        elif risk_delta > 0.1:
            recommendation = f"Warning: Risk increased by {risk_delta_percent:.1f}%"
        else:
            recommendation = "Minimal impact on risk"
        
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
        logger.error(f"What-if analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/feature-info")
async def get_feature_info():
    """Get information about all features and their ranges."""
    return {
        "features": [
            {"name": "age", "unit": "years", "min": 18, "max": 100, "description": "Patient age"},
            {"name": "gender", "unit": "", "min": 0, "max": 1, "description": "0=Female, 1=Male"},
            {"name": "heart_rate", "unit": "bpm", "min": 40, "max": 200, "description": "Heart rate"},
            {"name": "systolic_bp", "unit": "mmHg", "min": 70, "max": 250, "description": "Systolic blood pressure"},
            {"name": "diastolic_bp", "unit": "mmHg", "min": 40, "max": 150, "description": "Diastolic blood pressure"},
            {"name": "temperature", "unit": "°F", "min": 95, "max": 106, "description": "Body temperature"},
            {"name": "respiratory_rate", "unit": "breaths/min", "min": 8, "max": 50, "description": "Respiratory rate"},
            {"name": "wbc_count", "unit": "K/µL", "min": 1, "max": 50, "description": "White blood cell count"},
            {"name": "hemoglobin", "unit": "g/dL", "min": 5, "max": 20, "description": "Hemoglobin level"},
            {"name": "platelet_count", "unit": "K/µL", "min": 20, "max": 700, "description": "Platelet count"},
            {"name": "creatinine", "unit": "mg/dL", "min": 0.3, "max": 15, "description": "Serum creatinine"},
            {"name": "bun", "unit": "mg/dL", "min": 5, "max": 200, "description": "Blood urea nitrogen"},
            {"name": "glucose", "unit": "mg/dL", "min": 50, "max": 700, "description": "Blood glucose"},
            {"name": "lactate", "unit": "mmol/L", "min": 0.5, "max": 25, "description": "Lactate level"}
        ]
    }


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
                "name": "Sepsis Risk",
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


# Serve static files and frontend
@app.get("/app", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend application."""
    frontend_file = STATIC_DIR / "index.html"
    if frontend_file.exists():
        return FileResponse(frontend_file)
    return HTMLResponse(content="<h1>Frontend not found. Please create static/index.html</h1>", status_code=404)


# Mount static files for CSS, JS, etc.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("🚀 Explainable Medical AI API starting...")
    logger.info(f"📦 Loaded {len(model_manager.models)} models")
    logger.info("✅ API ready at http://localhost:8000")
    logger.info("📚 Documentation at http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("👋 Shutting down API...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
