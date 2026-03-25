"""
Disease-Specific Model Service Class
=====================================

A unified framework for disease prediction models following the Explainable AI methodology.

Supports:
- Multiple data types: EHR (tabular), Labs (time-series), Images (CNN)
- Multiple model types: XGBoost, Random Forest, Neural Networks, CNN
- Integrated SHAP/LIME explainability
- Clinical risk scoring

Usage:
    # Create a sepsis prediction service
    service = SepsisModelService(
        disease_name="Sepsis",
        data_type="EHR",
        model_type="XGBoost"
    )
    
    # Train the model
    metrics = service.train(X_train, y_train, X_val, y_val)
    
    # Make predictions
    predictions = service.predict(X_test)
    
    # Get explanations
    explanations = service.explain(X_sample, method="shap")
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Literal, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationInfo
import joblib
from pathlib import Path

# Optional imports (install as needed)
try:
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
    import shap
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    pass


# ============================================================================
# TYPE DEFINITIONS AND SCHEMAS
# ============================================================================

class DataType(str, Enum):
    """Supported data types for disease prediction."""
    EHR = "EHR"  # Electronic Health Records (tabular)
    LABS = "Labs"  # Laboratory values (time-series)
    IMAGES = "Images"  # Medical images (CNN)
    MULTIMODAL = "Multimodal"  # Combined data types


class ModelType(str, Enum):
    """Supported model architectures."""
    XGBOOST = "XGBoost"
    RANDOM_FOREST = "RandomForest"
    NEURAL_NETWORK = "NeuralNetwork"
    CNN = "CNN"
    ENSEMBLE = "Ensemble"


class ExplainerType(str, Enum):
    """Supported explainability methods."""
    SHAP = "shap"
    LIME = "lime"
    BOTH = "both"


# ============================================================================
# INPUT/OUTPUT SCHEMAS (Pydantic Models)
# ============================================================================

class ModelInput(BaseModel):
    """Input schema for model prediction."""
    patient_id: str = Field(..., description="Unique patient identifier")
    features: Dict[str, float] = Field(..., description="Feature values")
    feature_names: List[str] = Field(..., description="Ordered feature names")
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v, info: ValidationInfo):
        """Ensure all feature names have corresponding values."""
        values = info.data or {}
        if 'feature_names' in values:
            missing = set(values['feature_names']) - set(v.keys())
            if missing:
                raise ValueError(f"Missing features: {missing}")
        return v

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "patient_id": "P12345",
                "features": {
                    "age": 65.0,
                    "heart_rate": 110.0,
                    "temperature": 38.5,
                    "wbc_count": 15000.0
                },
                "feature_names": ["age", "heart_rate", "temperature", "wbc_count"]
            }
        }
    )


class ModelPrediction(BaseModel):
    """Output schema for model prediction."""
    patient_id: str = Field(..., description="Unique patient identifier")
    disease: str = Field(..., description="Disease being predicted")
    probability: float = Field(..., ge=0.0, le=1.0, description="Risk probability [0-1]")
    risk_level: Literal["Low", "Moderate", "High", "Critical"] = Field(..., description="Clinical risk category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")
    
    model_config = ConfigDict(json_schema_extra={
            "example": {
                "patient_id": "P12345",
                "disease": "Sepsis",
                "probability": 0.78,
                "risk_level": "High",
                "confidence": 0.92,
                "timestamp": "2026-01-13T10:30:00Z"
            }
        }
    )


class ExplanationOutput(BaseModel):
    """Output schema for model explanation."""
    patient_id: str
    disease: str
    method: ExplainerType
    feature_importances: Dict[str, float] = Field(..., description="Feature → importance mapping")
    top_features: List[str] = Field(..., description="Top contributing features (ordered)")
    clinical_translation: Dict[str, str] = Field(..., description="Feature → clinical meaning")
    base_value: Optional[float] = Field(None, description="SHAP base value (expected output)")
    
    model_config = ConfigDict(json_schema_extra={
            "example": {
                "patient_id": "P12345",
                "disease": "Sepsis",
                "method": "shap",
                "feature_importances": {
                    "temperature": 0.15,
                    "wbc_count": 0.12,
                    "heart_rate": 0.08
                },
                "top_features": ["temperature", "wbc_count", "heart_rate"],
                "clinical_translation": {
                    "temperature": "Elevated body temperature (fever) indicates infection",
                    "wbc_count": "High white blood cell count suggests immune response"
                },
                "base_value": 0.098
            }
        }
    )


class TrainingMetrics(BaseModel):
    """Output schema for model training evaluation."""
    disease: str
    model_type: ModelType
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    roc_auc: float = Field(..., ge=0.0, le=1.0)
    samples_train: int = Field(..., gt=0)
    samples_val: int = Field(..., gt=0)
    class_balance: Dict[str, float] = Field(..., description="Class distribution")
    
    model_config = ConfigDict(json_schema_extra={
            "example": {
                "disease": "Sepsis",
                "model_type": "XGBoost",
                "accuracy": 0.89,
                "precision": 0.85,
                "recall": 0.82,
                "f1_score": 0.83,
                "roc_auc": 0.91,
                "samples_train": 8000,
                "samples_val": 2000,
                "class_balance": {"negative": 0.902, "positive": 0.098}
            }
        }
    )


# ============================================================================
# BASE MODEL SERVICE (Abstract Class)
# ============================================================================

class BaseModelService(ABC):
    """
    Abstract base class for disease-specific model services.
    
    Implements the Unified Explainable AI methodology:
    1. Feature engineering from clinical data
    2. Model training with class balancing
    3. Risk prediction with probability calibration
    4. Integrated SHAP/LIME explainability
    5. Clinical translation of explanations
    
    Each disease extends this base class and implements disease-specific logic.
    """
    
    def __init__(
        self,
        disease_name: str,
        data_type: DataType,
        model_type: ModelType,
        feature_names: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the model service.
        
        Args:
            disease_name: Name of the disease (e.g., "Sepsis", "Acute Kidney Injury")
            data_type: Type of input data (EHR, Labs, Images)
            model_type: Model architecture to use
            feature_names: List of feature names (required for tabular data)
            model_params: Model-specific hyperparameters
        """
        self.disease_name = disease_name
        self.data_type = data_type
        self.model_type = model_type
        self.feature_names = feature_names or []
        self.model_params = model_params or {}
        
        # Model components
        self.model = None
        self.shap_explainer = None
        self.lime_explainer = None
        self.is_trained = False
        
        # Clinical metadata
        self.class_prevalence = None
        self.risk_thresholds = self._get_risk_thresholds()
        self.clinical_features = self._get_clinical_features()
    
    # ------------------------------------------------------------------------
    # ABSTRACT METHODS (Must be implemented by subclasses)
    # ------------------------------------------------------------------------
    
    @abstractmethod
    def _get_risk_thresholds(self) -> Dict[str, float]:
        """
        Define disease-specific risk thresholds for clinical interpretation.
        
        Returns:
            Dict with keys: 'low', 'moderate', 'high', 'critical'
        
        Example:
            {'low': 0.15, 'moderate': 0.35, 'high': 0.60, 'critical': 0.80}
        """
        pass
    
    @abstractmethod
    def _get_clinical_features(self) -> Dict[str, str]:
        """
        Map feature names to clinical descriptions.
        
        Returns:
            Dict mapping feature_name → clinical_description
        
        Example:
            {
                'temperature': 'Body temperature (fever indicator)',
                'wbc_count': 'White blood cell count (infection marker)'
            }
        """
        pass
    
    @abstractmethod
    def _create_model(self) -> Any:
        """
        Instantiate the ML model based on model_type.
        
        Returns:
            Untrained model instance (sklearn, xgboost, keras, etc.)
        """
        pass
    
    # ------------------------------------------------------------------------
    # CORE METHODS (Implemented with default behavior)
    # ------------------------------------------------------------------------
    
    def train(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: np.ndarray,
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> TrainingMetrics:
        """
        Train the disease prediction model.
        
        Methodology (from Unified XAI paper):
        1. Apply class balancing (SMOTE or class_weight)
        2. Train model with validation monitoring
        3. Initialize SHAP explainer on training data
        4. Compute evaluation metrics
        
        Args:
            X_train: Training features [n_samples, n_features]
            y_train: Training labels [n_samples]
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters
        
        Returns:
            TrainingMetrics with accuracy, precision, recall, F1, ROC-AUC
        """
        print(f"[{self.disease_name}] Training {self.model_type.value} model...")
        
        # Convert to numpy if needed
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        
        # Store class prevalence for risk calibration
        self.class_prevalence = {
            "negative": 1 - np.mean(y_train),
            "positive": np.mean(y_train)
        }
        
        # Create model instance
        self.model = self._create_model()
        
        # Train model
        if X_val is not None and y_val is not None:
            # Use validation set for early stopping (if supported)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)] if hasattr(self.model, 'eval_set') else None,
                **kwargs
            )
        else:
            self.model.fit(X_train, y_train, **kwargs)
        
        self.is_trained = True
        
        # Initialize explainers after training
        self._initialize_explainers(X_train)
        
        # Evaluate on validation set (or training set if no validation)
        X_eval = X_val if X_val is not None else X_train
        y_eval = y_val if y_val is not None else y_train
        
        metrics = self.evaluate(X_eval, y_eval, samples_train=len(y_train))
        
        print(f"[{self.disease_name}] Training complete. ROC-AUC: {metrics.roc_auc:.3f}")
        return metrics
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame, ModelInput],
        return_proba: bool = True
    ) -> Union[ModelPrediction, List[ModelPrediction]]:
        """
        Predict disease risk for patient(s).
        
        Args:
            X: Input features (array, DataFrame, or ModelInput schema)
            return_proba: If True, return probability; if False, return binary class
        
        Returns:
            ModelPrediction schema with risk probability and clinical interpretation
        """
        if not self.is_trained:
            raise RuntimeError(f"{self.disease_name} model has not been trained yet.")
        
        # Handle different input types
        if isinstance(X, ModelInput):
            # Single patient prediction
            patient_id = X.patient_id
            X_array = np.array([X.features[f] for f in X.feature_names]).reshape(1, -1)
            single_prediction = True
        elif isinstance(X, pd.DataFrame):
            patient_id = None
            X_array = X.values
            single_prediction = len(X) == 1
        else:
            patient_id = None
            X_array = np.array(X)
            single_prediction = X_array.shape[0] == 1
        
        # Get probability predictions
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X_array)[:, 1]
        else:
            probas = self.model.predict(X_array)
        
        # Create prediction objects
        predictions = []
        for i, proba in enumerate(probas):
            pred = ModelPrediction(
                patient_id=patient_id or f"Patient_{i}",
                disease=self.disease_name,
                probability=float(proba),
                risk_level=self._interpret_risk(proba),
                confidence=self._compute_confidence(proba),
                timestamp=pd.Timestamp.now().isoformat()
            )
            predictions.append(pred)
        
        return predictions[0] if single_prediction else predictions
    
    def evaluate(
        self,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: np.ndarray,
        samples_train: Optional[int] = None
    ) -> TrainingMetrics:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            TrainingMetrics with comprehensive evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError(f"{self.disease_name} model has not been trained yet.")
        
        # Convert to numpy if needed
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else y_pred
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        roc_auc = roc_auc_score(y_test, y_proba)
        
        return TrainingMetrics(
            disease=self.disease_name,
            model_type=self.model_type,
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            roc_auc=float(roc_auc),
            samples_train=samples_train if samples_train is not None else len(y_test),
            samples_val=len(y_test),
            class_balance=self.class_prevalence or {"negative": 0.5, "positive": 0.5}
        )
    
    def explain(
        self,
        X: Union[np.ndarray, pd.DataFrame, ModelInput],
        method: ExplainerType = ExplainerType.SHAP,
        top_k: int = 5
    ) -> ExplanationOutput:
        """
        Generate explanation for prediction using SHAP or LIME.
        
        Integration with SHAP/LIME (Unified XAI methodology):
        - SHAP: Global feature importance + local SHAP values
        - LIME: Local linear approximation
        - Clinical translation: Map features to clinical meaning
        
        Args:
            X: Input features (single patient)
            method: Explainer to use (shap, lime, or both)
            top_k: Number of top features to return
        
        Returns:
            ExplanationOutput with feature importances and clinical translation
        """
        if not self.is_trained:
            raise RuntimeError(f"{self.disease_name} model has not been trained yet.")
        
        # Handle input type
        if isinstance(X, ModelInput):
            patient_id = X.patient_id
            X_array = np.array([X.features[f] for f in X.feature_names]).reshape(1, -1)
        elif isinstance(X, pd.DataFrame):
            patient_id = "Unknown"
            X_array = X.values[0:1]
        else:
            patient_id = "Unknown"
            X_array = np.array(X).reshape(1, -1)
        
        # Generate explanations
        if method == ExplainerType.SHAP:
            importances = self._explain_shap(X_array)
            base_value = self.shap_explainer.expected_value if self.shap_explainer else None
        elif method == ExplainerType.LIME:
            importances = self._explain_lime(X_array)
            base_value = None
        else:  # Both
            shap_imp = self._explain_shap(X_array)
            lime_imp = self._explain_lime(X_array)
            # Average the two methods
            importances = {k: (shap_imp.get(k, 0) + lime_imp.get(k, 0)) / 2 
                          for k in set(shap_imp) | set(lime_imp)}
            base_value = self.shap_explainer.expected_value if self.shap_explainer else None
        
        # Get top features
        sorted_features = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = [f for f, _ in sorted_features[:top_k]]
        
        # Clinical translation
        clinical_translation = {
            f: self.clinical_features.get(f, f"Clinical factor: {f}")
            for f in top_features
        }
        
        return ExplanationOutput(
            patient_id=patient_id,
            disease=self.disease_name,
            method=method,
            feature_importances=importances,
            top_features=top_features,
            clinical_translation=clinical_translation,
            base_value=float(base_value) if base_value is not None else None
        )
    
    # ------------------------------------------------------------------------
    # UTILITY METHODS
    # ------------------------------------------------------------------------
    
    def _initialize_explainers(self, X_train: np.ndarray):
        """Initialize SHAP and LIME explainers after training."""
        print(f"[{self.disease_name}] Initializing explainers...")
        
        # SHAP explainer
        try:
            if self.model_type in [ModelType.XGBOOST, ModelType.RANDOM_FOREST]:
                self.shap_explainer = shap.TreeExplainer(self.model)
            else:
                # Use KernelExplainer for non-tree models
                background = shap.sample(X_train, min(100, len(X_train)))
                self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, background)
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer: {e}")
        
        # LIME explainer
        try:
            self.lime_explainer = LimeTabularExplainer(
                X_train,
                feature_names=self.feature_names,
                class_names=['Negative', 'Positive'],
                mode='classification'
            )
        except Exception as e:
            print(f"Warning: Could not initialize LIME explainer: {e}")
    
    def _explain_shap(self, X: np.ndarray) -> Dict[str, float]:
        """Generate SHAP feature importances."""
        if self.shap_explainer is None:
            return {}
        
        shap_values = self.shap_explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get positive class
        
        importances = dict(zip(self.feature_names, shap_values[0]))
        return importances
    
    def _explain_lime(self, X: np.ndarray) -> Dict[str, float]:
        """Generate LIME feature importances."""
        if self.lime_explainer is None:
            return {}
        
        exp = self.lime_explainer.explain_instance(
            X[0],
            self.model.predict_proba,
            num_features=len(self.feature_names)
        )
        
        importances = dict(exp.as_list())
        return importances
    
    def _interpret_risk(self, probability: float) -> str:
        """Convert probability to clinical risk level."""
        thresholds = self.risk_thresholds
        
        if probability < thresholds['low']:
            return "Low"
        elif probability < thresholds['moderate']:
            return "Moderate"
        elif probability < thresholds['high']:
            return "High"
        else:
            return "Critical"
    
    def _compute_confidence(self, probability: float) -> float:
        """
        Compute model confidence score.
        
        Confidence is higher when probability is far from decision boundary (0.5).
        Formula: confidence = 2 * |probability - 0.5|
        """
        return 2 * abs(probability - 0.5)
    
    def save(self, path: str):
        """Save trained model and explainers to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model.")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, save_path / "model.pkl")
        joblib.dump(self.feature_names, save_path / "features.pkl")
        joblib.dump(self.class_prevalence, save_path / "prevalence.pkl")
        
        print(f"[{self.disease_name}] Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model from disk."""
        load_path = Path(path)
        
        self.model = joblib.load(load_path / "model.pkl")
        self.feature_names = joblib.load(load_path / "features.pkl")
        self.class_prevalence = joblib.load(load_path / "prevalence.pkl")
        self.is_trained = True
        
        print(f"[{self.disease_name}] Model loaded from {path}")


# ============================================================================
# CONCRETE IMPLEMENTATION EXAMPLES
# ============================================================================

class SepsisModelService(BaseModelService):
    """
    Sepsis prediction model service.
    
    Clinical Context:
    - Sepsis is a life-threatening condition (mortality: 15-30%)
    - Early detection critical for treatment
    - Key indicators: fever, tachycardia, leukocytosis, hypotension
    
    Features (from MIMIC-III):
    - Vital signs: temperature, heart rate, blood pressure
    - Lab values: WBC count, lactate, creatinine
    - Demographics: age, gender
    """
    
    def __init__(self, model_type: ModelType = ModelType.XGBOOST, **kwargs):
        super().__init__(
            disease_name="Sepsis",
            data_type=DataType.EHR,
            model_type=model_type,
            **kwargs
        )
    
    def _get_risk_thresholds(self) -> Dict[str, float]:
        """Sepsis-specific risk thresholds (calibrated on MIMIC-III prevalence: 9.8%)."""
        return {
            'low': 0.05,      # < 5% risk: Low concern
            'moderate': 0.15,  # 5-15%: Monitor closely
            'high': 0.30,      # 15-30%: High risk - immediate attention
            'critical': 0.50   # > 50%: Critical - emergency intervention
        }
    
    def _get_clinical_features(self) -> Dict[str, str]:
        """Map features to clinical descriptions for sepsis."""
        return {
            'temperature': 'Body temperature (fever indicates infection)',
            'heart_rate': 'Heart rate (tachycardia in sepsis)',
            'respiratory_rate': 'Respiratory rate (tachypnea in sepsis)',
            'systolic_bp': 'Systolic blood pressure (hypotension indicates shock)',
            'wbc_count': 'White blood cell count (elevated or depressed in infection)',
            'lactate': 'Blood lactate (tissue hypoperfusion marker)',
            'creatinine': 'Serum creatinine (kidney function)',
            'age': 'Patient age (elderly at higher risk)',
            'glasgow_coma_scale': 'Mental status (altered in severe sepsis)'
        }
    
    def _create_model(self) -> Any:
        """Create XGBoost model with sepsis-optimized hyperparameters."""
        if self.model_type == ModelType.XGBOOST:
            return xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                scale_pos_weight=9.2,  # Class imbalance: 9.8% prevalence
                objective='binary:logistic',
                eval_metric='auc',
                **self.model_params
            )
        elif self.model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                **self.model_params
            )
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented for Sepsis.")


class KidneyFailureModelService(BaseModelService):
    """
    Acute Kidney Injury (AKI) prediction model service.
    
    Clinical Context:
    - AKI affects 10-15% of hospitalized patients
    - KDIGO criteria: creatinine increase or urine output decrease
    - Key indicators: creatinine, urea, urine output, fluid balance
    """
    
    def __init__(self, model_type: ModelType = ModelType.RANDOM_FOREST, **kwargs):
        super().__init__(
            disease_name="Acute Kidney Injury",
            data_type=DataType.EHR,
            model_type=model_type,
            **kwargs
        )
    
    def _get_risk_thresholds(self) -> Dict[str, float]:
        """AKI-specific risk thresholds (prevalence: 30.3%)."""
        return {
            'low': 0.15,
            'moderate': 0.35,
            'high': 0.55,
            'critical': 0.75
        }
    
    def _get_clinical_features(self) -> Dict[str, str]:
        """Map features to clinical descriptions for AKI."""
        return {
            'creatinine': 'Serum creatinine (primary AKI indicator)',
            'urea': 'Blood urea nitrogen (kidney function)',
            'urine_output': 'Urine output (oliguria indicates AKI)',
            'fluid_balance': 'Fluid intake vs output (fluid overload)',
            'potassium': 'Serum potassium (hyperkalemia in kidney failure)',
            'bicarbonate': 'Serum bicarbonate (metabolic acidosis)',
            'hemoglobin': 'Hemoglobin (anemia in chronic kidney disease)',
            'age': 'Patient age (risk increases with age)',
            'baseline_creatinine': 'Baseline creatinine (reference value)'
        }
    
    def _create_model(self) -> Any:
        """Create model optimized for AKI prediction."""
        if self.model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                class_weight='balanced',
                **self.model_params
            )
        elif self.model_type == ModelType.XGBOOST:
            return xgb.XGBClassifier(
                max_depth=8,
                learning_rate=0.05,
                n_estimators=200,
                scale_pos_weight=2.3,  # Prevalence: 30.3%
                **self.model_params
            )
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented for AKI.")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Example usage demonstrating the complete workflow.
    """
    
    print("=" * 80)
    print("DISEASE-SPECIFIC MODEL SERVICE - USAGE EXAMPLES")
    print("=" * 80)
    
    # Example 1: Create and train a Sepsis model
    print("\n[Example 1] Training Sepsis Prediction Model")
    print("-" * 80)
    
    sepsis_service = SepsisModelService(
        model_type=ModelType.XGBOOST,
        feature_names=['temperature', 'heart_rate', 'wbc_count', 'lactate', 'age']
    )
    
    # Generate synthetic training data
    np.random.seed(42)
    X_train = np.random.randn(1000, 5)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)  # Simple rule
    X_val = np.random.randn(200, 5)
    y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(int)
    
    # Train model
    metrics = sepsis_service.train(X_train, y_train, X_val, y_val)
    print(f"\nTraining Metrics:")
    print(f"  Accuracy: {metrics.accuracy:.3f}")
    print(f"  ROC-AUC: {metrics.roc_auc:.3f}")
    print(f"  F1-Score: {metrics.f1_score:.3f}")
    
    # Example 2: Make predictions
    print("\n[Example 2] Making Predictions")
    print("-" * 80)
    
    # Create input using Pydantic schema
    patient_input = ModelInput(
        patient_id="P12345",
        features={
            'temperature': 38.5,
            'heart_rate': 110.0,
            'wbc_count': 15000.0,
            'lactate': 2.5,
            'age': 65.0
        },
        feature_names=['temperature', 'heart_rate', 'wbc_count', 'lactate', 'age']
    )
    
    prediction = sepsis_service.predict(patient_input)
    print(f"\nPrediction:")
    print(f"  Patient: {prediction.patient_id}")
    print(f"  Disease: {prediction.disease}")
    print(f"  Risk Probability: {prediction.probability:.1%}")
    print(f"  Risk Level: {prediction.risk_level}")
    print(f"  Confidence: {prediction.confidence:.1%}")
    
    # Example 3: Generate explanations
    print("\n[Example 3] Generating SHAP Explanations")
    print("-" * 80)
    
    explanation = sepsis_service.explain(patient_input, method=ExplainerType.SHAP, top_k=3)
    print(f"\nExplanation:")
    print(f"  Method: {explanation.method}")
    print(f"  Top Features:")
    for i, feature in enumerate(explanation.top_features, 1):
        importance = explanation.feature_importances[feature]
        clinical = explanation.clinical_translation[feature]
        print(f"    {i}. {feature}: {importance:+.3f}")
        print(f"       → {clinical}")
    
    # Example 4: Save and load model
    print("\n[Example 4] Saving and Loading Model")
    print("-" * 80)
    
    sepsis_service.save("models/sepsis_xgboost")
    print("✓ Model saved")
    
    # Create new service and load
    new_service = SepsisModelService()
    new_service.load("models/sepsis_xgboost")
    print("✓ Model loaded")
    
    # Verify loaded model works
    prediction2 = new_service.predict(patient_input)
    print(f"✓ Loaded model prediction: {prediction2.probability:.1%}")
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
