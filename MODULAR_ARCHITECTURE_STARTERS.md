# MODULAR ARCHITECTURE - IMPLEMENTATION STARTERS

## Boilerplate Code for Each Module

This document provides starter code templates to accelerate implementation.

---

## 1. DATA LAYER STARTER CODE

### `src/1_data_layer/base_data_source.py`
```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseDataSource(ABC):
    """Abstract base class for data sources"""
    
    def __init__(self, source_path: str):
        self.source_path = source_path
        self.data: Dict[str, pd.DataFrame] = {}
    
    @abstractmethod
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load raw data tables from source"""
        pass
    
    @abstractmethod
    def validate_data(self) -> bool:
        """Validate loaded data"""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get data statistics"""
        pass
    
    def get_table(self, table_name: str) -> pd.DataFrame:
        """Retrieve specific table"""
        if table_name not in self.data:
            raise ValueError(f"Table {table_name} not found")
        return self.data[table_name]
    
    def list_tables(self) -> list:
        """List all loaded tables"""
        return list(self.data.keys())
```

### `src/1_data_layer/preprocessor.py`
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """Data preprocessing pipeline"""
    
    def __init__(self, test_size: float = 0.3, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
    
    def preprocess(self, raw_data: Dict) -> 'ProcessedDataset':
        """Execute full preprocessing pipeline"""
        logger.info("Starting data preprocessing...")
        
        # 1. Handle missing values
        df = self._handle_missing_values(raw_data)
        logger.info(f"✅ Handled missing values")
        
        # 2. Encode categorical
        df = self._encode_categorical(df)
        logger.info(f"✅ Encoded categorical features")
        
        # 3. Feature engineering
        df = self._engineer_features(df)
        logger.info(f"✅ Engineered features")
        
        # 4. Separate features and targets
        diseases = ['sepsis', 'kidney_failure', 'cardiovascular', 'mortality']
        for disease in diseases:
            if f'{disease}_target' not in df.columns:
                logger.warning(f"⚠️ {disease} target not found")
        
        # 5. Return processed dataset
        from src.core.types import ProcessedDataset
        return self._create_dataset(df)
    
    def _handle_missing_values(self, data: Dict) -> pd.DataFrame:
        """Handle missing values"""
        # Implementation
        pass
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features"""
        # Implement clinical feature engineering
        return df
    
    def _create_dataset(self, df: pd.DataFrame) -> 'ProcessedDataset':
        """Create ProcessedDataset object"""
        from sklearn.model_selection import train_test_split
        from src.core.types import ProcessedDataset
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col.endswith('_encoded') 
                       or col in ['age', 'heart_rate', 'systolic_bp']]
        
        X = df[feature_cols]
        
        # Create dataset for first disease (kidney_failure)
        if 'kidney_failure_target' in df.columns:
            y = df['kidney_failure_target']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, 
                random_state=self.random_state, stratify=y
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            X_train = pd.DataFrame(X_train_scaled, columns=feature_cols)
            X_test = pd.DataFrame(X_test_scaled, columns=feature_cols)
            
            return ProcessedDataset(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train.reset_index(drop=True),
                y_test=y_test.reset_index(drop=True),
                feature_names=feature_cols,
                scaler=self.scaler
            )
```

### `src/core/types.py`
```python
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler


@dataclass
class ProcessedDataset:
    """Output of data preprocessing"""
    X_train: pd.DataFrame           # (n_train, n_features)
    X_test: pd.DataFrame            # (n_test, n_features)
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    scaler: StandardScaler
    metadata: Dict[str, Any] = None


@dataclass
class ModelPrediction:
    """Output of model prediction"""
    disease: str
    risk_score: float               # 0-1
    risk_category: str              # LOW, MODERATE, HIGH, CRITICAL
    confidence: float
    probabilities: Dict
    model_metadata: Dict


@dataclass
class GlobalExplanation:
    """Global SHAP feature importance"""
    disease: str
    features: List[str]
    importances: List[float]
    positive_direction: List[bool]
    visualization_data: Dict


@dataclass
class LocalExplanation:
    """Local LIME explanation"""
    disease: str
    instance_id: int
    feature_contributions: Dict[str, float]
    top_features: List[str]
    clinical_summary: str
    confidence: float


@dataclass
class WhatIfAnalysis:
    """What-If scenario analysis results"""
    patient_id: int
    parameter: str
    original_value: float
    test_values: np.ndarray
    risks_per_disease: Dict[str, np.ndarray]
    optimal_value: float
    optimal_risks: Dict[str, float]
    recommendations: List[str]
```

---

## 2. MODEL SERVICES STARTER CODE

### `src/2_model_services/base_service.py`
```python
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
import pickle

logger = logging.getLogger(__name__)


class BaseModelService(ABC):
    """Abstract base class for disease-specific models"""
    
    def __init__(self, disease_name: str):
        self.disease_name = disease_name
        self.model = None
        self.metadata = {}
        self.is_trained = False
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """
        Train model
        
        Returns:
            Dict with metrics: {auc, accuracy, f1, precision, recall}
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities (positive class)
        
        Returns:
            1D array of probabilities [0-1]
        """
        pass
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model on test set
        
        Returns:
            Dict with all metrics
        """
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
        
        y_pred_proba = self.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        return {
            'auc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'prevalence': y_test.mean()
        }
    
    def save_model(self, path: str) -> None:
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"✅ Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model from disk"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info(f"✅ Model loaded from {path}")


class SepsisService(BaseModelService):
    """Sepsis prediction service"""
    
    def __init__(self):
        super().__init__("sepsis")
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, 
            class_weight='balanced', random_state=42
        )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        logger.info("🔄 Training Sepsis model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        metrics = self.evaluate(X_train, y_train)
        logger.info(f"✅ Sepsis: AUC={metrics['auc']:.3f}")
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]


class KidneyFailureService(BaseModelService):
    """Kidney failure prediction service"""
    
    def __init__(self):
        super().__init__("kidney_failure")
        import xgboost as xgb
        self.model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, 
            learning_rate=0.1, random_state=42
        )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        logger.info("🔄 Training Kidney Failure model...")
        self.model.fit(X_train, y_train, eval_metric='logloss')
        self.is_trained = True
        
        metrics = self.evaluate(X_train, y_train)
        logger.info(f"✅ Kidney Failure: AUC={metrics['auc']:.3f}")
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]


# Implement CardiovascularService and MortalityService similarly
```

### `src/2_model_services/model_registry.py`
```python
from typing import Dict
import logging
from pathlib import Path
from src.2_model_services.base_service import (
    BaseModelService, SepsisService, KidneyFailureService
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry for all disease models"""
    
    def __init__(self):
        self.services: Dict[str, BaseModelService] = {
            'sepsis': SepsisService(),
            'kidney_failure': KidneyFailureService(),
            # Add other services
        }
    
    def get_service(self, disease: str) -> BaseModelService:
        """Retrieve model service by disease"""
        if disease not in self.services:
            raise ValueError(f"Disease {disease} not registered")
        return self.services[disease]
    
    def train_all(self, X_train, y_train_dict) -> Dict:
        """Train all models"""
        results = {}
        
        for disease, service in self.services.items():
            if disease not in y_train_dict:
                logger.warning(f"⚠️ No target for {disease}, skipping")
                continue
            
            y_train = y_train_dict[disease]
            metrics = service.train(X_train, y_train)
            results[disease] = metrics
        
        return results
    
    def predict_all(self, X) -> Dict[str, list]:
        """Get predictions from all models"""
        predictions = {}
        
        for disease, service in self.services.items():
            if not service.is_trained:
                logger.warning(f"⚠️ {disease} model not trained")
                continue
            
            proba = service.predict(X)
            predictions[disease] = proba
        
        return predictions
    
    def save_all(self, model_dir: str) -> None:
        """Save all models"""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for disease, service in self.services.items():
            path = model_dir / f"{disease}_model.pkl"
            service.save_model(str(path))
    
    def load_all(self, model_dir: str) -> None:
        """Load all models"""
        model_dir = Path(model_dir)
        
        for disease, service in self.services.items():
            path = model_dir / f"{disease}_model.pkl"
            if path.exists():
                service.load_model(str(path))
```

---

## 3. XAI ENGINE STARTER CODE

### `src/3_xai_engine/shap_explainer.py`
```python
import shap
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP-based global feature importance"""
    
    def __init__(self, model, X_background: pd.DataFrame):
        self.model = model
        self.X_background = X_background
        
        # Initialize SHAP explainer
        try:
            self.explainer = shap.TreeExplainer(model)
            logger.info("✅ Using TreeExplainer for tree-based model")
        except:
            self.explainer = shap.KernelExplainer(
                model.predict_proba,
                X_background.sample(min(100, len(X_background)), random_state=42)
            )
            logger.info("⚠️ Using KernelExplainer (slower)")
    
    def explain_global(self, X: pd.DataFrame) -> Dict:
        """
        Compute global feature importance
        
        Returns:
            Dict with:
            - features: List of feature names
            - importances: Mean absolute SHAP values
            - ranking: Feature ranking by importance
        """
        logger.info("Computing SHAP values...")
        shap_values = self.explainer.shap_values(X)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        # Compute feature importance
        importances = np.abs(shap_values).mean(axis=0)
        
        # Create ranking
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return {
            'features': feature_importance['feature'].tolist(),
            'importances': feature_importance['importance'].tolist(),
            'shap_values': shap_values
        }
    
    def explain_local(self, instance: pd.Series) -> Dict:
        """
        Explain single prediction
        
        Returns:
            Dict with feature contributions
        """
        shap_values = self.explainer.shap_values([instance.values])[0]
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        contributions = pd.DataFrame({
            'feature': instance.index,
            'value': instance.values,
            'contribution': shap_values
        }).sort_values('contribution', ascending=False, key=abs)
        
        return {
            'feature_contributions': dict(zip(
                contributions['feature'], 
                contributions['contribution']
            )),
            'top_features': contributions['feature'].head(5).tolist()
        }
```

### `src/3_xai_engine/clinical_translator.py`
```python
class ClinicalTranslator:
    """Translate XAI outputs to clinical language"""
    
    # Clinical feature interpretations
    FEATURE_MEANINGS = {
        'creatinine': ('Kidney function marker', 'higher=worse'),
        'age': ('Patient age', 'higher=risk'),
        'systolic_bp': ('Blood pressure', 'higher=worse'),
        'glucose': ('Blood sugar', 'higher=worse'),
        'temperature': ('Body temperature', 'abnormal=bad'),
        'heart_rate': ('Pulse', 'abnormal=bad'),
        'white_blood_cells': ('Infection marker', 'higher=worse'),
    }
    
    @staticmethod
    def translate_contribution(feature: str, value: float, 
                             direction: str = 'positive') -> str:
        """
        Translate SHAP contribution to clinical language
        
        Example:
            Input: 'creatinine', 0.253, 'positive'
            Output: 'High creatinine (+0.253 ↗️) strongly increases risk'
        """
        if feature not in ClinicalTranslator.FEATURE_MEANINGS:
            return f"{feature}: {value:+.3f}"
        
        meaning, direction = ClinicalTranslator.FEATURE_MEANINGS[feature]
        arrow = "↗️" if value > 0 else "↘️"
        
        magnitude = "strongly" if abs(value) > 0.2 else "moderately"
        impact = "increases" if value > 0 else "decreases"
        
        return f"{meaning} ({value:+.3f} {arrow}) {magnitude} {impact} risk"
    
    @staticmethod
    def create_summary(top_features: list, contributions: dict) -> str:
        """Create human-readable clinical summary"""
        summary = "Clinical Risk Factors:\n"
        
        for i, feature in enumerate(top_features[:5], 1):
            contrib = contributions[feature]
            translation = ClinicalTranslator.translate_contribution(
                feature, contrib
            )
            summary += f"{i}. {translation}\n"
        
        return summary
```

---

## 4. WHAT-IF ENGINE STARTER CODE

### `src/4_whatif_engine/whatif_analyzer.py`
```python
import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class WhatIfAnalyzer:
    """What-If analysis engine"""
    
    def __init__(self, models):
        self.models = models
        self.feature_ranges = {
            'age': (20, 90),
            'creatinine': (0.5, 4.0),
            'systolic_bp': (90, 180),
            'glucose': (70, 300),
            'heart_rate': (60, 120),
            'temperature': (95, 105)
        }
    
    def analyze_feature_variation(self, patient: pd.Series, 
                                 feature: str, values: np.ndarray) -> Dict:
        """
        Vary single feature and analyze impact
        
        Returns:
            {feature_values, risks_per_disease}
        """
        logger.info(f"Analyzing {feature} variation...")
        
        risks_per_disease = {}
        
        for disease, service in self.models.services.items():
            if not service.is_trained:
                continue
            
            disease_risks = []
            
            for value in values:
                # Modify patient feature
                modified_patient = patient.copy()
                modified_patient[feature] = value
                
                # Get prediction
                risk = service.predict(pd.DataFrame([modified_patient]))[0]
                disease_risks.append(risk)
            
            risks_per_disease[disease] = np.array(disease_risks)
        
        return {
            'feature': feature,
            'values': values,
            'risks_per_disease': risks_per_disease,
            'original_value': float(patient[feature])
        }
    
    def optimize_parameters(self, patient: pd.Series, 
                           target_risk: float = 0.3) -> Dict:
        """
        Find parameter values that achieve target risk
        
        Simple greedy optimization
        """
        logger.info(f"Optimizing parameters to achieve {target_risk:.1%} risk...")
        
        optimal_params = {}
        
        for feature in self.feature_ranges.keys():
            min_val, max_val = self.feature_ranges[feature]
            values = np.linspace(min_val, max_val, 20)
            
            result = self.analyze_feature_variation(patient, feature, values)
            
            # Find value closest to target risk (avg across diseases)
            avg_risks = np.mean([
                result['risks_per_disease'][d] 
                for d in result['risks_per_disease'].keys()
            ], axis=0)
            
            closest_idx = np.argmin(np.abs(avg_risks - target_risk))
            optimal_params[feature] = float(values[closest_idx])
        
        return optimal_params
```

---

## 5. DASHBOARD STARTER CODE

### `src/5_dashboard_server/app.py`
```python
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import logging

logger = logging.getLogger(__name__)


class DashboardApp:
    """Main Dash application"""
    
    def __init__(self, models, xai_engine, whatif_engine):
        self.app = dash.Dash(__name__, external_stylesheets=[
            dbc.themes.BOOTSTRAP
        ])
        self.models = models
        self.xai = xai_engine
        self.whatif = whatif_engine
        
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Create dashboard layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                html.H1("🏥 Explainable Medical AI Dashboard")
            ], className="mt-4 mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Tabs([
                        dcc.Tab(label="Model Performance", children=[
                            self._create_performance_section()
                        ]),
                        dcc.Tab(label="Patient Analysis", children=[
                            self._create_patient_section()
                        ]),
                        dcc.Tab(label="What-If Analysis", children=[
                            self._create_whatif_section()
                        ])
                    ])
                ])
            ])
        ], fluid=True)
    
    def _create_performance_section(self):
        return dbc.Card([
            dbc.CardBody([
                html.H5("Model Performance Metrics"),
                dcc.Graph(id='performance-chart')
            ])
        ])
    
    def _create_patient_section(self):
        return dbc.Card([
            dbc.CardBody([
                html.H5("Patient Analysis"),
                dcc.Graph(id='risk-chart')
            ])
        ])
    
    def _create_whatif_section(self):
        return dbc.Card([
            dbc.CardBody([
                html.H5("What-If Analysis"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Age:"),
                        dcc.Slider(id='age-slider', 
                                  min=20, max=90, step=1, value=65)
                    ], width=4),
                    dbc.Col([
                        html.Label("Creatinine:"),
                        dcc.Slider(id='creatinine-slider',
                                  min=0.5, max=4, step=0.1, value=1.5)
                    ], width=4)
                ]),
                dcc.Graph(id='whatif-chart')
            ])
        ])
    
    def setup_callbacks(self):
        """Setup interactive callbacks"""
        
        @self.app.callback(
            Output('whatif-chart', 'figure'),
            [Input('age-slider', 'value'),
             Input('creatinine-slider', 'value')]
        )
        def update_whatif_chart(age, creatinine):
            # Implement what-if visualization
            import plotly.graph_objects as go
            
            fig = go.Figure()
            # Add traces
            return fig
    
    def run(self, port: int = 8051, debug: bool = False):
        """Launch dashboard"""
        logger.info(f"🌐 Starting dashboard on http://127.0.0.1:{port}")
        self.app.run_server(host='127.0.0.1', port=port, debug=debug)
```

---

## 6. EVALUATION MODULE STARTER CODE

### `src/6_evaluation_module/model_evaluator.py`
```python
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, 
    precision_score, recall_score, confusion_matrix
)
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate model performance"""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                       y_pred_proba: np.ndarray) -> dict:
        """
        Compute comprehensive metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'specificity': confusion_matrix(y_true, y_pred)[0, 0] / 
                          (confusion_matrix(y_true, y_pred)[0, 0] + 
                           confusion_matrix(y_true, y_pred)[0, 1])
        }
    
    @staticmethod
    def cross_validate(model, X, y, cv: int = 5) -> dict:
        """Perform cross-validation"""
        from sklearn.model_selection import cross_validate
        
        scoring = ['accuracy', 'roc_auc', 'f1']
        results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'accuracy_mean': results['test_accuracy'].mean(),
            'accuracy_std': results['test_accuracy'].std(),
            'auc_mean': results['test_roc_auc'].mean(),
            'auc_std': results['test_roc_auc'].std()
        }
```

---

## Quick Start Script

### `scripts/train_models.py`
```python
#!/usr/bin/env python3
"""Train all disease models using modular architecture"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_layer import Preprocessor, MIMICDataSource
from src.model_services import ModelRegistry
from src.xai_engine import SHAPExplainer
from src.evaluation_module import ModelEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("🚀 Starting modular training pipeline...")
    
    # 1. Load and preprocess data
    logger.info("\n📊 STEP 1: Data Loading & Preprocessing")
    data_source = MIMICDataSource('/path/to/mimic')
    raw_data = data_source.load_data()
    
    preprocessor = Preprocessor()
    dataset = preprocessor.preprocess(raw_data)
    
    # 2. Train all models
    logger.info("\n🤖 STEP 2: Model Training")
    registry = ModelRegistry()
    metrics = registry.train_all(dataset.X_train, {
        'kidney_failure': dataset.y_train
        # Add other diseases
    })
    
    # 3. Evaluate
    logger.info("\n📈 STEP 3: Model Evaluation")
    evaluator = ModelEvaluator()
    
    for disease, service in registry.services.items():
        if service.is_trained:
            eval_metrics = service.evaluate(dataset.X_test, dataset.y_test)
            logger.info(f"{disease}: {eval_metrics}")
    
    # 4. Save models
    logger.info("\n💾 STEP 4: Saving Models")
    registry.save_all('models/')
    
    logger.info("\n✅ Training complete!")


if __name__ == '__main__':
    main()
```

---

**END OF IMPLEMENTATION STARTERS**

Use these templates as starting points for implementing each module.
