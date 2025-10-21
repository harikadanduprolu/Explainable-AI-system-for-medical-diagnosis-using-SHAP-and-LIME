#!/usr/bin/env python3
"""
Explainable Medical Diagnosis using SHAP and LIME

This module implements explainable AI techniques (SHAP and LIME) for medical diagnosis
prediction using the MIMIC-III dataset. It provides:

1. Model training for diagnosis prediction (in-hospital mortality, disease classification)
2. SHAP explanations for global and local interpretability
3. LIME explanations for individual patient predictions
4. Clinical validation and monitoring capabilities
5. Visualization dashboards for clinicians

Author: GitHub Copilot
License: Academic/Research Use Only
"""

import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import joblib
import json
from datetime import datetime

# ML and preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_curve, roc_curve,
    classification_report, confusion_matrix, f1_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

# Explainability
import shap
import lime
from lime import lime_tabular

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Utilities
import pyllars.utils
import pyllars.logging_utils as logging_utils

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ExplainableMedicalDiagnosis:
    """
    Main class for explainable medical diagnosis prediction.
    
    This class handles the complete pipeline from data preprocessing to 
    model training, explanation generation, and clinical validation.
    """
    
    def __init__(self, config_path: str):
        """Initialize the explainable diagnosis system."""
        self.config = pyllars.utils.load_config(config_path)
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Explainability components
        self.shap_explainer = None
        self.lime_explainer = None
        self.shap_values = None
        
        # Results storage
        self.results = {
            'model_performance': {},
            'feature_importance': {},
            'explanations': {},
            'clinical_validation': {}
        }
        
    def load_processed_data(self) -> pd.DataFrame:
        """
        Load the preprocessed MIMIC dataset with all features.
        
        Returns:
            DataFrame with processed features and targets
        """
        logger.info("Loading processed MIMIC dataset")
        
        try:
            # Load the extended episodes dataset
            df_extended = pd.read_csv(self.config['extended_episodes'])
            logger.info(f"Loaded extended episodes: {df_extended.shape}")
            
            # Load time series features
            df_ts_features = pd.read_csv(self.config['time_series_features'])
            logger.info(f"Loaded time series features: {df_ts_features.shape}")
            
            # Merge datasets
            df_complete = df_extended.merge(
                df_ts_features, 
                on=['SUBJECT_ID', 'EPISODE'], 
                how='inner'
            )
            
            logger.info(f"Complete dataset shape: {df_complete.shape}")
            return df_complete
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_diagnosis_task(self, df: pd.DataFrame, 
                             task_type: str = 'mortality') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare the dataset for a specific diagnosis prediction task.
        
        Args:
            df: Complete dataset
            task_type: Type of prediction task ('mortality', 'sepsis', 'pneumonia')
            
        Returns:
            Tuple of (features, target)
        """
        logger.info(f"Preparing diagnosis task: {task_type}")
        
        # Define target based on task type
        if task_type == 'mortality':
            target_col = 'HOSPITAL_EXPIRE_FLAG'
        elif task_type == 'sepsis':
            # Create sepsis diagnosis from ICD codes
            sepsis_codes = ['DIAGNOSIS_ICD9_038', 'DIAGNOSIS_ICD9_995']  # Example ICD-9 codes
            df['SEPSIS'] = df[sepsis_codes].any(axis=1).astype(int)
            target_col = 'SEPSIS'
        elif task_type == 'pneumonia':
            # Create pneumonia diagnosis from ICD codes
            pneumonia_codes = ['DIAGNOSIS_ICD9_486', 'DIAGNOSIS_ICD9_487']  # Example ICD-9 codes
            df['PNEUMONIA'] = df[pneumonia_codes].any(axis=1).astype(int)
            target_col = 'PNEUMONIA'
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Select clinical features
        clinical_features = [
            'AGE', 'GENDER', 'ETHNICITY', 'ADMISSION_TYPE', 'INSURANCE',
            'HEIGHT', 'WEIGHT', 'LOS'
        ]
        
        # Add time series features (vital signs, lab values)
        ts_feature_cols = [col for col in df.columns if any(
            stat in col for stat in ['MEAN', 'MIN', 'MAX', 'STD', 'COUNT']
        )]
        
        # Combine all features
        feature_cols = clinical_features + ts_feature_cols
        
        # Filter features that exist in the dataset
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Prepare features and target
        X = df[available_features].copy()
        y = df[target_col].copy()
        
        # Handle categorical variables
        categorical_cols = ['GENDER', 'ETHNICITY', 'ADMISSION_TYPE', 'INSURANCE']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype('category').cat.codes
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Prepared dataset - Features: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   model_type: str = 'xgboost') -> None:
        """
        Train a machine learning model for diagnosis prediction.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_type: Type of model to train ('xgboost', 'lightgbm', 'random_forest', 'logistic')
        """
        logger.info(f"Training {model_type} model")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Convert back to DataFrame for SHAP/LIME
        self.X_train_scaled = pd.DataFrame(
            self.X_train_scaled, 
            columns=self.feature_names,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.X_test_scaled, 
            columns=self.feature_names,
            index=self.X_test.index
        )
        
        # Initialize model
        if model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate model
        self._evaluate_model()
        
        logger.info(f"Model training completed. AUC: {self.results['model_performance']['auc']:.3f}")
    
    def _evaluate_model(self) -> None:
        """Evaluate the trained model and store results."""
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_pred_proba)
        f1 = f1_score(self.y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, self.X_train_scaled, self.y_train, 
            cv=5, scoring='roc_auc'
        )
        
        self.results['model_performance'] = {
            'accuracy': accuracy,
            'auc': auc,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"Model Performance - AUC: {auc:.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
    
    def setup_shap_explainer(self) -> None:
        """Initialize SHAP explainer based on model type."""
        logger.info("Setting up SHAP explainer")
        
        if isinstance(self.model, (xgb.XGBClassifier, lgb.LGBMClassifier)):
            self.shap_explainer = shap.TreeExplainer(self.model)
        elif isinstance(self.model, RandomForestClassifier):
            self.shap_explainer = shap.TreeExplainer(self.model)
        else:
            # Use KernelExplainer for other models (slower but more general)
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict_proba,
                self.X_train_scaled.sample(100, random_state=42)  # Background dataset
            )
    
    def compute_shap_values(self, dataset: str = 'test') -> None:
        """
        Compute SHAP values for interpretability.
        
        Args:
            dataset: Which dataset to compute SHAP values for ('train' or 'test')
        """
        logger.info(f"Computing SHAP values for {dataset} set")
        
        if self.shap_explainer is None:
            self.setup_shap_explainer()
        
        if dataset == 'test':
            data = self.X_test_scaled
        else:
            data = self.X_train_scaled.sample(min(1000, len(self.X_train_scaled)), random_state=42)
        
        # Compute SHAP values
        self.shap_values = self.shap_explainer.shap_values(data)
        
        # For binary classification, get the positive class SHAP values
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Positive class
        
        # Store feature importance
        feature_importance = np.abs(self.shap_values).mean(0)
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        self.results['feature_importance']['shap'] = importance_df
        
        logger.info("SHAP values computed successfully")
    
    def setup_lime_explainer(self) -> None:
        """Initialize LIME explainer for tabular data."""
        logger.info("Setting up LIME explainer")
        
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=self.X_train_scaled.values,
            feature_names=self.feature_names,
            class_names=['No Event', 'Event'],
            mode='classification'
        )
        
        logger.info("LIME explainer setup completed")
    
    def explain_prediction_lime(self, patient_idx: int, num_features: int = 10) -> Dict:
        """
        Generate LIME explanation for a specific patient.
        
        Args:
            patient_idx: Index of patient in test set
            num_features: Number of top features to explain
            
        Returns:
            Dictionary containing LIME explanation
        """
        if self.lime_explainer is None:
            self.setup_lime_explainer()
        
        # Get patient data
        patient_data = self.X_test_scaled.iloc[patient_idx].values
        actual_label = self.y_test.iloc[patient_idx]
        predicted_proba = self.model.predict_proba([patient_data])[0]
        
        # Generate LIME explanation
        exp = self.lime_explainer.explain_instance(
            patient_data,
            self.model.predict_proba,
            num_features=num_features,
            top_labels=2
        )
        
        # Extract explanation data
        explanation_data = {
            'patient_idx': patient_idx,
            'actual_label': actual_label,
            'predicted_proba': predicted_proba.tolist(),
            'feature_contributions': exp.as_list(),
            'local_explanation': exp.as_map()[1],  # For positive class
            'intercept': exp.intercept[1]
        }
        
        return explanation_data
    
    def generate_clinical_report(self, patient_idx: int) -> Dict:
        """
        Generate a comprehensive clinical report for a patient.
        
        Args:
            patient_idx: Index of patient in test set
            
        Returns:
            Dictionary containing clinical report
        """
        # Get basic prediction info
        patient_data = self.X_test_scaled.iloc[patient_idx]
        predicted_proba = self.model.predict_proba([patient_data.values])[0, 1]
        actual_outcome = self.y_test.iloc[patient_idx]
        
        # Get SHAP explanation for this patient
        if self.shap_values is not None:
            patient_shap = self.shap_values[patient_idx]
            shap_contribution = dict(zip(self.feature_names, patient_shap))
        else:
            shap_contribution = {}
        
        # Get LIME explanation
        lime_explanation = self.explain_prediction_lime(patient_idx)
        
        # Create clinical report
        report = {
            'patient_id': patient_idx,
            'prediction': {
                'risk_score': predicted_proba,
                'risk_category': 'High' if predicted_proba > 0.7 else 'Medium' if predicted_proba > 0.3 else 'Low',
                'actual_outcome': actual_outcome
            },
            'feature_values': patient_data.to_dict(),
            'shap_contributions': shap_contribution,
            'lime_explanation': lime_explanation,
            'top_risk_factors': self._get_top_risk_factors(patient_shap if 'patient_shap' in locals() else None),
            'clinical_recommendations': self._generate_clinical_recommendations(predicted_proba, patient_data)
        }
        
        return report
    
    def _get_top_risk_factors(self, shap_values: Optional[np.ndarray], top_k: int = 5) -> List[Dict]:
        """Extract top risk factors from SHAP values."""
        if shap_values is None:
            return []
        
        # Get absolute SHAP values and sort
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-top_k:][::-1]
        
        risk_factors = []
        for idx in top_indices:
            risk_factors.append({
                'feature': self.feature_names[idx],
                'shap_value': float(shap_values[idx]),
                'impact': 'Increases Risk' if shap_values[idx] > 0 else 'Decreases Risk'
            })
        
        return risk_factors
    
    def _generate_clinical_recommendations(self, risk_score: float, patient_data: pd.Series) -> List[str]:
        """Generate clinical recommendations based on risk score and patient data."""
        recommendations = []
        
        if risk_score > 0.7:
            recommendations.append("High-risk patient: Consider immediate clinical intervention")
            recommendations.append("Monitor vital signs closely")
            recommendations.append("Consider ICU admission or increased monitoring frequency")
        elif risk_score > 0.3:
            recommendations.append("Medium-risk patient: Enhanced monitoring recommended")
            recommendations.append("Regular assessment of clinical status")
        else:
            recommendations.append("Low-risk patient: Standard care protocols")
        
        # Add specific recommendations based on feature values
        if 'AGE' in patient_data and patient_data['AGE'] > 70:
            recommendations.append("Elderly patient: Consider age-related complications")
        
        return recommendations
    
    def create_shap_visualizations(self, save_path: Optional[str] = None) -> None:
        """Create and save SHAP visualization plots."""
        if self.shap_values is None:
            logger.warning("SHAP values not computed. Run compute_shap_values() first.")
            return
        
        logger.info("Creating SHAP visualizations")
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values, 
            self.X_test_scaled, 
            feature_names=self.feature_names,
            show=False
        )
        if save_path:
            plt.savefig(f"{save_path}/shap_summary_plot.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance plot
        plt.figure(figsize=(10, 8))
        shap.plots.bar(
            shap.Explanation(
                values=self.shap_values,
                data=self.X_test_scaled.values,
                feature_names=self.feature_names
            ),
            show=False
        )
        if save_path:
            plt.savefig(f"{save_path}/shap_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_lime_visualization(self, patient_idx: int, save_path: Optional[str] = None) -> None:
        """Create LIME visualization for a specific patient."""
        explanation = self.explain_prediction_lime(patient_idx)
        
        # Create a simple bar plot for LIME results
        features, contributions = zip(*explanation['feature_contributions'])
        
        plt.figure(figsize=(10, 6))
        colors = ['red' if x < 0 else 'green' for x in contributions]
        plt.barh(features, contributions, color=colors, alpha=0.7)
        plt.xlabel('Feature Contribution')
        plt.title(f'LIME Explanation for Patient {patient_idx}\n'
                 f'Risk Score: {explanation["predicted_proba"][1]:.3f}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/lime_explanation_patient_{patient_idx}.png", 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def validate_explanations(self) -> Dict:
        """
        Validate explanation stability and consistency.
        
        Returns:
            Dictionary containing validation metrics
        """
        logger.info("Validating explanation stability")
        
        validation_results = {}
        
        # Test explanation stability for multiple patients
        stability_scores = []
        sample_patients = np.random.choice(len(self.X_test_scaled), 
                                         min(50, len(self.X_test_scaled)), 
                                         replace=False)
        
        for patient_idx in sample_patients:
            # Generate multiple LIME explanations with different random seeds
            explanations = []
            for seed in range(5):
                np.random.seed(seed)
                exp = self.explain_prediction_lime(patient_idx, num_features=10)
                feature_dict = dict(exp['feature_contributions'])
                explanations.append(feature_dict)
            
            # Calculate stability (correlation between explanations)
            if len(explanations) > 1:
                correlations = []
                for i in range(len(explanations)):
                    for j in range(i+1, len(explanations)):
                        # Get common features
                        common_features = set(explanations[i].keys()) & set(explanations[j].keys())
                        if len(common_features) > 1:
                            values_i = [explanations[i][f] for f in common_features]
                            values_j = [explanations[j][f] for f in common_features]
                            corr = np.corrcoef(values_i, values_j)[0, 1]
                            if not np.isnan(corr):
                                correlations.append(corr)
                
                if correlations:
                    stability_scores.append(np.mean(correlations))
        
        validation_results['lime_stability'] = {
            'mean_stability': np.mean(stability_scores) if stability_scores else 0,
            'std_stability': np.std(stability_scores) if stability_scores else 0,
            'num_patients_tested': len(stability_scores)
        }
        
        # SHAP consistency check
        if self.shap_values is not None:
            # Check if SHAP values sum to difference between prediction and expected value
            expected_value = self.shap_explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]  # For binary classification
            
            predictions = self.model.predict_proba(self.X_test_scaled)[:, 1]
            shap_sums = self.shap_values.sum(axis=1) + expected_value
            
            consistency_error = np.abs(predictions - shap_sums).mean()
            validation_results['shap_consistency'] = {
                'mean_absolute_error': consistency_error,
                'max_absolute_error': np.abs(predictions - shap_sums).max()
            }
        
        self.results['clinical_validation'] = validation_results
        
        logger.info(f"Explanation validation completed. LIME stability: {validation_results.get('lime_stability', {}).get('mean_stability', 'N/A')}")
        
        return validation_results
    
    def save_model_and_results(self, output_dir: str) -> None:
        """Save the trained model and all results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, output_path / 'diagnosis_model.pkl')
        joblib.dump(self.scaler, output_path / 'feature_scaler.pkl')
        
        # Save results
        with open(output_path / 'results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save feature names
        with open(output_path / 'feature_names.json', 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        logger.info(f"Model and results saved to {output_path}")
    
    def run_complete_pipeline(self, task_type: str = 'mortality', 
                            model_type: str = 'xgboost',
                            output_dir: str = 'explainable_diagnosis_results') -> None:
        """
        Run the complete explainable diagnosis pipeline.
        
        Args:
            task_type: Type of diagnosis task
            model_type: Type of ML model to use
            output_dir: Directory to save results
        """
        logger.info("Starting complete explainable diagnosis pipeline")
        
        try:
            # 1. Load and prepare data
            df = self.load_processed_data()
            X, y = self.prepare_diagnosis_task(df, task_type)
            
            # 2. Train model
            self.train_model(X, y, model_type)
            
            # 3. Setup explainability
            self.compute_shap_values()
            self.setup_lime_explainer()
            
            # 4. Validate explanations
            self.validate_explanations()
            
            # 5. Create visualizations
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            self.create_shap_visualizations(output_dir)
            
            # Generate sample patient reports
            sample_patients = np.random.choice(len(self.X_test_scaled), 5, replace=False)
            for patient_idx in sample_patients:
                self.create_lime_visualization(patient_idx, output_dir)
                
                # Generate clinical report
                report = self.generate_clinical_report(patient_idx)
                with open(output_path / f'clinical_report_patient_{patient_idx}.json', 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            
            # 6. Save everything
            self.save_model_and_results(output_dir)
            
            logger.info("Complete pipeline execution finished successfully")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Explainable Medical Diagnosis with SHAP and LIME"
    )
    
    parser.add_argument('config', help="Path to the yaml configuration file")
    
    parser.add_argument('--task-type', choices=['mortality', 'sepsis', 'pneumonia'],
                       default='mortality',
                       help="Type of diagnosis prediction task")
    
    parser.add_argument('--model-type', 
                       choices=['xgboost', 'lightgbm', 'random_forest', 'logistic'],
                       default='xgboost',
                       help="Type of machine learning model")
    
    parser.add_argument('--output-dir', default='explainable_diagnosis_results',
                       help="Output directory for results")
    
    parser.add_argument('--patient-idx', type=int,
                       help="Specific patient index to explain (optional)")
    
    logging_utils.add_logging_options(parser)
    
    args = parser.parse_args()
    logging_utils.update_logging(args)
    
    return args


def main():
    """Main function."""
    args = parse_arguments()
    
    # Initialize the explainable diagnosis system
    diagnosis_system = ExplainableMedicalDiagnosis(args.config)
    
    if args.patient_idx is not None:
        # Explain specific patient
        logger.info(f"Explaining prediction for patient {args.patient_idx}")
        # First need to run the pipeline to get trained model
        diagnosis_system.run_complete_pipeline(args.task_type, args.model_type, args.output_dir)
        
        # Generate detailed report for specific patient
        report = diagnosis_system.generate_clinical_report(args.patient_idx)
        print(f"\nClinical Report for Patient {args.patient_idx}:")
        print(f"Risk Score: {report['prediction']['risk_score']:.3f}")
        print(f"Risk Category: {report['prediction']['risk_category']}")
        print(f"Actual Outcome: {report['prediction']['actual_outcome']}")
        
        print("\nTop Risk Factors:")
        for i, factor in enumerate(report['top_risk_factors'][:5], 1):
            print(f"{i}. {factor['feature']}: {factor['impact']} (SHAP: {factor['shap_value']:.3f})")
        
        print("\nClinical Recommendations:")
        for i, rec in enumerate(report['clinical_recommendations'], 1):
            print(f"{i}. {rec}")
    
    else:
        # Run complete pipeline
        diagnosis_system.run_complete_pipeline(
            args.task_type, 
            args.model_type, 
            args.output_dir
        )
        
        # Print summary results
        performance = diagnosis_system.results['model_performance']
        print(f"\nModel Performance Summary:")
        print(f"AUC: {performance['auc']:.3f}")
        print(f"Accuracy: {performance['accuracy']:.3f}")
        print(f"F1-Score: {performance['f1_score']:.3f}")
        print(f"Cross-validation AUC: {performance['cv_mean']:.3f} ± {performance['cv_std']:.3f}")
        
        if 'feature_importance' in diagnosis_system.results:
            print(f"\nTop 10 Most Important Features (SHAP):")
            top_features = diagnosis_system.results['feature_importance']['shap'].head(10)
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                print(f"{i}. {row['feature']}: {row['importance']:.3f}")


if __name__ == '__main__':
    main()
