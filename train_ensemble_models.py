"""
Ensemble Training Pipeline - Improved Accuracy
===============================================
Uses stacking ensemble of multiple models for better accuracy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import xgboost as xgb
import argparse


class EnsembleTrainer:
    """Train ensemble models for improved accuracy."""
    
    def __init__(self, output_dir="trained_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.scaler = StandardScaler()
    
    def create_stacking_ensemble(self, scale_pos_weight=1.0):
        """Create stacking ensemble with diverse base models."""
        
        # Base models with different characteristics
        base_models = [
            ('xgb', xgb.XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )),
            ('rf', RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                class_weight='balanced'
            )),
            ('gbc', GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ))
        ]
        
        # Meta-learner with L2 regularization
        meta_learner = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        
        # Create stacking classifier
        ensemble = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        return ensemble
    
    def train_ensemble(self, X_train, y_train, X_val, y_val, disease):
        """Train ensemble model for a disease."""
        
        print(f"\n{'='*60}")
        print(f"Training Ensemble for {disease.upper()}")
        print(f"{'='*60}")
        
        # Feature engineering
        X_train_eng = self.add_features(X_train)
        X_val_eng = self.add_features(X_val)
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train_eng)
        X_val_scaled = self.scaler.transform(X_val_eng)
        
        # Calculate class imbalance
        n_positive = y_train.sum()
        n_negative = len(y_train) - n_positive
        scale_pos_weight = n_negative / (n_positive + 1)
        
        print(f"⏳ Training ensemble (XGBoost + RandomForest + GradientBoosting)...")
        print(f"   Class balance: {n_positive} positive, {n_negative} negative (weight={scale_pos_weight:.2f})")
        
        # Create and train ensemble
        ensemble = self.create_stacking_ensemble(scale_pos_weight)
        ensemble.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_proba = ensemble.predict_proba(X_val_scaled)[:, 1]
        
        # Find optimal threshold
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Metrics
        auroc = roc_auc_score(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        print(f"✅ AUROC: {auroc:.3f}")
        print(f"✅ Accuracy: {accuracy:.3f}")
        print(f"✅ F1 Score: {f1:.3f}")
        print(f"   Optimal threshold: {optimal_threshold:.3f}")
        
        # Save model
        model_path = self.output_dir / f"{disease}_ensemble_v1.0.0.pkl"
        model_bundle = {
            'model': ensemble,
            'scaler': self.scaler,
            'feature_names': list(X_train_eng.columns),
            'disease': disease,
            'model_type': 'ensemble',
            'optimal_threshold': optimal_threshold,
            'metrics': {
                'auroc': auroc,
                'accuracy': accuracy,
                'f1_score': f1,
                'prevalence': float(y_train.mean())
            }
        }
        
        joblib.dump(model_bundle, model_path)
        print(f"💾 Saved: {model_path}")
        
        return model_bundle['metrics']
    
    def add_features(self, X):
        """Add engineered features."""
        X_new = X.copy()
        
        # Clinical interaction features
        X_new['hr_bp_ratio'] = X['heart_rate'] / (X['systolic_bp'] + 1)
        X_new['creat_bun_ratio'] = X['creatinine'] / (X['bun'] + 1)
        X_new['age_glucose'] = X['age'] * X['glucose'] / 100
        X_new['shock_index'] = X['heart_rate'] / (X['systolic_bp'] + 1)
        X_new['map'] = (X['systolic_bp'] + 2 * X['diastolic_bp']) / 3  # Mean arterial pressure
        X_new['pulse_pressure'] = X['systolic_bp'] - X['diastolic_bp']
        
        # Risk scores (simplified)
        X_new['sepsis_score'] = (
            (X['temperature'] > 100).astype(int) +
            (X['heart_rate'] > 90).astype(int) +
            (X['respiratory_rate'] > 20).astype(int) +
            (X['wbc_count'] > 12).astype(int)
        )
        
        X_new['kidney_score'] = (
            (X['creatinine'] > 1.5).astype(int) +
            (X['bun'] > 25).astype(int)
        )
        
        return X_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='mimic_training_data.csv')
    parser.add_argument('--output-dir', type=str, default='trained_models')
    args = parser.parse_args()
    
    # Load data
    print("📂 Loading data...")
    df = pd.read_csv(args.data_path)
    
    # Features and labels
    feature_cols = ['age', 'gender', 'heart_rate', 'systolic_bp', 'diastolic_bp',
                   'temperature', 'respiratory_rate', 'wbc_count', 'hemoglobin',
                   'platelet_count', 'creatinine', 'bun', 'glucose', 'lactate']
    X = df[feature_cols]
    
    disease_cols = ['sepsis', 'kidney_failure', 'heart_disease', 'diabetes',
                   'anemia', 'mortality']
    
    # Train ensemble for each disease
    trainer = EnsembleTrainer(args.output_dir)
    results = {}
    
    for disease in disease_cols:
        if disease not in df.columns:
            continue
            
        y = df[disease].values
        
        # Skip if insufficient data
        if y.sum() < 5:
            print(f"\n⚠️  Skipping {disease}: insufficient positive cases")
            continue
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Train
        metrics = trainer.train_ensemble(X_train, y_train, X_val, y_val, disease)
        results[disease] = metrics
    
    # Summary
    print("\n" + "="*60)
    print("🎉 Ensemble Training Complete!")
    print("="*60)
    
    for disease, metrics in results.items():
        print(f"{disease:20s} AUROC: {metrics['auroc']:.3f}  "
              f"Acc: {metrics['accuracy']:.3f}  "
              f"F1: {metrics['f1_score']:.3f}")
    
    print(f"\n💾 Models saved to: {args.output_dir}/")
    print("\n📝 Ensemble models combine XGBoost + RandomForest + GradientBoosting")
    print("   for improved accuracy and robustness!")


if __name__ == "__main__":
    main()
