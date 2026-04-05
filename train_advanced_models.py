"""
Advanced Training Pipeline - Maximum Accuracy
==============================================
Trains advanced models using real MIMIC-IV mini data.

Default behavior:
- Load raw MIMIC-IV tables from dataset/mimic4/...
- Build a supervised training CSV using load_mimic_for_training.py logic
- Train advanced models with feature engineering and model selection
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    average_precision_score,
)
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import warnings

from load_mimic_for_training import MIMICDataLoader

warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """Advanced feature engineering for clinical data."""
    
    @staticmethod
    def engineer_features(df):
        """Create advanced clinical features."""
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
        X['bmi_proxy'] = X['age'] / 2  # Placeholder for BMI
        
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
        
        # Interaction terms for non-linear relationships
        X['age_squared'] = X['age'] ** 2
        X['glucose_squared'] = (X['glucose'] / 100) ** 2
        X['lactate_squared'] = X['lactate'] ** 2
        
        # Polynomial features for key vitals
        X['hr_map_interaction'] = X['heart_rate'] * X['map'] / 1000
        X['temp_wbc_interaction'] = X['temperature'] * X['wbc_count'] / 100
        
        return X


class AdvancedModelTrainer:
    """Train models with advanced techniques."""
    
    def __init__(self, output_dir="trained_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.scaler = StandardScaler()
    
    @staticmethod
    def _find_optimal_threshold(y_true, y_pred_proba, objective='f1'):
        """Choose a decision threshold for the selected objective."""
        if objective == 'youden':
            from sklearn.metrics import roc_curve

            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        thresholds = np.linspace(0.05, 0.95, 181)
        best_threshold = 0.5
        best_score = -np.inf

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            if objective == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            elif objective == 'balanced_accuracy':
                score = balanced_accuracy_score(y_true, y_pred)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return float(best_threshold)

    def train_optimized_model(self, X_train, y_train, X_val, y_val, disease, threshold_objective='f1'):
        """Train with grid search and optimal hyperparameters."""
        
        print(f"\n{'='*60}")
        print(f"Training {disease.upper()} - Advanced Pipeline")
        print(f"{'='*60}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Calculate class weights
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / (n_pos + 1)
        
        print(f"📊 Data: {len(X_train)} train, {len(X_val)} val")
        print(f"   Class balance: {n_pos} pos ({100*n_pos/len(y_train):.1f}%), {n_neg} neg")
        print(f"   Scale weight: {scale_pos_weight:.2f}")
        
        # Try multiple models and select best
        models = {}
        
        # XGBoost with aggressive tuning
        print("⏳ Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.15,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=0.15,
            reg_lambda=1.5,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            early_stopping_rounds=30
        )
        xgb_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
        models['xgb'] = xgb_model
        
        # Neural Network
        print("⏳ Training Neural Network...")
        nn_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42
        )
        nn_model.fit(X_train_scaled, y_train)
        models['nn'] = nn_model
        
        # Evaluate all models
        best_model = None
        best_auroc = 0
        best_name = ""
        model_aurocs = {}
        
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            auroc = roc_auc_score(y_val, y_pred_proba)
            model_aurocs[name] = float(auroc)
            if auroc > best_auroc:
                best_auroc = auroc
                best_model = model
                best_name = name
        
        print(f"🏆 Best model: {best_name.upper()} (AUROC: {best_auroc:.3f})")
        
        # Use best model for final predictions
        y_pred_proba = best_model.predict_proba(X_val_scaled)[:, 1]

        # Select threshold for the chosen operating objective.
        optimal_threshold = self._find_optimal_threshold(
            y_val,
            y_pred_proba,
            objective=threshold_objective,
        )
        
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        y_pred_default = (y_pred_proba >= 0.5).astype(int)
        
        # Metrics
        auroc = roc_auc_score(y_val, y_pred_proba)
        pr_auc = average_precision_score(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)
        accuracy_default = accuracy_score(y_val, y_pred_default)
        balanced_acc = balanced_accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        print(f"✅ AUROC: {auroc:.3f}")
        print(f"✅ PR-AUC: {pr_auc:.3f}")
        print(f"✅ Accuracy: {accuracy:.3f}")
        print(f"✅ Accuracy@0.50: {accuracy_default:.3f}")
        print(f"✅ Balanced Acc: {balanced_acc:.3f}")
        print(f"✅ F1 Score: {f1:.3f}")
        print(f"   Threshold ({threshold_objective}): {optimal_threshold:.3f}")

        # Persist per-model variants so serving can load both XGB and NN.
        for name, model in models.items():
            variant_bundle = {
                'model': model,
                'model_type': name,
                'scaler': self.scaler,
                'optimal_threshold': 0.5,
                'threshold_objective': 'fixed_0.5',
                'metrics': {
                    'auroc': model_aurocs.get(name, 0.0),
                },
            }
            variant_path = self.output_dir / f"{disease}_advanced_{name}_v1.0.0.pkl"
            joblib.dump(variant_bundle, variant_path)
            print(f"💾 Saved variant: {variant_path}")
        
        # Save
        model_path = self.output_dir / f"{disease}_advanced_v1.0.0.pkl"
        bundle = {
            'model': best_model,
            'model_type': best_name,
            'scaler': self.scaler,
            'optimal_threshold': optimal_threshold,
            'threshold_objective': threshold_objective,
            'metrics': {
                'auroc': auroc,
                'pr_auc': pr_auc,
                'accuracy': accuracy,
                'accuracy_at_0_50': accuracy_default,
                'balanced_accuracy': balanced_acc,
                'f1_score': f1,
            },
        }
        joblib.dump(bundle, model_path)
        print(f"💾 Saved: {model_path}")
        
        return bundle['metrics']


DEFAULT_MIMIC4_PATH = Path("dataset/mimic4/physionet.org/files/mimiciv/3.1")
DEFAULT_MIMIC4_MINI_PATH = Path("dataset/mimic4_mini/physionet.org/files/mimiciv/3.1")


def load_dataset_from_mimic4_mini(
    mimic_path: Path,
    output_csv: Path,
    max_patients: int = None,
    max_event_rows: int = 200000,
) -> pd.DataFrame:
    """Build a supervised training dataset from local MIMIC-IV files."""
    if not mimic_path.exists():
        raise FileNotFoundError(
            f"MIMIC-IV path not found: {mimic_path}. "
            "Provide --mimic-mini-path with a directory containing hosp/ and icu/."
        )

    print(f"\n📂 Building dataset from MIMIC-IV at: {mimic_path}")
    loader = MIMICDataLoader(str(mimic_path))
    dataset = loader.create_training_dataset(
        max_patients=max_patients,
        max_event_rows=max_event_rows,
        output_file=str(output_csv),
    )
    print(f"✅ Built dataset with {len(dataset)} rows")
    return dataset


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Optional prebuilt CSV path. If omitted, dataset is built from --mimic-mini-path.',
    )
    parser.add_argument(
        '--mimic-mini-path',
        type=str,
        default=str(DEFAULT_MIMIC4_PATH if DEFAULT_MIMIC4_PATH.exists() else DEFAULT_MIMIC4_MINI_PATH),
        help='Path to local MIMIC-IV folder containing hosp/ and icu/. Defaults to dataset/mimic4 when present, otherwise dataset/mimic4_mini.',
    )
    parser.add_argument(
        '--prepared-output',
        type=str,
        default='mimic4_mini_training_data.csv',
        help='CSV file path to write extracted MIMIC-IV mini training data.',
    )
    parser.add_argument(
        '--max-patients',
        type=int,
        default=None,
        help='Optional maximum ICU stays to sample while building dataset from MIMIC-IV mini. If omitted, use all available data.',
    )
    parser.add_argument(
        '--max-event-rows',
        type=int,
        default=1000000,
        help='Maximum event rows to read per source table while building features (higher uses more data, slower).',
    )
    parser.add_argument(
        '--threshold-objective',
        choices=['f1', 'accuracy', 'balanced_accuracy', 'youden'],
        default='f1',
        help='Objective used to choose the classification threshold.',
    )
    args = parser.parse_args()
    
    print("="*60)
    print("ADVANCED TRAINING PIPELINE - MAXIMUM ACCURACY")
    print("="*60)
    
    if args.data_path:
        print(f"\n📂 Loading data from {args.data_path}...")
        df = pd.read_csv(args.data_path)
        print(f"✅ Loaded {len(df)} samples from file")
    else:
        df = load_dataset_from_mimic4_mini(
            mimic_path=Path(args.mimic_mini_path),
            output_csv=Path(args.prepared_output),
            max_patients=args.max_patients,
            max_event_rows=args.max_event_rows,
        )
    
    # Feature engineering
    print("\n🔧 Advanced feature engineering...")
    feature_cols = ['age', 'gender', 'heart_rate', 'systolic_bp', 'diastolic_bp',
                   'temperature', 'respiratory_rate', 'wbc_count', 'hemoglobin',
                   'platelet_count', 'creatinine', 'bun', 'glucose', 'lactate']
    
    X_raw = df[feature_cols]
    X = AdvancedFeatureEngineer.engineer_features(X_raw)
    print(f"✅ Engineered {X.shape[1]} features (from {len(feature_cols)} base features)")
    
    # Train models
    diseases = ['sepsis', 'kidney_failure', 'diabetes',
                'anemia', 'thrombocytopenia', 'hypertension', 'mortality']
    
    trainer = AdvancedModelTrainer()
    results = {}

    use_predefined_splits = (
        'split' in df.columns and
        set(df['split'].dropna().unique()).issuperset({'train'})
    )
    if use_predefined_splits:
        train_mask = df['split'] == 'train'
        if (df['split'] == 'val').any():
            val_mask = df['split'] == 'val'
            eval_split_name = 'val'
        elif (df['split'] == 'test').any():
            val_mask = df['split'] == 'test'
            eval_split_name = 'test'
        else:
            use_predefined_splits = False
            eval_split_name = 'random'
    else:
        eval_split_name = 'random'

    if use_predefined_splits:
        print(f"\n📌 Using predefined split column for training/evaluation (train vs {eval_split_name})")
    else:
        print("\n📌 No usable predefined split found, falling back to random stratified split")
    
    for disease in diseases:
        y = df[disease].values
        
        if y.sum() < 20:
            print(f"\n⚠️  Skipping {disease}: insufficient cases")
            continue
        
        if use_predefined_splits:
            X_train = X.loc[train_mask]
            X_val = X.loc[val_mask]
            y_train = y[train_mask.values]
            y_val = y[val_mask.values]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.15, stratify=y, random_state=42
            )
        
        metrics = trainer.train_optimized_model(
            X_train,
            y_train,
            X_val,
            y_val,
            disease,
            threshold_objective=args.threshold_objective,
        )
        results[disease] = metrics
    
    # Summary
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE - RESULTS SUMMARY")
    print("="*60)
    
    for disease, metrics in results.items():
        print(f"{disease:20s} AUROC: {metrics['auroc']:.3f}  "
              f"PR-AUC: {metrics['pr_auc']:.3f}  "
              f"Acc: {metrics['accuracy']:.3f}  "
              f"F1: {metrics['f1_score']:.3f}")
    
    avg_auroc = np.mean([m['auroc'] for m in results.values()])
    avg_acc = np.mean([m['accuracy'] for m in results.values()])
    
    print(f"\n📊 Average Performance:")
    print(f"   AUROC: {avg_auroc:.3f}")
    print(f"   Accuracy: {avg_acc:.3f}")


if __name__ == "__main__":
    main()
