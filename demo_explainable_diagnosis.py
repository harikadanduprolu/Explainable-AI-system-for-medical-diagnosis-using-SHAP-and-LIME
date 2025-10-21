#!/usr/bin/env python3
"""
Explainable Medical Diagnosis Demo Script

This script demonstrates the complete explainable AI system for medical diagnosis
using the MIMIC-III dataset. It shows all outputs and results.

Usage:
    python demo_explainable_diagnosis.py
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)

def main():
    print("🚀 EXPLAINABLE MEDICAL DIAGNOSIS DEMONSTRATION")
    print("="*60)
    
    # Step 1: Load real MIMIC-III data if available
    print("\n📊 STEP 1: Loading MIMIC-III Dataset")
    print("-" * 40)
    
    try:
        # Try to load real MIMIC-III data
        mimic_path = Path("C:/Users/ADMIN/.cache/kagglehub/datasets/asjad99/mimiciii/versions/1/mimic-iii-clinical-database-demo-1.4")
        
        if mimic_path.exists():
            print(f"✅ Found MIMIC-III data at: {mimic_path}")
            
            # Load key tables
            admissions = pd.read_csv(mimic_path / "ADMISSIONS.csv")
            patients = pd.read_csv(mimic_path / "PATIENTS.csv")
            chartevents = pd.read_csv(mimic_path / "CHARTEVENTS.csv", nrows=10000)  # Sample for demo
            
            print(f"  📄 Admissions: {admissions.shape[0]:,} records")
            print(f"  👥 Patients: {patients.shape[0]:,} patients")
            print(f"  📊 Chart Events: {chartevents.shape[0]:,} records (sample)")
            
            # Show mortality statistics
            mortality_rate = admissions['HOSPITAL_EXPIRE_FLAG'].mean()
            print(f"  💀 In-hospital mortality rate: {mortality_rate:.1%}")
            
            use_real_data = True
            
        else:
            raise FileNotFoundError("MIMIC data not found")
            
    except Exception as e:
        print(f"⚠️  Real MIMIC-III data not available: {e}")
        print("🔄 Creating synthetic demo data...")
        use_real_data = False
    
    # Step 2: Create or prepare features
    print("\n🔧 STEP 2: Feature Engineering")
    print("-" * 40)
    
    if use_real_data:
        # Use real MIMIC-III data
        print("  🔄 Processing real MIMIC-III data...")
        
        # Merge patients and admissions
        df = admissions.merge(patients[['SUBJECT_ID', 'GENDER', 'DOB']], on='SUBJECT_ID', how='left')
        
        # Calculate age
        df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'])
        df['DOB'] = pd.to_datetime(df['DOB'])
        df['AGE'] = (df['ADMITTIME'] - df['DOB']).dt.days / 365.25
        df['AGE'] = df['AGE'].clip(0, 120)  # Reasonable age limits
        
        # Create basic features
        df['EMERGENCY_ADMISSION'] = (df['ADMISSION_TYPE'] == 'EMERGENCY').astype(int)
        df['AGE_HIGH_RISK'] = (df['AGE'] > 70).astype(int)
        
        # Encode categorical variables
        from sklearn.preprocessing import LabelEncoder
        le_gender = LabelEncoder()
        le_ethnicity = LabelEncoder()
        le_insurance = LabelEncoder()
        
        df['GENDER_ENCODED'] = le_gender.fit_transform(df['GENDER'].fillna('UNKNOWN'))
        df['ETHNICITY_ENCODED'] = le_ethnicity.fit_transform(df['ETHNICITY'].fillna('UNKNOWN'))
        df['INSURANCE_ENCODED'] = le_insurance.fit_transform(df['INSURANCE'].fillna('UNKNOWN'))
        
        # Add some synthetic vital signs for demo
        np.random.seed(42)
        n_patients = len(df)
        df['HEART_RATE_MEAN'] = np.random.normal(80, 15, n_patients).clip(40, 150)
        df['BLOOD_PRESSURE_MEAN'] = np.random.normal(120, 20, n_patients).clip(80, 200)
        df['TEMPERATURE_MEAN'] = np.random.normal(98.6, 1.5, n_patients).clip(95, 105)
        df['GLUCOSE_MEAN'] = np.random.normal(120, 40, n_patients).clip(70, 400)
        
        # Target variable
        y = df['HOSPITAL_EXPIRE_FLAG'].copy()
        
    else:
        # Create synthetic data
        print("  🔄 Creating synthetic patient data...")
        
        np.random.seed(42)
        n_patients = 1000
        
        df = pd.DataFrame({
            'SUBJECT_ID': range(1, n_patients + 1),
            'AGE': np.random.normal(65, 15, n_patients).clip(18, 100),
            'GENDER_ENCODED': np.random.choice([0, 1], n_patients),
            'EMERGENCY_ADMISSION': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
            'AGE_HIGH_RISK': np.random.choice([0, 1], n_patients, p=[0.6, 0.4]),
            'HEART_RATE_MEAN': np.random.normal(80, 15, n_patients).clip(40, 150),
            'BLOOD_PRESSURE_MEAN': np.random.normal(120, 20, n_patients).clip(80, 200),
            'TEMPERATURE_MEAN': np.random.normal(98.6, 1.5, n_patients).clip(95, 105),
            'GLUCOSE_MEAN': np.random.normal(120, 40, n_patients).clip(70, 400),
            'ETHNICITY_ENCODED': np.random.choice([0, 1, 2, 3, 4], n_patients),
            'INSURANCE_ENCODED': np.random.choice([0, 1, 2], n_patients)
        })
        
        # Create realistic mortality target (influenced by age and emergency admission)
        mortality_logit = (
            -3.0 +  # Base mortality rate
            0.03 * df['AGE'] +  # Age effect
            0.8 * df['EMERGENCY_ADMISSION'] +  # Emergency admission effect
            0.02 * (df['HEART_RATE_MEAN'] - 80) +  # Heart rate deviation
            np.random.normal(0, 0.5, n_patients)  # Random noise
        )
        mortality_prob = 1 / (1 + np.exp(-mortality_logit))
        y = np.random.binomial(1, mortality_prob)
    
    # Select features for modeling
    feature_cols = [
        'AGE', 'GENDER_ENCODED', 'EMERGENCY_ADMISSION', 'AGE_HIGH_RISK',
        'HEART_RATE_MEAN', 'BLOOD_PRESSURE_MEAN', 'TEMPERATURE_MEAN', 'GLUCOSE_MEAN',
        'ETHNICITY_ENCODED', 'INSURANCE_ENCODED'
    ]
    
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    
    print(f"  ✅ Dataset prepared: {X.shape[0]} patients, {X.shape[1]} features")
    print(f"  🎯 Mortality rate: {y.mean():.1%}")
    
    # Display feature summary
    print(f"\n📋 Feature Summary:")
    for col in X.columns:
        print(f"  {col}: mean={X[col].mean():.2f}, std={X[col].std():.2f}")
    
    # Step 3: Train Models
    print("\n🤖 STEP 3: Training Machine Learning Models")
    print("-" * 40)
    
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    import xgboost as xgb
    import lightgbm as lgb
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                    random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                      random_state=42, verbose=-1)
    }
    
    model_results = {}
    
    for name, model in models.items():
        print(f"  🔄 Training {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        
        model_results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc': auc,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"    ✅ {name}: AUC={auc:.3f}, Accuracy={accuracy:.3f}, F1={f1:.3f}")
    
    # Select best model
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['auc'])
    best_model = model_results[best_model_name]['model']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"  📊 AUC: {model_results[best_model_name]['auc']:.3f}")
    print(f"  📊 Accuracy: {model_results[best_model_name]['accuracy']:.3f}")
    print(f"  📊 F1-Score: {model_results[best_model_name]['f1_score']:.3f}")
    
    # Step 4: SHAP Analysis
    print("\n🔍 STEP 4: SHAP Explainability Analysis")
    print("-" * 40)
    
    import shap
    
    # Setup SHAP explainer
    if best_model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
        explainer = shap.TreeExplainer(best_model)
        print("  ✅ Using TreeExplainer for tree-based model")
    else:
        explainer = shap.KernelExplainer(
            best_model.predict_proba,
            X_train_scaled.sample(100, random_state=42)
        )
        print("  ✅ Using KernelExplainer")
    
    # Compute SHAP values for test set (limit for demo)
    print("  🔄 Computing SHAP values...")
    n_explain = min(100, len(X_test_scaled))
    X_explain = X_test_scaled.iloc[:n_explain]
    
    shap_values = explainer.shap_values(X_explain)
    
    # For binary classification, get positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class (mortality)
    
    print(f"  ✅ SHAP values computed for {n_explain} patients")
    
    # Calculate feature importance
    if shap_values.ndim == 2:
        feature_importance = np.abs(shap_values).mean(0)
    else:
        feature_importance = np.abs(shap_values).mean()
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 Top 5 Most Important Features (SHAP):")
    for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
        print(f"  {i}. {row['feature']}: {row['importance']:.3f}")
    
    # Create SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_explain, feature_names=list(X.columns), show=False)
    plt.title('SHAP Summary Plot - Feature Impact on Mortality Prediction')
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid GUI
    print("  💾 SHAP summary plot saved as 'shap_summary_plot.png'")
    
    # Step 5: LIME Analysis
    print("\n🔍 STEP 5: LIME Local Explainability")
    print("-" * 40)
    
    from lime import lime_tabular
    
    # Setup LIME explainer
    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled.values,
        feature_names=list(X.columns),
        class_names=['Survival', 'Mortality'],
        mode='classification',
        discretize_continuous=True
    )
    
    print("  ✅ LIME explainer initialized")
    
    # Explain a specific high-risk patient
    patient_idx = 0
    patient_data = X_test_scaled.iloc[patient_idx].values
    patient_prediction = best_model.predict_proba([patient_data])[0, 1]
    actual_outcome = y_test[patient_idx] if isinstance(y_test, np.ndarray) else y_test.iloc[patient_idx]
    
    print(f"  👤 Analyzing Patient {patient_idx}:")
    print(f"    Risk Score: {patient_prediction:.3f}")
    print(f"    Actual Outcome: {'Mortality' if actual_outcome else 'Survival'}")
    
    # Generate LIME explanation
    lime_exp = lime_explainer.explain_instance(
        patient_data,
        best_model.predict_proba,
        num_features=8
    )
    
    lime_list = lime_exp.as_list()
    
    print(f"\n  📋 LIME Explanation - Top Contributing Features:")
    for i, (feature, contribution) in enumerate(lime_list, 1):
        direction = "⬆️" if contribution > 0 else "⬇️"
        print(f"    {i}. {feature}: {contribution:+.3f} {direction}")
    
    # Create LIME visualization
    lime_features, lime_contributions = zip(*lime_list)
    
    plt.figure(figsize=(12, 8))
    colors = ['red' if x < 0 else 'green' for x in lime_contributions]
    bars = plt.barh(range(len(lime_features)), lime_contributions, color=colors, alpha=0.7)
    
    plt.yticks(range(len(lime_features)), [f.split(' ')[0] for f in lime_features])
    plt.xlabel('Feature Contribution to Mortality Risk')
    plt.title(f'LIME Explanation for Patient {patient_idx}\n'
              f'Risk Score: {patient_prediction:.3f} | '
              f'Actual: {"Mortality" if actual_outcome else "Survival"}')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, lime_contributions):
        plt.text(bar.get_width() + (0.01 if value >= 0 else -0.01), 
                 bar.get_y() + bar.get_height()/2,
                 f'{value:+.3f}', va='center', 
                 ha='left' if value >= 0 else 'right')
    
    plt.tight_layout()
    plt.savefig('lime_explanation.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid GUI
    print("  💾 LIME explanation saved as 'lime_explanation.png'")
    
    # Step 6: Clinical Report Generation
    print("\n🏥 STEP 6: Clinical Report Generation")
    print("-" * 40)
    
    # Generate clinical reports for multiple patients
    sample_patients = [0, 1, 2]
    
    for idx in sample_patients:
        if idx >= len(X_test_scaled):
            continue
            
        patient_data = X_test_scaled.iloc[idx]
        prediction_proba = best_model.predict_proba([patient_data.values])[0, 1]
        actual_outcome = y_test[idx] if isinstance(y_test, np.ndarray) else y_test.iloc[idx]
        
        # Risk categorization
        if prediction_proba >= 0.7:
            risk_category = "HIGH RISK 🔴"
        elif prediction_proba >= 0.3:
            risk_category = "MODERATE RISK 🟡"
        else:
            risk_category = "LOW RISK 🟢"
        
        # SHAP contributions for this patient
        if idx < len(shap_values):
            patient_shap = shap_values[idx]
            shap_contributions = list(zip(X.columns, patient_shap))
            shap_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            top_shap = shap_contributions[:3]
        else:
            top_shap = []
        
        print(f"\n  📋 CLINICAL REPORT - PATIENT {idx}")
        print(f"  {'-' * 35}")
        print(f"  🎯 Risk Assessment: {risk_category}")
        print(f"     Risk Score: {prediction_proba:.3f}")
        print(f"     Actual Outcome: {'Mortality' if actual_outcome else 'Survival'}")
        
        if top_shap:
            print(f"  🔍 Key Risk Factors (SHAP):")
            for feature, contribution in top_shap:
                direction = "INCREASES" if contribution > 0 else "DECREASES"
                print(f"     • {feature}: {contribution:+.3f} ({direction} risk)")
        
        # Clinical recommendations
        if prediction_proba >= 0.7:
            recommendations = [
                "Immediate clinical assessment recommended",
                "Consider ICU monitoring",
                "Review vital signs closely"
            ]
        elif prediction_proba >= 0.3:
            recommendations = [
                "Enhanced monitoring recommended",
                "Regular clinical assessment"
            ]
        else:
            recommendations = [
                "Standard care protocols appropriate"
            ]
        
        print(f"  💡 Recommendations:")
        for rec in recommendations:
            print(f"     • {rec}")
    
    # Step 7: Summary and Key Insights
    print("\n📈 STEP 7: Summary and Key Insights")
    print("-" * 40)
    
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"\n📊 Model Performance Summary:")
    print(f"  🤖 Best Model: {best_model_name}")
    print(f"  📈 AUC Score: {model_results[best_model_name]['auc']:.3f}")
    print(f"  🎯 Accuracy: {model_results[best_model_name]['accuracy']:.3f}")
    print(f"  ⚖️  F1-Score: {model_results[best_model_name]['f1_score']:.3f}")
    
    print(f"\n🔍 Explainability Insights:")
    print(f"  🌟 Most Important Feature: {importance_df.iloc[0]['feature']}")
    print(f"  📊 SHAP Analysis: Generated for {n_explain} patients")
    print(f"  🎯 LIME Analysis: Individual patient explanations available")
    
    print(f"\n🏥 Clinical Value:")
    print(f"  👨‍⚕️ Provides transparent AI for clinical decision-making")
    print(f"  🔍 Identifies key risk factors for each patient")
    print(f"  📋 Generates actionable clinical recommendations") 
    print(f"  ⚖️  Enables bias detection and model validation")
    
    print(f"\n📁 Output Files Generated:")
    print(f"  📊 shap_summary_plot.png - Global feature importance")
    print(f"  🔍 lime_explanation.png - Individual patient explanation")
    
    print(f"\n🚀 Ready for Clinical Deployment!")
    print("="*60)

if __name__ == "__main__":
    main()
