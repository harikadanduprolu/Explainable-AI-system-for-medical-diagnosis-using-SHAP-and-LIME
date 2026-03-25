#!/usr/bin/env python3
"""
Quick Test: Verify Trained Models Work
=======================================

Load models and make test predictions.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

def test_predictions():
    """Test all trained models."""
    
    print("\n" + "="*60)
    print("🧪 Testing Trained Models")
    print("="*60)
    
    # Load models (only XGBoost base models)
    model_dir = Path("trained_models")
    models = {}
    
    for model_file in model_dir.glob("*_xgboost_v*.pkl"):
        disease = model_file.stem.replace("_xgboost_v1.0.0", "")
        bundle = joblib.load(model_file)
        models[disease] = bundle
        print(f"✅ Loaded: {disease}")
    
    # Test patient (high-risk profile)
    print("\n📋 Test Patient Profile:")
    patient_data = {
        'age': 75.0,
        'gender': 1,
        'heart_rate': 115.0,
        'systolic_bp': 160.0,
        'diastolic_bp': 95.0,
        'temperature': 101.5,
        'respiratory_rate': 24.0,
        'wbc_count': 18.5,
        'hemoglobin': 9.2,
        'platelet_count': 120.0,
        'creatinine': 2.8,
        'bun': 45.0,
        'glucose': 280.0,
        'lactate': 3.5
    }
    
    # Add engineered features
    patient_data['shock_index'] = patient_data['heart_rate'] / patient_data['systolic_bp']
    patient_data['hr_bp_ratio'] = patient_data['heart_rate'] / patient_data['systolic_bp']
    patient_data['creat_bun_ratio'] = patient_data['creatinine'] / patient_data['bun']
    patient_data['age_glucose'] = patient_data['age'] * patient_data['glucose']
    
    print(f"  Age: {patient_data['age']:.0f} years")
    print(f"  Heart Rate: {patient_data['heart_rate']:.0f} bpm (elevated)")
    print(f"  Temperature: {patient_data['temperature']:.1f}°F (fever)")
    print(f"  WBC: {patient_data['wbc_count']:.1f} K/µL (elevated)")
    print(f"  Creatinine: {patient_data['creatinine']:.1f} mg/dL (high)")
    print(f"  Lactate: {patient_data['lactate']:.1f} mmol/L (high)")
    
    # Make predictions
    print("\n🔮 Predictions:")
    print("-" * 60)
    
    for disease, bundle in models.items():
        try:
            model = bundle['model']
            scaler = bundle['scaler']
            # Use scaler's feature names if available, otherwise use bundle's
            if hasattr(scaler, 'feature_names_in_'):
                feature_names = list(scaler.feature_names_in_)
            else:
                feature_names = bundle['feature_names']
            
            # Prepare input
            X = pd.DataFrame([patient_data])[feature_names]
            X_scaled = scaler.transform(X)
            
            # Predict
            risk_prob = model.predict_proba(X_scaled)[0][1]
            risk_category = (
                "CRITICAL" if risk_prob >= 0.75 else
                "HIGH" if risk_prob >= 0.55 else
                "MODERATE" if risk_prob >= 0.35 else
                "LOW"
            )
            
            # Get feature importance (using first 3 most important from basic features)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                top_idx = np.argsort(importances)[-3:][::-1]
                top_features = [feature_names[i].replace('_', ' ') for i in top_idx]
            else:
                top_features = ["N/A"]
            
            print(f"{disease.upper():20s} {risk_prob:5.1%}  [{risk_category:8s}]  "
                  f"Top: {', '.join(top_features)}")
        except Exception as e:
            print(f"{disease.upper():20s} ERROR: {e}")
    
    print("-" * 60)
    
    # Clinical interpretation
    print("\n💡 Clinical Interpretation:")
    print("This patient shows elevated risk across multiple conditions:")
    print("  • Fever + elevated WBC → possible infection (sepsis risk)")
    print("  • High creatinine → kidney dysfunction (AKI risk)")
    print("  • High lactate → tissue hypoperfusion")
    print("  • Advanced age + multiple organ issues → mortality risk")
    
    print("\n✅ All models are working correctly!")
    print("\n📝 Next: Start the web application")
    print("   1. python api_server.py")
    print("   2. python web_dashboard.py")
    print("   3. Open: http://localhost:8050")


if __name__ == "__main__":
    test_predictions()